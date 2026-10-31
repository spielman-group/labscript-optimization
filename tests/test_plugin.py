"""Reading costs out of the lyse dataframe, and saving the status back into it.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is a column of its own, filled
for every shot runmanager compiled and empty for one of its default shots.
"""

import itertools
import time
from types import SimpleNamespace

try:
    # The lock lyse puts over h5py refuses to be imported once h5py has been,
    # and the routine writes its results through lyse. A test file that
    # reached for h5py first would make lyse unimportable in the test run and
    # nowhere else.
    import labscript_utils.h5_lock  # noqa: F401
except ImportError:
    # The suite is an optional dependency, and the tests that need it skip
    # themselves.
    pass

import h5py
import pandas as pd
import pytest

from conftest import SESSION_CONFIG, FakeInterface

from labscript_optimization import config as config_module
from labscript_optimization.plugin import Plugin, extract

CONFIG = """
[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
groups = ["G"]
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


def config_for(maximize=True):
    return config_module.loads(
        CONFIG.replace('maximize = true', f'maximize = {str(maximize).lower()}')
    )


def frame(rows):
    """Build a dataframe shaped the way lyse shapes one.

    Every column label is a tuple, padded with empty levels out to the depth of
    the deepest one, which is what lyse does, and sorted. Two levels is the
    shallowest it ever makes; a shot carrying images makes it deeper.
    """

    def label(key, depth):
        key = key if isinstance(key, tuple) else (key,)
        return key + ('',) * (depth - len(key))

    depth = max([2] + [len(k) for k in rows[0] if isinstance(k, tuple)])
    rows = [{label(k, depth): v for k, v in r.items()} for r in rows]
    columns = pd.MultiIndex.from_tuples(sorted(rows[0]))
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def shot(tmp_path):
    """Make a dataframe row with a real shot file behind it, for lyse to read.

    ``cost=None`` leaves out the cost column.
    """
    numbers = itertools.count()

    def build(shot_id='row-3', cost=7.0, uncer=None):
        path = tmp_path / f'shot{next(numbers)}.h5'
        with h5py.File(path, 'w') as f:
            # The least lyse needs to read a file into a dataframe row: the
            # globals group it takes columns from, and the sequence it indexes by.
            f.create_group('globals')
            f.attrs['sequence_id'] = '20260922T120000_optimization'
        row = {'filepath': str(path), 'shot_id': shot_id}
        if cost is not None:
            row[('zTOF', 'Nb')] = cost
        if uncer is not None:
            row[('zTOF', 'u_Nb')] = uncer
        return row

    return build


@pytest.mark.parametrize(
    'maximize, uncer, expected',
    [
        (True, 0.5, ('row-3', -7.0, 0.5, False)),
        (False, None, ('row-3', 7.0, None, False)),
    ],
    ids=['maximized, with an uncertainty', 'minimized, without one'],
)
def test_a_shot_is_read_as_the_proposal_it_answers_and_its_cost(
    shot, maximize, uncer, expected
):
    """A maximized cost has its sign flipped on the way in, so everything
    downstream minimizes, and an uncertainty that is missing is absent rather
    than zero.
    """
    row = shot(uncer=uncer)
    assert extract(frame([row]), config_for(maximize)) == (
        [row['filepath']],
        [expected],
    )


@pytest.mark.parametrize(
    'cost', [float('nan'), float('inf'), None], ids=['nan', 'inf', 'no column']
)
def test_a_shot_with_no_usable_cost_is_still_read_and_marked_bad(shot, cost):
    _, [(shot_id, _, _, bad)] = extract(frame([shot(cost=cost)]), config_for())
    assert shot_id == 'row-3' and bad


def test_a_shot_carrying_no_identifier_is_not_the_sessions_to_read(shot):
    """One of runmanager's default shots goes to BLACS already compiled, so no
    queue-row id is written into it: lyse has the column, and it is empty. A
    dataframe without the column claims nothing, rather than claiming every
    shot and matching costs to proposals at random.
    """
    assert extract(frame([shot(shot_id='')]), config_for()) == ([], [])
    shots = frame([{'filepath': '/p', ('zTOF', 'Nb'): 5.0}])
    assert extract(shots, config_for()) == ([], [])


def wait_until(qt_application, condition):
    """Process the GUI thread's events until the condition holds, as lyse would."""
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline
        qt_application.processEvents()
        time.sleep(0.01)


@pytest.fixture
def lyse_plugin(qt_application, tmp_path, monkeypatch):
    """The plugin set up as lyse sets it up, with lyse's real shots table and
    runmanager faked, and what lyse's output box was sent."""
    import lyse.filebox
    from qtutils.qt import QtWidgets

    # What the shots table uses of lyse's application, which a test cannot start.
    app = SimpleNamespace(
        qapplication=qt_application,
        output_box=SimpleNamespace(output=lambda text, red=False: None),
        filebox=SimpleNamespace(
            add_shots_start_time=None,
            fire_event=lambda *args: None,
            set_add_shots_progress=lambda *args: None,
        ),
    )
    exp_config = SimpleNamespace(getboolean=lambda *args: False)
    table = lyse.filebox.DataFrameModel(app, QtWidgets.QTableView(), exp_config)
    output = []
    plugin = Plugin()
    plugin.interface_factory = FakeInterface
    # The runmanager light would ask the lab's runmanager.
    monkeypatch.setattr(plugin.window.link, 'start', lambda: None)
    plugin.plugin_setup_complete(
        {
            'shots_model': table,
            'output_box': SimpleNamespace(
                output=lambda text, red=False: output.append(text)
            ),
        }
    )
    yield plugin, table, output
    plugin.close()


def test_the_configuration_file_is_chosen_saved_and_driven_remotely(
    lyse_plugin, qt_application, tmp_path
):
    """With no file chosen the session cannot open; a file chosen remotely is
    read, saved with lyse's configuration, and the session started from it."""
    plugin, _, _ = lyse_plugin

    def phase():
        return plugin.handle_get_status()['phase']

    wait_until(qt_application, lambda: phase().startswith('Ended: No configuration'))
    config_file = tmp_path / 'config.toml'
    config_file.write_text(SESSION_CONFIG)
    plugin.handle_set_config_file(config_file)
    wait_until(qt_application, lambda: phase() == 'Paused')
    assert plugin.get_save_data()['config_file'] == str(config_file)
    plugin.handle_start()
    wait_until(qt_application, lambda: phase() == 'Running')
    assert plugin.handle_get_status()['submitted'] == 2


def test_each_analysed_shot_the_session_proposed_gets_its_status_columns(
    lyse_plugin, qt_application, tmp_path
):
    """After all analysis of a shot, its cost goes to the session, and the
    shot's row gets what proposed it and the best so far, in lyse's dataframe
    and not in the file. The first shot's cost is unusable, so its stand-ins
    fix the columns' types before the next has values to put in them."""
    plugin, table, _ = lyse_plugin
    config_file = tmp_path / 'config.toml'
    config_file.write_text(SESSION_CONFIG)
    plugin.restore_save_data({'config_file': str(config_file)})
    plugin.handle_start()
    wait_until(qt_application, lambda: plugin.handle_get_status().get('submitted'))

    from lyse.dataframe_utilities import get_dataframe_from_shots

    paths = []
    for number, cost in enumerate([float('nan'), 2.0]):
        path = str(tmp_path / f'shot{number}.h5')
        with h5py.File(path, 'w') as f:
            f.create_group('globals')
            f.attrs['sequence_id'] = '20260922T120000_optimization'
            f.attrs['shot_id'] = f'shot-{number}'
            f.create_group('results/r').attrs['c'] = cost
        paths.append(path)
    table.add_files(paths, get_dataframe_from_shots(paths), ['runmanager'] * 2)
    phase = ('labscript_optimization', 'phase')
    for number, path in enumerate(paths):
        plugin.on_analysis_done(path, {})
        wait_until(
            qt_application,
            lambda: phase in table.dataframe and table.dataframe[phase][number] == 'main',
        )
    columns = table.dataframe['labscript_optimization']
    assert list(columns['phase']) == ['main', 'main']
    assert list(columns['best_cost'].isna()) == [True, False]
    assert columns['best_params'][0] == [] and len(columns['best_params'][1]) == 1
    with h5py.File(paths[1], 'r') as f:
        assert 'labscript_optimization' not in f['results']
