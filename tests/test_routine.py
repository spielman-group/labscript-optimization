"""Reading costs out of the lyse dataframe, and saving the status back into it.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is a column of its own, filled
for every shot runmanager compiled and empty for one of its default shots.
"""

import itertools
import warnings
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

from labscript_optimization import config as config_module
from labscript_optimization import routine as routine_module
from labscript_optimization.routine import extract
from labscript_optimization.session import Session

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


@pytest.fixture
def routine():
    """A routine built without its window, worker or runmanager, as lyse
    allocates one before its ``__init__``.
    """
    routine = routine_module.OptimizationRoutine.__new__(
        routine_module.OptimizationRoutine
    )
    routine.path = routine.paths = None
    return routine


def test_a_pass_reads_its_own_files_whatever_order_lyse_keeps_its_columns_in(
    shot, routine, monkeypatch
):
    """lyse keeps its columns in the order they were added, and pandas warns
    about lexsort depth when an unsorted MultiIndex is read by a shallower key,
    which would print into lyse's output as though the optimizer had failed.
    The rows carry an image, which deepens every column label past the cost
    key, as it does in a sequence of shots with images.
    """
    rows = [shot(shot_id=f'row-{i}', cost=i + 1.0) for i in range(3)]
    for row in rows:
        row[('image', 'raw', 'width')] = 1
    lyse_rows = frame(rows)
    lyse_rows = lyse_rows[lyse_rows.columns[::-1]]
    monkeypatch.setattr(
        routine_module.lyse,
        'data',
        lambda filepath=None, where=None: lyse_rows[
            lyse_rows['filepath'].isin(where['filepath'])
        ],
    )
    handed = []
    routine.worker = SimpleNamespace(
        config=config_for(), hand_over=lambda *args: handed.append(args)
    )
    routine.paths = [rows[0]['filepath'], rows[1]['filepath']]
    with warnings.catch_warnings():
        warnings.simplefilter('error', pd.errors.PerformanceWarning)
        routine.run()
    [(filepaths, observations, _)] = handed
    assert filepaths == routine.paths
    assert [o[:2] for o in observations] == [('row-0', -1.0), ('row-1', -2.0)]


def status(**overrides):
    """What a shot carries: the session's own status, and the shot's phase."""
    session = Session(config_for(), None)
    return session.status() | {'phase': 'main'} | overrides


@pytest.fixture
def results(monkeypatch):
    """Read back what was saved against a shot, as lyse's dataframe gets it.

    Inside lyse, ``save_result`` hands each value over in ``_updated_data``,
    by file and then by column, and lyse sets it into that shot's row. Outside
    lyse nothing is handed over, so this stands in for the analysis
    subprocess.
    """
    pytest.importorskip('lyse')
    from lyse.utils import worker

    monkeypatch.setattr(worker, 'spinning_top', True)
    monkeypatch.setattr(worker, '_updated_data', {})

    def read(row):
        # The group name is the promise: df[('labscript_optimization', ...)].
        saved = worker._updated_data.get(row['filepath'], {})
        return {
            name: value
            for (group, name), value in saved.items()
            if group == 'labscript_optimization'
        }

    return read


def test_the_status_is_written_onto_the_shot_as_lyse_results_and_not_into_its_file(
    shot, routine, results
):
    """The lab reads it as ``df[('labscript_optimization', 'best_cost')]``
    alongside the shot it belongs to. The dataframe is where it is read:
    writing the file as well would take its h5 lock once for every shot the
    session took, inline in lyse.
    """
    row = shot()
    routine.save_status(
        row['filepath'],
        status(phase='warmup', best_cost=7.0, best_params=[0.25]),
    )
    assert results(row) == {
        'phase': 'warmup',
        'best_cost': 7.0,
        'best_params': [0.25],
    }
    with h5py.File(row['filepath'], 'r') as f:
        assert 'results' not in f


def test_the_shot_that_first_has_a_value_can_be_written_into_its_column(
    shot, routine, results, qt_application
):
    """A run spends its first shots on costs the fits cannot use, and then has
    an answer. The stand-in written while there was nothing to report fixes the
    column's type, and decides whether the shot that finally has something can
    be recorded at all. Each pass's saved results go into lyse's shots table
    through its ``update_rows``, as lyse hands them over.
    """
    import lyse.filebox
    from lyse.dataframe_utilities import get_dataframe_from_shots
    from lyse.utils import worker
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
    paths = [shot()['filepath'] for _ in range(4)]
    table.add_files(paths, get_dataframe_from_shots(paths), ['remote'] * len(paths))
    for number, path in enumerate(paths):
        # The last shot is the first with an answer.
        answer = {'best_params': [0.25, 1.5]} if number == 3 else {}
        routine.save_status(path, status(**answer))
        table.update_rows(worker._updated_data)
        worker._updated_data.clear()
    saved = list(table.dataframe[('labscript_optimization', 'best_params')])
    assert saved == [[]] * 3 + [[0.25, 1.5]]
