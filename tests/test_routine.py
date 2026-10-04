"""Reading costs out of the lyse dataframe, and saving the status back into it.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is a column of its own, filled
for every shot runmanager compiled and empty for one of its default shots.
"""

import contextlib
import sys
import types
import warnings

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


@pytest.fixture
def config():
    return config_module.loads(CONFIG)


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
    """Make a dataframe row with a real shot file behind it.

    lyse reads the identifier runmanager wrote into the file as a column, and
    an empty one for a shot that carries none, so the row is where the routine
    reads it. The file itself is there for lyse to read the row from, which
    ``lyse_column`` has it do.
    """
    made = []

    def build(shot_id='row-3', cost=7.0, uncer=None, with_cost=True):
        path = tmp_path / f'shot{len(made)}.h5'
        made.append(path)
        with h5py.File(path, 'w') as f:
            # The least a shot file needs for lyse to read it into a dataframe
            # row: the globals group the row takes its columns from, and the
            # sequence it is indexed by.
            f.create_group('globals')
            f.attrs['sequence_id'] = '20260922T120000_optimization'
        row = {'filepath': str(path), 'shot_id': shot_id}
        if with_cost:
            row[('zTOF', 'Nb')] = cost
        if uncer is not None:
            row[('zTOF', 'u_Nb')] = uncer
        return row

    return build


def test_the_shot_id_says_which_proposal_the_shot_answers(config, shot):
    _, [(shot_id, _, _, _)] = extract(frame([shot()]), config)
    assert shot_id == 'row-3'


def test_a_maximized_quantity_has_its_sign_flipped(config, shot):
    _, [(_, cost, _, bad)] = extract(frame([shot(cost=7.0)]), config)
    assert cost == -7.0 and not bad


def test_a_minimized_quantity_is_passed_through(shot):
    config = config_module.loads(CONFIG.replace('maximize = true', 'maximize = false'))
    _, [(_, cost, _, _)] = extract(frame([shot(cost=7.0)]), config)
    assert cost == 7.0


def test_an_uncertainty_column_is_picked_up_when_present(config, shot):
    _, [(_, _, uncer, _)] = extract(frame([shot(uncer=0.5)]), config)
    assert uncer == 0.5


def test_a_missing_uncertainty_is_absent_rather_than_zero(config, shot):
    _, [(_, _, uncer, _)] = extract(frame([shot()]), config)
    assert uncer is None


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_shot_with_no_usable_cost_is_bad(config, shot, value):
    _, [(_, _, _, bad)] = extract(frame([shot(cost=value)]), config)
    assert bad


def test_a_shot_carrying_no_identifier_has_nothing_to_read(config, shot):
    """One of runmanager's default shots. It goes to BLACS already compiled, so
    no queue-row id is ever written into it and no cost can be matched to a
    proposal by one.

    Empty rather than missing: lyse writes the column for every shot.
    """
    assert extract(frame([shot(shot_id='')]), config) == ([], [])


def test_a_dataframe_with_no_shot_id_column_yields_nothing(config):
    """A dataframe without the column claims nothing, rather than claiming
    every shot and matching costs to proposals at random.
    """
    shots = frame([{'filepath': '/p', ('zTOF', 'Nb'): 5.0}])
    assert extract(shots, config) == ([], [])


def test_a_cost_column_that_does_not_exist_yet_reads_as_bad(config, shot):
    _, [(shot_id, _, _, bad)] = extract(frame([shot(with_cost=False)]), config)
    assert shot_id == 'row-3' and bad


def test_a_shot_carrying_images_deepens_every_column_label(config, shot):
    """An image's attributes nest a level deeper than an analysis result does,
    and lyse pads every label out to the deepest, so the cost column is
    ``('zTOF', 'Nb', '')`` in such a sequence and ``('zTOF', 'Nb')`` in one
    whose shots have no images. Both name the cost.
    """
    row = shot()
    row[('side', 'atoms', 'exposure_time')] = 0.01
    _, [(shot_id, cost, _, _)] = extract(frame([row]), config)
    assert shot_id == 'row-3' and cost == -7.0


def test_the_shots_lyse_names_are_asked_of_its_dataframe_in_one_request(
    shot, monkeypatch
):
    """Their rows are asked of lyse's dataframe together, by file. A pass that
    names none has nothing to ask for, and lyse is not asked.
    """
    rows = [shot(shot_id='row-1'), shot(shot_id='row-2')]
    asked = []

    def data(where):
        asked.append(where)
        return frame(rows)

    monkeypatch.setattr(routine_module.lyse, 'data', data)
    paths = [row['filepath'] for row in rows]
    assert routine_module.analyzed([]) == []
    assert len(routine_module.analyzed(paths)) == 2
    assert asked == [{'filepath': paths}]


def test_lyse_s_unsorted_columns_are_read_without_a_warning(
    config, shot, monkeypatch
):
    """lyse keeps its columns in the order they were added, and pandas warns
    about lexsort depth when an unsorted MultiIndex is read by a shallower key,
    which would print into lyse's output as though the optimizer had failed.
    The frame is made deeper than the cost key, as a shot carrying images makes
    it, because a key as deep as the columns is looked up whole and does not
    warn."""
    row = shot(shot_id='row-1', cost=1.0)
    row[('image', 'raw', 'width')] = 1
    rows = frame([row])
    unsorted = rows[rows.columns[::-1]]
    monkeypatch.setattr(routine_module.lyse, 'data', lambda where: unsorted)
    with warnings.catch_warnings():
        warnings.simplefilter('error', pd.errors.PerformanceWarning)
        _, observations = extract(routine_module.analyzed([row['filepath']]), config)
    assert [o[0] for o in observations] == ['row-1']


def status(**overrides):
    """A status of the shape the session sends, with nothing found yet.

    It holds no phase: that is each shot's own, and travels with the verdict.
    """
    return {
        'submitted': 1,
        'completed': 0,
        'awaiting': 1,
        'dropped': 0,
        'blocked': 0,
        'starved': 0,
        'best_cost': None,
        'best_params': None,
        'best_shot_id': None,
        'paused': False,
        'pause_reason': None,
        'stopped': None,
    } | overrides


def test_the_stand_in_status_carries_the_keys_the_session_sends(config):
    """Every test below reads the worker's reply out of that fixture, so a key
    the session gained and the fixture did not is a key nothing here ever sees
    the routine handle -- and save_status walks its keys by name.
    """
    from labscript_optimization.session import Session

    session = Session(config, None, learner=types.SimpleNamespace())
    assert set(status()) == set(session.status())


@pytest.fixture
def results(monkeypatch):
    """Read back what was saved against a shot, as lyse's dataframe gets it.

    Inside lyse, ``save_result`` hands each value over in ``_updated_data``,
    by file and then by column, and lyse sets it into that shot's row, so
    reading it there is also what says the status was saved where lyse will
    find it. Outside lyse nothing is handed over, so this stands in for the
    analysis subprocess. A shot nothing was saved against reads as empty. lyse
    itself is an optional dependency -- the learners do not need the suite --
    so a checkout without it skips what reaches lyse rather than passing on a
    save that never happened.
    """
    pytest.importorskip('lyse')
    from lyse.utils import worker

    monkeypatch.setattr(worker, 'spinning_top', True)
    monkeypatch.setattr(worker, '_updated_data', {})

    def read(row):
        # Spelled out rather than read back from the module that wrote it: the
        # group name is the promise, df[('labscript_optimization', ...)].
        saved = worker._updated_data.get(row['filepath'], {})
        return {
            name: value
            for (group, name), value in saved.items()
            if group == 'labscript_optimization'
        }

    return read


def test_the_status_is_written_onto_the_shot_as_lyse_results(shot, results):
    """lyse sets each value saved into a column of the shot's row, so this is
    the whole point of computing a status: the lab reads it as
    ``df[('labscript_optimization', 'best_cost')]`` alongside the shot it
    belongs to.
    """
    row = shot()
    routine_module.save_status(
        row['filepath'],
        status(phase='warmup', best_cost=7.0, best_params=[0.25], best_shot_id='row-3'),
    )
    written = results(row)
    assert written['best_cost'] == 7.0
    assert written['best_params'] == [0.25]
    assert written['best_shot_id'] == 'row-3'
    assert written['phase'] == 'warmup'


def test_every_key_written_onto_a_shot_has_an_empty_to_stand_in_for_it(config):
    """save_status walks SHOT_RESULTS by name and looks each one up, so a key
    the routine gained without an empty is one that raises on the first shot
    the session has nothing to report it for -- which is every first shot.
    """
    assert set(routine_module.NO_VALUE_YET) == set(routine_module.SHOT_RESULTS)


def test_the_status_is_saved_to_the_dataframe_alone(monkeypatch):
    """The dataframe is where the status is read. Writing it into the shot
    file as well would open the file, and take its h5 lock, once for every
    shot the session took, inline in lyse.
    """
    calls = []

    class Run:
        def __init__(self, filepath):
            pass

        def set_group(self, group):
            pass

        def open(self, mode):
            calls.append(('open', mode))
            return contextlib.nullcontext()

        def save_result(self, name, value, **kwargs):
            calls.append(('save_result', name, kwargs))

    monkeypatch.setitem(sys.modules, 'lyse', types.SimpleNamespace(Run=Run))
    routine_module.save_status('shot.h5', status(phase='main'))
    assert calls == [
        ('save_result', name, {'save_to_h5': False})
        for name in routine_module.SHOT_RESULTS
    ]


@pytest.fixture
def lyse_column(shot, results):
    """Drive lyse from the statuses saved against a run's shots to the column
    they make.

    ``dataframe_utilities`` turns the shot files into rows, as lyse does when
    each file appears, and they carry nothing of the optimizer's: the status is
    saved to the dataframe alone. Then, a shot at a time,
    ``lyse.Run.save_result`` records each value in ``_updated_data``, the
    analysis subprocess hands that dict back to the file box, and
    ``FileBox.update_row`` sets each value with ``dataframe.at``. So the first
    shot's values make the columns and fix their dtypes, and every later value
    has to go into them. That assignment is the line the lab's traceback ends
    on. It is spelled out here rather than called because ``update_row`` is
    welded to the Qt model, but everything either side of it is lyse's own
    code.

    Its recovery is spelled out with it: when the assignment raises
    ``ValueError``, which is what a list raises, lyse makes the column if it is
    missing and widens it to ``object`` if not, and retries, so leaving it out
    would fail a case lyse survives. It does not catch ``TypeError``, which is
    what a string into a float column raises, and that is the crash.

    Takes the number of shots the session has nothing to report on and the
    status it finally has, and returns the column they produce.
    """
    from lyse.dataframe_utilities import get_dataframe_from_shots

    def build(empty_shots, reported):
        rows = [shot() for _ in range(empty_shots + 1)]
        frame = get_dataframe_from_shots([r['filepath'] for r in rows])
        depth = frame.columns.nlevels
        for row_number, row in enumerate(rows):
            # What a shot carries: the session's status, and its own phase.
            answer = reported if row_number == empty_shots else {}
            routine_module.save_status(row['filepath'], status(phase='main', **answer))
            for name, value in results(row).items():
                column = ('labscript_optimization', name) + ('',) * (depth - 2)
                try:
                    frame.at[row_number, column] = value
                except ValueError:
                    if column not in frame.columns:
                        frame.at[row_number, column] = None
                    else:
                        frame[column] = frame[column].astype('object')
                    frame.at[row_number, column] = value
        return frame

    return build


@pytest.mark.parametrize(
    'key, empty, reported',
    [
        ('stopped', '', 'reached max_num_runs (400)'),
        ('best_shot_id', '', '701dd468d82d4fc5b523ef69c0b2aaf2'),
        ('best_params', [], [0.25, 1.5]),
    ],
)
@pytest.mark.parametrize('empty_shots', [1, 3])
def test_the_shot_that_first_has_a_value_can_be_written_into_its_column(
    lyse_column, key, empty, reported, empty_shots
):
    """A run reaches its budget, or spends its first shots on costs the fits
    cannot use, and then has an answer. lyse fixes a column's dtype on the
    shots that came before, and writes the new value in with ``dataframe.at``,
    which refuses a value of another type. So the stand in written while there
    was nothing to report decides whether the shot that finally has something
    can be recorded at all.

    Three shots before the answer as well as one, because the lab saw this on
    a run whose opening shots all went to costs the fits could not use: the
    column is established across several rows before the value arrives.
    """
    frame = lyse_column(empty_shots, {key: reported})
    column = list(frame[('labscript_optimization', key)])
    assert column[-1] == reported
    assert column[:-1] == [empty] * empty_shots
