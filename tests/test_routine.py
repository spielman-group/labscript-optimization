"""Reading a cost out of the lyse dataframe, and handing it to the worker.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is a column of its own, filled
for every shot runmanager compiled and empty for one of its default shots.

What the routine then does with that cost is the other half. Those tests drive
the entry point against a pair of fake pipes and a fake process, so they say
what the routine sends, what it makes of the answer, what it writes back onto
the shot, and how it shuts the worker down, without depending on zprocess
starting anything.
"""

import math
import os
import queue
import subprocess
import threading
import time
import types

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
[MLOOP]
session = "run-a"
[MLOOP_PARAMS.G.x]
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
    the deepest one and sorted, which is what lyse does. Two levels is the
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
    reads it. The file itself is there for the results written back onto it.
    """
    made = []

    def build(shot_id='row-3', cost=7.0, uncer=None, with_cost=True):
        path = tmp_path / f'shot{len(made)}.h5'
        made.append(path)
        with h5py.File(path, 'w'):
            pass
        row = {'filepath': str(path), 'shot_id': shot_id}
        if with_cost:
            row[('zTOF', 'Nb')] = cost
        if uncer is not None:
            row[('zTOF', 'u_Nb')] = uncer
        return row

    return build


def test_the_shot_id_says_which_proposal_the_shot_answers(config, shot):
    shot_id, _, _, _ = extract(frame([shot()]), config)
    assert shot_id == 'row-3'


def test_a_maximised_quantity_has_its_sign_flipped(config, shot):
    _, cost, _, bad = extract(frame([shot(cost=7.0)]), config)
    assert cost == -7.0 and not bad


def test_a_minimised_quantity_is_passed_through(shot):
    config = config_module.loads(CONFIG.replace('maximize = true', 'maximize = false'))
    _, cost, _, _ = extract(frame([shot(cost=7.0)]), config)
    assert cost == 7.0


def test_an_uncertainty_column_is_picked_up_when_present(config, shot):
    _, _, uncer, _ = extract(frame([shot(uncer=0.5)]), config)
    assert uncer == 0.5


def test_a_missing_uncertainty_is_absent_rather_than_zero(config, shot):
    _, _, uncer, _ = extract(frame([shot()]), config)
    assert uncer is None


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_shot_with_no_usable_cost_is_bad(config, shot, value):
    _, _, _, bad = extract(frame([shot(cost=value)]), config)
    assert bad


def test_the_most_recent_shot_is_the_one_read(config, shot):
    rows = [shot(shot_id='row-1', cost=1.0), shot(shot_id='row-2', cost=2.0)]
    shot_id, cost, _, _ = extract(frame(rows), config)
    assert shot_id == 'row-2' and cost == -2.0


def test_a_shot_carrying_no_identifier_has_nothing_to_read(config, shot):
    """One of runmanager's default shots. It goes to BLACS already compiled, so
    no queue-row id is ever written into it and no cost can be matched to a
    proposal by one.

    Empty rather than missing: lyse writes the column for every shot.
    """
    assert extract(frame([shot(shot_id='')]), config) is None


def test_a_dataframe_with_no_shot_id_column_yields_nothing(config):
    """A dataframe without the column claims nothing, rather than claiming
    every shot and matching costs to proposals at random.
    """
    assert extract(frame([{'filepath': '/p', ('zTOF', 'Nb'): 5.0}]), config) is None


def test_an_empty_dataframe_yields_nothing(config, shot):
    assert extract(frame([shot()]).iloc[0:0], config) is None


def test_a_cost_column_that_does_not_exist_yet_reads_as_bad(config, shot):
    shot_id, _, _, bad = extract(frame([shot(with_cost=False)]), config)
    assert shot_id == 'row-3' and bad


def test_a_shot_carrying_images_deepens_every_column_label(config, shot):
    """An image's attributes nest a level deeper than an analysis result does,
    and lyse pads every label out to the deepest, so the cost column is
    ``('zTOF', 'Nb', '')`` in such a sequence and ``('zTOF', 'Nb')`` in one
    whose shots have no images. Both name the cost.
    """
    row = shot()
    row[('side', 'atoms', 'exposure_time')] = 0.01
    shot_id, cost, _, _ = extract(frame([row]), config)
    assert shot_id == 'row-3' and cost == -7.0


class Pipe:
    """Stands in for a zprocess queue: a get waits up to its timeout.

    A timeout of zero is the poll zprocess makes of it, and an empty queue
    raises the TimeoutError zprocess raises.
    """

    def __init__(self):
        self.sent = []
        self.incoming = queue.Queue()

    def put(self, item):
        self.sent.append(item)

    def get(self, timeout=None):
        try:
            return self.incoming.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('get() timed out')


class Answering(Pipe):
    """The end the routine writes to, with a worker behind it that answers.

    The real worker answers over a socket from another process, so its reply is
    on its way while the routine is looking for it and is never already there.
    Answering from a thread, a moment later, is what tells a routine that waits
    for the reply apart from one that glances and takes whatever it finds: the
    second sees nothing on the first shot and the shot before's status after
    that. Each answer names the message it answers, so a status belonging to an
    earlier shot is recognisable as one.
    """

    def __init__(self, from_worker, delay=0.02):
        super().__init__()
        self.from_worker = from_worker
        self.delay = delay
        #: Answers to send in place of the default, one per message, in order.
        self.replies = []

    def put(self, item):
        super().put(item)
        if self.replies:
            reply = self.replies.pop(0)
        else:
            reply = ('status', (False, {'answered': len(self.sent)}))
        timer = threading.Timer(self.delay, self.from_worker.incoming.put, [reply])
        # Nothing need wait at the end of a test for an answer nobody is
        # listening for any more.
        timer.daemon = True
        timer.start()


class Worker:
    """A worker process that exits only when signalled in a particular way.

    Records what it was sent, and whether anything waited for it once it had
    exited: a child nobody waits for is left a zombie.
    """

    def __init__(self, exits_on='quit'):
        self.exits_on = exits_on
        self.state = 'quit'
        self.signals = []
        self.reaped = False

    def wait(self, timeout=None):
        if self.state != self.exits_on:
            raise subprocess.TimeoutExpired('worker', timeout)
        self.reaped = True
        return 0

    def terminate(self):
        self.signals.append('terminate')
        self.state = 'terminate'

    def kill(self):
        self.signals.append('kill')
        self.state = 'kill'


@pytest.fixture
def session(monkeypatch, tmp_path):
    """A running session: a configuration file, and a worker made of fakes.

    ``worker`` is the end the routine writes to: what it was ``sent``, and the
    ``replies`` it answers with in place of its default.
    """
    path = tmp_path / 'mloop_config.toml'
    path.write_text(CONFIG)
    from_worker = Pipe()
    to_worker = Answering(from_worker)
    handles = (to_worker, from_worker, Worker())
    monkeypatch.setattr(routine_module, 'start_worker', lambda config_path: handles)
    storage = types.SimpleNamespace()
    yield types.SimpleNamespace(storage=storage, path=path, worker=to_worker)
    # Leaves the atexit hook that optimise() registered with nothing to stop.
    storage.optimisation_worker = None


def test_only_the_shot_it_was_called_on_is_asked_of_lyse(session, shot, monkeypatch):
    """A run is one sequence, and it grows by a row every time the routine is
    called on one of its shots. Asking for the sequence would make each
    invocation cost more than the one before, for rows nothing reads: the
    routine reads the most recent shot and no other.
    """
    lyse = pytest.importorskip('lyse')
    asked = []

    def data(**kwargs):
        asked.append(kwargs)
        return frame([shot()])

    monkeypatch.setattr(lyse, 'data', data)
    monkeypatch.setattr(lyse, 'routine_storage', session.storage)

    routine_module.optimise(session.path)

    assert asked == [{'n_sequences': 1, 'n_shots': 1}]


def test_a_shot_with_no_usable_cost_is_reported_as_a_bad_observation(session, shot):
    """It has run and lyse has analysed it, so no cost for it is coming.

    Unreported, its id would be awaited until a reconcile quietly recorded it
    as dropped, and the run it spent would go uncounted.
    """
    routine_module.optimise(
        session.path, session.storage, frame([shot(cost=float('nan'))])
    )
    (command, (shot_id, _, _, bad)), = session.worker.sent
    assert command == 'observe' and shot_id == 'row-3' and bad


def test_the_configuration_is_read_once_for_the_whole_session(session, shot):
    """The worker keeps the configuration it was started with, so this must too.

    Re-reading it every shot lets an edit mid-session leave the two disagreeing
    about what the cost is: a flipped maximize would drive the search the wrong
    way with nothing said.
    """
    routine_module.optimise(session.path, session.storage, frame([shot(cost=7.0)]))
    session.path.write_text(CONFIG.replace('maximize = true', 'maximize = false'))
    routine_module.optimise(session.path, session.storage, frame([shot(cost=7.0)]))
    assert [payload[1] for _, payload in session.worker.sent] == [-7.0, -7.0]


def test_the_status_returned_answers_this_shot_and_not_the_one_before(session, shot):
    """It is written back as lyse results against the shot the routine ran on,
    so a status from the invocation before is a row of numbers belonging to
    another shot, and a glance that finds nothing yet is no row at all.
    """
    first = routine_module.optimise(
        session.path, session.storage, frame([shot(cost=1.0)])
    )
    second = routine_module.optimise(
        session.path, session.storage, frame([shot(cost=2.0)])
    )
    assert (first, second) == ({'answered': 1}, {'answered': 2})


def test_a_worker_that_failed_says_so_through_the_routine(session, shot):
    """The worker is a process of its own with nowhere to report to. Raising
    here is the only route by which its failure reaches a person: lyse shows
    what a routine raises.
    """
    session.worker.replies.append(
        ('error', 'Traceback (most recent call last):\nRuntimeError: no runmanager')
    )
    with pytest.raises(RuntimeError, match='no runmanager'):
        routine_module.optimise(session.path, session.storage, frame([shot()]))


def test_a_worker_that_does_not_answer_in_time_is_given_up_on(
    session, shot, monkeypatch
):
    """lyse runs multishot routines inline and one at a time, so every moment
    spent waiting here delays the analysis of every shot behind this one. A
    worker that has stopped answering must cost one wait rather than the
    session.
    """
    monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', 0.05)
    session.worker.delay = 30.0
    started = time.monotonic()
    status = routine_module.optimise(
        session.path, session.storage, frame([shot()])
    )
    assert status is None and time.monotonic() - started < 5.0


def test_an_answer_still_on_its_way_is_left_for_the_next_shot(session, shot):
    """Behind this shot's answer only what is already waiting is swept up.

    That is how an error from the slow work behind an earlier reply arrives
    without being waited for. Waiting for more would hand lyse back the delay
    the worker exists to absorb, once per shot.
    """
    straggler = threading.Timer(
        1.0,
        session.worker.from_worker.incoming.put,
        [('status', (False, {'answered': 99}))],
    )
    straggler.daemon = True
    straggler.start()
    status = routine_module.optimise(session.path, session.storage, frame([shot()]))
    assert status == {'answered': 1}


def status(**overrides):
    """A status of the shape the session sends, with nothing found yet."""
    return {
        'session': 'run-a',
        'phase': 'main',
        'submitted': 1,
        'completed': 0,
        'awaiting': 1,
        'dropped': 0,
        'starved': 0,
        'best_cost': None,
        'best_params': None,
        'best_shot_id': None,
        'stopped': None,
    } | overrides


@pytest.fixture
def results():
    """Read a shot's results back the way lyse reads them into its dataframe.

    Only the attributes of ``/results/<group>`` become columns, so reading
    them is also what says the status was saved where lyse will find it. lyse
    itself is an optional dependency -- the learners do not need the suite --
    so a checkout without it skips what reaches a shot file rather than
    passing on a write that never happened.
    """
    pytest.importorskip('lyse')
    from labscript_utils.properties import get_attributes

    def read(row):
        # Spelt out rather than read back from the module that wrote it: the
        # group name is the promise, df[('labscript_optimization', ...)].
        with h5py.File(row['filepath'], 'r') as f:
            return get_attributes(f['results/labscript_optimization'])

    return read


def test_the_status_is_written_onto_the_shot_as_lyse_results(session, shot, results):
    """lyse turns each attribute into a column, so this is the whole point of
    computing a status: the lab reads it as ``df[('labscript_optimization',
    'best_cost')]`` alongside the shot it belongs to.
    """
    row = shot()
    session.worker.replies.append(
        (
            'status',
            (True, status(best_cost=-7.0, best_params=[0.25], best_shot_id='row-3')),
        )
    )
    routine_module.optimise(session.path, session.storage, frame([row]))
    written = results(row)
    assert written['best_cost'] == -7.0
    assert list(written['best_params']) == [0.25]
    assert written['best_shot_id'] == 'row-3'
    assert written['phase'] == 'main'


def test_the_sessions_own_counters_are_not_written_onto_every_shot(
    session, shot, results
):
    """They are one answer for the whole run rather than anything about a shot,
    and an attribute overwritten shot after shot grows the file for nothing. A
    routine that wants them has them: optimise returns the whole status.
    """
    row = shot()
    session.worker.replies.append(('status', (True, status())))
    answer = routine_module.optimise(session.path, session.storage, frame([row]))
    # Spelt out rather than read back from the module that wrote them: these
    # names are the promise, df[('labscript_optimization', 'best_cost')].
    assert set(results(row)) == {
        'phase',
        'best_cost',
        'best_params',
        'best_shot_id',
        'stopped',
    }
    assert answer['starved'] == 0 and answer['submitted'] == 1


def test_a_value_the_session_does_not_have_yet_is_written_as_nan(
    session, shot, results
):
    """An h5 attribute cannot be None, and a column that changes type partway
    through a session is one lyse cannot plot.
    """
    row = shot()
    session.worker.replies.append(('status', (True, status())))
    routine_module.optimise(session.path, session.storage, frame([row]))
    written = results(row)
    assert all(
        math.isnan(written[key])
        for key in ('best_cost', 'best_params', 'best_shot_id', 'stopped')
    )


def test_a_shot_carrying_no_identifier_is_not_written_to(session, shot):
    """One of runmanager's default shots. There is no id to send an observation
    under, so the session has nothing to take and nothing of the optimiser's
    belongs on the shot.
    """
    row = shot(shot_id=None)
    session.worker.replies.append(('status', (False, status())))
    routine_module.optimise(session.path, session.storage, frame([row]))
    with h5py.File(row['filepath'], 'r') as f:
        assert 'results' not in f


def test_a_shot_the_session_never_proposed_is_not_written_to(session, shot):
    """runmanager mints a shot id for every queue row it compiles, so a user's
    own shot arrives carrying one exactly as the optimiser's do. Only the
    session knows which ids it proposed, and its answer is what says so:
    writing on the strength of the id alone puts a column of the optimiser's
    numbers onto somebody else's shot.
    """
    row = shot(shot_id='someone-elses-shot')
    session.worker.replies.append(('status', (False, status())))
    routine_module.optimise(session.path, session.storage, frame([row]))
    with h5py.File(row['filepath'], 'r') as f:
        assert 'results' not in f


def test_a_status_that_cannot_be_written_does_not_stop_the_session(
    session, shot, capsys
):
    """The shot can go between the routine reading it and the status being
    written, because the routine waits for the worker in between. A progress
    report that cannot be saved is worth saying so about and nothing more.
    """
    pytest.importorskip('lyse')
    row = shot()
    session.worker.replies.append(('status', (True, status())))
    sending = session.worker.put
    session.worker.put = lambda item: (os.unlink(row['filepath']), sending(item))

    answer = routine_module.optimise(session.path, session.storage, frame([row]))

    assert answer == status()
    assert 'shot0.h5' in capsys.readouterr().err


@pytest.fixture
def stopping():
    """Stop a session whose worker exits only when signalled the named way."""

    def stop(exits_on):
        worker = Worker(exits_on)
        storage = types.SimpleNamespace(optimisation_worker=(Pipe(), Pipe(), worker))
        routine_module.stop_worker(storage)
        return worker

    return stop


def test_a_worker_that_quits_when_asked_is_not_signalled(stopping):
    worker = stopping('quit')
    assert worker.signals == [] and worker.reaped


def test_a_worker_that_will_not_quit_is_terminated_and_then_reaped(stopping):
    worker = stopping('terminate')
    assert worker.signals == ['terminate'] and worker.reaped


def test_a_worker_that_survives_being_terminated_is_killed_and_then_reaped(stopping):
    worker = stopping('kill')
    assert worker.signals == ['terminate', 'kill'] and worker.reaped
