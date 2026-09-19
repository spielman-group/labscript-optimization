"""Reading a cost out of the lyse dataframe, and handing it to the worker.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is not a column at all: lyse
does not carry it into the dataframe, so it is read from the shot file that
``filepath`` names.

What the routine then does with that cost is the other half. Those tests drive
the entry point against a pair of fake pipes and a fake process, so they say
what the routine sends, what it makes of the answer, and how it shuts the
worker down, without depending on zprocess starting anything.
"""

import queue
import subprocess
import threading
import time
import types

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
    """Make a dataframe row with a real shot file behind it."""
    made = []

    def build(shot_id='row-3', cost=7.0, uncer=None, with_cost=True):
        path = tmp_path / f'shot{len(made)}.h5'
        made.append(path)
        with h5py.File(path, 'w') as f:
            if shot_id is not None:
                f.attrs['shot_id'] = shot_id
        row = {'filepath': str(path)}
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


def test_a_shot_carrying_no_identifier_is_not_ours(config, shot):
    """A user's own shot, or one of runmanager's defaults, has no shot_id."""
    assert extract(frame([shot(shot_id=None)]), config) is None


def test_a_shot_whose_file_has_gone_is_not_ours(config, shot, tmp_path):
    row = shot()
    (tmp_path / 'shot0.h5').unlink()
    assert extract(frame([row]), config) is None


def test_a_dataframe_with_no_filepath_yields_nothing(config):
    assert extract(frame([{('zTOF', 'Nb'): 5.0}]), config) is None


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


@pytest.mark.parametrize('failure', [OSError, KeyError, RuntimeError, ValueError])
def test_an_unreadable_shot_file_is_not_ours(config, shot, monkeypatch, failure):
    """A damaged or half-written shot file cannot be identified either.

    h5py has several ways of saying so, and none of them is a reason to take
    the whole session down through lyse's error path.
    """

    def damaged(*args, **kwargs):
        raise failure('the shot file cannot be read')

    row = shot()
    monkeypatch.setattr(h5py, 'File', damaged)
    assert extract(frame([row]), config) is None


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
            reply = ('status', {'answered': len(self.sent)})
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
