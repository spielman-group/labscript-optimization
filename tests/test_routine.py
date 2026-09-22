"""Reading costs out of the lyse dataframe, and handing them to the worker.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is a column of its own, filled
for every shot runmanager compiled and empty for one of its default shots.

What the routine then does with those costs is the other half. lyse runs a
multishot routine once per drained batch of singleshot analyses, so an
invocation answers for every shot analysed since the one before it. Most of
those tests drive the entry point against a pair of fake pipes and a fake
process, so they say which shots the routine sends and in how many messages,
what it makes of the answer, what it writes back onto each shot, and how it
shuts the worker down, without depending on zprocess starting anything. The
last two drive it against the worker's own message loop in a thread, where
the replies come when the worker really sends them.
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
import numpy as np
import pandas as pd
import pytest

from labscript_optimization import config as config_module
from labscript_optimization import routine as routine_module
from labscript_optimization import runmanager_interface as interface_module
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
        with h5py.File(path, 'w') as f:
            # The least a shot file needs for lyse to read it into a dataframe
            # row: the globals group the row takes its columns from, and the
            # sequence it is indexed by.
            f.create_group('globals')
            f.attrs['sequence_id'] = '20260922T120000_optimisation'
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
    that. Each answer carries the number of the request it answers, so a status
    belonging to an earlier request is recognisable as one.

    The default answer takes no observation: a test about what is written onto
    a shot says so itself, with one verdict per observation the way the worker
    replies.
    """

    def __init__(self, from_worker, delay=0.02):
        super().__init__()
        self.from_worker = from_worker
        self.delay = delay
        #: Answers to send in place of the default, one ``(kind, payload)`` per
        #: request, in order, each under the number of the request it answers.
        self.replies = []

    def put(self, item):
        super().put(item)
        _, number, _ = item
        if self.replies:
            kind, payload = self.replies.pop(0)
        else:
            kind, payload = 'status', ((), {'answered': len(self.sent)})
        self.send(kind, number, payload)

    def send(self, kind, number, payload, delay=None):
        """Put one message under ``number`` into the pipe the routine reads.

        A delay of zero is a message already waiting when the routine next
        looks, which is what a worker coming out of an earlier request's
        trailing work leaves behind it.
        """
        message = (kind, number, payload)
        delay = self.delay if delay is None else delay
        if not delay:
            self.from_worker.incoming.put(message)
            return
        timer = threading.Timer(delay, self.from_worker.incoming.put, [message])
        # Nothing need wait at the end of a test for an answer nobody is
        # listening for any more.
        timer.daemon = True
        timer.start()


class Worker:
    """A worker process that exits only when signalled in a particular way.

    Records what it was sent, and whether anything waited for it once it had
    exited: a child nobody waits for is left a zombie. ``exited`` is what
    ``poll`` reports, and a worker nothing has waited for is running: a test
    about a worker that has died sets it.
    """

    def __init__(self, exits_on='quit'):
        self.exits_on = exits_on
        self.state = 'quit'
        self.signals = []
        self.reaped = False
        self.exited = False

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout=None):
        if self.state != self.exits_on:
            raise subprocess.TimeoutExpired('worker', timeout)
        self.reaped = True
        self.exited = True
        return 0

    def terminate(self):
        self.signals.append('terminate')
        self.state = 'terminate'

    def kill(self):
        self.signals.append('kill')
        self.state = 'kill'


def patch_spawned_worker(monkeypatch, to_worker, from_worker, child):
    """Make ``start_worker`` own the supplied pipe ends and child."""

    class SpawnedWorker:
        def __init__(self, *args, **kwargs):
            self.child = child

        def start(self):
            return to_worker, from_worker

    from labscript_optimization import worker as worker_module

    monkeypatch.setattr(worker_module, 'Worker', SpawnedWorker)


def test_start_worker_waits_for_and_consumes_the_configuration_reply(
    monkeypatch, tmp_path
):
    """A cold connection may take longer than one shot's reply allowance."""
    from_worker = Pipe()

    class Configuring(Pipe):
        def __init__(self):
            super().__init__()
            self.answered = threading.Event()

        def put(self, item):
            self.sent.append(item)

            def answer():
                from_worker.incoming.put(
                    (
                        'status',
                        routine_module.CONFIGURE_REQUEST,
                        ((), {'configured': True}),
                    )
                )
                self.answered.set()

            timer = threading.Timer(0.02, answer)
            timer.daemon = True
            timer.start()

    to_worker = Configuring()
    child = Worker()
    patch_spawned_worker(monkeypatch, to_worker, from_worker, child)
    monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', 0.001)
    monkeypatch.setattr(routine_module, 'configure_timeout', lambda: 0.2)
    handles = routine_module.start_worker(tmp_path / 'config.toml', object())

    assert handles == (to_worker, from_worker, child)
    assert to_worker.sent == [
        (
            'configure',
            routine_module.CONFIGURE_REQUEST,
            os.path.abspath(tmp_path / 'config.toml'),
        )
    ]
    assert to_worker.answered.is_set()
    assert from_worker.incoming.empty()


def test_start_worker_reaps_a_worker_that_rejects_its_configuration(
    monkeypatch, tmp_path
):
    from_worker = Pipe()

    class Rejecting(Pipe):
        def put(self, item):
            self.sent.append(item)
            if item[0] == 'configure':
                from_worker.incoming.put(
                    ('error', item[1], 'invalid configuration')
                )

    to_worker = Rejecting()
    child = Worker()
    patch_spawned_worker(monkeypatch, to_worker, from_worker, child)
    monkeypatch.setattr(routine_module, 'configure_timeout', lambda: 0.2)

    with pytest.raises(RuntimeError, match='invalid configuration'):
        routine_module.start_worker(tmp_path / 'config.toml', object())

    assert [command for command, _, _ in to_worker.sent] == [
        'configure',
        'quit',
    ]
    assert child.reaped


def test_start_worker_reaps_a_worker_that_does_not_configure(
    monkeypatch, tmp_path
):
    to_worker, from_worker, child = Pipe(), Pipe(), Worker()
    patch_spawned_worker(monkeypatch, to_worker, from_worker, child)
    monkeypatch.setattr(routine_module, 'configure_timeout', lambda: 0.01)

    with pytest.raises(TimeoutError, match='did not configure within'):
        routine_module.start_worker(tmp_path / 'config.toml', object())

    assert [command for command, _, _ in to_worker.sent] == [
        'configure',
        'quit',
    ]
    assert child.reaped


@pytest.fixture
def lab(monkeypatch):
    """labconfig as the routine reads it, in place of the workstation's.

    The real one reads whatever file this machine happens to have, so a test
    left with it would assert on the workstation rather than on the code. The
    dictionary is ``{(section, option): value}``, and a key not in it is one
    the lab has not set.
    """
    labconfig_module = pytest.importorskip('labscript_utils.labconfig')
    settings = {}

    class FakeLabConfig:
        def getfloat(self, section, option, fallback=None):
            return settings.get((section, option), fallback)

    monkeypatch.setattr(labconfig_module, 'LabConfig', FakeLabConfig)
    return settings


@pytest.mark.parametrize('set_by_the_lab', [None, 300.0])
def test_the_configure_deadline_covers_the_waits_inside_it(
    lab, monkeypatch, set_by_the_lab
):
    """The worker greets runmanager and then asks it questions, each of which
    the client waits ``communication_timeout`` for -- a minute where the lab
    has set nothing, and whatever the lab says where it has. A deadline below
    their sum fires first and reports a worker that was slow, when what
    happened is that runmanager stopped answering.

    The greeting is one of those waits. Left out of the sum, the worker is
    killed while the greeting it is inside still has time to run, and a
    runmanager that is not running is never named as the cause.

    The margin is held below the greeting's deadline so that no term of the
    sum is carried by another: at the margin the package ships, an allowance
    that had dropped the greeting's term would still cover the questions.
    """
    # BLACS's key, a different number for a different job: a lab that has set
    # it must not have it taken for the one the client waits.
    lab[('timeouts', 'liveness_timeout')] = 1.0
    if set_by_the_lab is not None:
        lab[('timeouts', 'communication_timeout')] = set_by_the_lab
    # What runmanager.remote.Client will wait, which is labconfig's number or
    # the fallback the client itself falls back to.
    client_waits = 60.0 if set_by_the_lab is None else set_by_the_lab
    monkeypatch.setattr(routine_module, 'CONFIGURE_MARGIN', 0.01)

    assert routine_module.configure_timeout() > (
        interface_module.GREETING_TIMEOUT
        + interface_module.CHECK_READY_REQUESTS * client_waits
    )


def test_a_runmanager_that_stops_answering_after_the_greeting_is_named(
    lab, monkeypatch, tmp_path
):
    """The greeting covers the first request and nothing after it. A
    runmanager whose GUI thread is inside a compile, or behind a modal dialog
    somebody left open, greets and then answers nothing, and the question that
    goes unanswered takes the client's full deadline to fail. The worker sends
    what that question raised, and the routine has to still be listening for
    the lab to read it.
    """
    client_waits = 0.2
    from_worker = Pipe()

    class Stalling(Pipe):
        """A worker whose runmanager greets and then goes quiet."""

        def put(self, item):
            self.sent.append(item)
            if item[0] != 'configure':
                return
            # The client gives up on its question at its own deadline, and the
            # worker sends on what it raised.
            timer = threading.Timer(
                client_waits,
                from_worker.incoming.put,
                [
                    (
                        'error',
                        item[1],
                        'Traceback (most recent call last):\n'
                        'TimeoutError: no response from runmanager',
                    )
                ],
            )
            timer.daemon = True
            timer.start()

    lab[('timeouts', 'communication_timeout')] = client_waits
    monkeypatch.setattr(interface_module, 'GREETING_TIMEOUT', 0.01)
    monkeypatch.setattr(routine_module, 'CONFIGURE_MARGIN', 0.01)
    to_worker, child = Stalling(), Worker()
    patch_spawned_worker(monkeypatch, to_worker, from_worker, child)

    with pytest.raises(RuntimeError, match='no response from runmanager'):
        routine_module.start_worker(tmp_path / 'config.toml', object())


@pytest.fixture
def session(monkeypatch, tmp_path):
    """A running session: a configuration file, and a worker made of fakes.

    ``worker`` is the end the routine writes to: what it was ``sent``, and the
    ``replies`` it answers with in place of its default.
    """
    path = tmp_path / 'optimisation_config.toml'
    path.write_text(CONFIG)
    from_worker = Pipe()
    to_worker = Answering(from_worker)
    child = Worker()
    handles = (to_worker, from_worker, child)
    monkeypatch.setattr(routine_module, 'start_worker', lambda config_path: handles)
    storage = types.SimpleNamespace()
    yield types.SimpleNamespace(
        storage=storage, path=path, worker=to_worker, child=child
    )
    # Leaves the atexit hook that optimise() registered with nothing to stop.
    storage.optimisation_worker = None


@pytest.fixture
def analysed(session, shot):
    """Run the routine on a session that has already seen a row of its sequence.

    Each call adds shots to the sequence lyse has analysed and invokes the
    routine on the whole of it, returning the status. The first invocation of
    a session hands nothing over -- it remembers where the sequence had got to
    and no further -- so everything about what reaches the worker starts from
    the second, and the priming invocation's message is cleared away here.
    """
    sequence = [shot(shot_id='before-this-session')]
    routine_module.optimise(session.path, session.storage, frame(sequence))
    session.worker.sent.clear()

    def analyse(*rows):
        sequence.extend(rows)
        return routine_module.optimise(
            session.path, session.storage, frame(sequence)
        )

    return analyse


@pytest.fixture
def box(session, shot, monkeypatch):
    """lyse's file box: the shots it has analysed, and what it was asked for.

    ``rows`` is the sequence, oldest first, and ``asked`` records the
    ``n_shots`` of each request, which is how a routine that fetches a pile-up
    whole is told from one that reads the end of it.
    """
    lyse = pytest.importorskip('lyse')
    sequence, asked = [], []

    def data(n_sequences=None, n_shots=None):
        assert n_sequences == 1, 'a run is one sequence'
        asked.append(n_shots)
        return frame(sequence[-n_shots:] if n_shots else sequence)

    monkeypatch.setattr(lyse, 'data', data)
    monkeypatch.setattr(lyse, 'routine_storage', session.storage)

    def add(*shot_ids):
        sequence.extend(shot(shot_id=shot_id) for shot_id in shot_ids)

    return types.SimpleNamespace(add=add, asked=asked)


def ids_sent(worker):
    """The shot ids of each message the routine sent, one list per message."""
    return [
        [observation[0] for observation in payload] if command == 'observe' else []
        for command, _, payload in worker.sent
    ]


def test_every_shot_analysed_since_the_last_invocation_is_handed_over(
    session, shot, analysed
):
    """lyse runs a multishot routine once per drained batch of singleshot
    analyses, not once per shot. Every shot in the batch ran and spent a run,
    so a routine that reads only the last one leaves the rest awaited until a
    reconcile drops them -- costs the learner never sees, counted against the
    run as lost.
    """
    analysed(shot(shot_id='row-1', cost=1.0), shot(shot_id='row-2', cost=2.0))
    assert ids_sent(session.worker) == [['row-1', 'row-2']]


def test_a_shot_already_handed_over_is_not_sent_again(session, shot, analysed):
    """The session would ignore a second cost for a shot it has already
    recorded, so this is invisible in what the optimisation does and plain in
    what crosses the pipe: every invocation would re-send the whole frame it
    can see, growing the message for as long as the run lasts.
    """
    analysed(shot(shot_id='row-1', cost=1.0))
    analysed(shot(shot_id='row-2', cost=2.0))
    assert ids_sent(session.worker) == [['row-1'], ['row-2']]


def test_several_observations_travel_in_one_message(session, shot, analysed):
    """The routine waits for one reply, and the worker answers one request at
    a time. Three messages would earn three replies, of which this invocation
    would read the first; the verdicts for the other two shots would arrive
    only on later invocations, leaving them unwritten until then and holding
    the worker to three rounds of reconciling where one would do.
    """
    analysed(*(shot(shot_id=f'row-{n}', cost=float(n)) for n in range(3)))
    assert len(session.worker.sent) == 1


def test_a_pile_up_larger_than_the_first_request_is_fetched_whole(session, box):
    """Analysis paused and resumed, or lyse started with shots already in the
    box, and a single invocation answers for a batch of any size. The routine
    asks for more until the row it handled last is in the frame; stopping at
    the first request would hand over the end of the batch and lose the rest.
    """
    box.add('row-0')
    routine_module.optimise(session.path)
    box.add('row-1', 'row-2', 'row-3', 'row-4')
    routine_module.optimise(session.path)
    assert ids_sent(session.worker)[-1] == ['row-1', 'row-2', 'row-3', 'row-4']
    assert box.asked == [1, 2, 4, 8]


def test_the_steady_state_costs_one_request(session, box):
    """Where analysis keeps up there is one new shot an invocation, and the
    frame that holds it and the row handled last is two rows. Asking for a
    third would be a second round trip to lyse for every shot of every run.
    """
    box.add('row-0')
    routine_module.optimise(session.path)
    box.add('row-1')
    routine_module.optimise(session.path)
    assert ids_sent(session.worker)[-1] == ['row-1']
    assert box.asked == [1, 2]


def test_the_first_invocation_hands_over_nothing_and_reads_one_row(session, box):
    """A session opens with whatever lyse already has in its box, which was
    analysed before the session existed. It proposed none of those shots, so
    it can make nothing of them -- and reaching back over a sequence hundreds
    of rows long to be told so would cost the whole frame to learn nothing.
    """
    box.add(*(f'row-{n}' for n in range(5)))
    routine_module.optimise(session.path)
    assert session.worker.sent == [('shot', 1, None)]
    assert box.asked == [1]


def test_a_frame_that_no_longer_reaches_the_handled_row_is_handed_over_whole(
    session, shot, analysed
):
    """A sequence that has run on further than the frame reaches, or a new one
    entirely, leaves lyse holding rows none of which is the one this routine
    handled last. Every one of them may be a run the session spent. Handing a
    row over twice is harmless -- the session takes a cost for a shot id once
    -- and skipping one is a run it never hears about at all.
    """
    analysed(shot(shot_id='row-1', cost=1.0))
    routine_module.optimise(
        session.path,
        session.storage,
        frame([shot(shot_id='row-2', cost=2.0), shot(shot_id='row-3', cost=3.0)]),
    )
    assert ids_sent(session.worker)[-1] == ['row-2', 'row-3']


def test_an_invocation_with_nothing_new_still_sends_one_message(
    session, shot, analysed
):
    """Reconciling and refilling happen in the worker's trailing work, after
    it has replied. A routine that returned early with nothing to report would
    stop the session giving up on shots that are not coming, and a generation
    an operator had unblocked would never be revived.
    """
    analysed(shot(shot_id='row-1', cost=1.0))
    analysed()
    assert [command for command, _, _ in session.worker.sent] == [
        'observe',
        'shot',
    ]


def test_a_shot_with_no_usable_cost_is_reported_as_a_bad_observation(
    session, analysed, shot
):
    """It has run and lyse has analysed it, so no cost for it is coming.

    Unreported, its id would be awaited until a reconcile quietly recorded it
    as dropped, and the run it spent would go uncounted.
    """
    analysed(shot(cost=float('nan')))
    (command, _, ((shot_id, _, _, bad),)), = session.worker.sent
    assert command == 'observe' and shot_id == 'row-3' and bad


def test_a_default_shot_is_passed_over_rather_than_handed_on(
    session, analysed, shot
):
    """One of runmanager's default shots, carrying no queue-row id. It is a row
    of the sequence like any other and the routine has handled it, but there is
    no id to send a cost under.
    """
    analysed(shot(shot_id=''), shot(shot_id='row-9'))
    assert ids_sent(session.worker) == [['row-9']]


def test_the_configuration_is_read_once_for_the_whole_session(
    session, analysed, shot
):
    """The worker keeps the configuration it was started with, so this must too.

    Re-reading it every shot lets an edit mid-session leave the two disagreeing
    about what the cost is: a flipped maximize would drive the search the wrong
    way with nothing said.
    """
    analysed(shot(cost=7.0))
    session.path.write_text(CONFIG.replace('maximize = true', 'maximize = false'))
    analysed(shot(cost=7.0))
    assert [
        observation[1]
        for _, _, payload in session.worker.sent
        for observation in payload
    ] == [-7.0, -7.0]


def test_the_status_returned_answers_this_shot_and_not_the_one_before(
    session, analysed, shot
):
    """It is written back as lyse results against the shots the routine was
    handed, so a status from the invocation before is a row of numbers
    belonging to other shots, and a glance that finds nothing yet is no row at
    all.
    """
    first = analysed(shot(cost=1.0))
    second = analysed(shot(cost=2.0))
    assert (first, second) == ({'answered': 1}, {'answered': 2})


def test_a_worker_that_failed_says_so_through_the_routine(session, analysed, shot):
    """The worker is a process of its own with nowhere to report to. Raising
    here is the only route by which its failure reaches a person: lyse shows
    what a routine raises.
    """
    session.worker.replies.append(
        ('error', 'Traceback (most recent call last):\nRuntimeError: no runmanager')
    )
    with pytest.raises(RuntimeError, match='no runmanager'):
        analysed(shot())


def test_a_worker_that_does_not_answer_in_time_is_given_up_on(
    session, analysed, shot, monkeypatch
):
    """lyse runs multishot routines inline and one at a time, so every moment
    spent waiting here delays the analysis of every shot behind this one. A
    worker that has stopped answering must cost one wait rather than the
    session.
    """
    monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', 0.05)
    session.worker.delay = 30.0
    started = time.monotonic()
    status = analysed(shot())
    assert status is None and time.monotonic() - started < 5.0


def test_a_worker_whose_child_has_exited_is_reported_as_dead(
    session, analysed, shot, monkeypatch
):
    """A reply that has not come yet means the worker is busy, not that it is
    gone: it answers before the reconciling and submitting behind the reply
    before, and that work is a whole population of files under generational
    submission. So the deadline cannot be what reports a death. Looking at the
    child between short waits is what tells the two apart, within about a
    second and saying which it was.
    """
    monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', 5.0)
    session.worker.delay = 30.0
    session.child.exited = True
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='died'):
        analysed(shot())
    assert time.monotonic() - started < 2.0


def test_a_failure_behind_an_earlier_reply_names_the_request_it_came_from(
    session, analysed, shot
):
    """Reconciling and submitting run after the reply they follow, so a
    failure in them is a second message for a request already answered, and it
    arrives while the routine is waiting on a later one. Read as that
    request's reply it condemns a healthy shot and sends whoever goes looking
    to the wrong invocation.
    """
    assert analysed(shot()) is not None
    _, answered, _ = session.worker.sent[-1]
    session.worker.send('error', answered, 'runmanager went away', delay=0)

    with pytest.raises(RuntimeError, match=f'request {answered}:'):
        analysed(shot())


def test_an_answer_still_on_its_way_is_left_for_the_next_shot(
    session, analysed, shot
):
    """Behind this invocation's answer only what is already waiting is swept up.

    That is how an error from the slow work behind an earlier reply arrives
    without being waited for. Waiting for more would hand lyse back the delay
    the worker exists to absorb, once per shot.
    """
    # The failure of the work behind an earlier reply, still on its way: the
    # next invocation raises it, this one is not held up for it.
    session.worker.send('error', 99, 'runmanager went away', delay=1.0)
    started = time.monotonic()
    assert analysed(shot()) == {'answered': 1}
    assert time.monotonic() - started < 0.5


def status(**overrides):
    """A status of the shape the session sends, with nothing found yet."""
    return {
        'phase': 'main',
        'submitted': 1,
        'completed': 0,
        'awaiting': 1,
        'dropped': 0,
        'blocked': 0,
        'starved': 0,
        'best_cost': None,
        'best_params': None,
        'best_shot_id': None,
        'stopped': None,
    } | overrides


def test_the_stand_in_status_carries_the_keys_the_session_sends(config):
    """Every test below reads the worker's reply out of that fixture, so a key
    the session gained and the fixture did not is a key nothing here ever sees
    the routine handle -- and save_status walks its keys by name.
    """
    from labscript_optimization.session import Session

    session = Session(config, None, learner=types.SimpleNamespace(last_phase='main'))
    assert set(status()) == set(session.status())


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


def test_the_status_is_written_onto_the_shot_as_lyse_results(
    session, analysed, shot, results
):
    """lyse turns each attribute into a column, so this is the whole point of
    computing a status: the lab reads it as ``df[('labscript_optimization',
    'best_cost')]`` alongside the shot it belongs to.
    """
    row = shot()
    session.worker.replies.append(
        (
            'status',
            (
                (True,),
                status(best_cost=7.0, best_params=[0.25], best_shot_id='row-3'),
            ),
        )
    )
    analysed(row)
    written = results(row)
    assert written['best_cost'] == 7.0
    assert list(written['best_params']) == [0.25]
    assert written['best_shot_id'] == 'row-3'
    assert written['phase'] == 'main'


def test_the_sessions_own_counters_are_not_written_onto_every_shot(
    session, analysed, shot, results
):
    """They are one answer for the whole run rather than anything about a shot,
    and an attribute overwritten shot after shot grows the file for nothing. A
    routine that wants them has them: optimise returns the whole status.
    """
    row = shot()
    session.worker.replies.append(('status', ((True,), status())))
    answer = analysed(row)
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


def test_a_value_the_session_does_not_have_yet_is_written_as_its_own_empty(
    session, analysed, shot, results
):
    """An h5 attribute cannot be None, so a key without a value needs a stand
    in, and lyse gives a column one dtype: the stand in has to be the type of
    the value it holds a place for. NaN is that only for the one key that is a
    float.
    """
    row = shot()
    session.worker.replies.append(('status', ((True,), status())))
    analysed(row)
    written = results(row)
    assert written['best_shot_id'] == ''
    assert written['stopped'] == ''
    assert list(written['best_params']) == []
    assert math.isnan(written['best_cost'])


def test_every_key_written_onto_a_shot_has_an_empty_to_stand_in_for_it(config):
    """save_status walks SHOT_RESULTS by name and looks each one up, so a key
    the routine gained without an empty is one that raises on the first shot
    the session has nothing to report it for -- which is every first shot.
    """
    assert set(routine_module.NO_VALUE_YET) == set(routine_module.SHOT_RESULTS)


@pytest.fixture
def lyse_column(shot, results):
    """Drive both halves of lyse that stand between a status and a column.

    ``dataframe_utilities`` is what turns the attributes of the shots already
    on disk into columns, and so what fixes a column's dtype. Then
    ``lyse.Run.save_result`` records the value it wrote in ``_updated_data``,
    the analysis subprocess hands that dict back to the file box, and
    ``FileBox.update_row`` sets each value with ``dataframe.at``. That
    assignment is the line the lab's traceback ends on. It is spelt out here
    rather than called because ``update_row`` is welded to the Qt model, but
    everything either side of it is lyse's own code.

    Its recovery is spelt out with it: lyse widens the column to ``object`` and
    retries when the assignment raises ``ValueError``, which is what a list
    into a float column raises, so leaving it out would fail a case lyse
    survives. It does not catch ``TypeError``, which is what a string into a
    float column raises, and that is the crash.

    Takes the number of shots the session has nothing to report on and the
    status it finally has, and returns the column they produce.
    """
    pytest.importorskip('lyse')
    from lyse import utils as lyse_utils
    from lyse.dataframe_utilities import get_dataframe_from_shots

    def build(empty_shots, reported):
        rows = [shot() for _ in range(empty_shots + 1)]
        for row in rows[:-1]:
            routine_module.save_status(row['filepath'], status())

        # The shots on disk when the last one is analysed. Its own row is
        # there -- lyse adds a row when the file appears -- and carries
        # nothing of the optimiser's yet.
        frame = get_dataframe_from_shots([r['filepath'] for r in rows])

        lyse_utils.worker.spinning_top = True
        lyse_utils.worker._updated_data = {}
        routine_module.save_status(rows[-1]['filepath'], status(**reported))
        updated = lyse_utils.worker._updated_data[rows[-1]['filepath']]

        depth = frame.columns.nlevels
        row_number = len(rows) - 1
        for (group, name), value in updated.items():
            column = (group, name) + ('',) * (depth - 2)
            try:
                frame.at[row_number, column] = value
            except ValueError:
                frame[column] = frame[column].astype('object')
                frame.at[row_number, column] = value
        return frame

    return build


def plain(value):
    """A column entry as the status wrote it.

    h5 gives a list back as an array, and a column of arrays holds arrays, so
    the one key whose value is a vector needs saying which it is before it can
    be compared with what was reported. Anything scalar is left alone, so a
    column holding the wrong type is reported by the assertion rather than
    raising here.
    """
    return list(value) if np.ndim(value) else value


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
    column = [plain(value) for value in frame[('labscript_optimization', key)]]
    assert column[-1] == reported
    assert column[:-1] == [empty] * empty_shots


def test_a_shot_carrying_no_identifier_is_not_written_to(session, analysed, shot):
    """One of runmanager's default shots. There is no id to send an observation
    under, so the session has nothing to take and nothing of the optimiser's
    belongs on the shot.
    """
    row = shot(shot_id=None)
    session.worker.replies.append(('status', ((), status())))
    analysed(row)
    with h5py.File(row['filepath'], 'r') as f:
        assert 'results' not in f


def test_a_shot_the_session_never_proposed_is_handed_over_and_not_written_to(
    session, analysed, shot
):
    """runmanager mints a shot id for every queue row it compiles, so a user's
    own shot, engaged alongside the optimisation, arrives carrying one exactly
    as the optimiser's do. The routine cannot tell them apart and does not try:
    it hands the shot over, and the session's answer -- that it took nothing --
    is what says nothing of the optimiser's belongs on it. Writing on the
    strength of the id alone puts a column of the optimiser's numbers onto
    somebody else's shot.
    """
    row = shot(shot_id='someone-elses-shot')
    session.worker.replies.append(('status', ((False,), status())))
    analysed(row)
    assert ids_sent(session.worker) == [['someone-elses-shot']]
    with h5py.File(row['filepath'], 'r') as f:
        assert 'results' not in f


def test_the_status_is_written_onto_each_shot_the_session_took(
    session, analysed, shot, results
):
    """A batch can hold the optimiser's shots and a user's own together, and
    the worker answers with a verdict for each in the order they were sent.
    One answer for the whole message could only write onto all of them or none.
    """
    ours, theirs = shot(shot_id='row-1'), shot(shot_id='someone-elses-shot')
    session.worker.replies.append(('status', ((True, False), status())))
    analysed(ours, theirs)
    assert results(ours)['phase'] == 'main'
    with h5py.File(theirs['filepath'], 'r') as f:
        assert 'results' not in f


def test_a_status_the_routine_gave_up_waiting_for_is_written_onto_its_own_shots(
    session, analysed, shot, results, monkeypatch
):
    """A reply that misses the deadline is neither lost nor the next
    invocation's. The shots its request handed over are remembered against
    that request's number, and its status is written onto them when it
    arrives, while the invocation that happens to read it writes its own
    answer onto its own shots.
    """
    monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', 0.05)
    waited, current = shot(shot_id='row-1'), shot(shot_id='row-2')

    session.worker.delay = 30.0
    assert analysed(waited) is None
    _, gave_up_on, _ = session.worker.sent[-1]

    # The worker comes out of the work behind an earlier reply and answers the
    # request it had in hand, an invocation late.
    session.worker.delay = 0.02
    session.worker.send(
        'status', gave_up_on, ((True,), status(best_shot_id='row-1')), delay=0
    )
    session.worker.replies.append(
        ('status', ((True,), status(best_shot_id='row-2')))
    )

    assert analysed(current)['best_shot_id'] == 'row-2'
    assert results(waited)['best_shot_id'] == 'row-1'
    assert results(current)['best_shot_id'] == 'row-2'


def test_a_status_that_cannot_be_written_does_not_stop_the_session(
    session, analysed, shot, capsys
):
    """The shot can go between the routine reading it and the status being
    written, because the routine waits for the worker in between. A progress
    report that cannot be saved is worth saying so about and nothing more.
    """
    pytest.importorskip('lyse')
    row = shot()
    session.worker.replies.append(('status', ((True,), status())))
    sending = session.worker.put
    session.worker.put = lambda item: (os.unlink(row['filepath']), sending(item))

    answer = analysed(row)

    assert answer == status()
    assert os.path.basename(row['filepath']) in capsys.readouterr().err


WORKER_CONFIG = """
[ANALYSIS]
cost_key = ["zTOF", "Nb"]
groups = ["G"]
[GENERAL]
learner = "random"
num_buffered_runs = 2
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""

#: Seconds the worker below spends writing shot files, once per refill, and
#: longer than the routine will wait for a reply. That is what a generation
#: submitted at once is: a population of files evaluated and written through
#: runmanager's GUI thread, many seconds by construction.
TRAILING_WORK = 0.5


class Link:
    """A pipe end joined to a worker that is really running.

    ``Pipe`` records what the routine sent and answers from a script. This one
    carries messages both ways, so the replies the routine reads are the ones
    the worker's own code sends, when it sends them.
    """

    def __init__(self):
        self.messages = queue.Queue()

    def put(self, item):
        self.messages.put(item)

    def get(self, timeout=None):
        try:
            return self.messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('get() timed out')


@pytest.fixture
def running(monkeypatch, tmp_path):
    """Start a session whose worker is the real message loop, in a thread.

    ``interface`` is what the worker turns the configuration into. The thread
    stands in for the child process, and the process handle beside it reports
    a worker that is alive, which is what a slow reply must not be mistaken
    for.
    """
    started = []

    def start(interface, reply_timeout=0.2):
        from labscript_optimization import worker as worker_module

        to_worker, from_worker, child = Link(), Link(), Worker()
        worker = worker_module.Worker(None, interface_factory=interface)
        worker.from_parent, worker.to_parent = to_worker, from_worker
        thread = threading.Thread(target=worker.run, daemon=True)

        class Spawned:
            def __init__(self, *args, **kwargs):
                self.child = child

            def start(self):
                thread.start()
                return to_worker, from_worker

        monkeypatch.setattr(worker_module, 'Worker', Spawned)
        monkeypatch.setattr(routine_module, 'REPLY_TIMEOUT', reply_timeout)
        # A deadline for configuring that a test can outrun if it has to, and
        # one these tests do not read the workstation's labconfig for.
        monkeypatch.setattr(routine_module, 'configure_timeout', lambda: 5.0)
        path = tmp_path / 'optimisation_config.toml'
        path.write_text(WORKER_CONFIG)
        storage = types.SimpleNamespace()
        started.append((storage, to_worker, thread))
        return types.SimpleNamespace(path=path, storage=storage, child=child)

    yield start
    for storage, to_worker, thread in started:
        # Leaves the atexit hook that optimise() registered with nothing to
        # stop, and lets the loop out of its wait for the next request.
        storage.optimisation_worker = None
        to_worker.put(('quit', None, None))
        thread.join(timeout=5)


def test_a_request_the_worker_could_not_handle_ends_the_wait(running, shot):
    """The error is that request's reply: the worker sends it in place of a
    status, and nothing else for that request is coming. A routine that held
    out for a status would wait out its deadline on a worker that is alive and
    well, and report it as slow. The fixture's deadline for configuring is the
    five seconds this beats.
    """

    class Refusing:
        def __init__(self, config):
            pass

        def check_ready(self):
            raise RuntimeError('runmanager reports an error in its globals')

    session = running(Refusing)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match='error in its globals'):
        routine_module.optimise(
            session.path, session.storage, frame([shot(shot_id='row-0')])
        )

    assert time.monotonic() - started < 2.0


def test_each_status_reaches_the_shot_that_earned_it_behind_slow_trailing_work(
    running, shot, results
):
    """The worker replies before it reconciles and submits, so a reply that
    misses the deadline says it is inside the previous request's trailing
    work, not that it is dead. Under generational submission that load does
    not let up: every invocation times out and every reply arrives an
    invocation or two after the shots it belongs to were handed over.

    Attributed by arrival, a verdict of ``True`` earned by the optimiser's own
    shot lands on whatever the invocation reading it is holding -- a user's
    shot engaged alongside the run gets a column of the optimiser's numbers,
    and the shots that earned them get none.
    """
    pytest.importorskip('lyse')

    class SlowToSubmit:
        """A runmanager that takes longer over a submission than the routine
        will wait for a reply.
        """

        def __init__(self, config):
            self.submitted = []

        def check_ready(self):
            pass

        def check_unchanged(self):
            pass

        def submit(self, proposals):
            time.sleep(TRAILING_WORK)
            ids = [
                f'shot-{len(self.submitted) + n}' for n in range(len(proposals))
            ]
            self.submitted.extend(ids)
            return ids

        def shot_status(self, shot_ids):
            return {i: {'pending': True, 'state': 'running'} for i in shot_ids}

    def has_results(row):
        with h5py.File(row['filepath'], 'r') as f:
            return 'results/labscript_optimization' in f

    session = running(SlowToSubmit)
    sequence = [shot(shot_id='before-this-session')]
    def invoke():
        return routine_module.optimise(
            session.path, session.storage, frame(sequence)
        )

    invoke()

    # Costs improving shot by shot, so that the best shot id in a status names
    # the shot whose request produced it.
    rows = {}
    for n, shot_id in enumerate(['shot-0', 'shot-1', 'someone-elses', 'shot-2']):
        rows[shot_id] = shot(shot_id=shot_id, cost=5.0 - n)
        sequence.append(rows[shot_id])
        invoke()

    ours = ['shot-0', 'shot-1', 'shot-2']
    for _ in range(40):
        if all(has_results(rows[shot_id]) for shot_id in ours):
            break
        # Nothing new to hand over, and the worker catches up behind it.
        invoke()

    assert [results(rows[shot_id])['best_shot_id'] for shot_id in ours] == ours
    with h5py.File(rows['someone-elses']['filepath'], 'r') as f:
        assert 'results' not in f


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
