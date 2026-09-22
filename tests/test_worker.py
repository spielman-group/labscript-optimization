"""The worker's message loop, driven through fake pipes.

The pipes zprocess attaches to a worker are attributes, so all but the last of
these tests put fakes in their place and call the child's entry point in this
process: they say what the protocol is without depending on zprocess starting
anything.
"""

import queue

import pytest

from labscript_optimization.worker import Worker

CONFIG = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "random"
num_buffered_runs = 2
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


class Pipe:
    """Stands in for a zprocess queue."""

    def __init__(self, items=()):
        self._queue = queue.Queue()
        for item in items:
            self._queue.put(item)
        self.sent = []

    def get(self, timeout=None):
        return self._queue.get_nowait()

    def put(self, item):
        self.sent.append(item)


class FakeInterface:
    instances = []

    def __init__(self, config):
        self.config = config
        self.submitted = []
        self.gone = set()
        FakeInterface.instances.append(self)

    def check_ready(self):
        pass

    def check_unchanged(self):
        pass

    def submit(self, proposals):
        ids = [f'shot-{len(self.submitted) + i}' for i in range(len(proposals))]
        self.submitted.extend(ids)
        return ids

    def shot_status(self, shot_ids):
        """The shape runmanager answers with: a verdict and a state per id."""
        return {
            i: {'pending': False, 'state': 'cancelled'}
            if i in self.gone
            else {'pending': True, 'state': 'running'}
            for i in shot_ids
        }


class RefusingInterface(FakeInterface):
    def check_ready(self):
        raise RuntimeError('runmanager reports an error in its globals')


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / 'config.toml'
    path.write_text(CONFIG)
    return str(path)


@pytest.fixture(autouse=True)
def clear_instances():
    FakeInterface.instances.clear()
    yield
    FakeInterface.instances.clear()


def driven(messages, interface=FakeInterface):
    """A worker holding fake pipes, its inbox already filled.

    ``messages`` are ``(command, payload)`` pairs, numbered from one in the
    order given the way the routine numbers its requests. Quitting is answered
    with nothing, so it carries no number.

    Nothing here starts a child, so the process tree is immaterial.
    """
    requests = [
        (command, number, payload)
        for number, (command, payload) in enumerate(messages, start=1)
    ]
    worker = Worker(None, interface_factory=interface)
    worker.from_parent = Pipe(requests + [('quit', None, None)])
    worker.to_parent = Pipe()
    return worker


def run(messages, interface=FakeInterface):
    worker = driven(messages, interface)
    worker.run()
    return worker.to_parent.sent


def status_of(message):
    """The status in one reply, ``('status', number, (recorded, status))``.

    ``recorded`` is one verdict per observation the request carried; the tests
    that are about that, and about the number, unpack it themselves.
    """
    _, _, (_, status) = message
    return status


def answers(sent):
    """What each message sent back is, and which request it belongs to."""
    return [(kind, number) for kind, number, _ in sent]


def test_configuring_fills_the_queue(config_file):
    sent = run([('configure', config_file)])
    assert [kind for kind, _, _ in sent] == ['status']
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1']


def test_an_observation_is_answered_before_the_next_shots_are_proposed(config_file):
    """The reply must not wait on a fit, or lyse waits with it."""
    sent = run(
        [('configure', config_file), ('observe', [('shot-0', 1.0, None, False)])]
    )
    assert [kind for kind, _, _ in sent] == ['status', 'status']
    assert status_of(sent[1])['completed'] == 1
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1', 'shot-2']


def test_an_observation_the_session_proposed_is_answered_as_taken(config_file):
    """The routine writes its results onto a shot on this answer and no other.

    runmanager mints a shot id for every queue row it compiles, so a user's own
    shot reaches the routine carrying one too. Whether the session proposed
    that id is the only thing that tells the two apart, and the session is the
    only thing that knows.
    """
    sent = run(
        [('configure', config_file), ('observe', [('shot-0', 1.0, None, False)])]
    )
    _, _, (recorded, _) = sent[-1]
    assert recorded == (True,)


def test_an_observation_the_session_never_proposed_is_answered_as_not_taken(
    config_file,
):
    """A user's own shot, engaged alongside the optimisation, carries an id
    runmanager minted for its queue row -- and the session still did not
    propose it, so nothing of the optimiser's belongs on it.
    """
    sent = run(
        [
            ('configure', config_file),
            ('observe', [('someone-elses-shot', 1.0, None, False)]),
        ]
    )
    _, _, (recorded, _) = sent[-1]
    assert recorded == (False,)


def test_every_observation_in_one_message_is_taken(config_file):
    """lyse hands the routine every shot it analysed in one batch, and all of
    them are runs this session spent. A message that carried two and recorded
    one would leave the other awaited until a reconcile dropped it.
    """
    sent = run(
        [
            ('configure', config_file),
            (
                'observe',
                [('shot-0', 1.0, None, False), ('shot-1', 2.0, None, False)],
            ),
        ]
    )
    assert status_of(sent[-1])['completed'] == 2


def test_one_verdict_comes_back_per_observation_in_the_order_sent(config_file):
    """The routine writes each shot's status into that shot's own file, so a
    single answer for a message carrying several would either write the
    optimiser's numbers onto somebody else's shot or leave one of its own
    without them.
    """
    sent = run(
        [
            ('configure', config_file),
            (
                'observe',
                [
                    ('someone-elses-shot', 1.0, None, False),
                    ('shot-1', 2.0, None, False),
                    ('another-of-theirs', 3.0, None, False),
                ],
            ),
        ]
    )
    _, _, (recorded, _) = sent[-1]
    assert recorded == (False, True, False)


def test_one_message_carrying_several_observations_is_answered_once(config_file):
    """One status per request, however many observations the request carried.

    The routine reads the first message carrying a request's number as the
    answer to it, and writes that one status onto every shot of the batch the
    session took. A verdict per observation in separate messages would leave
    the routine to collect a batch's answer a piece at a time, with a status
    apiece to choose between.
    """
    sent = run(
        [
            ('configure', config_file),
            (
                'observe',
                [('shot-0', 1.0, None, False), ('shot-1', 2.0, None, False)],
            ),
        ]
    )
    assert [kind for kind, _, _ in sent] == ['status', 'status']


def test_a_status_message_frees_the_places_of_lost_shots(config_file):
    class LosesEverything(FakeInterface):
        def shot_status(self, shot_ids):
            return {
                i: {'pending': False, 'state': 'cancelled'} for i in shot_ids
            }

    sent = run(
        [('configure', config_file), ('shot', None), ('shot', None)],
        LosesEverything,
    )
    # The reconciliation that dropped the first two shots ran after the second
    # invocation had already replied, so the count reaches the routine on the
    # third. A status is a report of finished work, not of work in progress.
    assert status_of(sent[-1])['dropped'] == 2
    # Having dropped them, it refilled rather than waiting on them for ever.
    assert FakeInterface.instances[0].submitted[:4] == [
        'shot-0',
        'shot-1',
        'shot-2',
        'shot-3',
    ]


def test_the_reply_is_sent_before_runmanager_is_asked_which_shots_remain(config_file):
    """Reconciling is a blocking round trip to runmanager, and the routine is
    blocked on this reply for as long as it takes -- up to the configured
    communication timeout if runmanager is busy or wedged. Asking before
    replying would hand lyse the very delay the worker exists to absorb, once
    per shot. Every question to runmanager must come after the reply.
    """
    outbox_when_asked = []

    class NotesTheOutbox(FakeInterface):
        def shot_status(self, shot_ids):
            outbox_when_asked.append(
                [kind for kind, _, _ in worker.to_parent.sent]
            )
            return super().shot_status(shot_ids)

    worker = driven([('configure', config_file), ('shot', None)], NotesTheOutbox)
    worker.run()

    # Configuring has nothing awaiting to ask about, so the one question comes
    # on the second invocation -- by which time that invocation's reply, and
    # the first one's, had both already gone out.
    assert outbox_when_asked == [['status', 'status']]


def test_every_reply_names_the_request_it_answers(config_file):
    """Order alone does not say which request an answer belongs to. The worker
    replies before the reconciling and submitting behind that reply, so by the
    time a reply crosses the pipe the routine may have sent two more requests
    and given up waiting for the answer to both.
    """
    sent = run(
        [
            ('configure', config_file),
            ('observe', [('shot-0', 1.0, None, False)]),
            ('shot', None),
        ]
    )
    assert answers(sent) == [('status', 1), ('status', 2), ('status', 3)]


def test_a_request_whose_handling_raises_is_answered_by_its_error_alone(
    config_file,
):
    """The routine reads the first message carrying a request's number as the
    answer to that request. One that failed before it could be replied to owes
    that number an error and nothing else: a request answered with neither
    would leave the routine waiting out its deadline on a worker that is alive
    and well, once per invocation for as long as the session lasted.
    """
    sent = run([('observe', [('shot-0', 1.0, None, False)])])
    assert answers(sent) == [('error', 1)]


def test_a_failure_behind_a_reply_carries_the_number_it_followed(config_file):
    """Reconciling and submitting run after the request they follow has been
    answered, so a failure in them is a second message for a request already
    replied to. Numbered as its own the routine would condemn the healthy
    request it is waiting on and leave that request's reply in the pipe.
    """

    class FailsOnSubmit(FakeInterface):
        def submit(self, proposals):
            raise RuntimeError('runmanager went away')

    sent = run([('configure', config_file), ('shot', None)], FailsOnSubmit)
    assert answers(sent) == [('status', 1), ('error', 1), ('status', 2)]


def test_a_runmanager_that_cannot_sustain_the_session_is_refused(config_file):
    sent = run([('configure', config_file)], RefusingInterface)
    kind, _, payload = sent[-1]
    assert kind == 'error'
    assert 'error in its globals' in payload


def test_a_shot_arriving_before_configuring_is_answered_with_nothing(config_file):
    """There is no session to report on yet, and the routine is waiting: an
    empty status is the answer, not an error and not silence.
    """
    assert run([('shot', None)]) == [('status', 1, ((), {}))]


def test_an_observation_before_configuring_is_an_error(config_file):
    sent = run([('observe', [('shot-0', 1.0, None, False)])])
    assert sent[-1][0] == 'error'
    assert 'before being configured' in sent[-1][2]


def test_an_unknown_command_is_an_error(config_file):
    sent = run([('configure', config_file), ('nonsense', None)])
    assert sent[-1][0] == 'error'


def test_a_failure_stops_the_session_proposing(config_file):
    """Carrying on past a runmanager that is not doing what it should would
    spend the run budget on shots nobody is counting. The error reaches the
    routine, and the session it stopped says so from then on.
    """

    class FailsOnSubmit(FakeInterface):
        def submit(self, proposals):
            raise RuntimeError('runmanager went away')

    sent = run([('configure', config_file), ('shot', None)], FailsOnSubmit)
    assert [kind for kind, _, _ in sent] == ['status', 'error', 'status']
    assert 'runmanager went away' in sent[1][2]
    assert status_of(sent[-1])['stopped'] == 'stopped by an error'


def test_quit_returns_without_replying(config_file):
    assert run([]) == []


def test_the_worker_starts_in_a_process_of_its_own(monkeypatch, tmp_path):
    """The one test that spawns anything: zprocess enters the child through a
    wrapper module of its own, which imports this class by name on the path
    the parent hands over. An import that does not resolve there looks like
    the child never connecting rather than like an import error, so the real
    thing is pinned here, where the message is plain.

    The path handed over is the parent's own ``sys.path``, not the child's
    working directory, which is what the run from a directory the package
    cannot be found from says.
    """
    import zprocess

    monkeypatch.chdir(tmp_path)
    worker = Worker(zprocess.ProcessTree(allow_insecure=True), startup_timeout=60)
    to_worker, from_worker = worker.start()
    try:
        to_worker.put(('shot', 7, None))
        assert from_worker.get(timeout=60) == ('status', 7, ((), {}))
    finally:
        to_worker.put(('quit', None, None))
        assert worker.child.wait(timeout=60) == 0
