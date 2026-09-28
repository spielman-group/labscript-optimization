"""The worker's message loop, driven through fake pipes.

The pipes zprocess attaches to a worker are attributes, so all but the last of
these tests put fakes in their place and call the child's entry point in this
process: they say what the protocol is without depending on zprocess starting
anything.
"""

from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from unittest.mock import patch

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
        self.inbox = queue.Queue()
        for item in items:
            self.inbox.put(item)
        self.sent = []
        self.condition = threading.Condition()

    def get(self, timeout=None):
        return self.inbox.get(timeout=timeout)

    def put(self, item):
        with self.condition:
            self.sent.append(item)
            self.condition.notify_all()

    def wait_sent(self, count):
        with self.condition:
            assert self.condition.wait_for(
                lambda: len(self.sent) >= count, timeout=5
            )


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
            i: {'pending': False, 'state': 'rejected'}
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


def driven(interface=FakeInterface):
    """A worker holding fake pipes, without starting a child."""
    worker = Worker(None, interface_factory=interface)
    worker.from_parent = Pipe()
    worker.to_parent = Pipe()
    return worker


def command(worker, name):
    worker.command_queue.put((name, None, None))


def run_headless(worker):
    from labscript_optimization.session import Session

    window = SimpleNamespace(
        ui=SimpleNamespace(show=lambda: None), update=lambda *args: None
    )
    application = SimpleNamespace(exit=lambda code: None)
    reader = threading.Thread(target=worker._read_requests, daemon=True)
    reader.start()
    with patch('qtutils.inmain_later', lambda fn, *args: fn(*args)):
        worker._run_session(Session, window, application)
    reader.join(timeout=10)
    assert not reader.is_alive()


def run(messages, interface=FakeInterface, *, start=True, inspect=None):
    worker = driven(interface)
    if inspect is not None:
        inspect(worker)
    requests = [
        (command, number, payload)
        for number, (command, payload) in enumerate(messages, start=1)
    ]

    def drive():
        if requests and requests[0][0] == 'configure':
            worker.from_parent.inbox.put(requests.pop(0))
            worker.to_parent.wait_sent(1)
            if start and worker.to_parent.sent[0][0] == 'status':
                command(worker, 'start')
        for request in requests:
            worker.from_parent.inbox.put(request)
        worker.from_parent.inbox.put(('quit', None, None))

    driver = threading.Thread(target=drive)
    driver.start()
    run_headless(worker)
    driver.join(timeout=10)
    assert not driver.is_alive()
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


def test_configuring_opens_a_paused_session(config_file):
    sent = run([('configure', config_file)], start=False)
    assert [kind for kind, _, _ in sent] == ['status']
    assert status_of(sent[0])['paused'] is True
    assert FakeInterface.instances[0].submitted == []


def test_start_fills_the_queue_without_a_routine_message(config_file):
    sent = run([('configure', config_file)])
    assert [kind for kind, _, _ in sent] == ['status']
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1']


def test_reset_discards_history_but_keeps_the_loaded_config(config_file):
    worker = driven()

    def drive():
        worker.from_parent.inbox.put(('configure', 1, config_file))
        worker.to_parent.wait_sent(1)
        command(worker, 'start')
        worker.from_parent.inbox.put(('shot', 2, None))
        worker.to_parent.wait_sent(2)

        Path(config_file).write_text('this is not TOML')

        command(worker, 'reset')
        worker.from_parent.inbox.put(('shot', 3, None))
        worker.to_parent.wait_sent(3)
        command(worker, 'start')
        worker.from_parent.inbox.put(('quit', None, None))

    driver = threading.Thread(target=drive)
    driver.start()
    run_headless(worker)
    driver.join(timeout=10)
    assert not driver.is_alive()

    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1']
    reset_status = status_of(worker.to_parent.sent[-1])
    assert reset_status['paused'] is True
    assert reset_status['submitted'] == 0
    assert FakeInterface.instances[1].submitted == ['shot-0', 'shot-1']


def test_an_observation_is_answered_before_the_next_shots_are_proposed(config_file):
    """The reply must not wait on a fit, or lyse waits with it."""
    sent = run(
        [('configure', config_file), ('observe', [('shot-0', 1.0, None, False)])]
    )
    assert [kind for kind, _, _ in sent] == ['status', 'status']
    assert status_of(sent[1])['completed'] == 1
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1', 'shot-2']


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


def test_a_status_message_frees_the_places_of_lost_shots(config_file):
    class LosesEverything(FakeInterface):
        def shot_status(self, shot_ids):
            return {
                i: {'pending': False, 'state': 'rejected'} for i in shot_ids
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
                [kind for kind, _, _ in workers[0].to_parent.sent]
            )
            return super().shot_status(shot_ids)

    workers = []
    run(
        [('configure', config_file), ('shot', None)],
        NotesTheOutbox,
        inspect=workers.append,
    )

    # Configuring has nothing awaiting to ask about, so the one question comes
    # on the second invocation -- by which time that invocation's reply, and
    # the first one's, had both already gone out.
    assert outbox_when_asked == [['status', 'status']]


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
    assert [kind for kind, _, _ in sent] == ['status', 'status']
    assert status_of(sent[-1])['stopped'] == 'stopped by an error'


class FailsOnStatus(FakeInterface):
    def shot_status(self, shot_ids):
        raise RuntimeError('runmanager went away')


@pytest.mark.parametrize(
    'limit',
    ['max_num_runs = 2', 'max_num_runs_without_better_params = 1'],
    ids=['the budget', 'the patience'],
)
def test_a_session_stopped_by_an_error_keeps_that_reason(tmp_path, limit):
    """The shots in flight when the error stopped the session go on reporting,
    and between them they reach a limit. The error is still why the run
    stopped, and the status goes on saying so.
    """
    path = tmp_path / 'config.toml'
    path.write_text(
        CONFIG.replace('num_buffered_runs = 2', f'num_buffered_runs = 2\n{limit}')
    )
    sent = run(
        [
            ('configure', str(path)),
            ('shot', None),
            ('observe', [('shot-0', 1.0, None, False), ('shot-1', 2.0, None, False)]),
        ],
        FailsOnStatus,
    )
    assert answers(sent) == [('status', 1), ('status', 2), ('error', 2), ('status', 3)]
    assert status_of(sent[-1])['completed'] == 2
    assert status_of(sent[-1])['stopped'] == 'stopped by an error'


def test_an_error_after_a_session_has_stopped_leaves_its_reason(tmp_path):
    """The session ran out of patience first, and the trailing work that failed
    after it is reported as the error it is -- but it is not why the run
    stopped.
    """
    path = tmp_path / 'config.toml'
    path.write_text(
        CONFIG.replace(
            'num_buffered_runs = 2',
            'num_buffered_runs = 3\nmax_num_runs_without_better_params = 1',
        )
    )
    sent = run(
        [
            ('configure', str(path)),
            ('observe', [('shot-0', 1.0, None, False), ('shot-1', 2.0, None, False)]),
            ('shot', None),
        ],
        FailsOnStatus,
    )
    # The shot still in flight is asked about after every reply, so each
    # request after the stop fails its trailing work again.
    assert answers(sent) == [
        ('status', 1),
        ('status', 2),
        ('error', 2),
        ('status', 3),
        ('error', 3),
    ]
    assert 'runmanager went away' in sent[2][2]
    assert status_of(sent[3])['stopped'] == (
        'no better parameters in 1 runs (max_num_runs_without_better_params)'
    )


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
    to_worker, _ = worker.start()
    to_worker.put(('quit', None, None))
    assert worker.child.wait(timeout=60) == 0
