"""The worker's message loop, driven through fake pipes.

Exercised directly rather than through a spawned process, so these tests say
what the protocol is without depending on zprocess starting anything.
"""

import os
import queue
import subprocess
import sys

import pytest

from labscript_optimization.worker import serve

CONFIG = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP]
session = "s"
learner = "random"
num_buffered_runs = 2
[MLOOP_PARAMS.G.x]
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
        raise RuntimeError("runmanager's empty-queue policy is 'nothing'")


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


def run(messages, interface=FakeInterface):
    to_parent = Pipe()
    serve(Pipe(list(messages) + [('quit', None)]), to_parent, interface)
    return to_parent.sent


def test_configuring_fills_the_queue(config_file):
    sent = run([('configure', config_file)])
    assert [kind for kind, _ in sent] == ['status']
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1']


def test_an_observation_is_answered_before_the_next_shots_are_proposed(config_file):
    """The reply must not wait on a fit, or lyse waits with it."""
    sent = run(
        [('configure', config_file), ('observe', ('shot-0', 1.0, None, False))]
    )
    assert [kind for kind, _ in sent] == ['status', 'status']
    assert sent[1][1]['completed'] == 1
    assert FakeInterface.instances[0].submitted == ['shot-0', 'shot-1', 'shot-2']


def test_a_status_message_frees_the_places_of_lost_shots(config_file):
    class LosesEverything(FakeInterface):
        def shot_status(self, shot_ids):
            return {
                i: {'pending': False, 'state': 'cancelled'} for i in shot_ids
            }

    sent = run(
        [('configure', config_file), ('status', None), ('status', None)],
        LosesEverything,
    )
    # The reconciliation that dropped the first two shots ran after the second
    # invocation had already replied, so the count reaches the routine on the
    # third. A status is a report of finished work, not of work in progress.
    assert sent[-1][1]['dropped'] == 2
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
    to_parent = Pipe()
    outbox_when_asked = []

    class NotesTheOutbox(FakeInterface):
        def shot_status(self, shot_ids):
            outbox_when_asked.append([kind for kind, _ in to_parent.sent])
            return super().shot_status(shot_ids)

    serve(
        Pipe([('configure', config_file), ('status', None), ('quit', None)]),
        to_parent,
        NotesTheOutbox,
    )

    # Configuring has nothing awaiting to ask about, so the one question comes
    # on the second invocation -- by which time that invocation's reply, and
    # the first one's, had both already gone out.
    assert outbox_when_asked == [['status', 'status']]


def test_a_runmanager_that_cannot_sustain_the_session_is_refused(config_file):
    sent = run([('configure', config_file)], RefusingInterface)
    kind, payload = sent[-1]
    assert kind == 'error'
    assert 'empty-queue policy' in payload


def test_an_observation_before_configuring_is_an_error(config_file):
    sent = run([('observe', ('shot-0', 1.0, None, False))])
    assert sent[-1][0] == 'error'
    assert 'before being configured' in sent[-1][1]


def test_an_unknown_command_is_an_error(config_file):
    sent = run([('configure', config_file), ('nonsense', None)])
    assert sent[-1][0] == 'error'


def test_a_failure_stops_the_session_proposing(config_file):
    class FailsOnSubmit(FakeInterface):
        def submit(self, proposals):
            raise RuntimeError('runmanager went away')

    sent = run([('configure', config_file)], FailsOnSubmit)
    assert any(kind == 'error' for kind, _ in sent)


def test_quit_returns_without_replying(config_file):
    assert run([]) == []


def test_the_worker_runs_as_a_script():
    """zprocess executes the worker file directly, with no package around it.

    Relative imports fail there, and the failure looks like the child never
    connecting rather than like an import error, so it is pinned here where
    the message is plain.
    """
    from labscript_optimization.worker import __file__ as worker_file

    finished = subprocess.run(
        [sys.executable, worker_file],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, 'LABSCRIPT_NO_ERROR_DIALOG': '1'},
    )
    assert 'ZPROCESS_PARENTINFO' in finished.stderr, finished.stderr
    assert 'ImportError' not in finished.stderr, finished.stderr
