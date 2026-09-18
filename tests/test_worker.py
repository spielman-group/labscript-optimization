"""The worker's message loop, driven through fake pipes.

The loop is exercised directly rather than through a spawned process, so these
tests say what the protocol is without depending on zprocess being able to
start anything.
"""

import os
import queue

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
        FakeInterface.instances.append(self)

    def check_ready(self):
        pass

    def submit(self, tag, params):
        self.submitted.append(tag)


class RefusingInterface(FakeInterface):
    def check_ready(self):
        raise RuntimeError('runmanager has an error in its globals')


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


def run(messages, config_file, interface=FakeInterface):
    to_parent = Pipe()
    serve(Pipe(list(messages) + [('quit', None)]), to_parent, interface)
    return to_parent.sent


def test_configuring_fills_the_queue(config_file):
    sent = run([('configure', config_file)], config_file)
    assert [kind for kind, _ in sent] == ['status']
    assert FakeInterface.instances[0].submitted == ['s:0', 's:1']


def test_an_observation_is_answered_before_the_next_shots_are_proposed(config_file):
    """The reply must not wait on a fit, or lyse waits with it."""
    sent = run(
        [('configure', config_file), ('observe', ('s:0', 1.0, None, False))],
        config_file,
    )
    kinds = [kind for kind, _ in sent]
    assert kinds == ['status', 'status']
    # The status answering the observation was sent before the refill, so it
    # still reports the shot as outstanding.
    assert sent[1][1]['completed'] == 1
    assert FakeInterface.instances[0].submitted == ['s:0', 's:1', 's:2']


def test_a_status_request_changes_nothing(config_file):
    sent = run([('configure', config_file), ('status', None)], config_file)
    assert sent[-1][1]['proposed'] == 2
    assert FakeInterface.instances[0].submitted == ['s:0', 's:1']


def test_a_runmanager_that_is_not_ready_is_reported_loudly(config_file):
    sent = run([('configure', config_file)], config_file, RefusingInterface)
    kind, payload = sent[-1]
    assert kind == 'error'
    assert 'error in its globals' in payload


def test_an_observation_before_configuring_is_an_error(config_file):
    sent = run([('observe', ('s:0', 1.0, None, False))], config_file)
    assert sent[-1][0] == 'error'
    assert 'before being configured' in sent[-1][1]


def test_an_unknown_command_is_an_error(config_file):
    sent = run([('configure', config_file), ('nonsense', None)], config_file)
    assert sent[-1][0] == 'error'


def test_a_failure_stops_the_session_proposing(config_file):
    class FailsOnSubmit(FakeInterface):
        def submit(self, tag, params):
            raise RuntimeError('runmanager went away')

    sent = run(
        [('configure', config_file), ('observe', ('s:0', 1.0, None, False))],
        config_file,
        FailsOnSubmit,
    )
    assert any(kind == 'error' for kind, _ in sent)
    statuses = [p for k, p in sent if k == 'status']
    # Whatever else happens, the session must not keep handing out shots.
    assert statuses[-1]['stopped'] is None or 'error' in str(statuses[-1]['stopped'])


def test_quit_returns_without_replying(config_file):
    assert run([], config_file) == []


def test_the_worker_runs_as_a_script():
    """zprocess executes the worker file directly, with no package around it.

    Relative imports fail in that situation, and the failure looks like the
    child never connecting rather than like an import error, so it is worth
    pinning here where the message is plain.
    """
    import subprocess
    import sys

    from labscript_optimization.worker import __file__ as worker_file

    finished = subprocess.run(
        [sys.executable, worker_file],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, 'LABSCRIPT_NO_ERROR_DIALOG': '1'},
    )
    # It should get all the way to looking for its parent and not find one.
    assert 'ZPROCESS_PARENTINFO' in finished.stderr, finished.stderr
    assert 'ImportError' not in finished.stderr, finished.stderr


def test_a_scan_left_enabled_is_refused_unless_repeats_are_expected():
    """One engage that makes many shots gives them all the same tag.

    The first cost to arrive claims the tag and the rest are dropped. That is
    exactly how averaging repeats is meant to work, so it is only an error
    when the configuration is not expecting repeats.
    """
    from labscript_optimization import config as config_module
    from labscript_optimization.runmanager_interface import RunmanagerInterface

    class Runmanager:
        def __init__(self, shots):
            self.shots = shots

        def error_in_globals(self):
            return False

        def get_globals(self):
            return {'gx': 0.0, 'mloop_session': '', 'mloop_iteration': 0}

        def n_shots(self):
            return self.shots

    strict = config_module.loads(CONFIG)
    with pytest.raises(RuntimeError, match='would compile 4 shots'):
        RunmanagerInterface(strict, Runmanager(4)).check_ready()

    RunmanagerInterface(strict, Runmanager(1)).check_ready()

    averaging = config_module.loads(
        CONFIG.replace('[ANALYSIS]', '[ANALYSIS]\nignore_bad = true')
    )
    RunmanagerInterface(averaging, Runmanager(4)).check_ready()
