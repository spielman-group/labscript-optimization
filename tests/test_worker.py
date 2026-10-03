"""The session thread, driven in process with runmanager faked."""

import queue
import threading
from types import SimpleNamespace

import pytest

from labscript_optimization import config as config_module
from labscript_optimization import worker as worker_module
from labscript_optimization.worker import Worker

from conftest import SESSION_CONFIG as CONFIG


class FakeInterface:
    def __init__(self, config):
        self.submitted = []

    def check_ready(self):
        pass

    def check_unchanged(self):
        pass

    def submit(self, proposals):
        ids = [f'shot-{len(self.submitted) + i}' for i in range(len(proposals))]
        self.submitted.extend(ids)
        return ids

    def shot_status(self, shot_ids):
        return {i: {'pending': True, 'state': 'running'} for i in shot_ids}


class Shown(list):
    """The statuses the window was shown, which a test can wait on."""

    def __init__(self):
        super().__init__()
        self.changed = threading.Condition()

    def append(self, status):
        with self.changed:
            super().append(status)
            self.changed.notify_all()

    def wait_for(self, condition):
        with self.changed:
            assert self.changed.wait_for(lambda: any(map(condition, self)), timeout=5)


def start(interface=FakeInterface, text=CONFIG):
    """A worker whose opening has been handled, and what its window was shown."""
    shown = Shown()
    window = SimpleNamespace(update=lambda status, *args: shown.append(status))
    commands = queue.Queue()
    worker = Worker(config_module.loads(text), window, commands, interface)
    # Answered after the opening, which is ahead of it in the queue.
    worker.hand_over([], [], None)
    return worker, commands, shown


def hand_over(worker, *shot_ids):
    """Hand over a cost for each id, and return the files whose status was saved."""
    saved = []
    worker.hand_over(
        [f'{i}.h5' for i in shot_ids],
        [(i, 1.0, None, False) for i in shot_ids],
        lambda filepath, status: saved.append((filepath, status['phase'])),
    )
    return saved


def test_a_session_opens_paused_and_start_fills_the_queue():
    worker, commands, shown = start()
    assert shown[-1]['paused'] and shown[-1]['submitted'] == 0
    commands.put(('start', None, None))
    hand_over(worker)
    assert shown[-1]['submitted'] == 2


def test_only_the_shots_the_session_proposed_get_their_status():
    worker, commands, _ = start()
    commands.put(('start', None, None))
    assert hand_over(worker, 'shot-0', 'someone-elses') == [('shot-0.h5', 'main')]


def test_a_reply_later_than_the_deadline_is_saved_by_the_next_hand_over(monkeypatch):
    """The reply precedes the runmanager round trips behind it, and a request
    queued behind slow ones is answered when they finish."""
    monkeypatch.setattr(worker_module, 'REPLY_TIMEOUT', 0.2)
    release = threading.Event()

    class Slow(FakeInterface):
        def shot_status(self, shot_ids):
            release.wait(timeout=10)
            return super().shot_status(shot_ids)

    worker, commands, _ = start(Slow)
    commands.put(('start', None, None))
    assert hand_over(worker, 'shot-0') == [('shot-0.h5', 'main')]
    assert hand_over(worker, 'shot-1') == []
    release.set()
    assert hand_over(worker) == [('shot-1.h5', 'main')]


def test_a_failure_after_a_reply_stops_the_session_and_is_raised_next():
    class FailsOnStatus(FakeInterface):
        def shot_status(self, shot_ids):
            raise RuntimeError('runmanager went away')

    worker, commands, shown = start(FailsOnStatus)
    commands.put(('start', None, None))
    # Raised by this hand-over if it has already failed by its end, else by the next.
    with pytest.raises(RuntimeError, match='runmanager went away'):
        hand_over(worker, 'shot-0')
        hand_over(worker)
    shown.wait_for(lambda status: status.get('stopped') == 'stopped by an error')


def test_a_failed_opening_leaves_the_worker_up_and_reset_retries():
    attempts = []

    class AbsentAtFirst(FakeInterface):
        def check_ready(self):
            attempts.append(None)
            if len(attempts) == 1:
                raise RuntimeError('runmanager did not answer')

    worker, commands, shown = start(AbsentAtFirst)
    assert shown[-1] == {'stopped': 'runmanager did not answer'}
    assert hand_over(worker, 'shot-0') == []
    commands.put(('reset', None, None))
    commands.put(('start', None, None))
    hand_over(worker)
    assert shown[-1]['submitted'] == 2


def test_a_session_thread_that_has_died_is_reported(monkeypatch):
    monkeypatch.setattr(worker_module, 'REPLY_TIMEOUT', 0.2)

    def update(status, *args):
        raise RuntimeError('the window is broken')

    window = SimpleNamespace(update=update)
    worker = Worker(config_module.loads(CONFIG), window, queue.Queue(), FakeInterface)
    worker.thread.join(timeout=10)
    with pytest.raises(RuntimeError, match='thread has stopped'):
        hand_over(worker)


def test_a_session_that_reaches_a_limit_says_why_once(capsys):
    text = CONFIG.replace(
        'num_buffered_runs = 2', 'num_buffered_runs = 2\nmax_num_runs = 2'
    )
    worker, commands, _ = start(text=text)
    commands.put(('start', None, None))
    hand_over(worker, 'shot-0', 'shot-1')
    hand_over(worker)
    assert capsys.readouterr().out.count('The optimization has stopped') == 1
