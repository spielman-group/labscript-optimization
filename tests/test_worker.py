"""The session thread, driven in process with runmanager faked."""

import queue
import threading
from types import SimpleNamespace

import numpy as np
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


def test_a_failure_after_a_reply_stops_the_session_and_is_raised_next(capsys):
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
    # Shown at once, in case no later pass comes to raise it.
    assert 'runmanager went away' in capsys.readouterr().err


def test_a_session_opens_paused_without_asking_runmanager_anything():
    class Absent(FakeInterface):
        def check_ready(self):
            raise TimeoutError('No response from server: timed out')

    _, _, shown = start(Absent)
    assert shown[-1]['paused'] and shown[-1]['pause_reason'] is None


def test_a_start_runmanager_refuses_leaves_the_session_paused_and_a_later_one_goes():
    refusals = ['runmanager reports an error in its globals']

    class Refuses(FakeInterface):
        def check_ready(self):
            if refusals:
                raise RuntimeError(refusals.pop())

    worker, commands, shown = start(Refuses)
    commands.put(('start', None, None))
    hand_over(worker)
    assert shown[-1]['paused'] and shown[-1]['submitted'] == 0
    assert shown[-1]['pause_reason'] == 'runmanager reports an error in its globals'
    commands.put(('start', None, None))
    hand_over(worker)
    assert not shown[-1]['paused'] and shown[-1]['submitted'] == 2
    assert shown[-1]['pause_reason'] is None


def test_runmanager_not_answering_mid_run_pauses_the_session_and_start_resumes_it(
    capsys,
):
    silent = [True]

    class Silent(FakeInterface):
        def shot_status(self, shot_ids):
            if silent:
                raise TimeoutError('No response from server: timed out')
            return super().shot_status(shot_ids)

    worker, commands, shown = start(Silent)
    commands.put(('start', None, None))
    # Neither hand-over raises: the failure is not one to report.
    hand_over(worker, 'shot-0')
    hand_over(worker)
    shown.wait_for(lambda s: s.get('pause_reason') == 'runmanager is not answering')
    assert shown[-1]['paused'] and shown[-1]['awaiting'] == 1
    assert capsys.readouterr() == ('', '')
    silent.clear()
    commands.put(('start', None, None))
    hand_over(worker)
    # The run goes on: the shot in flight is kept, and the queue topped up.
    assert not shown[-1]['paused'] and shown[-1]['submitted'] == 3


def test_a_session_thread_that_has_died_is_reported(monkeypatch):
    monkeypatch.setattr(worker_module, 'REPLY_TIMEOUT', 0.2)

    def update(status, *args):
        raise RuntimeError('the window is broken')

    window = SimpleNamespace(update=update)
    worker = Worker(config_module.loads(CONFIG), window, queue.Queue(), FakeInterface)
    worker.thread.join(timeout=10)
    with pytest.raises(RuntimeError, match='thread has stopped'):
        hand_over(worker)
    # Nothing is left awaiting a reply that cannot come.
    assert not worker.unsaved


def test_a_session_that_reaches_a_limit_says_why_once(capsys):
    text = CONFIG.replace(
        'num_buffered_runs = 2', 'num_buffered_runs = 2\nmax_num_runs = 2'
    )
    worker, commands, _ = start(text=text)
    commands.put(('start', None, None))
    hand_over(worker, 'shot-0', 'shot-1')
    hand_over(worker)
    assert capsys.readouterr().out.count('The optimization has stopped') == 1


def test_runmanagers_values_are_set_from_the_window_and_when_the_session_ends(capsys):
    writes, failures = [], []

    class Writes(FakeInterface):
        # What check_ready records at the first Start.
        original = {'x': '0.5'}

        def set_values(self, params=None):
            if failures:
                raise failures[0]
            writes.append(None if params is None else list(params))

    text = CONFIG.replace(
        'num_buffered_runs = 2', 'num_buffered_runs = 2\nmax_num_runs = 2'
    )
    worker, commands, shown = start(Writes, text)
    commands.put(('start', None, None))
    hand_over(worker, 'shot-0')
    commands.put(('pause', None, None))
    commands.put(('set_best', None, None))
    commands.put(('restore', None, None))
    hand_over(worker)
    best = shown[-1]['best_params']
    assert writes == [best, None]

    # A write runmanager does not take is one line, and the run is as it was.
    capsys.readouterr()
    for failure in (TimeoutError('timed out'), RuntimeError('refused')):
        failures[:] = [failure]
        commands.put(('set_best', None, None))
        hand_over(worker)
    out, err = capsys.readouterr()
    assert out.splitlines() == [
        "Could not set runmanager's values: runmanager is not answering",
        "Could not set runmanager's values: refused",
    ]
    assert err == ''
    assert shown[-1]['paused'] and shown[-1]['stopped'] is None

    # The limit ends the run, and runmanager is left showing its best once.
    failures.clear()
    commands.put(('start', None, None))
    hand_over(worker, 'shot-1')
    hand_over(worker)
    assert shown[-1]['stopped']
    assert writes == [best, None, shown[-1]['best_params']]


def test_a_start_from_runmanagers_values_opens_the_run_there_or_is_refused():
    reason = 'gx is 2, outside the range 0 to 1 of parameter x'
    refusals, sent = [reason], []

    class Reads(FakeInterface):
        def get_start(self):
            if refusals:
                raise RuntimeError(refusals.pop())
            return np.array([0.25])

        def submit(self, proposals):
            sent.append(proposals[0].tolist())
            return super().submit(proposals)

    worker, commands, shown = start(Reads)
    commands.put(('start', None, True))
    hand_over(worker)
    assert shown[-1]['paused'] and shown[-1]['pause_reason'] == reason
    assert shown[-1]['submitted'] == 0
    commands.put(('start', None, True))
    assert hand_over(worker, 'shot-0') == [('shot-0.h5', 'start')]
    assert sent[0] == [0.25] and shown[-1]['start'] == [0.25]

    # A Start that resumes the run does not read runmanager again.
    refusals.append(reason)
    commands.put(('pause', None, None))
    commands.put(('start', None, True))
    hand_over(worker)
    assert not shown[-1]['paused']
