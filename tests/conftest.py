import os

# Every unhandled exception in suite code otherwise spawns a tkinter window,
# up to ten per process. A conftest is imported by pytest and by nothing else,
# so a real run is untouched; setdefault leaves an explicit choice alone.
os.environ.setdefault('LABSCRIPT_NO_ERROR_DIALOG', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from concurrent.futures import wait

import numpy as np
import pytest
from runmanager.client import (
    BLACS_COMPLETED,
    BLACS_REJECTED,
    BLACS_RUNNING,
    BLACS_WAITING,
    LYSE_SENT,
    LYSE_WAITING,
    QUEUE_BLOCKED,
    QUEUE_LEFT,
    QUEUE_QUEUED,
)

from labscript_optimization.observations import COMPLETE, Observation
from labscript_optimization.space import Parameter, ParameterSpace


#: A one-parameter session under the random learner, keeping two shots queued.
SESSION_CONFIG = """
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


#: What runmanager says of a shot that is queued and in BLACS's hands.
STILL_COMING = {
    'pending': True,
    'queue': QUEUE_QUEUED,
    'blacs': BLACS_RUNNING,
    'lyse': LYSE_WAITING,
    'message': '',
}


@pytest.fixture(scope='session')
def qt_application():
    from qtutils.qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeRunmanager:
    """Stands in for runmanager, and decides what is still coming.

    It answers as runmanager does: one record per id asked about, and ``None``
    for an id it never had. A rejected shot keeps its row and says so, and a
    shot that has run leaves the queue ``completed`` and is on its way to lyse.
    """

    def __init__(self):
        self.submitted: list[str] = []
        self.rejected: set[str] = set()
        self.blocked: set[str] = set()
        self.finished: dict[str, str] = {}
        self.labscript_changed = False

    def check_unchanged(self):
        if self.labscript_changed:
            raise RuntimeError('the labscript file changed')

    def submit(self, proposals):
        ids = [f'shot-{len(self.submitted) + i}' for i in range(len(proposals))]
        self.submitted.extend(ids)
        return ids

    def shot_status(self, shot_ids):
        answers = {}
        for shot_id in shot_ids:
            if shot_id not in self.submitted:
                answers[shot_id] = None
                continue
            answers[shot_id] = record = dict(STILL_COMING)
            if shot_id in self.blocked:
                # Still pending: a blocked row is dropped whatever runmanager
                # says of it.
                record.update(queue=QUEUE_BLOCKED, blacs=BLACS_WAITING)
            elif shot_id in self.rejected:
                record.update(pending=False, blacs=BLACS_REJECTED)
            elif shot_id in self.finished:
                record.update(
                    pending=False,
                    queue=QUEUE_LEFT,
                    blacs=BLACS_COMPLETED,
                    lyse=self.finished[shot_id],
                )
        return answers

    def lose(self, *shot_ids):
        """An operator disposes of these shots, so they will never run."""
        self.rejected.update(shot_ids)

    def finish(self, *shot_ids, lyse=LYSE_SENT):
        """These shots run and leave the queue, as every healthy shot does.

        ``lyse`` is what became of the file: ``'sent'`` for a shot lyse has,
        or ``'not sent'`` or ``'rejected'`` for one whose cost will not come.
        """
        self.finished.update(dict.fromkeys(shot_ids, lyse))


@pytest.fixture
def runmanager():
    return FakeRunmanager()


@pytest.fixture
def space():
    """A two-parameter space on [-5, 5]^2."""
    return ParameterSpace(
        [Parameter('x', -5.0, 5.0), Parameter('y', -5.0, 5.0)]
    )


@pytest.fixture
def rng():
    return np.random.default_rng(20260918)


def observe(shot_id, params, cost, uncer=None, bad=False, state=COMPLETE, source=None):
    """Build an Observation without ceremony.

    A cost of ``None`` with a state of PENDING or DROPPED is a proposal that
    has produced nothing: a position the session spent. ``source`` is what a
    session would have recorded as proposing it.
    """
    return Observation(
        str(shot_id), np.asarray(params, dtype=float), cost, uncer, bad, state, source
    )


def run_loop(learner, space, cost_function, batches, k, rng, history=None):
    """Drive a learner in closed loop and return the history it produced.

    Every proposal is measured before the next call, so nothing is in flight
    when the learner is asked: ``k`` is the hint, which a learner declaring a
    generation does not read.
    """
    history = list(history or [])
    for batch in range(batches):
        for params, _ in learner.propose(history, k):
            history.append(
                observe(f'{batch}-{len(history)}', params, cost_function(params))
            )
    return history


def settle(learner):
    """Wait for the batch a Gaussian process learner is computing, if it is.

    It computes on a thread of its own, so a test that wants the batch in hand
    at the next refill waits for it here. Bounded, so that a batch that never
    finishes fails the test rather than hanging it.
    """
    if learner.computation is not None:
        done, _ = wait([learner.computation], timeout=60)
        assert done, 'the batch did not finish computing'
