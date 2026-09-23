import os

# Every unhandled exception in suite code otherwise spawns a tkinter window,
# up to ten per process. A conftest is imported by pytest and by nothing else,
# so a real run is untouched; setdefault leaves an explicit choice alone.
os.environ.setdefault('LABSCRIPT_NO_ERROR_DIALOG', '1')

import numpy as np
import pytest

from labscript_optimization.observations import COMPLETE, Observation
from labscript_optimization.space import Parameter, ParameterSpace


class FakeRunmanager:
    """Stands in for runmanager, and decides what is still coming.

    It answers as runmanager does: one ``{'pending', 'state'}`` per id asked
    about. A cancelled shot keeps its row and says so, while a shot that has
    run leaves the queue and becomes indistinguishable from an id runmanager
    never had -- both are ``unknown``.
    """

    def __init__(self):
        self.submitted: list[str] = []
        self.cancelled: set[str] = set()
        self.blocked: set[str] = set()
        self.finished: set[str] = set()
        self.labscript_changed = False

    def check_ready(self):
        pass

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
            if shot_id in self.blocked:
                answers[shot_id] = {'pending': False, 'state': 'blocked'}
            elif shot_id in self.cancelled:
                answers[shot_id] = {'pending': False, 'state': 'cancelled'}
            elif shot_id in self.finished or shot_id not in self.submitted:
                answers[shot_id] = {'pending': False, 'state': 'unknown'}
            else:
                answers[shot_id] = {'pending': True, 'state': 'running'}
        return answers

    def lose(self, *shot_ids):
        """An operator disposes of these shots, so they will never run."""
        self.cancelled.update(shot_ids)

    def finish(self, *shot_ids):
        """These shots run and leave the queue, as every healthy shot does."""
        self.finished.update(shot_ids)


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


def observe(shot_id, params, cost, uncer=None, bad=False, state=COMPLETE):
    """Build an Observation without ceremony.

    A cost of ``None`` with a state of PENDING or DROPPED is a proposal that
    has produced nothing: a position the session spent.
    """
    return Observation(
        str(shot_id), np.asarray(params, dtype=float), cost, uncer, bad, state
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


@pytest.fixture
def loop():
    return run_loop
