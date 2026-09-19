import os

# Every unhandled exception in suite code otherwise spawns a tkinter window,
# up to ten per process. A conftest is imported by pytest and by nothing else,
# so a real run is untouched; setdefault leaves an explicit choice alone.
os.environ.setdefault('LABSCRIPT_NO_ERROR_DIALOG', '1')

import numpy as np
import pytest

from labscript_optimization.observations import Observation
from labscript_optimization.space import Parameter, ParameterSpace


@pytest.fixture
def space():
    """A two-parameter space on [-5, 5]^2."""
    return ParameterSpace(
        [Parameter('x', -5.0, 5.0), Parameter('y', -5.0, 5.0)]
    )


@pytest.fixture
def rng():
    return np.random.default_rng(20260918)


def observe(shot_id, params, cost, uncer=None, bad=False):
    """Build an Observation without ceremony."""
    return Observation(str(shot_id), np.asarray(params, dtype=float), cost, uncer, bad)


def run_loop(learner, space, cost_function, batches, k, rng, history=None):
    """Drive a learner in closed loop and return the history it produced."""
    history = list(history or [])
    for batch in range(batches):
        for params in np.atleast_2d(learner.propose(history, k)):
            history.append(
                observe(f'{batch}-{len(history)}', params, cost_function(params))
            )
    return history


@pytest.fixture
def loop():
    return run_loop
