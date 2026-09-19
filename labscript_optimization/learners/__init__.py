"""The learners, and how a configuration names one."""

import inspect

import numpy as np

from .base import InsufficientData, Learner
from .differential_evolution import DifferentialEvolutionLearner
from .gaussian_process import GaussianProcessLearner
from .random import DirectedRandomLearner, RandomLearner
from .two_phase import TwoPhaseLearner

__all__ = [
    "DifferentialEvolutionLearner",
    "DirectedRandomLearner",
    "GaussianProcessLearner",
    "InsufficientData",
    "Learner",
    "RandomLearner",
    "TwoPhaseLearner",
    "build",
]

#: Learners that can be named in a configuration.
LEARNERS = {
    "random": RandomLearner,
    "directed_random": DirectedRandomLearner,
    "differential_evolution": DifferentialEvolutionLearner,
    "gaussian_process": GaussianProcessLearner,
}

#: Learners that need a training phase before their proposals mean anything,
#: and the learner that provides it.
NEEDS_TRAINING = {"gaussian_process": "directed_random"}


def _make(name: str, space, rng, options):
    try:
        cls = LEARNERS[name]
    except KeyError:
        raise ValueError(
            f"unknown learner {name!r}; choose one of {sorted(LEARNERS)}"
        ) from None
    accepted = inspect.signature(cls).parameters
    # A shared table carries knobs for every learner, so pass on the ones this
    # learner actually takes rather than making the user split them by hand.
    # The question is what the constructor accepts, which is its parameters and
    # nothing else: a name it happens to use as a local variable is not a knob.
    return cls(space, rng, **{k: v for k, v in options.items() if k in accepted})


def build(config, rng: np.random.Generator | None = None):
    """Build the learner a configuration asks for.

    A learner that needs a training phase is wrapped in a
    :class:`~labscript_optimization.learners.two_phase.TwoPhaseLearner` with
    the trainer named in :data:`NEEDS_TRAINING`, which is also the fallback for
    proposals the main learner cannot make.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    name = config.learner
    main = _make(name, config.space, rng, config.options_for(name))
    if name not in NEEDS_TRAINING:
        return main

    trainer_name = NEEDS_TRAINING[name]
    trainer = _make(
        trainer_name, config.space, rng, config.options_for(trainer_name)
    )
    return TwoPhaseLearner(trainer, main, config.num_training_runs)
