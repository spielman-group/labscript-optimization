"""The learners, and how a configuration names the ones a session runs."""

import inspect

import numpy as np

from .. import knobs
from .base import Learner, ParameterSpaceLearner
from .differential_evolution import DifferentialEvolutionLearner
from .gaussian_process import EXPLORERS, GaussianProcessLearner
from .random import DirectedRandomLearner, RandomLearner

__all__ = [
    "DifferentialEvolutionLearner",
    "DirectedRandomLearner",
    "GaussianProcessLearner",
    "Learner",
    "ParameterSpaceLearner",
    "RandomLearner",
    "build",
    "validate_options",
]


#: The learners a configuration can name. Loading a configuration resolves every
#: class, to hold each table to its constructor, and builds the selected one, to
#: read its ``generation``. So none may import the scientific stack before a
#: batch is asked for; see :mod:`labscript_optimization.learners.gaussian_process`.
#:
#: :meta hide-value:
LEARNERS = {
    "random": RandomLearner,
    "directed_random": DirectedRandomLearner,
    "differential_evolution": DifferentialEvolutionLearner,
    "gaussian_process": GaussianProcessLearner,
}


def _constructor_parameters(name: str):
    try:
        cls = LEARNERS[name]
    except KeyError:
        raise ValueError(
            f"unknown learner {name!r}; choose one of {sorted(LEARNERS)}"
        ) from None
    return inspect.signature(cls).parameters


def _option_names(name: str) -> frozenset[str]:
    return frozenset(_constructor_parameters(name)) - {"space", "rng"}


def _explorer_name(config) -> str | None:
    """The explorer the selected learner runs, or ``None`` if it runs none."""
    accepted = _constructor_parameters(config.learner)
    if "explorer" not in accepted:
        return None
    # The default is read off the constructor, the schema for the table, so no
    # copy drifts. The name is checked here because it is looked up to build
    # the explorer before the constructor sees it.
    named = config.learner_options.get(config.learner, {}).get(
        "explorer", accepted["explorer"].default
    )
    return knobs.choice("explorer", named, EXPLORERS)


def validate_options(config) -> None:
    """Hold a configuration to the learners it names, before any is built.

    Checks that the learner and the explorer it runs are available, and that each
    ``[LEARNER.<name>]`` table holds only keys its learner takes. What only a
    built learner can answer is checked in :func:`build`.

    Parameters
    ----------
    config : labscript_optimization.config.Config

    Raises
    ------
    ValueError
        If the learner or its explorer is unknown, or a table holds a key its
        learner does not take.
    """
    if config.learner not in LEARNERS:
        raise ValueError(
            f"unknown learner {config.learner!r}; choose one of {sorted(LEARNERS)}"
        )
    # Refuses an explorer that is not one of EXPLORERS.
    _explorer_name(config)
    for name, table in config.learner_options.items():
        knobs.reject_unknown(table, _option_names(name), f"[LEARNER.{name}]")


def build(config, rng: np.random.Generator | None = None) -> Learner:
    """Build the learner a configuration asks for, and hold it to the budget.

    A learner that runs an explorer, the Gaussian process, is handed one built
    from the explorer's own ``[LEARNER.<name>]`` table, so the two take a shared
    knob separately. The budget is checked against the built learner, because
    how many proposals it makes at a time is its own to say.

    Parameters
    ----------
    config : labscript_optimization.config.Config
        Names the learner, holds the tables of knobs, and sets the budget.
    rng : numpy.random.Generator, optional
        Source of randomness; ``None`` makes one seeded from ``config.seed``.

    Raises
    ------
    ValueError
        As :func:`validate_options`; if a learner refuses a knob's value; or if
        ``config.max_num_runs`` leaves less than two generations of a learner
        that proposes them.
    """
    # Config objects may be constructed directly instead of parsed from TOML.
    validate_options(config)
    if rng is None:
        rng = np.random.default_rng(config.seed)

    name = config.learner
    options = dict(config.learner_options.get(name, {}))
    explorer = _explorer_name(config)
    if explorer is not None:
        options["explorer"] = EXPLORERS[explorer](
            config.space, rng, **config.learner_options.get(explorer, {})
        )
    learner = LEARNERS[name](config.space, rng, **options)

    generation = learner.generation
    if (
        config.max_num_runs is not None
        and generation is not None
        and config.max_num_runs < 2 * generation
    ):
        raise ValueError(
            f"max_num_runs {config.max_num_runs} leaves less than two whole "
            f"generations of the {generation} proposals {name!r} makes at a "
            f"time, and a budget under two never evolves anything. Set it to "
            f"at least {2 * generation}, or configure {name!r} to propose "
            f"fewer at a time."
        )
    return learner
