"""The learners, and how a configuration names the ones a session runs."""

import inspect

import numpy as np

from .base import InsufficientData, Learner, ParameterSpaceLearner
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
    "ParameterSpaceLearner",
    "RandomLearner",
    "TwoPhaseLearner",
    "build",
    "knobs_by_learner",
    "validate_options",
]


#: Learners that can be named in a configuration. Every class here is
#: resolved whenever a configuration is loaded, because its constructor is the
#: schema for its own table, and the selected one is built there as well,
#: because what it proposes at a time is a fact about an instance. So none of
#: them may import the scientific stack either to be imported or to be built;
#: see :mod:`labscript_optimization.learners.gaussian_process`.
LEARNERS = {
    "random": RandomLearner,
    "directed_random": DirectedRandomLearner,
    "differential_evolution": DifferentialEvolutionLearner,
    "gaussian_process": GaussianProcessLearner,
}

#: Learners whose proposals mean nothing until the history holds a spread of
#: points, so a configuration naming one runs a trainer first. Which learner
#: trains is the file's to say, in ``[GENERAL] trainer``; this only says who
#: needs one.
NEEDS_TRAINING = frozenset({"gaussian_process"})


def _constructor_parameters(name: str):
    """The constructor signature of one named learner, by parameter name."""
    try:
        cls = LEARNERS[name]
    except KeyError:
        raise ValueError(
            f"unknown learner {name!r}; choose one of {sorted(LEARNERS)}"
        ) from None
    accepted = inspect.signature(cls).parameters
    if any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
        raise TypeError(
            f"learner {name!r} collects its knobs in **kwargs, which names "
            f"none of them, so its table has no schema: every key written in "
            f"it would be refused and the learner built from its defaults "
            f"alone; spell the knobs out"
        )
    return accepted


def _option_names(name: str) -> set[str]:
    """The configuration options accepted by one named learner."""
    return set(_constructor_parameters(name)) - {"space", "rng"}


def knobs_by_learner() -> dict[str, tuple[str, ...]]:
    """Every knob any learner takes, and the learners that take it.

    A configuration reads this to tell a knob written in the wrong table from a
    spelling nothing here knows: the two want different messages, and only the
    constructors can say which is which.
    """
    knobs: dict[str, list[str]] = {}
    for name in LEARNERS:
        for key in _option_names(name):
            knobs.setdefault(key, []).append(name)
    return {key: tuple(names) for key, names in knobs.items()}


def validate_options(config) -> None:
    """Hold a configuration to the learners it names.

    The learner and the trainer each have to be one of :data:`LEARNERS`, or the
    misspelling is found at ``build()`` -- which is worker configure, with the
    session already starting. A ``[LEARNER.<name>]`` table has one constructor
    that defines its keys, so anything else in it is a knob that learner
    ignores.

    The learners that will be built are read for their signatures whether or
    not a table names them, because a learner collecting its knobs in
    ``**kwargs`` leaves its table without a schema: every key written in it
    would be refused, and the learner built from its defaults alone.

    Everything here is answered by a name and a table, before any learner
    exists. What only a learner can answer is checked in :func:`build`.
    """
    for role in ("learner", "trainer"):
        name = getattr(config, role)
        if name not in LEARNERS:
            raise ValueError(
                f"unknown {role} {name!r}; choose one of {sorted(LEARNERS)}"
            )
    built = [config.learner]
    if config.learner in NEEDS_TRAINING:
        built.append(config.trainer)
    for name in dict.fromkeys([*built, *config.learner_options]):
        accepted = _option_names(name)
        unknown = sorted(set(config.learner_options.get(name, {})) - accepted)
        if unknown:
            # A learner may take no knobs at all, and then "it accepts:"
            # trails off into nothing, which reads as a message that failed to
            # finish rather than as the answer it is.
            takes = (
                f"It accepts: {', '.join(sorted(accepted))}."
                if accepted
                else "That learner takes no knobs at all, so its table holds "
                "nothing and is as well left out."
            )
            raise ValueError(
                f"[LEARNER.{name}] does not accept "
                f"{', '.join(repr(key) for key in unknown)}. {takes}"
            )


def build(config, rng: np.random.Generator | None = None):
    """Build the learner a configuration asks for, and hold it to the budget.

    A learner in :data:`NEEDS_TRAINING` is wrapped in a
    :class:`~labscript_optimization.learners.two_phase.TwoPhaseLearner` with
    the learner ``[GENERAL] trainer`` names, which runs the training shots and,
    where ``[GENERAL] num_runs_between_trainer_runs`` asks for them, the
    periodic runs after them. Each is built
    from its own ``[LEARNER.<name>]`` table, so a trainer and a main learner
    that take the same knob take it separately.

    The budget is then measured against the learner that came back, because
    how many proposals it makes at a time is its own to say and no signature
    or class attribute answers for it.
    """
    # Config objects may be constructed directly instead of parsed from TOML.
    validate_options(config)
    if rng is None:
        rng = np.random.default_rng(config.seed)

    name = config.learner
    options = config.learner_options
    main = LEARNERS[name](config.space, rng, **options.get(name, {}))
    if name in NEEDS_TRAINING:
        trainer = LEARNERS[config.trainer](
            config.space, rng, **options.get(config.trainer, {})
        )
        learner = TwoPhaseLearner(
            trainer,
            main,
            config.num_training_runs,
            config.num_runs_between_trainer_runs,
        )
    else:
        learner = main

    generation = learner.generation
    if (
        config.max_num_runs is not None
        and generation is not None
        and config.max_num_runs < 2 * generation
    ):
        raise ValueError(
            f"max_num_runs {config.max_num_runs} leaves less than two whole "
            f"generations of the {generation} proposals {name!r} makes at a "
            f"time, so set it to at least {2 * generation}, or configure "
            f"{name!r} to propose fewer at a time. The first generation is "
            f"the population itself and the second is the first to evolve it; "
            f"a second generation cut short evolves some of its slots rather "
            f"than a generation."
        )
    return learner
