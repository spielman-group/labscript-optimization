"""The learners, and how a configuration names the ones a session runs."""

import inspect

import numpy as np

from .. import knobs
from .base import InsufficientData, Learner, ParameterSpaceLearner
from .differential_evolution import DifferentialEvolutionLearner
from .gaussian_process import EXPLORERS, GaussianProcessLearner
from .random import DirectedRandomLearner, RandomLearner

__all__ = [
    "DifferentialEvolutionLearner",
    "DirectedRandomLearner",
    "GaussianProcessLearner",
    "InsufficientData",
    "Learner",
    "ParameterSpaceLearner",
    "RandomLearner",
    "build",
    "knobs_by_learner",
    "validate_options",
]


#: Learners that can be named in a configuration. Every class here is
#: resolved whenever a configuration is loaded, because its constructor is the
#: schema for its own table, and the selected one is built there as well and
#: asked what it proposes as a run opens, because what it proposes at a time
#: is a fact about an instance. So none of them may import the scientific
#: stack to be imported, to be built, or to open a run; see
#: :mod:`labscript_optimization.learners.gaussian_process`.
LEARNERS = {
    "random": RandomLearner,
    "directed_random": DirectedRandomLearner,
    "differential_evolution": DifferentialEvolutionLearner,
    "gaussian_process": GaussianProcessLearner,
}


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


def _explorer_name(config) -> str | None:
    """The learner the selected one explores with, by name, or ``None``.

    A learner whose constructor takes an ``explorer`` runs one beside itself:
    the one its own table names, or the knob's default where the table names
    none. The default is read off the constructor because the constructor is
    the schema for the table, and a second copy of it here could drift. The
    name is held to :data:`~labscript_optimization.learners.gaussian_process.EXPLORERS`
    in the words the constructor uses, because it is looked up to build the
    explorer before the constructor ever sees it.
    """
    accepted = _constructor_parameters(config.learner)
    if "explorer" not in accepted:
        return None
    named = config.learner_options.get(config.learner, {}).get(
        "explorer", accepted["explorer"].default
    )
    return knobs.choice("explorer", named, EXPLORERS)


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

    The learner has to be one of :data:`LEARNERS`, and the explorer it runs,
    if it runs one, one of the explorers; otherwise the misspelling is found
    at ``build()`` -- which is worker configure, with the session already
    starting. A ``[LEARNER.<name>]`` table has one constructor that defines
    its keys, so anything else in it is a knob that learner ignores.

    The learners that will be built -- the selected one and its explorer --
    are read for their signatures whether or not a table names them, because
    a learner collecting its knobs in ``**kwargs`` leaves its table without a
    schema: every key written in it would be refused, and the learner built
    from its defaults alone.

    Everything here is answered by a name and a table, before any learner
    exists. What only a learner can answer is checked in :func:`build`.
    """
    if config.learner not in LEARNERS:
        raise ValueError(
            f"unknown learner {config.learner!r}; choose one of {sorted(LEARNERS)}"
        )
    built = [config.learner]
    explorer = _explorer_name(config)
    if explorer is not None:
        built.append(explorer)
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

    A learner that runs an explorer -- the Gaussian process -- is handed it
    built: the learner its ``explorer`` knob names, from that learner's own
    ``[LEARNER.<name>]`` table, so an explorer and the learner running it that
    take the same knob take it separately.

    The budget is then measured against the learner that came back, because
    how many proposals it makes at a time is its own to say and no signature
    or class attribute answers for it.
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
            f"time, so set it to at least {2 * generation}, or configure "
            f"{name!r} to propose fewer at a time. The first generation is "
            f"the population itself and the second is the first to evolve it, "
            f"so a budget under two of them never evolves anything at all. "
            f"The budget need not be a whole number of generations: the last "
            f"one is cut short where it runs out, and its trials compete for "
            f"their own slots like any other."
        )
    return learner
