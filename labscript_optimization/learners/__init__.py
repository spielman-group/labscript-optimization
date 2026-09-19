"""The learners, and how a configuration names one."""

import inspect
from collections.abc import MutableMapping
from importlib import import_module

import numpy as np

from .base import InsufficientData, Learner, ParameterSpaceLearner
from .differential_evolution import DifferentialEvolutionLearner
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
    "make_learner",
    "validate_options",
]


class _LearnerRegistry(MutableMapping):
    """Classes available by configuration name, imported when first needed."""

    def __init__(self, entries):
        self._entries = dict(entries)

    def __getitem__(self, name):
        entry = self._entries[name]
        if isinstance(entry, tuple):
            module_name, class_name = entry
            entry = getattr(import_module(module_name, __name__), class_name)
            self._entries[name] = entry
        return entry

    def __contains__(self, name):
        # Answered from the names alone. The mapping default asks
        # ``__getitem__``, which would import the module and the whole
        # scientific stack under it just to say whether a name is spelt right.
        return name in self._entries

    def __setitem__(self, name, cls):
        self._entries[name] = cls

    def __delitem__(self, name):
        del self._entries[name]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)


#: Learners that can be named in a configuration.
LEARNERS = _LearnerRegistry(
    {
        "random": RandomLearner,
        "directed_random": DirectedRandomLearner,
        "differential_evolution": DifferentialEvolutionLearner,
        "gaussian_process": (".gaussian_process", "GaussianProcessLearner"),
    }
)

#: Learners that need a training phase before their proposals mean anything,
#: and the learner that provides it.
NEEDS_TRAINING = {"gaussian_process": "directed_random"}

#: Learners that evolve a population, and whose budget therefore has to cover
#: more than one of it. Written out rather than read off the selected learner's
#: signature, because reading a signature resolves that learner's class: the
#: registry is lazy so that the lyse process, which loads the configuration and
#: never builds a learner, does not pay for the scientific stack. The two are
#: held to each other by a test rather than by one deriving from the other.
POPULATION_LEARNERS = frozenset({"differential_evolution"})


def __getattr__(name):
    if name == "GaussianProcessLearner":
        return LEARNERS["gaussian_process"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            f"none of them, so every option would be dropped and the learner "
            f"built entirely from its defaults; spell the knobs out"
        )
    return accepted


def _option_names(name: str) -> set[str]:
    """The configuration options accepted by one named learner."""
    return set(_constructor_parameters(name)) - {"space", "rng"}


def validate_options(config) -> None:
    """Hold a configuration to the learner it names.

    The learner has to be one of :data:`LEARNERS`, or the misspelling is found
    at ``build()`` -- which is worker configure, with the session already
    starting. A ``[LEARNER.<name>]`` table has one constructor that defines its
    keys, so anything else in it is a knob that learner ignores. And a budget
    is measured against the population the selected learner will evolve.
    """
    if config.learner not in LEARNERS:
        raise ValueError(
            f"unknown learner {config.learner!r}; choose one of {sorted(LEARNERS)}"
        )
    for name, options in config.learner_options.items():
        accepted = _option_names(name)
        unknown = sorted(set(options) - accepted)
        if unknown:
            raise ValueError(
                f"[LEARNER.{name}] does not accept "
                f"{', '.join(repr(key) for key in unknown)}. It accepts: "
                f"{', '.join(sorted(accepted))}."
            )

    if config.max_num_runs is None or config.learner not in POPULATION_LEARNERS:
        return
    default = _constructor_parameters(config.learner)["population_size"].default
    size = int(config.options_for(config.learner).get("population_size", default))
    if config.max_num_runs < 2 * size:
        raise ValueError(
            f"max_num_runs {config.max_num_runs} leaves less than two "
            f"generations of the {size} members {config.learner!r} evolves, so "
            f"set it to at least {2 * size} or lower population_size. The "
            f"first generation is the population itself and the second is the "
            f"first to evolve it; a second generation cut short evolves some "
            f"of its slots rather than a generation."
        )


def make_learner(name: str, space, rng, options):
    accepted = _option_names(name)
    cls = LEARNERS[name]
    # A shared table carries knobs for every learner, so pass on the ones this
    # learner actually takes. What it takes is its signature and nothing wider:
    # a name the constructor happens to use as a local variable is not a knob,
    # and letting one through blames the shared table for a collision the user
    # cannot see.
    return cls(space, rng, **{k: v for k, v in options.items() if k in accepted})


def build(config, rng: np.random.Generator | None = None):
    """Build the learner a configuration asks for.

    A learner that needs a training phase is wrapped in a
    :class:`~labscript_optimization.learners.two_phase.TwoPhaseLearner` with
    the trainer named in :data:`NEEDS_TRAINING`, which is also the fallback for
    proposals the main learner cannot make.
    """
    # Config objects may be constructed directly instead of parsed from TOML.
    validate_options(config)
    if rng is None:
        rng = np.random.default_rng(config.seed)

    name = config.learner
    main = make_learner(name, config.space, rng, config.options_for(name))
    if name not in NEEDS_TRAINING:
        return main

    trainer_name = NEEDS_TRAINING[name]
    trainer = make_learner(
        trainer_name, config.space, rng, config.options_for(trainer_name)
    )
    return TwoPhaseLearner(trainer, main, config.num_training_runs)
