"""Reading the TOML configuration.

``[PARAMETERS.<group>.<name>]``
    One optimized parameter, with ``min``, ``max``, optional ``start``,
    optional ``enable`` (default true), and optional ``global_name``. Giving
    ``global_name`` is shorthand for a runmanager global that takes this
    parameter's value directly.

``[RUNMANAGER_GLOBALS.<group>.<name>]``
    A runmanager global computed from one or more parameters, with ``args``
    naming them, ``expr`` a lambda taking them in that order, and optional
    ``enable`` (default true). Used when several parameters feed one global.

``[ANALYSIS]``
    ``cost_key`` is ``[routine_name, result_name]``, the lyse dataframe column
    holding the cost; optional ``maximize`` (default false) makes larger costs
    better; ``groups`` selects which groups take part.

A parameter in a group not listed in ``groups`` is left out entirely. One with
``enable = false`` in a listed group is carried but not searched and gets no
mapping, so its runmanager global keeps the value it already holds. A listed
name that no table defines is refused as a misspelled group. Among the enabled
entries of the listed groups, a parameter name and a global name are each unique.

Every key must be one this package knows; an unknown one is refused.

``[GENERAL]`` carries the session's own settings: ``learner``,
``num_buffered_runs``, ``max_num_runs``, ``max_num_runs_without_better_params``
and ``seed``. A learner's knobs are written in ``[LEARNER.<name>]``, the table of
the learner that takes them: every key must be a knob that learner takes, and
every value of the kind that knob takes. The Gaussian process runs an explorer
beside itself, so a knob both take is written in each table, and each gets its
own value. The keys of every named table are checked, but its values only when
its learner is built, so one file can carry the settings of several learners.

The Configuration page of the documentation describes every key.
"""

import inspect
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import numpy as np

from . import knobs
from .space import Parameter, ParameterSpace

#: The tables a file carries; anything else is a misspelling whose settings
#: would go unread.
TOP_LEVEL_TABLES = frozenset(
    {
        "ANALYSIS",
        "GENERAL",
        "LEARNER",
        "PARAMETERS",
        "RUNMANAGER_GLOBALS",
    }
)

#: The settings ``[ANALYSIS]`` carries.
ANALYSIS_KEYS = frozenset({"cost_key", "groups", "maximize"})

#: What ``[GENERAL]`` carries, each a :class:`Config` field handed over by name.
#: A learner's knobs are not here; they go in that learner's own table.
GENERAL_KEYS = frozenset(
    {
        "learner",
        "max_num_runs",
        "max_num_runs_without_better_params",
        "num_buffered_runs",
        "seed",
    }
)

#: The whole-number settings of :class:`Config`, each with its smallest value
#: and why. Below the floor of ``max_num_runs``, or of ``num_buffered_runs`` with a
#: learner that has no generation, a session idles with no stop reason.
INTEGER_SETTINGS = {
    "num_buffered_runs": (
        1,
        "a queue asked to hold none of the session's shots never gets a proposal",
    ),
    "seed": (0, "numpy seeds its generator from a non-negative integer"),
    "max_num_runs": (
        1,
        "a budget of no runs is a session that never starts, not one that stops",
    ),
    "max_num_runs_without_better_params": (
        1,
        "the run that sets the best cost always has no runs after it",
    ),
}

#: Of those, the ones a session may leave unset.
OPTIONAL_INTEGER_SETTINGS = frozenset(
    {
        "max_num_runs",
        "max_num_runs_without_better_params",
        "seed",
    }
)

#: The keys one ``[PARAMETERS.<group>.<name>]`` table carries.
PARAMETER_KEYS = frozenset({"enable", "global_name", "max", "min", "start"})

#: Of those, the ones a parameter table must carry: a search needs bounds.
PARAMETER_REQUIRED = frozenset({"max", "min"})

#: The keys one ``[RUNMANAGER_GLOBALS.<group>.<name>]`` table carries.
GLOBAL_KEYS = frozenset({"args", "enable", "expr"})

#: Of those, the one a globals table must carry; ``expr`` is optional.
GLOBAL_REQUIRED = frozenset({"args"})


@dataclass(frozen=True)
class GlobalMapping:
    """One runmanager global, and how to compute it from the parameters.

    Parameters
    ----------
    name : str
        The runmanager global to set.
    expr : str or None
        Source of a lambda taking ``args`` in order, evaluated in the namespace
        runmanager evaluates globals in; ``None`` passes the single argument
        through unchanged.
    args : tuple of str
        Names of the parameters feeding this global.
    """

    name: str
    expr: str | None
    args: tuple[str, ...]
    #: ``expr`` as a callable, or ``None`` when the single argument passes
    #: through unchanged. Not an init field: it exists only through the checks
    #: in ``__post_init__``.
    function: Callable[..., Any] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        """Build ``expr``'s callable now, so a bad one fails here, not mid-session."""
        if self.expr is None:
            if len(self.args) != 1:
                raise ValueError(
                    f"global {self.name!r} has no expr, so its value is the "
                    f"one parameter it takes; it names {len(self.args)}: "
                    f"{list(self.args)}. Give it an expr taking them all, or "
                    f"name one parameter."
                )
            return
        # The namespace runmanager evaluates globals in, so an expr sees the
        # same names. Imported here, so a file without an expr loads neither
        # matplotlib nor runmanager.
        namespace = {}
        exec("from pylab import *", namespace, namespace)
        exec("from runmanager.functions import *", namespace, namespace)
        try:
            # The lab's own configuration file is as trusted as its analysis
            # routines.
            function = eval(self.expr, namespace)  # noqa: S307
        except Exception as error:
            raise ValueError(
                f"the expr for global {self.name!r} cannot be evaluated: "
                f"{self.expr!r} ({error})"
            ) from error
        if not callable(function):
            raise ValueError(
                f"the expr for global {self.name!r} is not callable: "
                f"{self.expr!r}. It must be a lambda taking args in order."
            )
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            # Some callables expose no signature; refusing one would reject a
            # mapping that works.
            signature = None
        if signature is not None:
            try:
                signature.bind(*self.args)
            except TypeError as error:
                raise ValueError(
                    f"the expr for global {self.name!r} cannot take the "
                    f"{len(self.args)} parameters it is given "
                    f"({list(self.args)}): {self.expr!r} ({error})"
                ) from error
        object.__setattr__(self, "function", function)

    def evaluate(self, values: dict[str, float]) -> Any:
        """Compute this global's value from a parameter-name to value mapping."""
        arguments = [values[a] for a in self.args]
        if self.function is None:
            return arguments[0]
        result = self.function(*arguments)
        return result.item() if isinstance(result, np.generic) else result


@dataclass
class Config:
    """Everything a session needs to run."""

    #: The searched parameters and their bounds.
    space: ParameterSpace
    #: How each runmanager global the session sets is computed from the
    #: parameters.
    globals: tuple[GlobalMapping, ...]
    #: ``(routine_name, result_name)``, the lyse dataframe column holding the
    #: cost.
    cost_key: tuple[str, str]
    #: Whether larger costs are better.
    maximize: bool = False
    #: The learner that runs, a key of
    #: :data:`~labscript_optimization.learners.LEARNERS`.
    learner: str = "gaussian_process"
    #: One table of knobs per learner, by learner name. A knob reaches only the
    #: learner whose table holds it, so the Gaussian process and its explorer can
    #: take different values of the same knob.
    learner_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Shots of this session to keep in runmanager's queue, and the hint every
    #: learner is handed. One in flight is the one BLACS is running, so at 2 one
    #: is waiting. A generational learner sets its own depth; a file also
    #: writing this is refused.
    num_buffered_runs: int = 2
    #: The run budget, or ``None`` for no limit.
    max_num_runs: int | None = None
    #: Stop after this many completed shots without a better cost. Every
    #: completed shot counts, one with an unusable cost included, so a dead
    #: detector still ends the run.
    max_num_runs_without_better_params: int | None = None
    #: Seed of the random number generator the learners draw from, or ``None``
    #: to seed it from system entropy.
    seed: int | None = None

    def __post_init__(self) -> None:
        """Check the settings' kinds and floors, the cost key and the global names."""
        for key, (floor, why) in INTEGER_SETTINGS.items():
            value = getattr(self, key)
            if value is None and key in OPTIONAL_INTEGER_SETTINGS:
                continue
            knobs.integer(key, value)
            if value < floor:
                raise ValueError(f"{key} must be at least {floor}, got {value}: {why}.")

        knobs.boolean("maximize", self.maximize)
        knobs.string("learner", self.learner)

        named = all(isinstance(key, str) for key in self.cost_key)
        if len(self.cost_key) != 2 or not named:
            raise ValueError(
                f"cost_key must be two strings, [routine_name, result_name], "
                f"got {list(self.cost_key)!r}."
            )

        names = [g.name for g in self.globals]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"{', '.join(repr(n) for n in repeated)} names more than one "
                f"runmanager global; each is set once per shot. "
                f"The globals set are {names}"
            )

    @property
    def uncertainty_key(self) -> tuple[str, str]:
        """The column holding the uncertainty on the cost, by convention."""
        routine, result = self.cost_key
        return routine, f"u_{result}"

    def globals_for(self, params: Sequence[float]) -> dict[str, Any]:
        """The runmanager globals that realize one parameter vector.

        Values hold no numpy scalars: runmanager writes each one's ``repr``, and
        a numpy scalar's names numpy, as in ``np.float64(0.25)``.
        """
        values = {p.name: float(v) for p, v in zip(self.space.parameters, params)}
        return {g.name: g.evaluate(values) for g in self.globals}


def present(table: dict, keys: Iterable[str]) -> dict[str, Any]:
    """The ``keys`` the table carries, as written; the rest take Config's defaults."""
    return {key: table[key] for key in keys if key in table}


def require_present(table: dict, keys: frozenset[str], where: str) -> None:
    """Raise for a required key the table leaves out."""
    missing = sorted(keys - set(table))
    if missing:
        raise ValueError(
            f"{where} is missing {', '.join(repr(k) for k in missing)}. "
            f"It requires: {', '.join(sorted(keys))}."
        )


def require_tables(holder: Any, where: str, shape: str) -> dict:
    """Return ``holder`` if it is a table of tables, else raise naming ``where``."""
    # The usual slip is leaving out the group, so a parameter's own settings sit
    # where its group's parameters belong and a later check fails on a type. A
    # list is refused too, or the key check would blame its first element.
    if not isinstance(holder, dict):
        raise ValueError(
            f"{where} is written as {holder!r}, and {shape} expects a table."
        )
    wrong = sorted(key for key, value in holder.items() if not isinstance(value, dict))
    if wrong:
        named = ", ".join(repr(key) for key in wrong)
        raise ValueError(
            f"{where} gives {named} a value where {shape} expects a table. "
            f"This is one level short: only the innermost table of {shape} "
            f"carries settings, so write the table each goes under."
        )
    return holder


def check_keys(raw: dict) -> None:
    """Raise for an unknown key, a missing required one, or a wrongly typed value."""
    knobs.reject_unknown(raw, TOP_LEVEL_TABLES, "the top level of the configuration")
    require_tables(raw.get("LEARNER", {}), "[LEARNER]", "[LEARNER.<name>]")
    for name in ("ANALYSIS", "GENERAL"):
        if not isinstance(raw.get(name, {}), dict):
            raise ValueError(f"[{name}] must be a table, not {raw[name]!r}.")
    knobs.reject_unknown(raw.get("ANALYSIS", {}), ANALYSIS_KEYS, "[ANALYSIS]")
    knobs.reject_unknown(raw.get("GENERAL", {}), GENERAL_KEYS, "[GENERAL]")
    # Every group's tables are checked for keys and kinds of value, listed or not,
    # so a typo in a switched-off group does not wait for the day it is listed.
    # What bounds, args and expr mean is checked in from_dict, for listed groups.
    for table, allowed, required in (
        ("PARAMETERS", PARAMETER_KEYS, PARAMETER_REQUIRED),
        ("RUNMANAGER_GLOBALS", GLOBAL_KEYS, GLOBAL_REQUIRED),
    ):
        shape = f"[{table}.<group>.<name>]"
        groups = require_tables(raw.get(table, {}), f"[{table}]", shape)
        for group, entries in groups.items():
            require_tables(entries, f"[{table}.{group}]", shape)
            for name, entry in entries.items():
                where = f"[{table}.{group}.{name}]"
                knobs.reject_unknown(entry, allowed, where)
                require_present(entry, required, where)
                if "args" in entry:
                    knobs.array(f"{where} args", entry["args"])
                if "enable" in entry:
                    knobs.boolean(f"{where} enable", entry["enable"])
                for key in ("min", "max", "start"):
                    if key in entry:
                        knobs.number(f"{where} {key}", entry[key])
                if "global_name" in entry:
                    knobs.string(f"{where} global_name", entry["global_name"])
    # [LEARNER.<name>] is checked by learners.validate_options, which knows each
    # learner's keys.


def loads(text: str) -> Config:
    """Parse a configuration from TOML text, checked as :func:`from_dict` checks it.

    Raises
    ------
    ValueError
        If ``text`` is not valid TOML (:class:`tomllib.TOMLDecodeError`), or
        names a table or setting that is unknown, missing, of the wrong type,
        or refused by the learner it configures. The message names it.
    """
    return from_dict(tomllib.loads(text))


def load(path: str | PathLike) -> Config:
    """Read a configuration from a TOML file, checked as :func:`from_dict` checks it.

    Raises
    ------
    OSError
        If the file cannot be opened.
    ValueError
        For a file that is not valid TOML or is refused, as :func:`loads`.
    ModuleNotFoundError
        Without labscript-utils, which the ``lyse`` extra installs.
    """
    # Imported here: labscript-utils is the lyse extra, and loads and from_dict
    # need none of it.
    from labscript_profile.toml_config import load_toml_file

    return from_dict(load_toml_file(path))


def from_dict(raw: dict) -> Config:
    """Build a :class:`Config` from already-parsed TOML, checked as strictly as a file.

    Raises
    ------
    ValueError
        For every complaint about the file, a missing setting included: a
        ``KeyError`` would show its sentence in quotes.
    """
    check_keys(raw)

    analysis = raw.get("ANALYSIS", {})
    general = raw.get("GENERAL", {})

    if "groups" not in analysis:
        raise ValueError(
            "ANALYSIS.groups is required: the list of groups whose parameters "
            "and globals take part. A group not listed is switched off."
        )
    active_groups = knobs.array("ANALYSIS.groups", analysis["groups"])
    # A group exists through the tables written under it, so a listed name with
    # none is a misspelling; a defined group that is not listed is switched off.
    defined = sorted(
        {*raw.get("PARAMETERS", {}), *raw.get("RUNMANAGER_GLOBALS", {})}
    )
    undefined = [group for group in active_groups if group not in defined]
    if undefined:
        raise ValueError(
            f"ANALYSIS.groups lists {', '.join(repr(g) for g in undefined)}, "
            f"which no [PARAMETERS.<group>] or [RUNMANAGER_GLOBALS.<group>] "
            f"table defines. The groups this file defines are {defined}."
        )

    parameters: list[Parameter] = []
    mappings: list[GlobalMapping] = []

    for group, entries in raw.get("PARAMETERS", {}).items():
        if group not in active_groups:
            continue
        for name, entry in entries.items():
            enabled = entry.get("enable", True)
            parameters.append(
                Parameter(
                    name=name,
                    minimum=float(entry["min"]),
                    maximum=float(entry["max"]),
                    start=None if entry.get("start") is None else float(entry["start"]),
                    enable=enabled,
                )
            )
            # A switched-off parameter gets no mapping, so its global keeps
            # runmanager's value.
            if enabled and "global_name" in entry:
                mappings.append(
                    GlobalMapping(name=entry["global_name"], expr=None, args=(name,))
                )

    for group, entries in raw.get("RUNMANAGER_GLOBALS", {}).items():
        if group not in active_groups:
            continue
        for name, entry in entries.items():
            # A switched-off global is not set; unlike a parameter, it has no
            # bounds to carry.
            if not entry.get("enable", True):
                continue
            mappings.append(
                GlobalMapping(
                    name=name,
                    expr=entry.get("expr"),
                    args=tuple(entry["args"]),
                )
            )

    if not parameters:
        raise ValueError(
            f"no parameters are enabled. ANALYSIS.groups is {active_groups!r}; "
            f"PARAMETERS defines {sorted(raw.get('PARAMETERS', {}))}"
        )

    # A switched-off parameter is unmapped on purpose, so nothing may depend on it.
    known = {p.name for p in parameters if p.enable}
    for mapping in mappings:
        for arg in mapping.args:
            if arg not in known:
                raise ValueError(
                    f"global {mapping.name!r} takes {arg!r}, which is not an "
                    f"enabled parameter. Enabled: {sorted(known)}"
                )
    for name in known:
        if not any(name in m.args for m in mappings):
            raise ValueError(
                f"parameter {name!r} is not mapped to any runmanager global. "
                f"Give it a global_name, or name it in the args of an entry "
                f"under RUNMANAGER_GLOBALS."
            )

    # A name an expr cannot see fails only when its lambda is called, so call
    # each mapping once with its parameters at the middle of their ranges.
    middle = {p.name: (p.minimum + p.maximum) / 2 for p in parameters}
    for mapping in mappings:
        try:
            mapping.evaluate(middle)
        except NameError as error:
            raise ValueError(
                f"the expr for global {mapping.name!r} uses {error.name!r}, "
                f"which runmanager's globals cannot see: {mapping.expr!r}"
            ) from error
        except Exception:
            # One failing point says nothing about the rest of the range.
            pass

    if "cost_key" not in analysis:
        raise ValueError("ANALYSIS.cost_key is required: [routine_name, result_name]")
    # A bare string is a sequence of characters, so the list check stops
    # ``cost_key = "rc"`` passing as a pair. What the pair holds is Config's to
    # say.
    knobs.array("ANALYSIS.cost_key", analysis["cost_key"])
    cost_key = tuple(analysis["cost_key"])

    learner_options = {
        name: dict(table) for name, table in raw.get("LEARNER", {}).items()
    }

    settings: dict[str, Any] = {
        **present(analysis, ("maximize",)),
        **present(general, GENERAL_KEYS),
    }

    config = Config(
        space=ParameterSpace(parameters),
        globals=tuple(mappings),
        cost_key=cost_key,
        learner_options=learner_options,
        **settings,
    )
    # Imported here so importing this module alone stays light. The learners'
    # constructors are the schema for their named tables.
    from .learners import build

    # Built to ask how many it proposes at a time, a fact about the object. It
    # fits nothing and is discarded: a session builds its own.
    learner = build(config)
    if "num_buffered_runs" in general and learner.generation is not None:
        raise ValueError(
            f"num_buffered_runs is not accepted with learner "
            f"{config.learner!r}, which proposes a whole generation of "
            f"{learner.generation} at a time and sets the queue depth to it. "
            f"Delete num_buffered_runs."
        )
    return config
