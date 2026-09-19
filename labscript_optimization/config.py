"""Reading the TOML configuration.

``[MLOOP_PARAMS.<group>.<name>]``
    One optimised parameter, with ``min``, ``max``, optional ``start``,
    optional ``enable`` (default true), and optional ``global_name``. Giving
    ``global_name`` is shorthand for a runmanager global that takes this
    parameter's value directly.

``[RUNMANAGER_GLOBALS.<group>.<name>]``
    A runmanager global computed from one or more parameters, with ``args``
    naming them and ``expr`` a lambda taking them in that order. Used when
    several parameters feed one global.

``[ANALYSIS] groups`` selects which groups take part. A parameter in a group
that is not listed is left out of the session entirely; one with
``enable = false`` in a group that is listed is carried, but is not searched
and gets no mapping, so its runmanager global keeps whatever value it already
holds.

Every key is either acted on or rejected. A setting this package does not read
-- one M-LOOP needed, or one spelt wrongly -- stops the load with a message
naming it: accepting it and ignoring it is how a lab comes to believe an
option is in force when nothing reads it. No analysislib-mloop file therefore
loads as it stands: one carried over has to be cut down to the keys this
module names first.

``[LEARNER.<name>]`` is the exception. Its contents belong to the learners:
one lab's table serves whichever learner is selected, and the factory passes
each learner the knobs its constructor takes.
"""

import tomllib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .space import Parameter, ParameterSpace

#: The tables a configuration file carries. Anything else at the top level is
#: a misspelling, and every setting written under it would go unread.
TOP_LEVEL_TABLES = frozenset(
    {
        "ANALYSIS",
        "LEARNER",
        "MLOOP",
        "MLOOP_PARAMS",
        "RUNMANAGER_GLOBALS",
    }
)

#: The settings ``[ANALYSIS]`` carries.
ANALYSIS_KEYS = frozenset({"cost_key", "groups", "maximize"})

#: Learner knobs that a configuration puts directly in ``[MLOOP]``.
SHARED_LEARNER_KEYS = frozenset(
    {
        "trust_region",
        "trust_range",
        "trust_gaussian",
        "explore_fraction",
        "cost_has_noise",
        "population_size",
        "evolution_strategy",
        "mutation_scale",
        "cross_over_probability",
        "restart_tolerance",
        "cost_bias",
        "uncer_bias",
        "generation_size",
        "length_scale_bounds",
        "noise_level_bounds",
        "minimum_observations",
    }
)

#: The session settings ``[MLOOP]`` carries, alongside the learner knobs in
#: :data:`SHARED_LEARNER_KEYS`.
MLOOP_KEYS = frozenset(
    {
        "learner",
        "max_num_runs",
        "max_num_runs_without_better_params",
        "num_buffered_runs",
        "num_training_runs",
        "seed",
        "session",
    }
)

#: The keys one ``[MLOOP_PARAMS.<group>.<name>]`` table carries.
PARAMETER_KEYS = frozenset({"enable", "global_name", "max", "min", "start"})

#: The keys one ``[RUNMANAGER_GLOBALS.<group>.<name>]`` table carries.
GLOBAL_KEYS = frozenset({"args", "enable", "expr"})


@dataclass(frozen=True)
class GlobalMapping:
    """One runmanager global, and how to compute it from the parameters.

    Args:
        name: The runmanager global to set.
        expr: Source of a lambda taking ``args`` in order, or ``None`` to pass
            the single argument through unchanged.
        args: Names of the parameters feeding this global.
    """

    name: str
    expr: str | None
    args: tuple[str, ...]
    #: ``expr`` as a callable, or ``None`` when the single argument passes
    #: through unchanged. Built here rather than handed in, so that the only
    #: way to get one is through the checking in ``__post_init__``.
    function: Callable[..., Any] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        """Turn ``expr`` into its callable now, so that a bad one stops the load.

        Evaluated on demand instead, a mistyped lambda would first be found on
        a proposal: mid-session, out through the worker's error path, which is
        the late failure the rest of this module exists to prevent.
        """
        if self.expr is None:
            return
        try:
            # The expression comes from the lab's own configuration file,
            # which is as trusted as the analysis routines themselves.
            function = eval(self.expr)  # noqa: S307
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
        object.__setattr__(self, "function", function)

    def evaluate(self, values: dict[str, float]) -> Any:
        """Compute this global's value from a parameter-name to value mapping."""
        arguments = [values[a] for a in self.args]
        if self.function is None:
            return arguments[0]
        return self.function(*arguments)


@dataclass
class Config:
    """Everything a session needs to run.

    ``session`` is a label and nothing more. It is reported among the
    routine's results so that a row can be attributed to the run that produced
    it; no matching is done on it. A cost reaches the proposal it answers by
    the shot id runmanager mints for its queue row, which is unique across
    runs on its own.
    """

    space: ParameterSpace
    globals: tuple[GlobalMapping, ...]
    cost_key: tuple[str, str]
    maximize: bool = False
    session: str = "default"
    learner: str = "gaussian_process"
    learner_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_buffered_runs: int = 3
    num_training_runs: int = 5
    max_num_runs: int | None = None
    max_num_runs_without_better_params: int | None = None
    seed: int | None = None

    @property
    def uncertainty_key(self) -> tuple[str, str]:
        """The column holding the uncertainty on the cost, by convention."""
        routine, result = self.cost_key
        return routine, f"u_{result}"

    def options_for(self, learner: str) -> dict[str, Any]:
        """Knobs for one learner: the shared ones, overridden per learner."""
        options = dict(self.learner_options.get("shared", {}))
        options.update(self.learner_options.get(learner, {}))
        return options

    def globals_for(self, params: Sequence[float]) -> dict[str, Any]:
        """The runmanager globals that realise one parameter vector."""
        values = {p.name: v for p, v in zip(self.space.parameters, params)}
        return {g.name: g.evaluate(values) for g in self.globals}


def _present(
    table: dict, keys: Sequence[str], convert: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    """The ``keys`` this table actually carries, coerced by ``convert``.

    A key the file leaves out is left out of the result, so :class:`Config`
    supplies it from the field's own default. That is why no default appears
    here: each one is written down once, on the dataclass, and cannot drift
    away from a second copy kept for the files that omit it.
    """
    return {
        key: table[key] if convert is None else convert(table[key])
        for key in keys
        if key in table
    }


def _reject_unknown(table: dict, allowed: frozenset[str], where: str) -> None:
    """Fail on any key of ``table`` that nothing in this package reads.

    The message names the table as well as the key because the same spelling
    can be a setting in one table and meaningless in another, and somebody
    editing a lab file has nothing to go on but what is printed here.
    """
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(
            f"{where} does not accept {', '.join(repr(k) for k in unknown)}. "
            f"It accepts: {', '.join(sorted(allowed))}."
        )


def _reject_unknown_keys(raw: dict) -> None:
    """Fail on every key in the file that nothing would act on.

    Accepting one and ignoring it is how a lab comes to believe a setting is
    in force when it is not, so a stale file is stopped at the door instead.

    Parameter and global tables are checked whether or not their group is
    active: their shape does not depend on that, and a typo left to load in a
    switched-off group waits for the day somebody switches the group on.
    """
    _reject_unknown(raw, TOP_LEVEL_TABLES, "the top level of the configuration")
    _reject_unknown(raw.get("ANALYSIS", {}), ANALYSIS_KEYS, "[ANALYSIS]")
    _reject_unknown(raw.get("MLOOP", {}), MLOOP_KEYS | SHARED_LEARNER_KEYS, "[MLOOP]")
    for table, allowed in (
        ("MLOOP_PARAMS", PARAMETER_KEYS),
        ("RUNMANAGER_GLOBALS", GLOBAL_KEYS),
    ):
        for group, entries in raw.get(table, {}).items():
            for name, entry in entries.items():
                _reject_unknown(entry, allowed, f"[{table}.{group}.{name}]")
    # [LEARNER.<name>] is left alone: those knobs are the learners' own, and
    # the factory takes the ones each constructor accepts.


def loads(text: str) -> Config:
    """Parse a configuration from TOML text."""
    return from_dict(tomllib.loads(text))


def load(path) -> Config:
    """Read a configuration from a TOML file."""
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f))


def from_dict(raw: dict) -> Config:
    """Build a :class:`Config` from already-parsed TOML.

    Every complaint about the file is a :class:`ValueError`, a missing setting
    as much as a contradictory one. The message is the whole of what somebody
    with a stale file gets, and ``KeyError`` reprs its argument: a sentence
    raised as one reaches the reader wrapped in quotes with its own quotes
    escaped.
    """
    _reject_unknown_keys(raw)

    analysis = raw.get("ANALYSIS", {})
    mloop = raw.get("MLOOP", {})

    active_groups = analysis.get("groups", [])

    parameters: list[Parameter] = []
    mappings: list[GlobalMapping] = []

    for group, entries in raw.get("MLOOP_PARAMS", {}).items():
        if group not in active_groups:
            # A group nobody switched on is not part of this session at all.
            continue
        for name, entry in entries.items():
            enabled = entry.get("enable", True)
            parameters.append(
                Parameter(
                    name=name,
                    global_name=entry.get("global_name", name),
                    minimum=float(entry["min"]),
                    maximum=float(entry["max"]),
                    start=None if entry.get("start") is None else float(entry["start"]),
                    enable=enabled,
                )
            )
            # A parameter that is switched off keeps whatever value runmanager
            # already holds, so it gets no mapping and is never set.
            if enabled and "global_name" in entry:
                mappings.append(
                    GlobalMapping(name=entry["global_name"], expr=None, args=(name,))
                )

    for group, entries in raw.get("RUNMANAGER_GLOBALS", {}).items():
        if group not in active_groups:
            # A group nobody switched on is not part of this session at all.
            continue
        for name, entry in entries.items():
            # A switched-off global is simply not set, so there is nothing to
            # carry: unlike a parameter, it has no bounds anybody looks at.
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
            f"MLOOP_PARAMS defines {sorted(raw.get('MLOOP_PARAMS', {}))}"
        )

    # Only searched parameters take part: a switched-off one is deliberately
    # left unmapped, and nothing may depend on it.
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

    if "cost_key" not in analysis:
        raise ValueError("ANALYSIS.cost_key is required: [routine_name, result_name]")
    cost_key = tuple(analysis["cost_key"])
    if len(cost_key) != 2:
        raise ValueError(
            f"ANALYSIS.cost_key must be [routine_name, result_name], got {cost_key!r}"
        )

    # Learner knobs written straight into [MLOOP] are the shared defaults, and
    # [LEARNER.<name>] overrides them for one learner.
    learner_options: dict[str, dict[str, Any]] = {
        "shared": {k: v for k, v in mloop.items() if k in SHARED_LEARNER_KEYS}
    }
    for name, table in raw.get("LEARNER", {}).items():
        learner_options[name] = dict(table)

    # Only the settings the file actually carries are passed on; Config fills
    # in the rest from its field defaults, which are the one place a default
    # is written down.
    settings: dict[str, Any] = {
        **_present(analysis, ("maximize",), bool),
        **_present(mloop, ("learner", "session"), str),
        **_present(mloop, ("num_buffered_runs", "num_training_runs"), int),
        **_present(
            mloop, ("max_num_runs", "max_num_runs_without_better_params", "seed")
        ),
    }

    return Config(
        space=ParameterSpace(parameters),
        globals=tuple(mappings),
        cost_key=cost_key,
        learner_options=learner_options,
        **settings,
    )
