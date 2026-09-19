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
that is not listed is left out entirely; one with ``enable = false`` in a group
that is listed is carried but not searched, and gets no mapping, so its
runmanager global keeps whatever value it already holds.

Every key must be one this package knows: a spelling it does not is refused
rather than accepted and ignored, so a file carried over from analysislib-mloop
has to be cut down to the keys named here before it will load.

Known is not the same as acted on. The learner knobs in ``[MLOOP]`` are the
union over every learner, and a learner is built with the ones its own
constructor takes, so whichever learner is named leaves the rest of them
unused -- which is what lets one file serve several. A ``[LEARNER.<name>]``
table says which learner it is for, so its keys are held to that constructor
and none of them goes unread.
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

#: Of those, the ones a parameter table must carry: bounds are what a search
#: has to have.
PARAMETER_REQUIRED = frozenset({"max", "min"})

#: The keys one ``[RUNMANAGER_GLOBALS.<group>.<name>]`` table carries.
GLOBAL_KEYS = frozenset({"args", "enable", "expr"})

#: Of those, the one a globals table must carry: ``expr`` is optional, the
#: parameters it is given are not.
GLOBAL_REQUIRED = frozenset({"args"})


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
    #: through unchanged. Not an init field: the only way to get one is through
    #: the checking in ``__post_init__``.
    function: Callable[..., Any] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        """Build ``expr``'s callable now, so a bad one raises here.

        Left until a proposal needs it, a mistyped lambda would first be found
        mid-session, out through the worker's error path.
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

    ``session`` is a label and nothing more: it is reported among the routine's
    results so that a row can be attributed to the run that produced it, and
    nothing is matched on it.
    """

    space: ParameterSpace
    globals: tuple[GlobalMapping, ...]
    cost_key: tuple[str, str]
    maximize: bool = False
    session: str = "default"
    learner: str = "gaussian_process"
    shared_learner_options: dict[str, Any] = field(default_factory=dict)
    learner_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_buffered_runs: int = 3
    num_training_runs: int = 5
    max_num_runs: int | None = None
    #: Stop after this many completed shots without a better cost. Every
    #: completed shot counts, including one whose cost was not usable: it is
    #: still a shot spent, and counting only the usable ones would let a
    #: session whose detector has died run for ever on the limit meant to
    #: stop it.
    max_num_runs_without_better_params: int | None = None
    seed: int | None = None

    @property
    def uncertainty_key(self) -> tuple[str, str]:
        """The column holding the uncertainty on the cost, by convention."""
        routine, result = self.cost_key
        return routine, f"u_{result}"

    def options_for(self, learner: str) -> dict[str, Any]:
        """Knobs for one learner: the shared ones, overridden per learner."""
        options = dict(self.shared_learner_options)
        options.update(self.learner_options.get(learner, {}))
        return options

    def globals_for(self, params: Sequence[float]) -> dict[str, Any]:
        """The runmanager globals that realise one parameter vector."""
        values = {p.name: v for p, v in zip(self.space.parameters, params)}
        return {g.name: g.evaluate(values) for g in self.globals}


def present(
    table: dict, keys: Sequence[str], convert: Callable[[Any], Any] | None = None
) -> dict[str, Any]:
    """The ``keys`` this table actually carries, coerced by ``convert``.

    A key the file leaves out is left out of the result, so :class:`Config`
    supplies it from the field's own default. No default is written here: each
    one lives on the dataclass and nowhere else, so none can drift.
    """
    found: dict[str, Any] = {}
    for key in (k for k in keys if k in table):
        try:
            found[key] = table[key] if convert is None else convert(table[key])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{key} must be readable as {convert.__name__}, got {table[key]!r}"
            ) from error
    return found


def reject_unknown(table: dict, allowed: frozenset[str], where: str) -> None:
    """Fail on any key of ``table`` that nothing in this package reads.

    The message names the table as well as the key, because the same spelling
    can be a setting in one table and meaningless in another.
    """
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(
            f"{where} does not accept {', '.join(repr(k) for k in unknown)}. "
            f"It accepts: {', '.join(sorted(allowed))}."
        )


def require_present(table: dict, keys: frozenset[str], where: str) -> None:
    """Fail on a required key the table leaves out, in the voice a typo gets."""
    missing = sorted(keys - set(table))
    if missing:
        raise ValueError(
            f"{where} is missing {', '.join(repr(k) for k in missing)}. "
            f"It requires: {', '.join(sorted(keys))}."
        )


def require_type(value: Any, kind: type, where: str) -> Any:
    """Fail on a setting of the wrong type, which no spelling check catches.

    A quoted boolean is truthy and a bare string is a sequence of its own
    characters, so either goes through and acts as something nobody wrote.
    """
    if not isinstance(value, kind):
        written = {bool: "true or false, unquoted", list: "a list in [brackets]"}[kind]
        raise ValueError(f"{where} must be written as {written}, not {value!r}.")
    return value


def check_keys(raw: dict) -> None:
    """Fail on a key this package does not know, and on a required one left out.

    Accepting a key and ignoring it is how a lab comes to believe a setting is
    in force when it is not, so a stale file is stopped at the door instead. A
    key that is known may still go unused: the learner knobs in ``[MLOOP]``
    cover every learner between them and only the selected learner's are read,
    which is the price of a file that keeps working when the learner changes.

    Parameter and global tables are checked whether or not their group is
    active: a typo left to load in a switched-off group waits for the day
    somebody switches the group on.
    """
    reject_unknown(raw, TOP_LEVEL_TABLES, "the top level of the configuration")
    reject_unknown(raw.get("ANALYSIS", {}), ANALYSIS_KEYS, "[ANALYSIS]")
    reject_unknown(raw.get("MLOOP", {}), MLOOP_KEYS | SHARED_LEARNER_KEYS, "[MLOOP]")
    for table, allowed, required in (
        ("MLOOP_PARAMS", PARAMETER_KEYS, PARAMETER_REQUIRED),
        ("RUNMANAGER_GLOBALS", GLOBAL_KEYS, GLOBAL_REQUIRED),
    ):
        for group, entries in raw.get(table, {}).items():
            for name, entry in entries.items():
                where = f"[{table}.{group}.{name}]"
                reject_unknown(entry, allowed, where)
                require_present(entry, required, where)
                if "args" in entry:
                    require_type(entry["args"], list, f"{where} args")
                if "enable" in entry:
                    require_type(entry["enable"], bool, f"{where} enable")
    # [LEARNER.<name>] is left to learners.validate_options, which is the only
    # thing that knows one learner's keys: the table names its learner, so that
    # learner's constructor is the schema for it.


def loads(text: str) -> Config:
    """Parse a configuration from TOML text."""
    return from_dict(tomllib.loads(text))


def load(path) -> Config:
    """Read a configuration from a TOML file."""
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f))


def from_dict(raw: dict) -> Config:
    """Build a :class:`Config` from already-parsed TOML.

    ``raw`` is the whole file as nested dictionaries, checked as strictly as
    one read from disk.

    Every complaint about the file is a :class:`ValueError`, a missing setting
    as much as a contradictory one, because ``KeyError`` reprs its argument and
    a written-out sentence raised as one reaches the reader in quotes with its
    own quotes escaped.
    """
    check_keys(raw)

    analysis = raw.get("ANALYSIS", {})
    mloop = raw.get("MLOOP", {})

    if "maximize" in analysis:
        require_type(analysis["maximize"], bool, "ANALYSIS.maximize")
    active_groups = require_type(analysis.get("groups", []), list, "ANALYSIS.groups")

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
    require_type(analysis["cost_key"], list, "ANALYSIS.cost_key")
    cost_key = tuple(analysis["cost_key"])
    if len(cost_key) != 2:
        raise ValueError(
            f"ANALYSIS.cost_key must be [routine_name, result_name], got {cost_key!r}"
        )

    # Learner knobs written straight into [MLOOP] are the shared defaults, and
    # [LEARNER.<name>] overrides them for one learner.
    shared_learner_options = {
        key: value for key, value in mloop.items() if key in SHARED_LEARNER_KEYS
    }
    learner_options = {
        name: dict(table) for name, table in raw.get("LEARNER", {}).items()
    }

    settings: dict[str, Any] = {
        **present(analysis, ("maximize",)),
        **present(mloop, ("learner", "session"), str),
        **present(
            mloop,
            (
                "max_num_runs",
                "max_num_runs_without_better_params",
                "num_buffered_runs",
                "num_training_runs",
                "seed",
            ),
            int,
        ),
    }

    config = Config(
        space=ParameterSpace(parameters),
        globals=tuple(mappings),
        cost_key=cost_key,
        shared_learner_options=shared_learner_options,
        learner_options=learner_options,
        **settings,
    )
    # Learner constructors are the authoritative schema for their named
    # tables. Import lazily so importing this module alone stays lightweight.
    from .learners import validate_options

    validate_options(config)
    return config
