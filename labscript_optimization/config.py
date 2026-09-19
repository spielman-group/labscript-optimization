"""Reading the TOML configuration.

The schema is the one analysislib-mloop used, so existing lab configuration
files load unchanged:

``[MLOOP_PARAMS.<group>.<name>]``
    One optimised parameter, with ``min``, ``max``, optional ``start``,
    optional ``enable`` (default true), and optional ``global_name``. Giving
    ``global_name`` is shorthand for a runmanager global that takes this
    parameter's value directly.

``[RUNMANAGER_GLOBALS.<group>.<name>]``
    A runmanager global computed from one or more parameters, with ``args``
    naming them and ``expr`` a lambda taking them in that order. Used when
    several parameters feed one global.

``[ANALYSIS] groups`` selects which groups take part; a parameter in a group
that is not listed is left out entirely, as is one with ``enable = false``.

Keys M-LOOP needed and this package does not -- ``visualisations``,
``no_delay``, archive paths -- are ignored rather than rejected, so a file that
still carries them keeps working.
"""

import tomllib
from dataclasses import dataclass, field
from typing import Any, Sequence

from .space import Parameter, ParameterSpace

#: Keys that M-LOOP needed and this package has no use for. Ignored on load.
OBSOLETE_KEYS = frozenset(
    {
        "no_delay",
        "visualisations",
        "visualizations",
        "archive_type",
        "archive_filename",
        "controller_archive_filename",
        "learner_archive_filename",
        "controller_archive_file_type",
        "learner_archive_file_type",
    }
)

#: Learner knobs that older configurations put directly in ``[MLOOP]``.
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

    def evaluate(self, values: dict[str, float]) -> Any:
        """Compute this global's value from a parameter-name to value mapping."""
        arguments = [values[a] for a in self.args]
        if self.expr is None:
            return arguments[0]
        # The expression comes from the lab's own configuration file, which is
        # as trusted as the analysis routines themselves.
        return eval(self.expr)(*arguments)  # noqa: S307


@dataclass
class Config:
    """Everything a session needs to run."""

    space: ParameterSpace
    globals: tuple[GlobalMapping, ...]
    cost_key: tuple[str, str]
    maximize: bool = False
    ignore_bad: bool = False
    session: str = "default"
    learner: str = "gaussian_process"
    learner_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    num_buffered_runs: int = 3
    num_training_runs: int = 5
    max_num_runs: int | None = None
    max_num_runs_without_better_params: int | None = None
    mock: bool = False
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


def _enabled(entry: dict, group: str, active_groups: Sequence[str]) -> bool:
    return group in active_groups and entry.get("enable", True)


def loads(text: str) -> Config:
    """Parse a configuration from TOML text."""
    return from_dict(tomllib.loads(text))


def load(path) -> Config:
    """Read a configuration from a TOML file."""
    with open(path, "rb") as f:
        return from_dict(tomllib.load(f))


def from_dict(raw: dict) -> Config:
    """Build a :class:`Config` from already-parsed TOML."""
    analysis = raw.get("ANALYSIS", {})
    mloop = raw.get("MLOOP", {})
    compilation = raw.get("COMPILATION", {})

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
                    minimum=float(entry["minimum"] if "minimum" in entry else entry["min"]),
                    maximum=float(entry["maximum"] if "maximum" in entry else entry["max"]),
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
        for name, entry in entries.items():
            if not _enabled(entry, group, active_groups):
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
                raise KeyError(
                    f"global {mapping.name!r} takes {arg!r}, which is not an "
                    f"enabled parameter. Enabled: {sorted(known)}"
                )
    for name in known:
        if not any(name in m.args for m in mappings):
            raise KeyError(
                f"parameter {name!r} is not mapped to any runmanager global. "
                f"Give it a global_name, or name it in the args of an entry "
                f"under RUNMANAGER_GLOBALS."
            )

    if "cost_key" not in analysis:
        raise KeyError("ANALYSIS.cost_key is required: [routine_name, result_name]")
    cost_key = tuple(analysis["cost_key"])
    if len(cost_key) != 2:
        raise ValueError(
            f"ANALYSIS.cost_key must be [routine_name, result_name], got {cost_key!r}"
        )

    # Learner knobs: the ones old configurations put in [MLOOP] become the
    # shared defaults, and [LEARNER.<name>] overrides them per learner.
    learner_options: dict[str, dict[str, Any]] = {
        "shared": {k: v for k, v in mloop.items() if k in SHARED_LEARNER_KEYS}
    }
    for name, table in raw.get("LEARNER", {}).items():
        learner_options[name] = dict(table)

    return Config(
        space=ParameterSpace(parameters),
        globals=tuple(mappings),
        cost_key=cost_key,
        maximize=bool(analysis.get("maximize", False)),
        ignore_bad=bool(analysis.get("ignore_bad", False)),
        session=str(mloop.get("session", "default")),
        learner=mloop.get("learner", mloop.get("controller_type", "gaussian_process")),
        learner_options=learner_options,
        num_buffered_runs=int(mloop.get("num_buffered_runs", 3)),
        num_training_runs=int(mloop.get("num_training_runs", 5)),
        max_num_runs=mloop.get("max_num_runs"),
        max_num_runs_without_better_params=mloop.get(
            "max_num_runs_without_better_params"
        ),
        mock=bool(compilation.get("mock", False)),
        seed=mloop.get("seed"),
    )
