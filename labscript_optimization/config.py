"""Reading the TOML configuration.

``[PARAMETERS.<group>.<name>]``
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
rather than accepted and ignored, so a stale file has to be cut down to the
keys named here before it will load.

``[GENERAL]`` carries the session's own settings -- which learner runs, which
learner trains it, how deep the queue is, what stops the run -- and a learner's
knobs are written in ``[LEARNER.<name>]``, the table of the learner that takes
them. A named table is held to that learner's constructor, so every key in it
is a knob that learner takes, and a knob found in ``[GENERAL]`` is refused with
the tables it belongs in named. Two learners run whenever the one selected has
a trainer, so a knob both take is written twice, once in each table, and each
gets its own value.

Known is not the same as acted on: a table written for a learner no session
builds goes unread, which is what lets one file carry the settings for several
learners and switch between them.

A parameter name and a global name are each unique across the active groups:
both are looked up by name when a proposal is turned into runmanager globals,
so a repeat would quietly give one value to two places.
"""

import inspect
import tomllib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .space import Parameter, ParameterSpace

#: The tables a configuration file carries. Anything else at the top level is
#: a misspelling, and every setting written under it would go unread.
TOP_LEVEL_TABLES = frozenset(
    {
        "ANALYSIS",
        "GENERAL",
        "LEARNER",
        "PARAMETERS",
        "RUNMANAGER_GLOBALS",
    }
)

#: The tables this package used to name after M-LOOP, and what each is called
#: now. It replaces M-LOOP and carries none of its code, so a table named after
#: it was a name nothing written under it answered to. They are kept here to be
#: refused by name: aliased to their replacements instead, an old file would go
#: on loading and a lab would go on typing the name of a tool it is not
#: running.
RENAMED_TABLES = {"MLOOP": "GENERAL", "MLOOP_PARAMS": "PARAMETERS"}

#: The settings ``[ANALYSIS]`` carries.
ANALYSIS_KEYS = frozenset({"cost_key", "groups", "maximize"})

#: The whole of what ``[GENERAL]`` carries: the session's own settings and
#: nothing else. Each is the name of a :class:`Config` field and is handed over
#: under that name, so the two lists cannot drift apart. A learner's knobs are
#: not here; they are written in that learner's own table.
GENERAL_KEYS = frozenset(
    {
        "learner",
        "max_num_runs",
        "max_num_runs_without_better_params",
        "num_buffered_runs",
        "num_runs_between_trainer_runs",
        "num_training_runs",
        "seed",
        "trainer",
    }
)

#: The whole-number settings of :class:`Config`, the smallest value each one
#: accepts, and why that is the floor. Most of these floors stand between a
#: file and a session that dies without saying anything: it refills empty for
#: as long as it is asked to, counts no starvation, and leaves no stop reason,
#: because ``check_stop`` is reached only from ``record`` and nothing is ever
#: recorded to reach it with.
INTEGER_SETTINGS = {
    "num_buffered_runs": (1, "a queue holding none of our shots is never refilled"),
    "num_training_runs": (0, "a negative number of training shots is not a number"),
    "num_runs_between_trainer_runs": (
        1,
        "no runs between one trainer run and the next is every run a trainer "
        "run, and the learner the file names never proposing at all; leave it "
        "out for a run that never goes back to the trainer",
    ),
    "seed": (0, "numpy's generator is seeded from a non-negative integer"),
    "max_num_runs": (
        1,
        "a budget of no runs is a session that never starts rather than one "
        "that stops, and nothing is ever submitted to stop it",
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
        "num_runs_between_trainer_runs",
        "seed",
    }
)

#: The keys one ``[PARAMETERS.<group>.<name>]`` table carries.
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
        mid-session, out through the worker's error path. The arguments are
        held to the same moment: a callable that cannot take the ones it is
        configured with fails in exactly that place otherwise.
        """
        if self.expr is None:
            if len(self.args) != 1:
                raise ValueError(
                    f"global {self.name!r} has no expr, so its value is the "
                    f"one parameter it takes; it names {len(self.args)}: "
                    f"{list(self.args)}. Give it an expr taking them all, or "
                    f"name one parameter."
                )
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
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            # Some callables do not expose one. Refusing on a signature that
            # cannot be read would reject a mapping that works.
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
        return self.function(*arguments)


@dataclass
class Config:
    """Everything a session needs to run."""

    space: ParameterSpace
    globals: tuple[GlobalMapping, ...]
    cost_key: tuple[str, str]
    maximize: bool = False
    learner: str = "gaussian_process"
    #: The learner that runs the training shots for a ``learner`` that needs
    #: them, and the periodic runs after them. Read only for such a learner: a
    #: file naming it beside one that trains itself is refused rather than left
    #: with a setting nothing acts on.
    trainer: str = "directed_random"
    #: One table of knobs per learner, by learner name. A knob is written in
    #: the table of the learner that takes it and reaches no other, which is
    #: what lets a trainer and a main learner be given different values of the
    #: same knob.
    learner_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: How many of this session's shots to keep in runmanager's queue. Read
    #: for a learner asked for any number of proposals at a time; a learner
    #: declaring a generation sets its own depth from that declaration, and a
    #: file writing this beside one is refused rather than left with two
    #: settings for the same number.
    num_buffered_runs: int = 3
    #: How many usable observations the ``trainer`` gathers before ``learner``
    #: takes over. At least that learner's own ``minimum_observations``, or the
    #: handover could not happen where this says it does and the file is
    #: refused.
    num_training_runs: int = 5
    #: How many consecutive proposals come from ``learner`` between one
    #: proposal from ``trainer`` and the next, once training is over. Unset,
    #: the run never goes back to the trainer after the handover; set, one
    #: proposal in every cycle of this many plus one is the trainer's, which
    #: goes on widening the history under a learner that is narrowing onto the
    #: best point it has found. Read only for a ``learner`` that trains.
    num_runs_between_trainer_runs: int | None = None
    max_num_runs: int | None = None
    #: Stop after this many completed shots without a better cost. Every
    #: completed shot counts, including one whose cost was not usable: it is
    #: still a shot spent, and counting only the usable ones would let a
    #: session whose detector has died run for ever on the limit meant to
    #: stop it.
    max_num_runs_without_better_params: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Hold every field to what it is allowed to be.

        This is the one authority for them, so a :class:`Config` written out
        in a script or a test is checked exactly as one parsed from a file:
        :func:`from_dict` hands over what the file said, and this decides
        whether it can be run.
        """
        for key, (floor, why) in INTEGER_SETTINGS.items():
            value = getattr(self, key)
            if value is None and key in OPTIONAL_INTEGER_SETTINGS:
                continue
            # ``isinstance(True, int)`` is True, so bool is excluded by name:
            # a bare integer check reads ``true`` as a budget of one run.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{key} must be written as a whole number, got {value!r}."
                )
            if value < floor:
                raise ValueError(f"{key} must be at least {floor}, got {value}: {why}.")

        if not isinstance(self.maximize, bool):
            raise ValueError(
                f"maximize must be written as true or false, unquoted, not "
                f"{self.maximize!r}."
            )
        for key in ("learner", "trainer"):
            value = getattr(self, key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be written as a string, got {value!r}.")

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
                f"runmanager global. Each global is set once per shot, so the "
                f"last mapping written would be the only one that took effect; "
                f"the globals set are {names}"
            )

    @property
    def uncertainty_key(self) -> tuple[str, str]:
        """The column holding the uncertainty on the cost, by convention."""
        routine, result = self.cost_key
        return routine, f"u_{result}"

    def globals_for(self, params: Sequence[float]) -> dict[str, Any]:
        """The runmanager globals that realise one parameter vector."""
        values = {p.name: v for p, v in zip(self.space.parameters, params)}
        return {g.name: g.evaluate(values) for g in self.globals}


def present(table: dict, keys: Iterable[str]) -> dict[str, Any]:
    """The ``keys`` this table actually carries, exactly as they were written.

    A key the file leaves out is left out of the result, so :class:`Config`
    supplies it from the field's own default. No default is written here: each
    one lives on the dataclass and nowhere else, so none can drift. Nothing is
    coerced either: what a setting may be is
    :meth:`Config.__post_init__`'s to say, and a value converted on the way
    past would reach it as something nobody wrote.
    """
    return {key: table[key] for key in keys if key in table}


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


def reject_misplaced_knobs(general: dict) -> None:
    """Fail on a learner's knob written in ``[GENERAL]``, naming where it goes.

    ``[GENERAL]`` is read once for the whole session, and a session runs two
    learners whenever the one it names has a trainer. A knob written here would
    reach both of them with no way to tell them apart, so a trust region wide
    enough to train with and one tight enough to refine with cannot both be
    asked for; and it would be dropped in silence for every learner whose
    constructor does not take it.

    Which keys those are is read from the constructors, so a learner that gains
    a knob moves that knob's message with it.
    """
    # Lazily, so importing this module alone stays lightweight.
    from .learners import knobs_by_learner

    knobs = knobs_by_learner()
    misplaced = [key for key in sorted(general) if key in knobs]
    if misplaced:
        written = []
        for key in misplaced:
            tables = [f"[LEARNER.{name}]" for name in knobs[key]]
            last = tables.pop()
            joined = f"{', '.join(tables)} or {last}" if tables else last
            written.append(f"{key!r} in {joined}")
        where = "; ".join(written)
        raise ValueError(
            f"[GENERAL] carries the session's own settings and no learner's "
            f"knobs, so write {where}. A knob here reaches a learner and its "
            f"trainer alike, with no way to give them different values, and "
            f"goes unread by any learner that does not take it."
        )


def check_keys(raw: dict) -> None:
    """Fail on a key this package does not know, and on a required one left out.

    Accepting a key and ignoring it is how a lab comes to believe a setting is
    in force when it is not, so a stale file is stopped at the door instead. A
    learner knob written in ``[GENERAL]`` is refused by that rule and gets its
    own message, naming the tables it could have been written in: it is a real
    knob in the wrong place, not a spelling nothing here knows.

    Parameter and global tables are checked whether or not their group is
    active: a typo left to load in a switched-off group waits for the day
    somebody switches the group on.

    A table this package has renamed is refused ahead of both, naming its
    replacement, because the spelling check would call it a typo and leave the
    reader to guess which of the tables it lists was meant.
    """
    renamed = [
        f"[{table}] is now [{RENAMED_TABLES[table]}]"
        for table in sorted(raw)
        if table in RENAMED_TABLES
    ]
    if renamed:
        raise ValueError(
            f"the configuration no longer names its tables after M-LOOP, "
            f"which this package replaces and carries no code from: "
            f"{'; '.join(renamed)}. Rename the table; nothing is read under "
            f"the old name."
        )
    reject_unknown(raw, TOP_LEVEL_TABLES, "the top level of the configuration")
    reject_unknown(raw.get("ANALYSIS", {}), ANALYSIS_KEYS, "[ANALYSIS]")
    reject_misplaced_knobs(raw.get("GENERAL", {}))
    reject_unknown(raw.get("GENERAL", {}), GENERAL_KEYS, "[GENERAL]")
    for table, allowed, required in (
        ("PARAMETERS", PARAMETER_KEYS, PARAMETER_REQUIRED),
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
    general = raw.get("GENERAL", {})

    active_groups = require_type(analysis.get("groups", []), list, "ANALYSIS.groups")

    parameters: list[Parameter] = []
    mappings: list[GlobalMapping] = []

    for group, entries in raw.get("PARAMETERS", {}).items():
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
            f"PARAMETERS defines {sorted(raw.get('PARAMETERS', {}))}"
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
    # A bare string is a sequence of its own characters, so the list check is
    # what stops ``cost_key = "rc"`` passing for a pair of one-letter names.
    # What that pair may hold is Config's to say.
    require_type(analysis["cost_key"], list, "ANALYSIS.cost_key")
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
    # Learner constructors are the authoritative schema for their named
    # tables. Import lazily so importing this module alone stays lightweight.
    from .learners import NEEDS_TRAINING, build

    # Both settings describe the trainer's part in the run, so both go unread
    # for a learner that has no trainer. Named together because a file usually
    # carries both and would otherwise be refused twice over.
    idle = [
        key
        for key in ("num_runs_between_trainer_runs", "trainer")
        if key in general
    ]
    if idle and config.learner not in NEEDS_TRAINING:
        raise ValueError(
            f"{' and '.join(idle)} {'are' if len(idle) > 1 else 'is'} not "
            f"accepted with learner {config.learner!r}, which proposes from "
            f"the first shot and so runs no training phase: no trainer is "
            f"built, and what is written here would be a setting nothing acts "
            f"on. Delete it, or name a learner that trains: "
            f"{sorted(NEEDS_TRAINING)}."
        )

    # Build the selected learner and ask it, rather than predicting from its
    # class or its constructor what an instance would say. How many proposals
    # a learner makes at a time is a fact about the object, and nothing here
    # knows how it arrives at one. Construction is all this costs: no learner
    # fits anything until it is asked to propose, and the built learner is
    # discarded -- a session builds its own, from this same configuration.
    learner = build(config)
    if "num_buffered_runs" in general and learner.generation is not None:
        raise ValueError(
            f"num_buffered_runs is not accepted with learner "
            f"{config.learner!r}, which proposes one whole generation of "
            f"{learner.generation} at a time and waits for all of it. Its "
            f"queue depth is therefore that generation, and a second setting "
            f"for the same number is one that can disagree with it: delete "
            f"num_buffered_runs."
        )
    return config
