# labscript-optimization: second review

Work items from a second review of the package, each confirmed by loading a
configuration or driving a learner before it was written down. Three of the
five are a setting that reaches the session as something nobody wrote; one is
an algorithm that does not do what its docstring says; one is a stop reason
overwritten by a later one.

House rules that apply to every slice:

- Every test must be **proven to fail** by mutating the code it guards. Run the
  mutation, capture the failure, revert it, report both. A test that cannot be
  made to fail is a finding, not a deliverable. A surviving mutation is either
  answered by a stronger test or reported as semantically equivalent, with the
  reason.
- A comment or docstring states what the code does now. No "previously", no
  "used to". Prose that describes behaviour is a claim under test: when
  behaviour changes, grep the package, `README.md`, `UPGRADING.md` and
  `examples/` for every sentence that described the old behaviour.
- No settings accepted for compatibility. An unknown or unusable key is refused
  at load, naming the key and what is accepted instead.
- No tiny wrapper helpers unless they remove real duplication.
- The name guard in `tests/test_package.py` keeps its allowances as they are.
- Branch is `Development`, one commit per slice, not pushed. The suite stands
  at 397 tests.

Out of bounds for all five: the routine's `catch_up`/`unreported`/`n_sequences`
logic, `TwoPhaseLearner`, the session's refill and queue-depth logic,
`num_buffered_runs`, `num_training_runs`, the periodic trainer and the `phase`
column. `session.py` is touched only for A6.

## Checklist

- [x] A2: Learner knobs are refused, not coerced
- [x] A3: A group listed in `groups` that no table defines is refused
- [x] A4: Differential evolution redraws its weight once per generation
- [x] A5: runmanager receives plain Python values
- [x] A6: The first stop reason is kept

---

## A2: Learner knobs are refused, not coerced

### Type

`AFK`

### What to build

`[GENERAL]` refuses a mistyped setting through `Config.__post_init__`,
`INTEGER_SETTINGS` and `require_type`. A `[LEARNER.<name>]` table goes straight
to its learner's constructor, and the constructors coerce:

- `cost_has_noise = "false"` (Gaussian process) and `trust_gaussian = "false"`
  (directed random) load as `True`, because `bool()` of a non-empty string is
  true. That is the quoted-boolean trap the package already refuses for
  `maximize` and `enable`.
- `mutation_scale = 0.8` (differential evolution), `trust_range = 0.5`
  (directed random) and `length_scale_bounds = 5` (Gaussian process) crash with
  a bare `TypeError` out of `tuple()` or `len()`.
- `population_size = 8.9` becomes 8, and `refit_interval = true` becomes 1.
- `trust_region = "0.1"` passes `np.isscalar` in
  `ParameterSpace.absolute_trust_region` and is coerced by `float()`.

The constructors are the schema for their tables, so the check belongs there,
where direct Python construction meets it too. The same four kinds of check
recur across three learners, which is real duplication and justifies one shared
place for them:

- a boolean knob is a `bool`;
- an integer knob is an integer and not a `bool`;
- a number is a real number and not a `bool` or a `str`; numpy numeric scalars
  count;
- a pair is a two-element list, tuple or numpy array of numbers, not a string
  and not a scalar.

Each refusal is a `ValueError` naming the knob, what was written, and what is
accepted. `mutation_scale` given one number says to write `[0.8, 0.8]` for a
fixed weight.

### Acceptance criteria

- [x] Every knob of `RandomLearner`, `DirectedRandomLearner`,
      `DifferentialEvolutionLearner` and `GaussianProcessLearner` is checked
      for its kind, and refused with a `ValueError` naming it
- [x] A quoted boolean is refused for `cost_has_noise` and `trust_gaussian`
- [x] A single number is refused for `mutation_scale`, `trust_range`,
      `length_scale_bounds` and `noise_level_bounds`, and `mutation_scale`'s
      refusal says to write a pair for a fixed weight
- [x] A fractional or boolean integer knob is refused, not truncated
- [x] A quoted `trust_region` is refused by `ParameterSpace.absolute_trust_region`
- [x] Tuples, lists, numpy arrays and numpy scalars are still accepted wherever
      a pair or a number is
- [x] The same refusals reach a configuration loaded from TOML

### Blocked by

None - can start immediately.

---

## A3: A group listed in `groups` that no table defines is refused

### Type

`AFK`

### What to build

`[ANALYSIS] groups = ["CMOT", "SHIMSS"]` beside a `[PARAMETERS.SHIMS.b]` table
loads and searches CMOT's parameters alone. The misspelt group matches nothing,
SHIMS is left out, its globals are never set, and the lab believes it is
optimising `b`.

Refuse any name in `groups` that no `[PARAMETERS.<group>]` or
`[RUNMANAGER_GLOBALS.<group>]` table defines, naming it and the groups that do
exist. A group that is defined and not listed stays deliberately switched off,
which is not an error.

### Acceptance criteria

- [x] A name in `groups` that no table defines is refused, naming it and the
      groups the file defines
- [x] A group defined only under `[RUNMANAGER_GLOBALS]` counts as defined
- [x] A group defined and not listed still loads, switched off

### Blocked by

None - can start immediately.

---

## A4: Differential evolution redraws its weight once per generation

### Type

`AFK`

### What to build

The `mutation_scale` docstring says the differential weight is "redrawn each
generation", and scipy redraws it once per generation in
`DifferentialEvolutionSolver.__next__`. The code draws it in `trial()`, once
per member. The package claims textbook differential evolution, so the code is
brought to the claim: one draw per generation.

Under the generation barrier one `propose` call is one generation. The first
call may be one short, because the session places a configured start at
position 0, and every proposal in that call is a founder.

The benchmark sweep is not re-run: `benchmarks/README.md` pins its numbers to
the commit that produced them, and the re-run is scheduled after the redesign.
Prose that the change makes false is corrected.

### Acceptance criteria

- [x] Every trial in one generation shares one differential weight, and
      successive generations draw different ones, shown by recovering the
      weight from the trials themselves
- [x] No sentence in the package, `README.md`, `UPGRADING.md`, `examples/` or
      `benchmarks/README.md` describes a per-member draw

### Blocked by

None - can start immediately.

---

## A5: runmanager receives plain Python values

### Type

`AFK`

### What to build

runmanager writes each submitted global into its expression field with
`repr(value)`. Under numpy 2, `repr(np.float64(0.25))` is `'np.float64(0.25)'`,
which is what the operator sees and what the shot file stores; it evaluates only
because runmanager's sandbox star-imports pylab. The values come from
`Config.globals_for`, via `RunmanagerInterface.submit`, fed from a numpy array.

Make every entry handed to `client.submit_shots` hold plain Python values. The
runmanager half is a separate item and is not touched here.

### Acceptance criteria

- [x] A `global_name` mapping submits a Python `float`
- [x] An `expr` mapping such as `lambda a, b: (a, b)` submits a tuple of Python
      `float`s, with no numpy scalar anywhere inside it
- [x] Each submitted value's `repr` evaluates without numpy in scope

### Blocked by

None - can start immediately.

---

## A6: The first stop reason is kept

### Type

`AFK`

### What to build

When trailing work raises, `worker.run` sets `session.stopped` to
`"stopped by an error"`. The shots already in flight go on reporting, and
`Session.check_stop` then overwrites the reason with `"reached max_num_runs
(N)"` or the patience message, so the run's recorded cause is wrong. Keep the
first reason.

The worker is the other writer of the reason, and it overwrites too: a session
that ran out of patience and then had its trailing work raise came to read
`"stopped by an error"`. The same rule covers it, so both writers go through
one method that sets the reason only when there is none.

### Acceptance criteria

- [x] A session stopped by an error still reads `"stopped by an error"` after
      in-flight shots report enough to reach `max_num_runs`
- [x] Likewise after they reach the patience limit
- [x] A session stopped by its patience keeps that reason when in-flight shots
      then reach `max_num_runs`, even with the patience limit no longer holding
- [x] A session already stopped keeps its reason when trailing work raises
      afterwards, and the error still reaches the routine
- [x] A running session still stops on either limit

### Blocked by

None - can start immediately.
