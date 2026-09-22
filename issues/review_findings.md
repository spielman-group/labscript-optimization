# labscript-optimization: crash paths a configuration can reach

Work items from a whole-package review — all 3277 lines of source, not a diff —
triaged jointly with a second model. Six of the eight findings are a
configuration or a history reaching a crash that says nothing useful; the other
two are a document cited from the wrong place and a test asserting through a
private attribute.

The thread running through the first three slices is one rule this package
already keeps everywhere else and breaks in these places: **a setting that
cannot work is refused at the first moment it is knowable, in this package's
own voice.** Every finding below is a place where the refusal is missing and
numpy, scikit-learn or the TOML reader speaks instead.

House rules that apply to every slice:

- Every test must be **proven to fail** by mutating the code it guards. Run the
  mutation, capture the failure, revert it, report both. A test that cannot be
  made to fail is a finding, not a deliverable.
- A comment or docstring states what the code does now. No "previously", no
  "used to". Prose that describes behaviour is a claim under test: when
  behaviour changes, grep for every sentence that described the old behaviour.
- No settings accepted for compatibility. An unknown or unusable key is refused
  at load, naming the key and what is accepted instead.
- Branch is `Development`. The suite stands at 397 tests. A count quoted inside
  a slice is the one that slice was written against; recount before relying on
  either.

## Checklist

- [x] Slice 1: A table written at the wrong depth is named, not crashed into
- [x] Slice 2: A record with no cost is not a complete one
- [x] Slice 3: The Gaussian process refuses what it cannot fit
- [x] Slice 4: The budget may cut the last generation short, and `build` says so
- [x] Slice 5: The benchmark evidence lives with the benchmark (the document's own fate is Ian's call)
- [x] Slice 6: Two tests that do not say what they mean

---

## Slice 1: A table written at the wrong depth is named, not crashed into

### Type

`AFK`

### What to build

`[PARAMETERS.<group>.<name>]` is three levels deep and
`[RUNMANAGER_GLOBALS.<group>.<name>]` likewise. A file that writes one of them
two levels deep — the group omitted, which is the ordinary hand-edit slip —
reaches the key check with a float where it expects a table, and the load dies
with `TypeError: 'float' object is not iterable`: no file, no table, no key.
`[LEARNER.<name>.<extra>]`, one level too deep, looked like the same shape from
the other direction and is not: the extra level is read as a knob, and the
learner's own constructor is the schema that refuses it by name. Nothing to do
there beyond a test that says so.

Check the shape before iterating it, and say what was expected and what was
found. One helper, three call sites. This is the most likely mistake there is
in a hand-edited configuration, and it currently gets the worst message in a
package whose whole discipline is naming the offending key.

### Acceptance criteria

- [x] `[PARAMETERS.OPT]` carrying `min`/`max` directly is refused with a message
      naming the table, the depth expected and the value found
- [x] `[RUNMANAGER_GLOBALS.OPT]` carrying `args` directly is refused the same way
- [x] `[LEARNER.gaussian_process.anything]` is refused by the learner's schema,
      naming the key
- [x] A correctly nested file of each kind still loads
- [x] No `TypeError` escapes `check_keys` for any shape of mis-nesting

### Blocked by

None - can start immediately.

---

## Slice 2: A record with no cost is not a complete one

### Type

`AFK`

### What to build

`Observation.state` defaults to `COMPLETE` while `cost` is typed
`float | None`, so the obvious construction of a costless record —
`Observation(shot_id, params, None)` — raises `TypeError` out of
`np.isfinite(None)` inside `usable`. The type signature advertises the value
the field default forbids.

One change, not two. The constructor refuses `COMPLETE` with `cost=None`,
because a shot that completed has a cost; the field default is then fine as it
stands. Guarding `usable` against a `None` cost as well was written and then
taken out: with the pair refused, "has no cost" and "is not complete" describe
the same records, so each guard makes the other dead and the mutation that
removes either survives. `usable` keeps the state test it always had.

`Observation` becomes a frozen dataclass to carry the refusal — `NamedTuple`
forbids `super()` in its methods, so there is no way to validate inside one
without restating every field default. That is the form `Parameter` and
`GlobalMapping` already take.

`Session.history` always passes `state=` explicitly, so the session is unaffected
either way. What this protects is every history built by hand — the benchmarks,
the tests, and any lab code that assembles one.

### Acceptance criteria

- [x] `Observation(id, params, None)` is refused at construction, naming the
      state and the missing cost
- [x] `Observation(id, params, None, state=PENDING).usable` is False, not a raise
- [x] `Observation(id, params, None, state=DROPPED).usable` is False
- [x] A complete observation with a finite cost is still usable, and one with a
      NaN cost is still not
- [x] Every session path still builds its history without raising

### Blocked by

None - can start immediately.

---

## Slice 3: The Gaussian process refuses what it cannot fit

### Type

`AFK`

### What to build

Three places where a configuration reaches scikit-learn or numpy and the lab is
told nothing useful. All three are the same rule: refuse at the first moment it
is knowable.

**`minimum_observations` has no floor.** Set to 0 alongside
`num_training_runs = 0`, it satisfies the warmup refusal (0 >= 0) and dies with
`ValueError: Found array with 0 sample(s) ... required by StandardScaler`. A
floor of 1 closes it. Nothing regressed here — the deleted `InsufficientData`
fallback caught only that exception — the gap is that every other unrunnable
setting is refused at load and this one is not.

**A degenerate posterior returns `None`.** `minimise_acquisition` compares
`result.fun < winning_value`, which is False for NaN at every restart, so
`winner` stays `None` and `np.clip(None, ...)` raises a `TypeError` naming
neither the learner nor the cause. Raise instead, saying the acquisition was
never finite and what to look at. The message does not name a cause: identical
costs are not one — scikit-learn's scaler floors a zero scale at 1.0 — and the
remaining routes are ill-conditioned fits rather than a single nameable
setting.

**`cost_has_noise = false` with a mixed history is a documented crash.** The
constructor docstring already spells out that a history where only some
observations carry an uncertainty gives the rest an `alpha` of exactly zero, and
the fit raises `numpy.linalg.LinAlgError`. It is documented and then permitted,
where every other unrunnable combination is refused with a sentence. The mixed
history is not knowable at load, so the refusal fires the first time one is
seen, naming both ways out: turn `cost_has_noise` on, or give every shot an
uncertainty.

### Acceptance criteria

- [x] `minimum_observations = 0` is refused at load, naming the floor
- [x] `minimum_observations = 1` still loads
- [x] An acquisition that is NaN at every restart raises naming the degenerate
      posterior, not a `TypeError` from `np.clip`
- [x] A non-degenerate acquisition still returns the winning point
- [x] `cost_has_noise = false` with a history where some observations carry an
      uncertainty and some do not raises before reaching sklearn, naming the
      first shot without one and both ways out
- [x] `cost_has_noise = false` with a history where none carries one still fits
- [x] `cost_has_noise = false` with a history where all carry one still fits
- [x] `cost_has_noise = true` with a mixed history still fits, which is the
      ordinary case

### Blocked by

None - can start immediately.

---

## Slice 4: The budget may cut the last generation short, and `build` says so

### Type

`AFK`

### What to build

No behaviour change. `session.refill`'s `max_num_runs` clamp can hand a
generational learner a partial generation: at `population_size = 5` and
`max_num_runs = 13` the run is 5, 5, 3. `build` raises a `ValueError` over a
budget under two generations and its message calls a short generation "a second
generation cut short evolves some of its slots rather than a generation" — which
reads as a promise that no generation is ever short, and the clamp breaks it.

The truncation is right and stays. A short generation is a defect only when
something follows it and reads the population afterwards; at the end of a run
nothing does. The walk is anchored to position, so a short final generation
misaligns no slot; every trial in it still competes for its own slot; and
refusing `max_num_runs = 13` or rounding it to 10 throws away three shots the
lab paid for to protect a property nothing reads — and rounding would alter a
number the lab wrote without saying so.

So the fix is the sentence. Keep the clamp, keep the two-generation refusal —
that one guards against never evolving at all, which is a real floor — and say
what is true: below two generations differential evolution never evolves; the
last generation may be cut short by the budget, and its trials still compete for
their slots.

### Acceptance criteria

- [x] `population_size = 5`, `max_num_runs = 13` runs generations of 5, 5, 3
- [x] In that third generation, slots 0-2 hold the better of founder and trial
      and slots 3-4 keep their founders
- [x] The two-generation refusal still fires below `2 * population_size`
- [x] `build`'s message no longer claims a generation is never cut short
- [x] No sentence anywhere in the package still makes that claim

### Blocked by

None - can start immediately.

---

## Slice 5: The benchmark evidence lives with the benchmark

### Type

`HITL`

### What to build

`codex_issues_proposal.md` is the record of one code-review round, named after
the agent that produced it. It is cited as the authority for a shipped default
from `differential_evolution.py`'s `population_size` docstring, from `README.md`
twice, from `UPGRADING.md`, from `benchmarks/README.md` and from
`tests/test_benchmarks.py`. A physicist reading why the population defaults to
eight is sent to a document about a code review.

A repository states its current condition, not its history. `benchmarks/README.md`
already exists and already describes the harness that produced the numbers, so
that is where the evidence belongs: the sweep's tables, the two figures the
source quotes, and the caveats. Every citation then points there, and the
proposal document is free to be archived or deleted without breaking anything.

HITL: whether that document goes, stays, or moves out of the repository is Ian's
call. This slice makes it *possible* by removing the last thing that depends on
it, and stops there.

### Acceptance criteria

- [x] `benchmarks/README.md` carries the dimension sweep's tables, the two
      figures the source quotes, and the caveats
- [x] `differential_evolution.py` cites `benchmarks/README.md`
- [x] `README.md`, `UPGRADING.md` and `tests/test_benchmarks.py` cite it too
- [x] No file outside `issues/` references `codex_issues_proposal.md`
- [x] The M-LOOP guard in `tests/test_package.py` still passes, with its
      allowance for that file adjusted to what is true

### Blocked by

None - can start immediately.

---

## Slice 6: Two tests that do not say what they mean

### Type

`AFK`

### What to build

**A test asserts on a private attribute.** `tests/test_learners.py` compares
`_kernel.theta` between an instance that has been fitting all session and a
fresh one handed the same history. The behaviour is worth keeping — that is the
whole cache-is-a-cache rule — but the leading underscore is this repository's
marker for what must never be touched from outside, and `predict` is the public
surface that makes the cache observable. Assert through it.

**A claim has no test.** `fit` keys its hyperparameter cache on which
observations it was fitted to "and not how many", because a cost arriving late
lands in proposal order and rewrites a prefix of unchanged length. The comment
says so and nothing tested it — the mutation that keys on `len(prefix)` passes
the whole suite, including the equivalence test above, both before and after
this slice's rewrite. A cost that turns up for the third of nine proposals is
the case: eight usable before it lands and eight in the prefix after, holding a
different eight.

**A path has no test.** The routine sends `("observe", ...)` when a pass carried
observations and `("shot", ...)` when it did not. The empty-frame case is
covered and the mixed case is covered, but a pass whose rows are *all* of
runmanager's default shots — rows present, no ids among them, so `shots` is
non-empty while `observations` is empty — is not. That is the dummy apparatus's
own behaviour when a queue runs dry, and it takes the `else` branch that keeps
the session reconciling.

### Acceptance criteria

- [x] The cache-equivalence test asserts through `predict`, not `_kernel`
- [x] A late cost that rewrites a prefix of unchanged length refits, and keying
      the cache on `len(prefix)` fails that test
- [x] A pass carrying only default shots sends one `"shot"` message and no
      `"observe"`
- [x] Mutating the `if observations:` branch fails that test

### Blocked by

None - can start immediately.
