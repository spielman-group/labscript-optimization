# labscript-optimization: `num_buffered_runs` as a hint, and the GP's own cycle

Work items for the design in [`buffer_hint_plan.md`](buffer_hint_plan.md), which
states the decision error being corrected, every decision taken, and the
reasons. Each slice names the plan sections it implements; read them before
starting. These items say *what*; the plan says *why*.

In one sentence: the session stops asking one learner to fill the whole queue.
`num_buffered_runs` becomes how many shots to keep queued, honoured by each
learner as far as its method allows — exactly by the random learners, not at
all by differential evolution, and by the Gaussian process through explorer
shots queued behind each of its batches.

House rules that apply to every slice:

- Every test must be **proven to fail** by mutating the code it guards. Run the
  mutation, capture the failure, revert it, report both. A test that cannot be
  made to fail is a finding, not a deliverable. The plan's §7 lists the tests
  B needs with the mutation each must fail.
- A comment or docstring states what the code does now. No "previously", no
  "used to". Prose that describes behaviour is a claim under test: when
  behaviour changes, grep for every sentence that described the old behaviour.
- No settings accepted for compatibility. A removed key is refused at load,
  naming its replacement.
- Branch is `Development`. The suite stands at 452 tests. A count quoted inside
  a slice is the one that slice was written against; recount before relying on
  either.

## Checklist

- [ ] Slice 1: Each shot carries the learner that proposed it
- [ ] Slice 2: Learners pace themselves
- [ ] Slice 3: The Gaussian process runs its own cycle
- [ ] Slice 4: Test cleanup
- [ ] Slice 5: Re-run the population measurement against the shipped dither

---

## Slice 1: Each shot carries the learner that proposed it

### Type

`AFK`

### What to build

Every proposal records the learner that made it, at the moment it is made, and
the record never changes. Two things follow.

The `phase` written onto a shot becomes that shot's own source. Today it is the
phase of whatever was proposed most recently, written onto the shot being
analysed — a different shot whenever more than one is in flight. A simulation
of the routine's order of events found 7 of 24 shots labelled wrongly: the
handover shots the trainer proposed read `main`, and the last shots of a run
read `periodic trainer` after the budget ran out. The worker's reply therefore
carries each observation's source alongside its verdict, and the routine writes
each shot's own.

The session's count of shots in flight is derived from the history's pending
records rather than kept separately, so a learner reading states from history
and the session counting what is awaited cannot disagree.

This slice changes what the routine writes, and so does the C2 work on how the
routine learns which shots are new. The two are sequenced, never run in
parallel.

### Acceptance criteria

- [ ] Every observation in the history carries the source recorded when it was
      proposed, including the configured start
- [ ] In the simulated routine run that found 7 of 24 wrong, every shot's
      `phase` is its own source; writing the latest proposal's phase instead
      fails the test
- [ ] The session's awaited shots and the history's pending records agree after
      drops, late costs and duplicate costs; a separately kept count fails the
      test
- [ ] The phase column's documentation says it is the shot's own source

### Blocked by

None - can start immediately.

### User stories covered

- Plan §1 defect #6; §2 "Across all learners"; §4 Q9

---

## Slice 2: Learners pace themselves

### Type

`AFK`

### What to build

The contract between the session and a learner changes from "give me k
proposals" to "here is the hint; propose whatever your method allows now": zero
or more proposals, each with its source. The session submits what comes back
and no longer carries a barrier of its own.

- The random learners fill to the hint, as the session does for them today.
- Differential evolution proposes a whole generation when none of its own
  proposals is outstanding and nothing otherwise, reading that off its own
  pending records. The barrier moves from the session into the learner; the
  algorithm is untouched.
- `generation` stays as a declaration. The budget refusal
  (`max_num_runs` under two generations) reads it off the built learner, and
  the starvation count still skips a learner that declares one.

No behaviour changes in this slice: every existing test of differential
evolution, the random learners and the session's queueing passes unaltered in
what it asserts.

### Acceptance criteria

- [ ] Learners answer the hint with zero or more proposals, each with its source
- [ ] Differential evolution proposes a whole generation only when none of its
      own is pending, and nothing otherwise; ignoring pending states sends a
      second generation out and fails the test
- [ ] The random learners keep exactly the hint in flight
- [ ] The budget refusal still fires at `max_num_runs < 2 × population_size`;
      deleting `generation` fails the test
- [ ] Starvation is not counted for differential evolution and is counted for
      the random learners
- [ ] The session has no barrier branch of its own

### Blocked by

- Slice 1: Each shot carries the learner that proposed it

### User stories covered

- Plan §2 "Random and directed random", "Differential evolution"; §4 Q2, Q3

---

## Slice 3: The Gaussian process runs its own cycle

### Type

`AFK`

### What to build

The Gaussian process owns its explorer and its warmup, and proposes in batches.

- **Warmup.** Below `warmup_observations` usable observations, the explorer
  alone keeps max(`explore_runs`, `num_buffered_runs`, 1) queued.
  `warmup_observations` defaults to `max(5, 2 × num_params)` and counts usable
  observations, not shots. Warmup ends at the count; explorer shots already
  queued still run, and the documentation says so as the design.
- **The cycle.** After warmup the GP proposes a batch of `batch_size` points,
  each conditioned on the ones before it, with its hyperparameters refit once
  per batch and cached on the usable observations in hand when the batch is
  computed. Behind each batch go max(`explore_runs`, `num_buffered_runs`)
  explorer shots. The next batch goes out when every GP shot of the last one is
  completed or dropped; explorer shots never hold it up, and their costs join
  the fit whenever they land. The exploration schedule is indexed by position
  within the batch, so each batch starts from the greedy weight. The GP does
  not condition on explorer shots in flight.
- **The budget** cuts the batch last: the batch first, then explorer shots until
  the room runs out.
- **Configuration.** `explorer` (`random` or `directed_random`, default
  `directed_random`), `explore_runs` (default 1), `batch_size` (default 4) and
  `warmup_observations` live in `[LEARNER.gaussian_process]`; the explorer's own
  knobs stay in its own table. `num_buffered_runs = 0` is legal for the GP, and
  the GP's default is 1. The random learners' default follows Ian's answer to
  the open item in plan §3.1 — settle it before this slice starts.
- **Removed**, each refused at load with its replacement named:
  `TwoPhaseLearner`, `[GENERAL] trainer`, `num_training_runs`,
  `num_runs_between_trainer_runs`, `refit_interval`, `minimum_observations`.
- **Documentation.** The GP's docstring and the README say, in one sentence,
  that raising `num_buffered_runs` to cover a slow fit also raises the share of
  explorer shots. The empty-queue caveat the README carries for differential
  evolution is stated for the GP beside it: a batch whose shots were all
  deleted is released only by a reconcile, which needs a shot to arrive.
  `UPGRADING.md` says what each removed key became. The example configuration
  and Ian's own configuration are rewritten to the new keys; in Ian's, values
  stay as he set them wherever a key survives.

### Acceptance criteria

- [ ] The GP proposes exactly `batch_size` points, and nothing while any
      GP-sourced proposal of its latest batch is pending
- [ ] A dropped GP shot releases the batch; requiring completion stalls the
      cycle and fails the test
- [ ] Explorer shots behind a batch number max(`explore_runs`,
      `num_buffered_runs`), including at `num_buffered_runs = 0`
- [ ] Warmup ends at `warmup_observations` usable observations; counting shots
      fails the test with NaN costs
- [ ] `warmup_observations` defaults to `max(5, 2 × num_params)`
- [ ] Each batch walks the exploration schedule from its first weight
- [ ] The budget cuts explorer shots before any of the batch
- [ ] A pure GP (`explore_runs = 0`, `num_buffered_runs = 0`) loads, and its
      starvation is counted
- [ ] Every removed key is refused naming its replacement
- [ ] The late-cost cache test still passes
- [ ] No sentence in the package, README, UPGRADING or the example still
      describes the trainer, the two-phase wrapper, the periodic trainer run, or
      `num_training_runs`
- [ ] Ian's configuration loads and builds

### Blocked by

- Slice 2: Learners pace themselves

### User stories covered

- Plan §1 defects #2, #7, #8; §2 "The Gaussian process"; §3; §4 Q1, Q4–Q8; §7

---

## Slice 4: Test cleanup

### Type

`AFK`

### What to build

Use the test-cleanup skill on what slices 1–3 left behind. Remove the tests
that only served the development loop and those that pin implementation
details; keep the behavioural safety net — every test in the plan's §7, the
refusals of removed keys, and the tests that describe what a lab sees. Tests of
`TwoPhaseLearner` and the periodic trainer run that survive slice 3 in
rewritten form are reviewed here for whether they still say anything.

### Acceptance criteria

- [ ] Every test the plan's §7 lists is still present and still fails under its
      mutation
- [ ] No remaining test reaches into a private attribute
- [ ] Each deleted test is listed in the commit message with the reason

### Blocked by

- Slice 3: The Gaussian process runs its own cycle

### User stories covered

- Plan §7

---

## Slice 5: Re-run the population measurement against the shipped dither

### Type

`AFK`

### What to build

Differential evolution now draws its differential weight once per generation
rather than once per member. The sweep behind the `population_size` default —
eight members for the budgets a lab usually runs, sixteen past a thousand shots
— was measured with the per-member draw, so the tables in the benchmark README
describe an algorithm that no longer ships. Re-run the dimension sweep against
the shipped learner, replace the tables and their commit, and change the
default or the learner's docstring figures only if the numbers move. Nothing in
slices 1–3 changes the algorithm, so this can run alongside them.

### Acceptance criteria

- [ ] The dimension sweep is re-run at the current commit, with its rows in the
      benchmark results and its header naming the commit and command
- [ ] The benchmark README's tables and figures are the new ones
- [ ] If the block winners move, the `population_size` default and its
      docstring are changed to match and the change is stated; if they do not,
      that is stated
- [ ] The benchmark smoke test still passes

### Blocked by

None - can start immediately.

### User stories covered

- Plan §6, "A (in flight)"
