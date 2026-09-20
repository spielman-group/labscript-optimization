# The seven codex issues: assessment and design

Read against the working tree at `88790ab` + the two agents' uncommitted work.
This document is a plan; no code in it has been written. The numbers in it were
measured against a `git archive` copy of the repository, in a scratch directory
that no longer exists — except Issue 2's benchmarks, whose harness and raw rows
are in `benchmarks/`, and which are reproducible from there.

**Baseline, after codex's review of this document:** the tree is at
`bec4dfc` + codex's uncommitted work, and the suite is **224 tests**, not the
216 the option-B blast radius was counted against — recount before
implementing. The two observations this document made about the landed
`start_worker` fix are both resolved: it now reaps the child in
`try/except BaseException` before re-raising, and `CONFIGURE_TIMEOUT` is 30 s
with an explicit raise rather than a silent 2 s `(False, None)`.

**Decisions taken by the repository owner, 2026-09-19:** issue 5 is settled
(report in lab units), issue 7b is settled (no change), and issue 2 is settled
*in principle* — DE must actually be DE, so the fix is in the plan.

**The Fable session's review, folded in below.** It found a defect nobody else
had — issue 0, the routine discarding shots lyse analysed — and broke the
generational design as first sketched. Its findings were verified here before
being folded in; each says where.

**Codex's review, folded in below.** Its blocking objection is upheld and
verified: **option B is necessary but not sufficient**, because it leaves a
proposal's *role* unstable during population founding. Issue 2 is therefore
held until the proposal-role semantics are defined. Its other eight points are
accepted; where this document previously said something different, it has been
rewritten rather than annotated.

## What a lab would feel, worst first

| # | Issue | What the lab sees | Verdict |
|---|---|---|---|
| 0 | Routine reads one row; lyse batches | Analysed shots silently dropped, learner never sees their costs | **Prerequisite — do it first** |
| 7a | Not installed | The routine does not run at all | **Just do it** |
| 1 | Worker reply offset | Every column one shot stale, for ever | **Fixed while I was writing — two follow-up observations** |
| 4 | `num_buffered_runs = 0`/`-1`, `max_num_runs = 0` | Session submits nothing, says nothing | **Just do it** |
| 3 | Parameter/global collisions | A global driven to 5.5 when its parameter says `[0, 1]` | **Just do it** |
| 6 | Cost must precede the routine | Every shot recorded bad, no explanation | **Just do it** (docs) |
| 5 | `best_cost` reports the internal sign | A maximising lab reads `-7` for a measurement of 7 | **Decided: report in lab units** |
| 2 | DE slot walk | Nothing measurable in search quality — but it is not DE | **Decided: textbook generational DE, position founding** |
| 2b | GP pending-point conditioning | Measurable duplication, worse results when fixed | **Decline** |
| 7b | `main`/`Production` at "Initial commit" | Public clone is one `.gitignore` | **Decided: no change; `main` is populated at release** |

---

## Issue 1 — worker reply offset: status report only

**It landed while I was writing.** `start_worker` now calls `_drain(from_worker)`
immediately after `to_worker.put(("configure", ...))`, and its docstring says so.
That removes the offset at its source. `worker.py` is unchanged, which is right:
the worker was always replying correctly, nobody was reading it.

The other agent also landed the **codex validation work**:
`learners.validate_options` (per-`[LEARNER.<name>]` table key checking, called
from both `config.from_dict` and `learners.build`), `learner_options["shared"]`
promoted to its own `Config.shared_learner_options` field, and the sklearn /
scipy imports made lazy — plus the committed removal of `Parameter.global_name`
(`88790ab`) and README / UPGRADING / example-config prose.

I designed nothing for issue 1. The two follow-up observations this document
raised have both since been fixed, verified in the tree:

- **Orphaned worker on a failed configure — fixed.** `start_worker` builds
  `handles` before configuring and wraps the configure exchange in
  `try/except BaseException: _stop_worker(handles); raise`, so the spawned
  child is reaped before the exception reaches `optimise`.
- **Silent 2 s configure timeout — fixed.** `_drain` takes a timeout,
  `CONFIGURE_TIMEOUT` is 30 s, and a timeout raises with a message naming the
  limit instead of returning `(False, None)` and re-establishing the offset.

**Residue, found by the Fable session and verified.** `CONFIGURE_TIMEOUT` is
30 s, but `runmanager.remote.Client` takes labconfig's `communication_timeout`,
which falls back to 60 s (`runmanager/remote.py:18`). So with runmanager down,
`check_ready()` is still inside its own ZMQ wait when our 30 s expires: the
worker is killed mid-wait and the failure reads "did not configure within 30
seconds", never naming the cause. Give the interface's client a short timeout,
or `say_hello` first.

---

## Issue 0 (new, and the prerequisite) — the routine discards shots lyse analysed

**Found by the Fable session; verified in lyse's source.** This is independent
of DE, and generational DE makes it certain to bite.

lyse runs multishot routines **once per drained analysis batch, not once per
shot**. `lyse/filebox.py` `analysis_loop`: the inner `while` singleshot-analyses
every incomplete file in turn, breaks when `get_first_incomplete()` returns
`None`, and only then — after the loop, at lines 943-945 — does
`do_multishot_analysis()` run, once.

`optimise` asks for `lyse.data(n_sequences=1, n_shots=1)`, the last row. So when
three shots are analysed in one pass, two of them are never handed to the
worker. runmanager answers `unknown` for them, the session drops them after two
reconciles, the learner never sees their costs, `save_status` never writes to
them, and `dropped` climbs for shots that ran and were analysed perfectly well
— which makes the README's "a `dropped` above zero is shots the run actually
lost" false.

A pile-up needs only a shot to arrive while the previous one is still in
singleshot analysis, or analysis paused and resumed, or lyse started with shots
already waiting. With dummy devices the shot period is seconds and
`example_absorption.py` draws a matplotlib figure per shot, so the validation
run will pile up.

**This is my error.** The `n_shots` feature was added to lyse and adopted here
on my framing — "the shot it was called on" — and lyse's model does not have
that premise for multishot routines. The lyse-side commit is fine; the
assumption in this package is not.

**Fix, in the routine — its own job, not lyse's.** Remember the last row
handled; ask for `n_shots = m`, doubling `m` until that row is in the frame or
the frame is shorter than `m`; then hand over every newer row carrying a shot
id, in order, writing the status onto each one the session records. `record`
already ignores unknown ids and duplicates, so overlap is harmless.

**On the first invocation**, there is no remembered row and nothing of the
session's can be in the frame yet: remember the last row and hand over nothing,
rather than doubling out to fetch the whole sequence. `record` would ignore
every row anyway, but a fresh session should not open by pulling a 400-row
frame.

**Tests, each with its mutation:**

- two shots analysed in one pass are both handed over [`n_shots=1` — the second
  is never sent];
- a row already handed over is not sent again [drop the remembered row, and
  assert on the messages, because `record` would otherwise mask it];
- a pile-up larger than the first request is fetched whole [remove the
  doubling — the oldest row is missed];
- **an invocation that finds no new rows still sends the worker one message**
  [return early when nothing is new]. This one is load-bearing and easy to lose
  while writing the others: `reconcile` and `refill` run only in the trailing
  work *after* a reply, so a routine that returns early stops reconciling
  entirely. A generation that was blocked and then cleared by an operator would
  never be revived. The `("shot", None)` path is what keeps it alive.

`test_only_the_shot_it_was_called_on_is_asked_of_lyse` pins the wrong contract
and goes.

**This ranks above issue 2**: a generation cannot complete cleanly until it is
done.

---

## Issue 2 — differential evolution: textbook generational DE

### The defect

`Session.history` omits unresolved proposals and DE's `replay()` reconstructs
the population and target slot by *counting* what it is given. At buffer depth
d every trial competes against the slot d−1 further on, and every NaN shifts it
further. Measured: at the shipped default of 3, essentially every trial has
always competed against the wrong incumbent.

### The decision

**Textbook generational DE** — scipy's `updating='deferred'`. Ian's ruling:
we cannot distribute something, call it differential evolution, and have it do
something else; and if the pipeline does not support it, that is a pipeline
problem, since DE has already been run on this apparatus by hand.

The whole generation is submitted at once. This dissolves the founding-phase
role instability that blocked the earlier design — the learner is never asked
for a proposal while any of its own are outstanding, so a founder cannot be
drawn against a half-built population and later reinterpreted as a trial.

### The walk: found by block position, not by usable count

Submitting whole generations is **not sufficient on its own.** Verified against
a scratch tree: option B founds the population from the first N *usable*
records, so a dropped or NaN founder spills founding into block 1.

Population 4, founder `p1` dropped, block 1 complete (trials at costs
1.0/2.0/3.0/4.0):

| history | population | next slot |
|---|---|---|
| founder `p1` dropped, omitted or unusable | `[2.0, 3.0, 4.0, 1.0]` | 3 |
| `p1` returns late, all four founders present | `[1.0, 2.0, 3.0, 4.0]` | 0 |

Under usable-count founding, `p4` — a trial — is promoted to founder 3, and
`p5`–`p7` are read as trials 0, 1, 2 rather than 1, 2, 3. When `p1` returns
late every role re-indexes. "Position in the block is the role" is exactly the
property usable-count founding loses at the first bad founder.

**The rule:** history holds every proposal in order with a state; slot =
position within its block, for founders and trials alike; a slot's member is
its best usable result; a slot whose founder produced no cost is **vacant**
until a later trial for that slot lands, and a trial for a vacant slot is drawn
founder-style, since there is no incumbent to cross over with. That is the only
generation-side change.

**On the record's shape.** Codex argued that pending and dropped must stay
distinct rather than both being `cost=None`, and that stands. The Fable session
narrows it usefully: the *learner* never needs the distinction — to it both are
"position spent, no usable cost" — only the session and diagnostics do, and the
session already has `dropped`. So a `state` field on the record is sufficient;
renaming `Observation` to `ProposalRecord` is honest but optional, not
load-bearing.

Measured at each refill's decision, N=4, 400 shots, 20 seeds, random completion
order, where "mismatch" is a completed proposal judged in a role other than the
one it was generated under:

| walk | restart | NaN | drop | late return | mismatch | role flips |
|---|---|---|---|---|---|---|
| usable-count (B) | off | 0 | 0 | 0 | 0% | 0 |
| usable-count (B) | off | 15% | 0 | 0 | 0.87% | 25 |
| usable-count (B) | off | 0 | 10% | 50% of drops | 0.89% | 13 |
| usable-count (B) | on | 15% | 0 | 0 | 6.65% | 352 |
| usable-count (B) | on | 0 | 10% | 50% | 11.07% | 459 |
| **position** | off | any | any | any | **0%** | **0** |
| position | on | 0 | 10% | 0 | 0% | 0 |
| position | on | 0 / 10% | 10% | 50% | 0.73% / 0.89% | 0 |

**Correction to an earlier claim in this document:** NaN costs do not shift the
walk *for trials* — a bad trial advances its slot without displacing the
incumbent, which is correct. The NaN problem is confined to founders, and
position founding is what fixes it.

### Remove `restart_tolerance`

The last two rows above are the residue no walk can remove: the restart
decision is taken at a boundary from the costs resolved by then, and a cost
arriving after the boundary can change it, turning a block generated as trials
into founders of a new epoch. Neither the learner nor the session can record
that decision without a new kind of state.

It is also harmful within lab budgets. On the 4-D sphere the shipped walk stops
at 0.018 at every depth with restart on, and reaches 0.000 with it off:
`restart_tolerance = 0.01` re-seeds a population that has converged to within
1% of its initial spread but not to the minimum. `max_num_runs_without_better_params`
is the stop a lab actually wants there.

Refuse the key at load per decision 3; list it in `UPGRADING.md` beside the
other removals.

### Late costs are an ordinary operator path

A shot behind a row that will not compile is `blocked`, and `reconcile` drops
it at the first answer. The README then tells the operator to delete the red
row — after which the blocked shots run and their costs arrive. Under
generational submission that reads: generation *g* blocked, all of it dropped,
`awaiting == 0`, generation *g+1* proposed, operator clears the head, *g* runs,
and its costs land after *g+1* was built without them.

Position founding keeps every slot aligned through this. What must be written
down is the semantics: **a late cost is taken and competes for its slot, and
the generation built meanwhile was built without it** — which is what textbook
DE would have produced had the cost arrived in time. Two generations sit in the
queue in that state, and the `starved` / `awaiting` prose must not call it a
fault.

### The generation boundary, as a learner property

One attribute on `Learner`: **`generation: int | None = None`** — "proposes
only whole groups of this many, and only when none of its proposals is
outstanding". DE sets it to `population_size`. `refill` becomes: if the learner
declares a generation, `wanted = generation if awaiting == 0 else 0`; otherwise
as today.

**Named `generation`** (Ian, 2026-09-19). I had proposed `block`, to avoid
colliding with `GaussianProcessLearner`'s public `generation_size`
(`gaussian_process.py:64`). Overruled, and rightly: "generation" is the term the
DE literature uses, and our code does not overrule standard convention.

The collision is resolved from the other side instead — **the Gaussian process
knob is renamed `batch_size`**, which is what batch Bayesian optimisation calls
it, so each learner now uses its own field's standard word. That is a public
configuration key, so it is refused under its old name at load and listed in
`UPGRADING.md`; the rename is carried in the configuration slice.

That is the same kind of declaration as `minimum_observations` and
`last_phase`, not a delivery-policy mechanism, and `propose(history, k)` is
unchanged. Decision 2 holds: `awaiting == 0` is runmanager's answer, asked at
the reconcile that precedes every refill.

Two consequences:

- **`num_buffered_runs` is refused, not checked**, for a learner that declares
  a generation. The ground is simply that it would duplicate `population_size`, and
  a setting that duplicates another is a setting that can disagree with it.
  (The earlier ground — that N moved whenever a parameter was enabled — died
  with the multiplier.)
- **`starved` fires by design at every boundary.** `refill` counts a starved
  when `awaiting == 0 and self.proposals`, which under generational submission
  is every generation. A counter that fires by design is noise in the one
  number the README tells a lab to watch. Skip it for a learner that declares a
  generation, and say in the README that such a learner empties the queue once
  per generation.

The budget refusal (`max_num_runs < 2 × population_size`) is stated with the
population decision below; it is not repeated here.

### Benchmark: what the generation barrier costs

Every figure here and in the next section comes from `benchmarks/de_pipeline.py`
and the rows in `benchmarks/results/`; the README beside them names the command
and the commit for each table. Each run is a `Session` built by `config.loads`
from a real configuration file, with a fake runmanager standing in for the
queue, so the generation barrier in `refill`, the position walk in `replay` and
the history's pending entries are exercised as they are in a lab; costs come
back out of order, one shot per step, drawn from those in flight. **Every arm
runs to the same number of completed shots, and nothing is submitted once the
budget is claimed.**

Three arms: what ships, and two references each differing from it in a single
thing.

- **generational** — the shipped learner, built by the shipped loader and
  driven by the shipped session, barrier in force.
- **asynchronous d3** — the same shipped learner subclassed so that
  `generation` is `None`, and nothing else, at `num_buffered_runs = 3`. What
  separates it from the first is the barrier and only the barrier: same
  replay, same mutation, same crossover, same bounds handling. It is a
  reference point for feedback latency rather than an alternative on offer,
  since it also runs the apparatus at a shallower queue.
- **former d3** — the learner as it shipped before slice 4, taken out of git
  (`842c0f6`), at the same depth, with its `restart_tolerance` on and off. Its
  walk counts usable records, which is the accidental slot mixing the second
  reading below is about. No copy of it is kept in the repository.

Four analytic functions on `[-5.12, 5.12]^D`, `best1` throughout, no drops and
no NaN costs.

**1. The barrier costs sample efficiency, and the gap does not close with
budget — it opens.** 4-D, N over 8 / 16 / 60, 16 seeds. The ratio is the
median over the twelve (function, N) cells of median generational over median
asynchronous.

| shots | generational / asynchronous, median over the 12 cells | cells where generational is worse |
|---|---|---|
| 120 | 1.01 | 6 of 12 |
| 240 | 1.05 | 9 of 12 |
| 600 | 1.64 | 10 of 12 |

The price grows from nothing measurable at 120 shots to 1.64× at 600. So
one-generation feedback latency is a standing cost of the decision rather than
a transient a longer run absorbs — and it is a number for the documentation,
not an argument against the decision: Ian's reason for textbook DE is that the
name must be true.

**2. The strongly multimodal function is where the correct algorithm loses to
the walk it replaced.** Same cells; a win is the shipped generational learner's
median beating the former walk's at depth 3.

| function | 120 shots | 240 shots | 600 shots | all |
|---|---|---|---|---|
| rastrigin | 1 of 3 | 0 of 3 | 0 of 3 | 1 of 9 |
| sphere | 1 of 3 | 1 of 3 | 2 of 3 | 4 of 9 |
| ackley | 1 of 3 | 2 of 3 | 2 of 3 | 5 of 9 |
| rosenbrock | 2 of 3 | 2 of 3 | 2 of 3 | 6 of 9 |
| **all four** | **5 of 12** | **5 of 12** | **6 of 12** | **16 of 36** |

Rastrigin is the correct algorithm's worst function by a distance — one cell
of nine, and none at all once the budget passes 120 shots — and Rosenbrock its
best, at six of nine. Over all four functions the split is roughly even rather
than uniformly worse: the correct algorithm wins decisively on Rosenbrock (600 shots, N=16: 1.19 against 13.65) and on Ackley
(600 shots, N=8: 0.001 against 0.039), and loses on Rastrigin, where the former
code's accidental slot mixing helps.

Note what the former walk pays where it loses. Turning `restart_tolerance` off
makes it beat the generational learner in 8, 9 and 9 of the twelve cells at the
three budgets, against 7, 7 and 6 with the restart on: the re-seeding costs the
old code more than its slot mixing wins it, which is independent support for
having removed the key.

**3. The population size is the dominant knob.** N=60 — which is M-LOOP's
default of 15 at four parameters, under the multiplier this package no longer
has — holds the worst median of the three N values in 12 of 12 (function,
budget) cells, in both variants, often by an order of magnitude (Rosenbrock at
240 shots: 103 at N=60 against 3.0 at N=8). It completes 2, 4 and 10
generations at the three budgets. Fifteen is scipy's default for *thousands* of
evaluations. What the right number is, and what shape the setting should have,
is the next section.

**The whole 4-D table**, median / mean of the best cost found, 16 seeds:

| function | N | shots | generational | asynchronous d3 | former d3 | former d3, no restart |
|---|---|---|---|---|---|---|
| rastrigin | 8 | 120 | 12.217 / 13.420 | 11.663 / 12.635 | 10.617 / 11.295 | 10.617 / 11.295 |
| rastrigin | 8 | 240 | 6.135 / 6.643 | 5.234 / 6.036 | 6.005 / 5.665 | 5.982 / 6.144 |
| rastrigin | 8 | 600 | 3.980 / 3.720 | 2.489 / 3.306 | 2.041 / 2.111 | 3.074 / 4.527 |
| rastrigin | 16 | 120 | 15.738 / 15.467 | 17.737 / 17.781 | 16.447 / 16.306 | 16.447 / 16.306 |
| rastrigin | 16 | 240 | 12.165 / 11.768 | 9.388 / 10.192 | 7.742 / 8.731 | 7.742 / 8.731 |
| rastrigin | 16 | 600 | 3.296 / 3.860 | 3.399 / 3.577 | 2.085 / 2.420 | 2.075 / 2.462 |
| rastrigin | 60 | 120 | 25.147 / 23.730 | 24.743 / 24.369 | 24.785 / 23.872 | 24.785 / 23.872 |
| rastrigin | 60 | 240 | 20.815 / 19.193 | 21.107 / 20.818 | 15.785 / 15.974 | 15.785 / 15.974 |
| rastrigin | 60 | 600 | 11.454 / 10.222 | 10.242 / 10.563 | 9.034 / 9.211 | 9.034 / 9.211 |
| sphere | 8 | 120 | 0.175 / 0.195 | 0.160 / 0.181 | 0.150 / 0.201 | 0.141 / 0.170 |
| sphere | 8 | 240 | 0.002 / 0.003 | 0.002 / 0.027 | 0.042 / 0.058 | 0.001 / 0.016 |
| sphere | 8 | 600 | 0.000 / 0.000 | 0.000 / 0.000 | 0.018 / 0.018 | 0.000 / 0.012 |
| sphere | 16 | 120 | 0.955 / 1.202 | 0.494 / 0.593 | 0.828 / 0.909 | 0.828 / 0.909 |
| sphere | 16 | 240 | 0.061 / 0.075 | 0.028 / 0.036 | 0.051 / 0.067 | 0.022 / 0.041 |
| sphere | 16 | 600 | 0.000 / 0.000 | 0.000 / 0.000 | 0.020 / 0.026 | 0.000 / 0.000 |
| sphere | 60 | 120 | 1.832 / 2.421 | 2.213 / 2.680 | 2.646 / 2.986 | 2.646 / 2.986 |
| sphere | 60 | 240 | 1.146 / 1.155 | 1.019 / 1.080 | 0.685 / 0.941 | 0.685 / 0.941 |
| sphere | 60 | 600 | 0.179 / 0.160 | 0.072 / 0.081 | 0.057 / 0.068 | 0.057 / 0.068 |
| ackley | 8 | 120 | 1.615 / 1.807 | 2.553 / 2.054 | 1.553 / 1.722 | 1.553 / 1.722 |
| ackley | 8 | 240 | 0.103 / 0.611 | 0.474 / 0.991 | 0.170 / 1.054 | 0.176 / 1.038 |
| ackley | 8 | 600 | 0.001 / 0.507 | 0.000 / 0.772 | 0.039 / 0.161 | 0.001 / 0.966 |
| ackley | 16 | 120 | 3.201 / 3.125 | 3.206 / 2.891 | 3.065 / 3.093 | 3.065 / 3.093 |
| ackley | 16 | 240 | 0.999 / 1.173 | 0.949 / 1.138 | 0.750 / 0.943 | 0.750 / 0.943 |
| ackley | 16 | 600 | 0.010 / 0.128 | 0.004 / 0.007 | 0.016 / 0.030 | 0.003 / 0.118 |
| ackley | 60 | 120 | 4.181 / 4.270 | 4.300 / 4.323 | 4.632 / 4.598 | 4.632 / 4.598 |
| ackley | 60 | 240 | 3.520 / 3.428 | 3.442 / 3.330 | 3.694 / 3.552 | 3.694 / 3.552 |
| ackley | 60 | 600 | 2.119 / 1.991 | 1.257 / 1.313 | 0.763 / 0.990 | 0.763 / 0.990 |
| rosenbrock | 8 | 120 | 21.885 / 35.458 | 9.036 / 16.637 | 62.358 / 70.231 | 21.556 / 34.956 |
| rosenbrock | 8 | 240 | 2.964 / 3.903 | 2.884 / 3.404 | 16.523 / 21.907 | 3.560 / 13.774 |
| rosenbrock | 8 | 600 | 1.582 / 1.953 | 0.754 / 1.799 | 7.200 / 9.245 | 1.697 / 3.574 |
| rosenbrock | 16 | 120 | 48.814 / 75.423 | 54.792 / 65.042 | 38.533 / 58.934 | 36.030 / 57.534 |
| rosenbrock | 16 | 240 | 5.946 / 10.765 | 5.688 / 12.022 | 20.321 / 23.888 | 3.922 / 3.993 |
| rosenbrock | 16 | 600 | 1.189 / 1.785 | 1.109 / 1.860 | 13.653 / 12.243 | 0.573 / 0.669 |
| rosenbrock | 60 | 120 | 218.300 / 387.929 | 188.883 / 318.572 | 244.177 / 260.017 | 244.177 / 260.017 |
| rosenbrock | 60 | 240 | 103.334 / 129.949 | 89.948 / 95.427 | 78.451 / 85.950 | 78.451 / 85.950 |
| rosenbrock | 60 | 600 | 15.509 / 17.832 | 7.773 / 9.746 | 10.280 / 12.980 | 5.888 / 6.900 |

### The dimension sweep: `population_size` is the wrong shape

A 4-D sweep cannot distinguish "multiplier 4 is right" from "N is about 16 at
four parameters", because `population_size` was a multiplier *per parameter*
and every cell held the parameter count fixed. The 2-D / 4-D / 8-D sweep
settles it: 12 seeds, both variants, N over the multiples of D from 4 to 32,
budgets 240 and 600 with 1200 added at 8-D, the same four functions.

**No multiplier is right across dimension.** k=4 gives the best cell at 2-D
(N=8), the runner-up at 4-D (N=16) and the worst at 8-D (N=32 — the worst of
the three in 11 of the 12 cells there; the exception is Ackley at 1200 shots).
k=2 stalls at 2-D (N=4), is best at 4-D (N=8), and is best at 8-D only at 1200
shots. The multiplier form holds fixed the one thing the data says varies.

**The member count is what is invariant.** A block goes to the N holding the
lowest median on the most of the four functions:

| D | shots | generational | asynchronous d3 |
|---|---|---|---|
| 2 | 240 | N=8 (N=8: 3, N=16: 1) | N=8 and N=16 (N=8: 2, N=16: 2) |
| 2 | 600 | N=8 and N=16 (N=8: 2, N=16: 2) | N=8 (N=8: 3, N=16: 1) |
| 4 | 240 | N=8 (N=8: 4) | N=8 (N=8: 4) |
| 4 | 600 | N=8 (N=8: 3, N=16: 1) | N=8 (N=8: 4) |
| 8 | 240 | N=8 (N=8: 4) | N=8 (N=8: 4) |
| 8 | 600 | N=8 (N=8: 3, N=16: 1) | N=8 and N=16 (N=8: 2, N=16: 2) |
| 8 | 1200 | N=16 (N=16: 4) | N=8 and N=16 (N=8: 2, N=16: 2) |

N=8 takes five of the seven blocks outright, ties one with N=16, and loses one
— 8-D at 1200 shots, the largest budget in the sweep. N=16 takes over only at
the largest budgets. Counted by single cells rather than by blocks, N=8 holds
the lowest median in 19 of the 28 (dimension, budget, function) cells and N=16
in the other 9; **neither N=4 nor N=32 is best in a single cell**, in either
variant, and at 8-D and 240 shots N=32 is the worst of the three on every
function. The asynchronous variant ranks N the same way, so this is a property
of the budget rather than of the generation barrier.

**The floor is about eight members, and it is a search floor, not the mutation
floor.** N=4 satisfies `draws + 1` for `best1` and all but stalls: given two
and a half times the budget it improves by under 1% on all eight (dimension,
function) cells, where N=8 improves by 10% to 100% on the same cells. Lifting
the barrier partly unsticks it, which says the stall is a collapsed population
that a generation of feedback latency cannot re-spread. Premature convergence
is precisely the failure a default must avoid, and it sits at N=4, not at N=8.

| variant | D | function | 240 shots | 600 shots | change |
|---|---|---|---|---|---|
| generational, N=4 | 2 | rastrigin | 2.029 | 2.029 | 0.0% |
| generational, N=4 | 2 | sphere | 0.009 | 0.009 | 0.1% |
| generational, N=4 | 2 | ackley | 0.356 | 0.356 | 0.0% |
| generational, N=4 | 2 | rosenbrock | 1.023 | 1.023 | 0.0% |
| generational, N=4 | 4 | rastrigin | 8.193 | 8.128 | 0.8% |
| generational, N=4 | 4 | sphere | 0.422 | 0.420 | 0.4% |
| generational, N=4 | 4 | ackley | 2.885 | 2.881 | 0.1% |
| generational, N=4 | 4 | rosenbrock | 18.015 | 17.959 | 0.3% |
| asynchronous, N=4 | 2 | rastrigin | 1.544 | 1.530 | 0.9% |
| asynchronous, N=4 | 2 | sphere | 0.000 | 0.000 | 36.9% |
| asynchronous, N=4 | 2 | ackley | 0.244 | 0.096 | 60.8% |
| asynchronous, N=4 | 2 | rosenbrock | 0.462 | 0.457 | 1.1% |
| asynchronous, N=4 | 4 | rastrigin | 7.624 | 7.050 | 7.5% |
| asynchronous, N=4 | 4 | sphere | 0.940 | 0.689 | 26.7% |
| asynchronous, N=4 | 4 | ackley | 2.711 | 2.685 | 1.0% |
| asynchronous, N=4 | 4 | rosenbrock | 12.268 | 6.319 | 48.5% |
| generational, N=8 | 2 | rastrigin | 0.556 | 0.497 | 10.5% |
| generational, N=8 | 2 | sphere | 0.000 | 0.000 | 100.0% |
| generational, N=8 | 2 | ackley | 0.001 | 0.000 | 100.0% |
| generational, N=8 | 2 | rosenbrock | 0.245 | 0.000 | 100.0% |
| generational, N=8 | 4 | rastrigin | 6.105 | 3.270 | 46.4% |
| generational, N=8 | 4 | sphere | 0.002 | 0.000 | 100.0% |
| generational, N=8 | 4 | ackley | 0.095 | 0.000 | 99.8% |
| generational, N=8 | 4 | rosenbrock | 3.370 | 1.582 | 53.0% |

**Decided (Ian, 2026-09-19): `population_size` means the population size.**
It keeps its name and becomes the number of members, default **8**. No
multiplier survives in any form — `population_multiplier` is not introduced,
and nothing is preserved for the sake of existing files. "Now is the time to do
it right."

This is also the shape the literature uses: Storn & Price's control parameter
is **NP**, the number of population vectors, given directly, with NP ≈ 5D–10D
offered as a rule of thumb for *choosing* it rather than as the parameter's
form. The multiplier came from M-LOOP, whose `population_size` is documented at
`mloop/learners.py:770` as "multiplier proportional to the number of parameters
in a generation" — and which nonetheless had to keep the true count as a second
attribute, `num_population_members`. scipy's `popsize` is a multiplier too. We
inherited the shape from M-LOOP; we are not keeping it.

Consequences:

- `num_members` is renamed `population_size` internally, so the key, the
  attribute and the literature all use one word for one thing. That also
  settles the naming defect on its own terms: the current name does not say
  what the thing does.
- **No migration handling, and none is needed.** A file carrying
  `population_size = 15` from M-LOOP now means 15 members rather than 15 × D.
  At three parameters that is 15 instead of 45 — which moves *towards* the
  measured optimum, not away from it. `UPGRADING.md` states the new meaning
  as a fact; nothing is refused and nothing is converted.
- **Hard floor stays `draws + 1`**, which is a correctness requirement of the
  chosen strategy. The ~8-member search floor is guidance in the prose, not a
  refusal: a smaller population searches badly, and searching badly is the
  user's business.
- **Budget check: refuse `max_num_runs < 2 × population_size`.** Strictly,
  nothing evolves only when the budget is at or below one population; between
  that and two, a truncated second generation evolves some slots. Two full
  generations is still the right line — a partial generation is not a
  generation — so keep the refusal at 2×, but the message must not claim the
  band below it "cannot evolve at all". The stronger `20 × population_size`
  from the sweep is *guidance* and belongs in the prose: refusing it would
  reject configurations that work, merely not well.
- Prose: raise it to 16 when the budget exceeds about a thousand shots.

One reading to treat as looser than it sounds: the "40–75 generations" rule of
thumb is a post-hoc fit. N=8 wins blocks at 30 and at 75 generations, and N=16
wins at 37 and at 75, so generation count alone does not separate them. The
defensible statement is the simple one — default 8, raise it at large budgets.

**The whole dimension table**, generational, median / mean, 12 seeds:

| D | N | N/D | shots | rastrigin | sphere | ackley | rosenbrock |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 2 | 240 | 2.029 / 2.636 | 0.009 / 0.209 | 0.356 / 1.531 | 1.023 / 1.945 |
| 2 | 8 | 4 | 240 | 0.556 / 0.508 | 0.000 / 0.000 | 0.001 / 0.216 | 0.245 / 0.477 |
| 2 | 16 | 8 | 240 | 0.609 / 0.841 | 0.000 / 0.000 | 0.044 / 0.090 | 0.031 / 0.147 |
| 2 | 32 | 16 | 240 | 1.574 / 1.792 | 0.011 / 0.013 | 0.417 / 0.514 | 0.099 / 0.211 |
| 2 | 4 | 2 | 600 | 2.029 / 2.635 | 0.009 / 0.209 | 0.356 / 1.531 | 1.023 / 1.942 |
| 2 | 8 | 4 | 600 | 0.497 / 0.497 | 0.000 / 0.000 | 0.000 / 0.215 | 0.000 / 0.200 |
| 2 | 16 | 8 | 600 | 0.000 / 0.332 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 2 | 32 | 16 | 600 | 0.054 / 0.281 | 0.000 / 0.000 | 0.009 / 0.010 | 0.001 / 0.003 |
| 4 | 4 | 1 | 240 | 8.193 / 9.310 | 0.422 / 0.824 | 2.885 / 2.777 | 18.015 / 79.704 |
| 4 | 8 | 2 | 240 | 6.105 / 6.527 | 0.002 / 0.003 | 0.095 / 0.400 | 3.370 / 4.264 |
| 4 | 16 | 4 | 240 | 12.165 / 12.561 | 0.066 / 0.083 | 1.064 / 1.252 | 5.922 / 11.507 |
| 4 | 32 | 8 | 240 | 18.450 / 15.916 | 0.436 / 0.715 | 3.092 / 2.917 | 39.716 / 51.342 |
| 4 | 4 | 1 | 600 | 8.128 / 9.160 | 0.420 / 0.780 | 2.881 / 2.767 | 17.959 / 79.297 |
| 4 | 8 | 2 | 600 | 3.270 / 3.377 | 0.000 / 0.000 | 0.000 / 0.307 | 1.582 / 1.967 |
| 4 | 16 | 4 | 600 | 3.970 / 4.365 | 0.000 / 0.000 | 0.013 / 0.167 | 1.189 / 1.665 |
| 4 | 32 | 8 | 600 | 7.100 / 6.890 | 0.009 / 0.011 | 0.548 / 0.483 | 3.782 / 3.866 |
| 8 | 8 | 1 | 240 | 36.070 / 37.234 | 0.457 / 1.273 | 2.085 / 2.318 | 67.422 / 130.667 |
| 8 | 16 | 2 | 240 | 45.996 / 47.873 | 1.932 / 1.929 | 3.536 / 3.584 | 249.013 / 440.579 |
| 8 | 32 | 4 | 240 | 52.540 / 54.398 | 6.119 / 6.683 | 4.717 / 4.706 | 1009.468 / 1092.445 |
| 8 | 8 | 1 | 600 | 14.437 / 15.505 | 0.026 / 0.377 | 1.343 / 1.418 | 7.784 / 18.508 |
| 8 | 16 | 2 | 600 | 29.704 / 28.839 | 0.033 / 0.038 | 0.575 / 0.636 | 17.090 / 32.281 |
| 8 | 32 | 4 | 600 | 37.689 / 38.768 | 0.573 / 0.612 | 2.747 / 2.778 | 87.084 / 119.783 |
| 8 | 8 | 1 | 1200 | 12.032 / 12.561 | 0.000 / 0.232 | 1.297 / 1.394 | 6.283 / 6.712 |
| 8 | 16 | 2 | 1200 | 11.137 / 12.353 | 0.000 / 0.000 | 0.010 / 0.117 | 4.895 / 9.688 |
| 8 | 32 | 4 | 1200 | 30.054 / 29.593 | 0.021 / 0.028 | 0.543 / 0.536 | 8.942 / 14.806 |

**Caveats, to be carried by anything citing these numbers.** Four analytic test
functions, all on the same box; two to eight parameters; budgets of 120 to 1200
shots; one mutation strategy, `best1`; no noise on the cost, no dropped shots
and no unusable ones. A lab's landscape is none of those things, and the sizing
guidance is guidance rather than a refusal for that reason. What the fake
runmanager does not exercise is everything above the session — the worker, the
routine, lyse and a real queue — and what these runs do not exercise at all is
the loss paths: a dropped founder, a NaN trial and a cost arriving after its
generation's boundary are covered by the test suite rather than measured here.

### Tests, each proven by a mutation

- a dropped founder leaves its slot vacant and shifts no other slot
  [mutation: usable-count founding];
- a late-returning founder changes no other proposal's role [B's walk];
- roles at generation equal roles at replay under random order, drops, NaN and
  late returns, checked after **every arrival** [any counting walk];
- a NaN trial leaves its slot's member unchanged [`usable(history)` walk];
- with `cross_over_probability = 0`, a trial for block position p shares all
  but one coordinate with slot p's member [target by completed count];
- the configured start point is proposed once [completed-only history];
- `refill` proposes nothing while a generation is outstanding, and exactly N
  when none is [the top-up formula];
- `starved` stays zero across a clean generational run [count at boundaries];
- `population_size = 15` builds fifteen members [multiply by `num_params` —
  forty-five];
- a `population_size` below `draws + 1` is refused at construction [delete the
  check];
- `max_num_runs < 2 × population_size` is refused at load [delete the check —
  it loads];
- `num_buffered_runs` written in a file whose learner declares a block is
  refused [delete the check — it loads, and `refill` tops up mid-generation].

Note the third: checking **after every arrival** is what distinguishes these
from the earlier measurements, which compared only final replays. A final
replay is order-independent by construction and reports 0% wrong however broken
the intermediate behaviour is.

### Corrections

Two things this document stated were wrong. The body above has been corrected;
what was wrong, and why, is recorded here rather than beside each number, so
that the decision reads as what it is rather than as an argument with itself.

**The harness truncation (2026-09-19).** Every benchmark taken before this date
was driven by a loop that counted proposals *submitted*: it landed one shot per
turn and stopped with whatever was still in the queue unscored. The cost scales
with queue depth, so it fell on the generational arm — at 120 shots with a
population of 60 it was scored on 61 of its 120 shots against the asynchronous
arm's 118 — and it biased the dimension sweep against large populations the
same way. Those readings were also taken from a scratch reimplementation of the
design driven by a hand-rolled pipeline, whereas the numbers above are the
shipped learner driven by the shipped session.

The barrier's price was recorded as a median ratio of generational to
asynchronous of **1.36 / 1.53 / 1.67** at 120 / 240 / 600 shots. At equal
completed shots it is **1.01 / 1.05 / 1.64**.

Reapplying the truncation deliberately to the shipped learner is what
identifies the harness rather than the learner as the cause: it reproduces the
old shape, moving the aggregate from 1.01 / 1.05 / 1.64 to 1.23 / 1.23 / 1.71.
That accounts for most of the difference at 120 shots, part of it at 240, and
nothing that needed explaining at 600. What is left over is the remaining
distance between the two harnesses — the earlier asynchronous runs completed
their shots in submission order where these complete them in a random one, and
the earlier generational and asynchronous arms were two separate scratch
functions where these are one shipped class and a two-line subclass of it.

| shots | N | shots the generational arm was scored on | ratio under that rule | ratio at equal completed shots |
|---|---|---|---|---|
| 120 | 8 | 113 of 120 | 1.19 | 1.07 |
| 120 | 16 | 113 of 120 | 1.11 | 0.94 |
| 120 | 60 | 61 of 120 | 1.52 | 0.99 |
| 240 | 8 | 233 of 240 | 1.04 | 0.89 |
| 240 | 16 | 225 of 240 | 1.56 | 1.17 |
| 240 | 60 | 181 of 240 | 1.26 | 1.07 |
| 600 | 8 | 593 of 600 | 1.57 | 1.52 |
| 600 | 16 | 593 of 600 | 1.64 | 1.78 |
| 600 | 60 | 541 of 600 | 2.10 | 1.84 |

**The conclusions survive, and the decision is cheaper than it was recorded as
being.** Nothing reopens. The gap still does not close with budget (it opens),
eight members is still the default across parameter counts, Rastrigin is still
where the correct algorithm loses to the walk it replaced, and the reason
for textbook DE was never the benchmark but that the name has to be true. What
changed is the size of the bill: the barrier costs nothing measurable at short
budgets rather than about a third of the search.

The rows are in `benchmarks/results/truncated_4d.csv`, and the one-line edit
that produced them is quoted in `benchmarks/README.md`. It is not a mode of the
harness, because a supported way to run two arms on different numbers of shots
is a supported way to make the mistake again; `tests/test_benchmarks.py` fails
if the harness ever counts that way.

**The N=4 stall.** This document said N=4's medians at 240 and 600 shots were
"identical to three significant figures", which was never literally true of the
numbers it quoted in the same sentence — 8.19 against 8.13. The defensible
statement, and the one above, is that N=4 improves by **under 1% for two and a
half times the budget**, on all eight (dimension, function) cells, where N=8
improves by between 10% and 100% on the same cells. The reading it supports —
that premature convergence sits at four members and not at eight — is
unchanged.

---

## Issue 3 — parameter-to-global collisions

Re-verified against the current tree (post-`global_name`-removal), all four
still load:

- two active groups both defining `x` → `globals_for([0.5, 5.5])` gives
  `{'ga': 5.5, 'gb': 5.5}`; `ga`'s parameter is bounded `[0, 1]`
- two parameters sharing one `global_name` → `{'g': 0.9}`, one global for two
  searched dimensions
- `[RUNMANAGER_GLOBALS]` with no `expr` and two args → `{'g': 0.1}`, second arg
  dropped
- with no `expr` and zero args → `IndexError: list index out of range` inside
  `globals_for`, mid-session

**Recommendation: refuse all four at load, in two places that already validate.**

- `GlobalMapping.__post_init__` (cases 3, 4 and 5): with `expr is None`,
  `args` must have exactly one entry. Beside the existing `expr` compilation,
  and for the same stated reason — "left until a proposal needs it, a mistyped
  lambda would first be found mid-session".
- **Case 5, added on codex's point and verified:** when there *is* an `expr`,
  check the compiled callable accepts `len(args)` arguments. Today
  `expr = "lambda x: x"` with `args = ["x", "y"]` loads without complaint and
  then raises `TypeError: <lambda>() takes 1 positional argument but 2 were
  given` inside `globals_for`, mid-session. That is the same failure the
  `__post_init__` compilation exists to prevent, so leaving it out would make
  the stated rationale only half true. `inspect.signature(function)` gives the
  arity; allow `*args` through.
- `Config.__post_init__` (cases 1 and 2): no two entries of `space.parameters`
  share a `name`; no two entries of `globals` share a `name`. This is Config's
  own invariant, because `globals_for`'s two dict comprehensions are what
  collapse. Putting it here also catches a `Config` built directly in a test or
  a script, which is the same argument the in-flight `build()` change makes for
  `validate_options`.

Not in `check_keys`: that runs before group activity is known and deliberately
inspects switched-off groups, where a duplicate name is legitimate.

**Placement correction from the Fable session.** Parameter-name uniqueness is
`ParameterSpace`'s invariant, not `Config`'s, by this document's own rule: a
standalone learner handed a space with two parameters named `x` is malformed
too, with no `Config` in sight. Global-name uniqueness stays in `Config`. Also:
`inspect.signature` raises `ValueError` for some callables, so let an `expr`
whose signature cannot be read pass rather than refusing it.

**Tests and mutations**

- Each of the four configurations raises `ValueError` at `loads`, with the
  colliding name in the message. **Mutation:** delete the corresponding check;
  each test then sees the load succeed.
- A parameter named `x` in an *inactive* group alongside an active `x` still
  loads. **Mutation:** move the duplicate check into `check_keys`; this fails.

**Blast radius:** none on existing tests — I ran all four configurations and
none resembles a fixture. `examples/config_example.toml` is unaffected.
`config.py`'s module docstring should gain a sentence that a parameter name and
a global name are each unique across the active groups. **Effort: ~2 h.**

---

## Issue 4 — coercive scalar settings

Re-verified, plus one the list missed: `seed = 1.9 → 1`.

The serious ones are silent deaths. Confirmed empirically — three successive
`refill()` calls on a fresh session:

| setting | refills | `starved` | `stopped` |
|---|---|---|---|
| `num_buffered_runs = 0` | `[], [], []` | 0 | `None` |
| `num_buffered_runs = -1` | `[], [], []` | 0 | `None` |
| `max_num_runs = 0` | `[], [], []` | 0 | `None` |

**That settles the "`max_num_runs = 0` is arguably a legitimate stop
immediately" question: it is not.** `check_stop` is only reached from `record`,
so with nothing ever submitted `stopped` stays `None` and the status explains
nothing. It is the same silent death as `num_buffered_runs = 0`, not a clean
stop. Refuse it.

`max_num_runs_without_better_params = 0` is the mild one: it submits a first
batch and then stops on the first cost *with a message*. Still incoherent — the
run that sets the best always has zero runs after it — so refuse, but it is the
least urgent of the four.

**Recommendation: `Config.__post_init__`, not a schema layer.** `config.py`
already validates in exactly this shape: `Parameter.__post_init__` checks its
own bounds, `GlobalMapping.__post_init__` compiles its own `expr`,
`ParameterSpace.__init__` checks it has a parameter, learner constructors check
their own knobs, and `from_dict` checks the cross-object relationships. A
schema layer would be a fourth idea about where validation lives. The dataclass
checking its own fields is the third *instance* of an idea already used twice,
and it is the only placement that covers a `Config` built directly.

**Two corrections from the Fable session, both verified.** First,
`validate_options` iterates `learner_options` only, so the *selected* learner is
never checked: `learner = "gaussain_process"` loads without complaint and fails
later at `build()`, i.e. at worker configure, with the session already starting.
Check `config.learner in LEARNERS` where the option validation runs. Second,
"`from_dict` coerces nothing" must be scoped to the `present` calls —
`Parameter(minimum=float(entry["min"]))` has to keep coercing, because the
example file legitimately writes `min = 5` and `max = 40` as integers.

**Consolidated on codex's point: `Config.__post_init__` is the single
authority, and `from_dict` coerces nothing beyond that.** This document previously split
exact-type checking into `from_dict` and range checking into `Config`, which
would put two rules about one field in two files and leave a directly
constructed `Config` half-checked. Instead `from_dict` passes the values
through as written — dropping the `int` and `str` converters from its `present`
calls — and `__post_init__` is authoritative for both TOML and direct
construction.

What it checks:

- exact integers, rejecting `bool`, for `num_buffered_runs`,
  `num_training_runs`, `max_num_runs`, `max_num_runs_without_better_params`
  and `seed`;
- ranges: `num_buffered_runs >= 1`; `num_training_runs >= 0`; `seed >= 0`;
  `max_num_runs` and `max_num_runs_without_better_params` each `None` or `>= 1`;
- `maximize` an exact `bool`;
- `learner` and `session` strings — `str()` coercion has exactly the problem
  `int()` coercion has, and this document missed it;
- `cost_key` two strings.

Two traps worth naming, because each is a check that would otherwise look
clean: `isinstance(True, int)` is `True` in Python, so every integer check must
exclude `bool` explicitly; and `require_type`'s `written` dict is keyed on
`bool` and `list` only and will `KeyError` on a new kind.

**Tests and mutations**

- `num_buffered_runs = 0`, `= -1`, `max_num_runs = 0`,
  `max_num_runs_without_better_params = 0` each raise at `loads`.
  **Mutation:** delete the clause; the load succeeds and, in a follow-on
  assertion, `refill()` returns `[]` three times with `stopped is None`.
- `num_buffered_runs = 2.9` raises and the message says integer.
  **Mutation:** restore `convert=int`; it loads as 2.
- `num_buffered_runs = true` raises. **Mutation:** use a bare
  `isinstance(value, int)`; it loads as 1.
- `cost_key = [1, 2]` raises. **Mutation:** drop the string check; it loads and
  `uncertainty_key` becomes `(1, 'u_2')`.

**Blast radius:** `tests/test_config.py` builds its minimal TOML without these
keys and `test_a_file_that_sets_no_options_gets_exactly_the_dataclass_defaults`
compares against a directly-constructed `Config` whose defaults all pass — but
any test that builds a `Config(...)` directly now runs `__post_init__`, so check
`tests/test_learners.py::a_config` and `tests/test_session.py::make_config`
first. Both look safe. **Effort: ~1.5 h.**

---

## Issue 5 — `best_cost` reports the internal sign

**Recommendation: unflip for reporting. Keep the column name.**

One line in `Session.status`: report `best.cost` negated when
`self.config.maximize`. It is the exact mirror of the single flip in
`routine.extract`, and it puts the reported number in the units the lab wrote
in its own cost column, beside a `best_params` that was never flipped.

Against renaming to `best_loss`: it changes a published column name *and*
leaves a maximising lab reading a negative number they must re-flip by hand.
The minimisation convention is the package's internal business; exporting it
into the lyse dataframe exports an implementation detail into the GUI a
physicist reads.

**What it costs an existing lab: nothing.** Issue 7 establishes the package is
not installed in the labscript env and has never run through lyse, so there is
no saved analysis reading `best_cost` anywhere. This is the cheapest moment
this decision will ever be available. That is the whole argument for doing it
now rather than deciding it later.

**Decided: report in lab units.** `best_cost` comes back in the units and sign
of the lab's own cost column; the column keeps its name and `best_loss` is not
adopted. Unflip in `Session.status` as above.

**What UPGRADING.md must say:** the "What stays the same" bullet currently reads
"`maximize` still flips the sign", which is true of the internal treatment and
misleading about the column. It should say that `maximize` means your cost
column holds something to be made large, and that `best_cost` comes back in the
same units and sign as that column. (Another agent is editing this file right
now — name the bullet, not a line number.)

**Test and mutation.** Codex is right that one observation is too weak: it
proves only that something was negated, not that the right point was chosen.
Use **at least two maximising observations**, say 7.0 and 3.0, and assert both
that `status()["best_cost"] == 7.0` and that `best_shot_id` names the shot that
measured 7.0 — so the test proves internal minimisation picked the larger
measurement *and* that reporting restored the lab's sign. **Mutation:** remove
the negation; it reports `-7.0`. **Second mutation:** make `Session.best` pick
the maximum internally; `best_shot_id` names the wrong shot. Add the mirror for
`maximize = false`, so a future refactor cannot satisfy the first by negating
unconditionally.

**Blast radius:** `Session.status` is the only producer; `routine.SHOT_RESULTS`
carries it through unchanged. No existing test asserts the sign — worth noting
as another check that was never there. **Effort: ~30 min.**

---

## Issue 6 — the cost must already exist when the routine runs

Documentation only. **One place: `README.md`, "Using it", the paragraph
beginning "You compute the cost yourself, in your own lyse routine".** That is
where the reader is standing when they add the routine to lyse, and lyse's
routine *ordering* is the actual subject — not the configuration file. Roughly:

> The column named by `cost_key` has to exist by the time this routine runs.
> lyse runs multishot routines in list order, so the routine that computes your
> cost must be a singleshot routine, or a multishot routine above this one in
> the list. A cost column that is not there yet is not waited for: the shot is
> recorded as a bad observation, and every shot of the run will be.

I would not repeat it in `examples/config_example.toml` or `UPGRADING.md` —
"say each thing once, where someone would be standing when they need it", and
the config file's `cost_key` comment already covers what the key means.

No test: this is prose about lyse's behaviour, not this package's.
**Effort: ~15 min.**

---

## Issue 7 — setup

### (a) Not installed — just do it

Confirmed: `pip show labscript-optimization` reports nothing in the labscript
env, and `import labscript_optimization` from `/tmp` raises
`ModuleNotFoundError`. Every *other* suite repo is editable-installed from
`/Users/ispielma/Code/Python/Labscript/` — `blacs`, `labscript`,
`labscript-devices`, `labscript-utils`, `lyse`, `runmanager`, `runviewer`,
`zprocess`. This one alone is not, which means nothing in it has ever been
exercised through lyse.

```
~/miniforge3/envs/labscript/bin/pip install -e /Users/ispielma/Code/Python/Labscript/labscript-optimization
```

`setuptools_scm` resolves `0.1.0.dev30` on `Development` with no tags, so the
build is fine as it stands.

**No unit test for it.** This document proposed one, and codex is right that
it would be wrong: a test that fails because *this workstation* was not
installed editable tests the environment, not the repository. It would be
tautological in any CI that installs the package before running the suite, and
would turn a plain source-tree test run red for a reason that has nothing to do
with the code. Treat 7a as environment setup and document it, nothing more.

If packaging ever needs permanent verification, the right shape is a separate
job that builds the distribution, installs it into a clean environment and
imports it from outside the checkout — which tests packaging rather than
whether someone ran a command on their laptop. Not needed today.
**Effort: ~5 min, and it is a command, not a change.**

### (b) `main` and `Production` at "Initial commit" — decided: leave them

Confirmed: both are `4a2cbad`, holding `.gitignore` and nothing else, and
`refs/remotes/origin/HEAD` points at `origin/main`.

**No merge forward is proposed.** Development leading Production is deliberate
across this suite. But something real does follow, and it is not about
promotion:

- Anyone cloning `https://github.com/spielman-group/labscript-optimization/`
  gets a repository containing one `.gitignore`. They must already know to
  `git checkout Development`.
- The GitHub landing page renders no README, so the project has no public
  description — and `pyproject.toml` publishes that URL as `Repository`.
- `pip install git+https://github.com/spielman-group/labscript-optimization`
  fails: no package at the default ref.
- `UPGRADING.md` step 1 tells a reader to `pip install -e
  /path/to/labscript-optimization`, which is a path they got by cloning.

**Decided: no change. `main` stays the public default and is populated when
testing finishes.** The consequences listed above are accepted as the cost of a
repository that is effectively private until its first release. Do not change
the default branch, and do not merge anything forward.

Two things follow that *are* in scope, because they are true now and will still
be true at release:

- `UPGRADING.md` step 1 and the README both assume a local checkout. Whatever
  they say about obtaining the package should be written once, at release,
  against whatever `main` then holds — not patched now to describe a clone that
  does not work.
- `pyproject.toml` publishes the repository URL. That is correct and should
  stay; it simply does not resolve to anything installable yet.

**Effort: none.**

---

## Summary

Three models have now reviewed this: me, codex, and the Fable session. The
Fable session found a defect the other two missed and broke the design the
other two had converged on. Everything below was verified here before being
accepted.

**Order of work:**

1. **Issue 0 — the routine must hand over every unseen row.** lyse runs
   multishot routines once per drained batch, not once per shot, so
   `n_shots=1` silently discards every analysed shot but the last. Their costs
   never reach the learner and `dropped` climbs for shots that ran perfectly.
   This is a prerequisite: a generation cannot complete cleanly until it is
   fixed, and it is my error — the `n_shots` feature was adopted here on a
   premise lyse's model does not have.
2. **7a** — install the package. A command, not a change; no unit test.
3. **4** — `Config.__post_init__` as the single authority, plus checking the
   *selected* learner, which `validate_options` never did.
4. **3** — name collisions and `expr` arity, with parameter-name uniqueness in
   `ParameterSpace` rather than `Config`.
5. **5** — unflip `best_cost`, two-observation test.
6. **6** — one README paragraph on cost ordering.
7. **2** — textbook generational DE.

**Issue 2, settled.** Whole generations submitted at once, which dissolves the
founding-phase role instability structurally. But generational submission is
**not sufficient alone**: founding must be by block position, not by usable
count, or a dropped founder promotes a trial to founder and re-indexes every
role after it. With position founding the mismatch rate is 0% under random
completion order, drops, NaNs and late returns; with usable-count founding it
is not. `restart_tolerance` is removed — it is the one remaining source of
order-dependence and it is harmful within lab budgets anyway. The boundary is a
learner attribute, `generation: int | None`, in the same family as
`minimum_observations`; `num_buffered_runs` is *determined* by it rather than
checked; `starved` is not counted at its boundaries; `max_num_runs < 2N` is
refused.

**The cost is real and must be documented.** Over four functions generational
DE beats the code it replaces in 5 of 12 cells at 120 shots, 5 of 12 at 240 and
6 of 12 at 600 — roughly even, not uniformly worse. It wins decisively on
Rosenbrock and Ackley and loses on Rastrigin, the multimodal one. What it does
pay, standingly, is the generation barrier: asynchronous DE leads it by a
median 1.01× / 1.05× / 1.64× at the three budgets — nothing measurable at the
shortest — and that gap does **not** close with budget. It opens.

**`population_size` is the wrong shape, not just the wrong number.** The
dimension sweep (2-D, 4-D, 8-D) shows no multiplier is right across dimension:
k=4 is best at 2-D and worst at 8-D. What is invariant is the **member count** —
N=8 takes five of seven (dimension, budget) blocks outright and ties a sixth,
N=16 the largest budget, and N=4 all but stalls, improving by under 1% for two
and a half times the budget. At four parameters M-LOOP's default of 15 is N=60,
the worst value tested in 12 of 12 cells.

**Decided: `population_size` becomes the population size**, in members,
default 8 — the name it already has, now meaning what it says, and the form
Storn & Price use (NP, given directly). No multiplier in any form. A file
carrying M-LOOP's `population_size = 15` now means 15 members instead of 15 × D,
which moves towards the measured optimum; `UPGRADING.md` states the new meaning
and nothing is refused or converted. Hard floor stays `draws + 1`; refuse
`max_num_runs < 2 × population_size`; the 20× rule is prose.

**Still declined:** the retry machine — deferral, not dismissal, and the record
keeps a `state` field so it stays buildable. GP pending-point conditioning — it
halves a real 28% duplication and makes the answer consistently worse. The
`uncer_bias` documentation this document once proposed is withdrawn: the GP
cannot see where queued shots are, so it would have asserted a mechanism the
code does not have.

**Re-baseline before implementing.** The suite is 224 tests, not the 216 the
earlier blast radius was counted against. The worker startup leak and the
silent configure timeout are both fixed; the 30 s / 60 s timeout mismatch
behind them is not.

**Nothing is open.** Every measurement has landed and every decision is taken.
The plan is ready to implement in the order above, starting with issue 0.
