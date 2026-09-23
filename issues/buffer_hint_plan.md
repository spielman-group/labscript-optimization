# B — `num_buffered_runs` as a hint, and the Gaussian process's own cycle

Design for `labscript-optimization` (branch `Development`), reconciled with a
second model's critique and decided by Ian. The work items are in
`issues/buffer_hint.md`; each names the sections here it implements.

## 1. The decision error

The session asks **one learner to fill the whole queue**:
`wanted = num_buffered_runs - awaiting`, every refill. A method with a cadence
of its own either breaks the method or grows a special case, and most of the
package's recent complexity is those special cases:

- **Generational DE** refuses `num_buffered_runs`, declares `Learner.generation`,
  and the session gains a barrier branch plus an exemption from the starvation
  count because the queue empties every generation by design.
- **The GP** is asked for one point at a time in the steady state, so its batch
  machinery never engages (#2 below).
- **Warmup** needed `TwoPhaseLearner`, its refusals (no generational learner, no
  trainer with a minimum), `num_training_runs`, the warmup refusal added in
  `b55eedf`, and the periodic trainer `num_runs_between_trainer_runs`
  (`8bae554`).

Four confirmed defects trace to it:

| # | Defect | Evidence |
|---|---|---|
| 2 | The GP proposes without regard to its own shots in flight: at `num_buffered_runs = 5` nearly every steady-state refill is k=1, and `condition_on` works only within one call. This was known and kept deliberately — conditioning on pending points was measured to halve a 28% duplication and make the answer consistently worse (`codex_issues_proposal.md`, "Still declined"). B closes it by construction: the GP never proposes while its own batch is out. | A k=1 refill at an exploring weight re-proposed the point in flight (distance 0.0). |
| 6 | The `phase` column labels the wrong shots. The status is the most recent *proposal's* phase, written onto the shot being *analysed*. | Simulated routine: 7 of 24 wrong — handover shots proposed by the trainer labelled `main`, the last shots stuck at `periodic trainer` after the budget ran out. It lined up in the steady state only because 4 in flight happened to equal a cycle of 4. |
| 7 | The trainer runs `num_training_runs` + the shots in flight at handover. B keeps this as the design, stated: warmup ends at the count, and shots already queued complete. | `num_training_runs = 4`, `num_buffered_runs = 5`: 8 training shots. |
| 8 | The default warmup is refused by the default learner. | `num_training_runs` defaults to 5; the GP needs 2×D; any GP search over ≥ 3 parameters that omits it is refused. |

## 2. The model

**`num_buffered_runs` keeps its name and means what it says — how many shots to
keep queued — and each learner honours it as far as its method allows.**
(settled)

### Random and directed random — unchanged (settled)
Exactly `num_buffered_runs` of their proposals in flight. Floor 1: below it
nothing is ever proposed. The default becomes 2 (§3).

### Differential evolution — unchanged (settled)
DE ignores `num_buffered_runs` and the explorer entirely. It keeps its
generation barrier: a whole generation when none of its own is outstanding,
otherwise nothing. The queue may run dry between generations, and runmanager
then idles or sends default shots according to its empty-queue policy.
`num_buffered_runs` stays refused beside DE, as today — a setting nothing acts
on. Its history then holds only its own proposals, so its position walk is
untouched. **Depends on C2:** today a single default shot makes the routine stop
handing over costs, so "run dry" is safe only with runmanager on "Send nothing"
until C2 lands.

### The Gaussian process — a batch cycle with its own explorer (settled)

- **`batch_size`** (default 4, see §3) replaces `refit_interval`. The GP proposes a
  batch, each point conditioned on the ones before it, and refits its
  hyperparameters once per batch.
- **`explorer`** (`random` or `directed_random`, default `directed_random`) and
  **`explore_runs`** (default 1) live in **`[LEARNER.gaussian_process]`** — no
  other learner uses them. The explorer's own knobs still come from its own
  table (`[LEARNER.directed_random]`).
- **The rule.** Explorer shots queued behind each batch =
  **max(`explore_runs`, `num_buffered_runs`)**.

  | `explore_runs` | `num_buffered_runs` | explorer shots per batch |
  |---|---|---|
  | 2 | 0, 1 or 2 | 2 |
  | 2 | 5 | 5 |

  Why: computing the next batch is slow, so it cannot go out the moment the
  last one returns. The explorer shots queued behind a batch are what keep the
  apparatus busy during that fit, which is exactly what `num_buffered_runs`
  asks for. `explore_runs` is the exploring the GP does regardless; the buffer
  only ever adds to it.
- **The next batch goes out when the GP's own batch is back** — every GP shot
  of it completed or dropped. Explorer shots do not hold it up; their costs
  land whenever they land and join the fit.
- **`num_buffered_runs = 0` is legal for the GP** and means no buffer beyond
  `explore_runs`. Exploring is not switched off by it. The default
  `num_buffered_runs` is 2 (§3).
- **`warmup_observations`** replaces both `minimum_observations` and
  `num_training_runs`, and defaults to `max(5, 2 × num_params)` (see §3) — a
  default that scales with the search, because a constant one is what #8 was.
  Below it the explorer alone keeps
  max(`explore_runs`, `num_buffered_runs`, 1) queued. It counts **usable
  observations**, not shots: a shot whose cost is NaN does not count. Warmup
  ends at the count, and the explorer shots already queued at handover — up to
  max(`explore_runs`, `num_buffered_runs`) — still run. That is the design, and
  the docs say so.
- **Deleted:** `TwoPhaseLearner`, `[GENERAL] trainer` (now the GP's `explorer`),
  `num_training_runs`, `num_runs_between_trainer_runs`, `refit_interval`,
  `minimum_observations`. Each old key is refused with its replacement named;
  none is aliased.
- **Defaults reproduce the cycle of M-LOOP, which this package replaces.**
  `batch_size = 4` with `uncer_bias = (0, 1, 2, 3)` walked across the batch,
  then one explorer run, is its four machine-learner runs at weights 0–3
  followed by one run from its training source (`controllers.py:918` and
  `learners.py:1866` in its source, where
  `generation_num = bias_func_cycle = 4`).

### Across all learners (settled)

- **The configured start** takes the main learner's first position (DE slot 0,
  as now; for the GP, the first warmup shot).
- **Every proposal records the learner that made it** (`source` on the record).
  The GP needs it to know when its batch is back; the `phase` column written
  onto a shot becomes that shot's own source, which closes #6.

### Configuration, before and after (Ian's file)

```toml
# before
[GENERAL]
learner = "gaussian_process"
num_buffered_runs = 5
num_training_runs = 20
trainer = "directed_random"
# num_runs_between_trainer_runs = 4
[LEARNER.gaussian_process]
# refit_interval = 4
# minimum_observations = 10

# after
[GENERAL]
learner = "gaussian_process"
num_buffered_runs = 5          # 5 explorer shots behind each batch of 4
[LEARNER.gaussian_process]
explorer = "directed_random"
explore_runs = 1
batch_size = 4
warmup_observations = 20
```

## 3. Decided (Ian)

1. **The default `num_buffered_runs` is 2, for every learner that reads it.**
   Of the shots in flight, one is always the shot BLACS is running: a shot
   comes back when lyse has analysed it, after BLACS has already asked for the
   next. At 1, a random learner therefore never has a shot waiting when BLACS
   asks, and the apparatus alternates with default shots; at 2, one is always
   waiting. For the GP the default gives max(1, 2) = 2 explorer shots behind
   each batch of 4, and a fit a shot of cover. Ian runs at 2 or more in
   practice, so the default is set to how the package is used rather than to
   the four-to-one ratio of the cycle in §2, and needs no per-learner
   declaration.
2. **`explore_runs = 0` with `num_buffered_runs = 0` is allowed**: a pure GP
   that idles the apparatus during fits. Unwise for most labs, impossible for
   none, and `starved` counts it truthfully.
3. **The max() rule stays**, so `num_buffered_runs` means one thing — shots
   queued — for every learner that reads it. For the GP it has a second effect,
   stated in one sentence in the GP's docstring and the README: raising it to
   cover a slow fit also raises the share of explorer shots.
4. **No measurement of the batch cadence; defaults from the literature.** The
   rules that scale with dimension are for the initial design, not the batch:
   about 10·D for an initial computer experiment (Loeppky, Sacks and Welch,
   *Technometrics* 51, 2009), and `max(5, min(2·D, num_trials // 5))` in Ax's
   generation strategy. Batch size is set by available parallelism, not by
   dimension — Ax's default Bayesian concurrency is the constant 3, and the cycle
   §2's defaults reproduce is the constant 4. So:
   - **`batch_size = 4`**, constant — which also pairs one batch with one pass
     of the default four-weight `uncer_bias` schedule.
   - **`warmup_observations = max(5, 2 × num_params)`** — Ax's rule, without
     the budget term, which this package's warmup has never had.

## 4. Design questions — resolved

**Q1. The cycle lives in `GaussianProcessLearner`**, in one method that takes
history and hint and returns proposals with their sources. The scheduler is
future work and this neither anticipates nor constrains it: whenever it comes,
these settings would be stripped out and moved into it either way, so making
the current arrangement logical now costs nothing later. (A sequential schedule
of disjoint stages could not express "the explorer runs while the next fit is
computed" in any case, so the cycle is learner-internal on its own terms.)

**Q2. The contract** becomes `propose(history, hint)`, returning zero or more
proposals, each with its source. Random: the hint minus its own in flight. DE:
a generation when none of its own is outstanding, else nothing — the barrier
moves from the session into DE, read off its own pending records. GP: the
cycle, or nothing while its batch is out.

**Q3. Keep `generation` as a declaration.** `build()` refuses
`max_num_runs < 2 × generation` by asking the built instance, and deleting the
declaration would force it to read `population_size` off DE specifically — the
class-specific proxy an earlier slice removed. A future schedule needs it too:
a barrier stage followed by another must be a multiple of it. The starvation
exemption stays keyed on it, as today, so no new declaration is added. The GP
declares none, so a pure GP's starvation counts.

**Q4. No conditioning on in-flight explorer shots.** Dropped: it contradicts
the measurement above, and a random explorer draw is a weaker thing to
fantasise than a GP proposal. If duplication matters, measure it first.

**Q5. The schedule index is position within the batch**, `uncer_bias[i % len]`,
so each batch starts from the greedy weight.

**Q6. The hyperparameter cache is keyed on the usable observations in hand when
the batch is computed** — a function of history; the late-cost test stands.

**Q7. "Back" means every GP-sourced proposal of the latest batch is completed
or dropped.** Dropping depends on reconcile, which runs on a routine
invocation, which needs a shot to arrive. If an operator deletes everything
queued under "Send nothing", the apparatus idles and the cycle waits with it —
the empty-queue caveat the README already carries for DE, now stated for the GP
beside it.

**Q8. The budget cuts the batch last**: the batch first, then explorer shots
until the room runs out.

**Q9. One implementation of "in flight."** Learners read record states from
history; `Session.awaiting` is proposals minus results minus dropped. Derive
`awaiting` from the history's pending records, so the two cannot disagree —
rather than documenting an invariant someone must keep. `Observation.source` is
recorded at proposal time and never changes, which is the kind of fact history
may carry; nothing keeps a mutable count.

## 5. What changes

- `learners/two_phase.py` — deleted.
- `learners/__init__.py` — `NEEDS_TRAINING` wrapping deleted; `build()` hands the
  GP its explorer.
- `learners/base.py` — the contract (Q2); `generation` stays (Q3), the
  `minimum_observations` declaration goes.
- `learners/gaussian_process.py` — batch cycle, explorer, `explore_runs`,
  `warmup_observations`, `batch_size`, refit per batch, Q5.
- `learners/differential_evolution.py`, `learners/random.py` — the contract
  only; DE's algorithm untouched.
- `observations.py` — `Observation.source`.
- `session.py` — refill passes the hint and submits what comes back; records
  sources; `awaiting` derived from the history (Q9).
- `worker.py`, `routine.py` — each shot's verdict carries its source, and
  `save_status` writes it as that shot's `phase` (#6).
- `config.py` — `[GENERAL]` loses `trainer`, `num_training_runs`,
  `num_runs_between_trainer_runs`; the `num_buffered_runs` floor becomes 0 for
  the GP; old keys refused with replacements named.
- `README.md`, `UPGRADING.md`, `examples/config_example.toml`, Ian's own
  configuration.
- Tests: `TwoPhaseLearner` and periodic-trainer tests deleted or rewritten; new
  tests for the rule, warmup, the batch barrier, sources, and the phase column —
  each proven by mutation.

## 6. Interactions

- **A (in flight).** A2 added `labscript_optimization/knobs.py`; the new GP knobs
  use it. A4 changes only DE's weight draw and B leaves DE untouched, so the
  benchmark sweep behind `population_size = 8` can be re-run as soon as A4
  lands. B starts after the A worker finishes: both touch
  `gaussian_process.py`, `session.py` and `config.py`.
- **C2.** A prerequisite for DE running dry under "Send default shot". The GP
  cycle also relies on the routine delivering each shot; today a shot arriving
  mid-pass is handed over with a NaN cost, which counts as "back" but loses the
  observation. C2 and B's phase-column work both edit `routine.py`, so they are
  sequenced, not parallel.
- **Closed by B:** #2 (by construction), #6, #8. **Stated as design, not
  closed:** #7.

## 7. Tests B needs, each proven by the mutation in brackets

- The GP proposes exactly `batch_size` points, each conditioned on the ones
  before it, and nothing while any GP-sourced proposal of its latest batch is
  pending [ignore states — a second batch goes out].
- A dropped GP shot releases the batch [require completed only — the cycle
  stalls forever on a deleted shot].
- Explorer shots behind a batch number max(`explore_runs`, `num_buffered_runs`)
  [off by one either way].
- Warmup ends at `warmup_observations` *usable* observations [count shots — warmup
  ends early on NaN costs].
- `warmup_observations` defaults to `max(5, 2 × num_params)` [a constant — #8
  returns; drop the floor — a one-parameter search warms up on two].
- Every shot's `phase` is its own source [write the latest proposal's phase —
  7 of 24 reproduces].
- The budget cuts explorer shots before any of the batch [reverse it].
- `starved` is not counted for DE and is counted for a GP at
  `explore_runs = 0, num_buffered_runs = 0` [count both, or neither].
- The budget refusal still fires at `max_num_runs < 2 × population_size` after
  the contract change [delete `generation`].
- `Session.awaiting` and the history's pending records agree after drops, late
  costs and duplicates [keep a separate count].
