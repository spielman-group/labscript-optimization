# Upgrading from M-LOOP and analysislib-mloop

Your existing `mloop_config.toml` will not load as it stands, and the two
runmanager globals the old plugin needed are no longer used. Everything else
carries over: the same parameter and globals tables, the same cost column, the
same algorithms.

Work through the six steps below. Each says what to change and why, so you can
tell whether it applies to your lab.

## 1. Install it, and point lyse at the new routine

```
pip install -e /path/to/labscript-optimization
```

Run that against the Python environment lyse itself runs in, the one holding
the rest of your suite: the routine runs inside a lyse analysis subprocess, so
an installation anywhere else is one lyse cannot import.

Needs Python 3.11 or newer, and numpy, scipy and scikit-learn. There is no
tensorflow: the neural-network learner is gone, and with it M-LOOP's hard
import of tensorflow through every other learner.

Your lyse analysis routine becomes two lines:

```python
import labscript_optimization.routine as optimisation

optimisation.optimise('mloop_config.toml')
```

Remove `mloop_multishot.py`, `mloop_interface.py`, `mloop_controller.py`,
`mloop_learner.py` and `monkey.py` from your analysis directory. Nothing from
analysislib-mloop is imported any more, and `mloop` itself is not a dependency.

## 2. Cut the settings that no longer exist

A key this package does not know is now **refused**, naming the key, the
table it was found in, and what that table accepts. A setting that is accepted
and then quietly ignored is how a lab comes to believe an option is in force
when it is not, so loading stops rather than carrying on.

Delete these from your configuration:

| Table | Delete | Why |
| --- | --- | --- |
| `[ANALYSIS]` | `ignore_bad` | A shot whose cost is `NaN` is now recorded as a bad observation: counted as a completed run, left out of the fits. Nothing waits for it, so there is nothing to switch off. |
| `[ANALYSIS]` | `analysislib_console_log_level`, `analysislib_file_log_level` | The plugin's own logging configuration. It has no logging of its own to configure. |
| `[MLOOP]` | `no_delay` | The Gaussian process runs in a worker process that never blocks the routine, so there is no delay to avoid. |
| `[MLOOP]` | `visualisations` | No plots and no GUI. Progress comes back as the routine's results. |
| `[MLOOP]` | `console_log_level`, `console_log_string` | As above. |
| `[MLOOP]`, `[LEARNER.differential_evolution]` | `restart_tolerance` | The population is not re-seeded when its costs converge. That decision was taken at a generation boundary from the costs resolved by then, so a cost arriving afterwards could change it and turn a block generated as trials into founders of a new epoch; and within a lab's budget it re-seeded populations that had converged to within a fraction of their initial spread but not to the minimum. `max_num_runs_without_better_params` is the stop to use instead. |
| whole table | `[COMPILATION]` | Its only key was `mock`, which selected a dry-run interface that has been removed. |

And rename two:

| Table | Rename | To |
| --- | --- | --- |
| `[MLOOP]` | `controller_type` | `learner` |
| `[MLOOP]`, `[LEARNER.gaussian_process]` | `generation_size` | `batch_size` |

`batch_size` does what `generation_size` did: it is the period of the Gaussian
process's exploration schedule and how many new observations it accepts before
refitting the kernel, which is what batch Bayesian optimisation calls a batch.
The word `generation` belongs to differential evolution, where it names the
population's step from one whole set of members to the next, and one document
cannot hold two senses of it.

Archive paths, `archive_type` and the other M-LOOP pass-through keys go the
same way if you have them. There is no archive: if the worker dies, the
history dies with it and a new session starts from nothing.

`min` and `max` in a parameter table keep those spellings; `minimum` and
`maximum` are not accepted.

## 3. Move every learner knob into its learner's table

`[MLOOP]` carries the session's own settings and nothing else. A learner's
knobs are written in `[LEARNER.<name>]`, the table of the learner that takes
them, and a knob found in `[MLOOP]` is **refused**, with the table it belongs
in named. Nothing is preserved: there is no shared table of learner knobs any
more.

| Move out of `[MLOOP]` | Into |
| --- | --- |
| `trust_range`, `trust_gaussian`, `explore_fraction` | `[LEARNER.directed_random]` |
| `population_size`, `evolution_strategy`, `mutation_scale`, `cross_over_probability` | `[LEARNER.differential_evolution]` |
| `cost_has_noise`, `cost_bias`, `uncer_bias`, `batch_size`, `length_scale_bounds`, `noise_level_bounds`, `minimum_observations` | `[LEARNER.gaussian_process]` |
| `trust_region` | all three of those tables take it — write it in each one you want it in, with the value you want there |

That split is the reason for the change. Of the fifteen keys `[MLOOP]` used to
accept as shared learner knobs, fourteen are taken by exactly one learner. The
sharing served a single knob, `trust_region`, and for the other fourteen it put
a setting where it read as though it might apply to any learner and then
dropped it in silence for the ones that do not take it.

The selectable trainer, under *What changed in the algorithms* below, is
what forces it. A session that trains runs two
learners at once, and a knob in `[MLOOP]` reaches both with no way to say which
was meant — so a wide trust region to train with and a tight one to refine with
could not both be asked for. In each learner's own table they can:

```toml
[LEARNER.directed_random]
trust_region = 0.2

[LEARNER.gaussian_process]
trust_region = 0.05
```

A table written for a learner this file does not build goes unread, so the
settings for several learners can sit side by side and be switched between by
changing `[MLOOP] learner`.

## 4. Remove the tag globals

**Delete `mloop_session` and `mloop_iteration` from runmanager.** The old
plugin stamped them on every shot so it could recognise its own results. A
shot is now identified by the id runmanager mints for its queue row and writes
into the shot file, which lyse reads as the `shot_id` column, so there is
nothing for you to create and nothing to keep in step.

## 5. Check `num_buffered_runs`

It now defaults to 3 rather than 1, and more than one is usually what you
want. BLACS asks for its next shot as soon as it finishes the last, which is
before the optimiser has seen the cost and proposed a replacement — so a queue
holding only one of its shots is empty at precisely that moment and runmanager
gives BLACS a default shot instead. At one buffered run roughly every second
shot is a default one. Those shots go to BLACS already compiled, so runmanager
never writes a shot id into them.

The status counts a `starved` for each time the routine found nothing of its
own queued. If it keeps climbing, the fit is taking longer than a shot: raise
`num_buffered_runs`.

**`differential_evolution` refuses the key.** It proposes one whole population
at a time and is not asked again until every member has been answered for, so
its queue depth is `population_size`; a second setting for the same number is
one that can disagree with it. Delete `num_buffered_runs` from a file that
names that learner. The queue empties once per generation there, by design,
and nothing counts a `starved` for it.

## 6. Read `population_size` again

**`population_size` is the number of members in the differential evolution
population.** It is NP as the literature gives it: a file saying
`population_size = 15` is asking for fifteen members, over however many
parameters. Nothing is converted and nothing is refused on account of the
number you already have — at three parameters, fifteen members is a smaller
population than M-LOOP would have built and a better one than that. Around
eight members is where a population stops converging prematurely, and a budget
over about a thousand shots is worth sixteen; the strategy's own minimum, three
members for `best1`, is refused below and is nowhere near enough to search
with.

Those two numbers are measured, and measured narrowly: four analytic test
functions, two to eight parameters, budgets of 120 to 1200 shots, and the
`best1` strategy only. Over that sweep eight members holds the best median in
nineteen of the twenty-eight (parameters, budget, function) cells and sixteen
members in the other nine; counted by blocks of four functions, eight wins
five of the seven and sixteen takes a block only at the largest budget. Four
members, at two and at four parameters, improves by under a per cent when
given two and a half times the budget, which is what converging prematurely
looks like. `codex_issues_proposal.md`, under "The dimension sweep", has the
tables, and `benchmarks/` has the harness and the rows behind them. Your landscape is not an analytic test function, so
treat the two numbers as a place to start.

A budget has to cover two whole generations of that population, so
`max_num_runs` below `2 × population_size` is refused: the first generation is
the population itself and the second is the first to evolve it.

The same number is how many shots are in the queue at once, since a generation
goes out whole.

## What stays the same

- `[MLOOP_PARAMS.<group>.<name>]` with `global_name`, `min`, `max`, `start`
  and `enable`, and `[RUNMANAGER_GLOBALS.<group>.<name>]` with `expr` and
  `args`, are unchanged. So is `ANALYSIS.groups` selecting which groups take
  part, and `enable = false` keeping a parameter in the file but out of the
  search.
- `cost_key` is still `[routine_name, result_name]`, `maximize` still means
  your cost column holds something to be made large, and an optional
  `u_<result>` column alongside is still read as the uncertainty and weights
  the Gaussian process fit. `best_cost` comes back in the same units and sign
  as that column.
- You still compute the cost yourself, in your own lyse routine. This package
  never computes one.
- `num_training_runs`, `max_num_runs` and `max_num_runs_without_better_params`
  mean what they meant.
- Adding the routine to lyse starts a session; removing or restarting it, or
  reaching the run budget, stops it.

## What changed in the algorithms

- **The learners are named** `random`, `directed_random`,
  `differential_evolution` and `gaussian_process`, in `[MLOOP] learner`.
- **The trainer is chosen**, in `[MLOOP] trainer`, and defaults to
  `directed_random`. `gaussian_process` is the only learner that needs one: it
  runs the trainer for its training shots and falls back to it for any proposal
  it cannot make. A name that is not a learner is refused, and so is a trainer
  named beside a learner that needs none, which would be a setting nothing acts
  on.

  **The default is not what M-LOOP did.** Its machine-learning controllers took
  `training_type`, defaulting to `differential_evolution`
  (`mloop/controllers.py`), and used that learner for the training shots and
  for any point the machine-learning learner was too slow to supply — so both
  came from a population clustered around the best points seen. This package's
  `directed_random` centres its draws on a band of *middling* costs instead,
  which is what makes it explore rather than refine, so training and fallback
  range much wider and produce stretches of poor shots that M-LOOP never
  showed. That is what a run against the dummy apparatus looks like.

  `differential_evolution` cannot be the trainer here: it proposes a whole
  population at a time and only when none of its proposals is outstanding, and
  a two-phase learner cannot hold that barrier across a handover, so a file
  naming it is refused rather than run in pieces. To train nearer the best
  points, narrow the trainer's own band — `trust_range = [0.9, 1.0]` in
  `[LEARNER.directed_random]` centres on the best point rather than on
  middling ones, and `[1, 1]` is the best point alone. `trainer = "random"` is
  the plain uniform spread over the whole space.
- **Nelder-Mead and the neural network are gone.** Nelder-Mead may return;
  the neural network will not.
- **The directed random learner's trust region now works.** Its guard sent
  every draw to the whole space once any cost had been recorded, which left
  `trust_region`, `trust_range` and `trust_gaussian` doing nothing — so this
  learner was a plain random one in every lab running it. A bad run, arriving
  as an infinite cost, also left the trust band undefined so that nothing fell
  inside it. Both are fixed, which means **this learner will behave
  differently from the one you have been running**, as it was always meant to.
  `trust_range` still defaults to `[0.1, 0.25]`, which centres the search on
  middling results rather than the best one: that is deliberate, and is what
  makes it explore rather than refine. A pair written the wrong way round is
  now refused at construction, where M-LOOP quietly sorted it.
- **Every learner knob goes under `[LEARNER.<name>]`**, which step 3 above is
  the move for. A named learner table is strict: an unknown learner, or a key
  that learner's constructor does not accept, stops the file from loading.
- **Differential evolution is generational**, which is what the textbook
  algorithm and scipy's deferred updating are. A whole population is proposed
  at once and none of its trials is judged until the generation is complete,
  so the incumbent a trial competes against is the one it was bred from. A
  proposal's position in the history is its role -- founders fill the first
  population, and every block after them is a generation of trials, one to a
  slot -- so a shot that never reports leaves its slot empty rather than
  shifting every role after it. The barrier costs sample efficiency against an
  asynchronous variant, and the cost grows with the budget rather than washing
  out: over four analytic test functions at four parameters it is nothing
  measurable at 120 shots and a factor of 1.6 in the best cost found at 600.
  The name has to be true.
- **`seed`** in `[MLOOP]` makes a run reproducible.

## What the routine reports

Where the search has got to is written onto each shot the session proposed, so
it comes back as dataframe columns under `labscript_optimization` —
`df[('labscript_optimization', 'best_cost')]` and so on. The keys are `phase`,
`best_cost`, `best_params`, `best_shot_id` and `stopped`. A key the session has
nothing to report for yet reads as `NaN`.

The session's counters — `submitted`, `completed`, `awaiting`, `dropped`,
`blocked` and `starved` — are one answer for the whole run rather than anything about a shot,
so they are not written onto every shot of it. `optimise` returns the whole
status, so a routine that wants them prints it:

```python
print(optimisation.optimise('mloop_config.toml'))
```

`dropped` counts shots the session proposed that will never produce a cost:
cancelled, deleted, refused, or gone from the queue with nothing reaching lyse.
A shot that simply ran is not among them. runmanager answers for a finished
shot exactly as it does for one that was deleted, so a shot is given up on only
when a second reconcile says the same, by which time a healthy shot's cost has
come back through lyse. A `dropped` above zero is shots the run actually lost,
and is worth looking into.

`blocked` is the part of that number worth acting on. It counts shots sitting
behind a row the queue will not hand over — a rejected head, or one whose
compile failed — and nothing moves them but an operator. Every other way a
shot stops coming is the apparatus getting on with things; this one means go
and look at the queue.

## Two failures worth recognising

**A shot that cannot compile stops everything.** Such a row stays at the head
of runmanager's queue until someone deletes it, and nothing behind it runs —
so nothing reaches lyse and the optimiser is never invoked again. The row is
red in runmanager and the queue has visibly halted. That is the signal; the
optimiser cannot report it, because it is not running. Delete the row and the
queue moves.

**The session stops if the labscript file changes underneath it.** Its shots
would no longer be the experiment it has been optimising, so it says so rather
than carrying on.

## One thing that looks like a failure and is not

A cost that arrives after the generation it belonged to. The shots behind a red
row are given up on and counted in `dropped`, and the search moves on without
them; you delete the row, they run, and their costs come back for a generation
that has already been replaced. Each is still taken and competes for its own
slot and no other -- which is what differential evolution would have done with
it had it arrived in time. Two generations of the optimiser's shots sit in the
queue while it catches up, and `awaiting` reads high for as long as they do.
