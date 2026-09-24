# Upgrading from M-LOOP and analysislib-mloop

Your existing `mloop_config.toml` will not load as it stands, and the two
runmanager globals the old plugin needed are no longer used. Everything else
carries over: the same parameters and globals, the same cost column, the same
algorithms.

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

## 2. Rename the two M-LOOP tables, and cut the settings that no longer exist

Do the renames first. The rest of this document names the new tables, and an
old name is **refused** before any other complaint about the file is reached.

| Rename | To |
| --- | --- |
| `[MLOOP]` | `[GENERAL]` |
| `[MLOOP_PARAMS.<group>.<name>]` | `[PARAMETERS.<group>.<name>]` |

This package replaces M-LOOP and carries none of its code, so a settings table
named after it was a name nothing written under it answered to. `[GENERAL]`
reads against `[ANALYSIS]` and `[LEARNER.<name>]` without naming the table
after the package every table in the file belongs to anyway.
`[RUNMANAGER_GLOBALS.<group>.<name>]` was already named for what it holds and
is untouched. Nothing is read under an old name, and a file carrying both is
told about both at once.

Then the keys. A key this package does not know is also **refused**, naming the
key, the table it was found in, and what that table accepts. A setting that is
accepted and then quietly ignored is how a lab comes to believe an option is in
force when it is not, so loading stops rather than carrying on.

Delete these from your configuration:

| Table | Delete | Why |
| --- | --- | --- |
| `[ANALYSIS]` | `ignore_bad` | A shot whose cost is `NaN` is now recorded as a bad observation: counted as a completed run, left out of the fits. Nothing waits for it, so there is nothing to switch off. |
| `[ANALYSIS]` | `analysislib_console_log_level`, `analysislib_file_log_level` | The plugin's own logging configuration. It has no logging of its own to configure. |
| `[GENERAL]` | `session` | A label. Nothing read it but the status `optimise` returns, where it was one more key in the printed dictionary; it was never written onto a shot and never matched on. A shot is attributed by the id runmanager mints for its queue row, which lyse reads as the `shot_id` column. |
| `[GENERAL]` | `no_delay` | The Gaussian process runs in a worker process that never blocks the routine, so there is no delay to avoid. |
| `[GENERAL]` | `visualisations` | No plots and no GUI. Progress comes back as the routine's results. |
| `[GENERAL]` | `console_log_level`, `console_log_string` | As above. |
| `[LEARNER.random]`, `[LEARNER.directed_random]`, `[LEARNER.differential_evolution]` | `first_params` | Where a run begins is written on the parameters, as each one's `start` beside its own `min` and `max`, and the session proposes that point first and once, whichever learner is running. `first_params` was a second way to say the same thing, and only three of the four learners took it, so the file's answer to where a run starts depended on which learner had been chosen. As a bare vector it could not be checked against the parameter tables it stood for either: reorder them and it silently meant a different experiment, which is not an error but a different run. Write `start` on the parameters instead. |
| `[GENERAL]`, `[LEARNER.differential_evolution]` | `restart_tolerance` | The population is not re-seeded when its costs converge. That decision was taken at a generation boundary from the costs resolved by then, so a cost arriving afterwards could change it and turn a block generated as trials into founders of a new epoch; and within a lab's budget it re-seeded populations that had converged to within a fraction of their initial spread but not to the minimum. `max_num_runs_without_better_params` is the stop to use instead. |
| whole table | `[COMPILATION]` | Its only key was `mock`, which selected a dry-run interface that has been removed. |

And rename these, each into `[LEARNER.gaussian_process]`, the table of the
Gaussian process, which runs its warmup and its explorer itself:

| Table | Rename | To |
| --- | --- | --- |
| `[GENERAL]` | `controller_type` | `learner`, staying in `[GENERAL]` |
| `[GENERAL]`, `[LEARNER.gaussian_process]` | `generation_size` | `batch_size` |
| `[GENERAL]` | `num_training_runs` | `warmup_observations` |
| `[GENERAL]` | `trainer` | `explorer` |
| `[GENERAL]` | `num_runs_between_trainer_runs` | `explore_runs`, a different number: see below |
| `[GENERAL]`, `[LEARNER.gaussian_process]` | `refit_interval` | `batch_size` |
| `[GENERAL]`, `[LEARNER.gaussian_process]` | `minimum_observations` | `warmup_observations` |

The last four appear only in a file already written for this package; an
analysislib-mloop file carries none of them. The last five are each
**refused** naming the key that replaces it and the table it goes in, and none
is read as its replacement: `explore_runs`, for one, counts the explorer shots
behind each batch, where `num_runs_between_trainer_runs` was a period between
one of them and the next. *What changed in the algorithms*, below, says what
each knob does.

`generation_size` set two things at once: how often the Gaussian process
refits its kernel hyperparameters, and the period of its exploration schedule.
The schedule is written out as `uncer_bias`, a list of weights whose length is
its own period, and `batch_size` is how many points the Gaussian process
proposes at a time, refitting its hyperparameters once per batch and walking
the schedule from its first weight at every batch. The word `generation`
belongs to differential evolution, where it names the population's step from
one whole set of members to the next, and one document cannot hold two senses
of it.

`uncer_bias` takes the schedule the two of them implied. Under
`generation_size = 4` and `uncer_bias = 1.0`, successive proposals weighted
the predicted uncertainty by 0, 1, 2 and 3 and then began again; write that as

```toml
[LEARNER.gaussian_process]
uncer_bias = [0.0, 1.0, 2.0, 3.0]
batch_size = 4
```

which is the default, so a file that wants it need write neither. A single
number is still accepted and is a cycle of one step -- `uncer_bias = 1.0`
alone is a weight of 1.0 on every point, with no greedy one among them.

Archive paths, `archive_type` and the other M-LOOP pass-through keys go the
same way if you have them. There is no archive: if the worker dies, the
history dies with it and a new session starts from nothing.

`min` and `max` in a parameter table keep those spellings; `minimum` and
`maximum` are not accepted.

`start` keeps its spelling and gains a rule. It is one coordinate of a single
opening point over every searched parameter, so it is written on all of them
or on none, and a file writing it on some is **refused**, naming which
parameters carry one and which do not. It used to be dropped instead: a start
on three parameters of five was ignored for all five, and the run opened on a
uniform draw with nothing said. A parameter carrying `enable = false` is not
searched, so it is not a coordinate of that point and needs no `start`.

## 3. Move every learner knob into its learner's table

`[GENERAL]` carries the session's own settings and nothing else. A learner's
knobs are written in `[LEARNER.<name>]`, the table of the learner that takes
them, and a knob found in `[GENERAL]` is **refused**, with the table it belongs
in named. Nothing is preserved: there is no shared table of learner knobs any
more.

| Move out of `[GENERAL]` | Into |
| --- | --- |
| `trust_range`, `trust_gaussian`, `explore_fraction` | `[LEARNER.directed_random]` |
| `population_size`, `evolution_strategy`, `mutation_scale`, `cross_over_probability` | `[LEARNER.differential_evolution]` |
| `cost_has_noise`, `cost_bias`, `uncer_bias`, `batch_size`, `length_scale_bounds`, `noise_level_bounds`, `warmup_observations`, `explorer`, `explore_runs` | `[LEARNER.gaussian_process]` |
| `trust_region` | all three of those tables take it — write it in each one you want it in, with the value you want there |

That split is the reason for the change. Every one of those knobs but
`trust_region` is taken by exactly one learner, so a shared table put each of
them where it read as though it might apply to any learner and then dropped it
in silence for the ones that do not take it.

The Gaussian process's explorer, under *What changed in the algorithms* below,
is what forces it. A Gaussian process runs its explorer beside itself, two
learners at once, and a knob in `[GENERAL]` reaches both with no way to say
which was meant — so a wide trust region to explore with and a tight one to
refine with could not both be asked for. In each learner's own table they
can:

```toml
[LEARNER.directed_random]
trust_region = 0.2

[LEARNER.gaussian_process]
trust_region = 0.05
```

A table written for a learner this file does not build goes unread, so the
settings for several learners can sit side by side and be switched between by
changing `[GENERAL] learner`.

## 4. Remove the tag globals

**Delete `mloop_session` and `mloop_iteration` from runmanager.** The old
plugin stamped them on every shot so it could recognise its own results. A
shot is now identified by the id runmanager mints for its queue row and writes
into the shot file, which lyse reads as the `shot_id` column, so there is
nothing for you to create and nothing to keep in step.

## 5. Check `num_buffered_runs`

It defaults to 2, because one of the shots in flight is always the one BLACS
is running. BLACS asks for its next shot as soon as it finishes
the last, which is before the optimiser has seen the cost and proposed a
replacement — so a queue holding only one of its shots is empty at precisely
that moment and runmanager gives BLACS a default shot instead. At one buffered
run roughly every second shot is a default one. Those shots go to BLACS
already compiled, so runmanager never writes a shot id into them.

The status counts a `starved` for each time the routine found nothing of its
own queued. If it keeps climbing, the fit is taking longer than a shot: raise
`num_buffered_runs`.

**Under `gaussian_process` it is the explorer shots behind each batch.** The
number queued behind a batch is the larger of `num_buffered_runs` and
`explore_runs`, so raising it to cover a slow fit also raises the share of
explorer shots. Zero is accepted there and queues `explore_runs` alone. The
random learners keep exactly `num_buffered_runs` in flight and so refuse zero,
which would be a run that never proposes.

**`differential_evolution` refuses the key.** It proposes one whole population
at a time and nothing more until every member has been answered for, so its
queue depth is `population_size`; a second setting for the same number is
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
looks like. `benchmarks/README.md`, under "What the sweep found", has the
tables, the caveats, and the harness and rows behind them. Your landscape is not an analytic test function, so
treat the two numbers as a place to start.

A budget has to cover two whole generations of that population, so
`max_num_runs` below `2 × population_size` is refused: the first generation is
the population itself and the second is the first to evolve it, so a budget
under two of them never evolves anything. It need not be a whole number of
generations above that. The last one is cut short where the budget runs out,
and its trials compete for their own slots like any other; nothing reads the
population after it.

The same number is how many shots are in the queue at once, since a generation
goes out whole.

## What stays the same

- What the parameter and globals tables carry is unchanged, and only the
  first of them was renamed: `[PARAMETERS.<group>.<name>]` takes
  `global_name`, `min`, `max`, `start` and `enable` -- `start` under the rule
  above -- and
  `[RUNMANAGER_GLOBALS.<group>.<name>]` takes `expr` and `args`.
  `ANALYSIS.groups` still selects which groups take part, and `enable = false`
  still keeps a parameter in the file but out of the search. A name in
  `groups` that no table defines is refused, naming it, so a misspelt group
  cannot leave the one it meant switched off.
- `cost_key` is still `[routine_name, result_name]`, `maximize` still means
  your cost column holds something to be made large, and an optional
  `u_<result>` column alongside is still read as the uncertainty and weights
  the Gaussian process fit. `best_cost` comes back in the same units and sign
  as that column.
- You still compute the cost yourself, in your own lyse routine. This package
  never computes one.
- `max_num_runs` and `max_num_runs_without_better_params` mean what they
  meant.
- Adding the routine to lyse starts a session; removing or restarting it, or
  reaching the run budget, stops it.

## What changed in the algorithms

- **The learners are named** `random`, `directed_random`,
  `differential_evolution` and `gaussian_process`, in `[GENERAL] learner`.
- **The Gaussian process runs its own cycle**, set in
  `[LEARNER.gaussian_process]`. Its `explorer` — `random` or
  `directed_random`, and `directed_random` by default — proposes a warmup of
  `warmup_observations` usable observations, which defaults to twice the
  number of searched parameters and never fewer than five. After that the
  Gaussian process proposes `batch_size` points at a time, 4 by default, each
  conditioned on the ones before it, and behind each batch go
  max(`explore_runs`, `num_buffered_runs`) explorer shots; `explore_runs`
  defaults to 1. The next batch goes out when every point of the last is back
  or given up on, and the explorer shots never hold it up. Warmup counts
  observations a fit can use, not shots, and ends at the count: explorer shots
  already queued then still run. The `phase` column reads `warmup`, `main` or
  `explore` for the three kinds of shot.

  With `num_buffered_runs = 1` those defaults are M-LOOP's own cycle: its
  machine-learning controller ran `generation_num` machine-learner runs, fixed
  at 4, at the four exploration weights `[0, 1, 2, 3]` that are `uncer_bias`
  here, and then one run from its training source, round and round
  (`mloop/controllers.py`, `mloop/learners.py`). At the default of 2, two
  explorer shots follow each batch rather than one. The other half of M-LOOP's
  condition — take a training point whenever the machine learner has none
  ready — was `no_delay`, which step 2 above deletes: the explorer shots queued
  behind each batch are what keep the apparatus busy while it is fitted.

  **The explorer's default is not what M-LOOP trained with.** Its
  machine-learning controllers took `training_type`, defaulting to
  `differential_evolution` (`mloop/controllers.py`), and used that learner for
  the training shots, for the periodic training runs among them, and for any
  point the machine-learning learner was too slow to supply — so all of them
  came from a population clustered around the best points seen. This package's
  `directed_random` centres its draws on a band of *middling* costs instead,
  which is what makes it explore rather than refine, so the warmup and the
  explorer shots range much wider and produce stretches of poor shots that
  M-LOOP never showed. That is what a run against the dummy apparatus looks
  like.

  `differential_evolution` cannot be the explorer: it proposes a whole
  population at a time and only when none of its proposals is outstanding, so
  it can neither open a warmup shot by shot nor fill the buffer behind a
  batch, and a file naming it is refused. To explore nearer the best points,
  narrow the explorer's own band — `trust_range = [0.9, 1.0]` in
  `[LEARNER.directed_random]` centres on the best point rather than on
  middling ones, and `[1, 1]` is the best point alone. `explorer = "random"`
  is the plain uniform spread over the whole space.
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
  that learner's constructor does not accept, stops the file from loading, and
  so does a value of the wrong kind -- a quoted `"false"`, a fraction where a
  whole number belongs, or one number where a pair belongs, as in
  `mutation_scale`, whose fixed weight is written `[0.8, 0.8]`.
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
- **`seed`** in `[GENERAL]` makes a run reproducible.

## What the routine reports

Where the search has got to is saved into lyse's dataframe, in the row of each
shot the session proposed, as columns under `labscript_optimization` —
`df[('labscript_optimization', 'best_cost')]` and so on. It is not written into
the shot files, so shots loaded into lyse afresh come without it. The keys are
`phase`, `best_cost`, `best_params`, `best_shot_id` and `stopped`. `phase` is
the shot's own: what proposed that shot, which is `start` for the configured
start and, under `gaussian_process`, `warmup`, `main` or `explore`. A key
the session has nothing to report for yet is empty — `NaN` for `best_cost`, an
empty string or list for the rest.

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
