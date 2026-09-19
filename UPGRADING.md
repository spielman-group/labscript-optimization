# Upgrading from M-LOOP and analysislib-mloop

Your existing `mloop_config.toml` will not load as it stands, and the two
runmanager globals the old plugin needed are no longer used. Everything else
carries over: the same parameter and globals tables, the same cost column, the
same algorithms.

Work through the four steps below. Each says what to change and why, so you can
tell whether it applies to your lab.

## 1. Install it, and point lyse at the new routine

```
pip install -e /path/to/labscript-optimization
```

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

A key this package does not act on is now **refused**, naming the key, the
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
| whole table | `[COMPILATION]` | Its only key was `mock`, which selected a dry-run interface that has been removed. |

And rename one:

| Table | Rename | To |
| --- | --- | --- |
| `[MLOOP]` | `controller_type` | `learner` |

Archive paths, `archive_type` and the other M-LOOP pass-through keys go the
same way if you have them. There is no archive: if the worker dies, the
history dies with it and a new session starts from nothing.

`min` and `max` in a parameter table keep those spellings; `minimum` and
`maximum` are not accepted.

## 3. Remove the tag globals

**Delete `mloop_session` and `mloop_iteration` from runmanager.** The old
plugin stamped them on every shot so it could recognise its own results. A
shot is now identified by the id runmanager mints for its queue row and writes
into the shot file, which lyse reads as the `shot_id` column, so there is
nothing for you to create and nothing to keep in step.

## 4. Check `num_buffered_runs`

It now defaults to 3 rather than 1, and more than one is usually what you
want. BLACS asks for its next shot as soon as it finishes the last, which is
before the optimiser has seen the cost and proposed a replacement — so a queue
holding only one of its shots is empty at precisely that moment and runmanager
gives BLACS a default shot instead. At one buffered run roughly every second
shot is a default one. Those shots carry no shot id, so they are never
mistaken for the optimiser's own.

The status counts a `starved` for each time the routine found nothing of its
own queued. If it keeps climbing, the fit is taking longer than a shot: raise
`num_buffered_runs`.

## What stays the same

- `[MLOOP_PARAMS.<group>.<name>]` with `global_name`, `min`, `max`, `start`
  and `enable`, and `[RUNMANAGER_GLOBALS.<group>.<name>]` with `expr` and
  `args`, are unchanged. So is `ANALYSIS.groups` selecting which groups take
  part, and `enable = false` keeping a parameter in the file but out of the
  search.
- `cost_key` is still `[routine_name, result_name]`, `maximize` still flips
  the sign, and an optional `u_<result>` column alongside is still read as the
  uncertainty and weights the Gaussian process fit.
- You still compute the cost yourself, in your own lyse routine. This package
  never computes one.
- `num_training_runs`, `max_num_runs` and `max_num_runs_without_better_params`
  mean what they meant.
- Adding the routine to lyse starts a session; removing or restarting it, or
  reaching the run budget, stops it.

## What changed in the algorithms

- **The learners are named** `random`, `directed_random`,
  `differential_evolution` and `gaussian_process`, in `[MLOOP] learner`.
  `gaussian_process` runs `directed_random` for its training shots and falls
  back to it for any proposal it cannot make, which is what
  `controller_type = "gaussian_process"` did before.
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
- **Per-learner settings** may go under `[LEARNER.<name>]`, overriding the
  same keys in `[MLOOP]`. Knobs in `[MLOOP]` still apply to whichever learner
  takes them, so nothing has to move.
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
