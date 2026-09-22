# Benchmarks

`de_pipeline.py` drives the differential evolution learner that ships through a
real `Session`, over analytic test functions, and writes one row per run. Every
measured figure this package quotes for differential evolution was produced by
a command below, from the rows in `results/`, and is set out under "What the
sweep found".

This is not a second test suite. A benchmark measures search quality, which is
a number that moves; the suite in `tests/` proves behaviour, which is a thing
that either holds or does not. What the suite does run is the `smoke` sweep
(`tests/test_benchmarks.py`), because a harness nobody runs rots, and a check
that does not run looks exactly like one that passes.

## The rule the numbers depend on

**Every arm runs to the same number of completed shots, and nothing is
submitted once the budget is claimed.** The session's `max_num_runs` counts
shots *completed*, `refill` leaves room for what is already in flight, and the
loop in `run` goes until the session stops.

Counting proposals *submitted* instead is the defect that invalidated every
benchmark taken before 2026-09-19. It stops with whatever is still in the queue
unscored — two shots for a depth-3 arm, up to `N - 1` for a generational one —
so the two arms are scored on different numbers of shots and the cost scales
with queue depth. At 120 shots with a population of 60 the generational arm was
scored on 61 of its 120 shots against the asynchronous arm's 118. It is
invisible in a sweep's output: the rows look ordinary and only the aggregate
moves.

## Running it

The package must be installed into the labscript environment (see the root
`README.md`); the harness imports it rather than reaching into a checkout.

```
~/miniforge3/envs/labscript/bin/python benchmarks/de_pipeline.py run <sweep> --out <file>
~/miniforge3/envs/labscript/bin/python benchmarks/de_pipeline.py report <sweep> <file>...
```

Seeds are 0 upwards, and the completion order of each run is drawn from
`default_rng(seed + 991)`, so a sweep re-run at the same commit reproduces its
rows exactly. `run` uses every core; `barrier` takes about ten seconds and
`dimension` about a minute on a ten-core machine.

## What produced each table

| table | command | rows | learner at |
|---|---|---|---|
| The whole 4-D table | `run barrier --out benchmarks/results/barrier_4d.csv` then `report barrier benchmarks/results/barrier_4d.csv benchmarks/results/barrier_4d_former.csv` | `benchmarks/results/barrier_4d.csv`, `benchmarks/results/barrier_4d_former.csv` | `24e7d0a`; former arms `9dd3994` |
| What the barrier costs | as above | as above | as above |
| Rastrigin against the former walk | as above | as above | as above |
| Members against parameters | `run dimension --out benchmarks/results/dimension.csv` then `report dimension benchmarks/results/dimension.csv` | `benchmarks/results/dimension.csv` | `24e7d0a` |
| Which N takes each block | as above | as above | `24e7d0a` |
| The floor at four members | as above | as above | `24e7d0a` |
| The withdrawn stopping rule | `report truncation benchmarks/results/truncated_4d.csv benchmarks/results/barrier_4d.csv` | `benchmarks/results/truncated_4d.csv` | `24e7d0a`, plus the edit below |

Each file in `results/` repeats its date, commit and command in its own header.

## What the sweep found

This is where `population_size`'s default of **8** comes from, and it is what
`DifferentialEvolutionLearner`'s docstring and the root `README.md` cite.

**The setting is a member count, not a multiplier on the parameter count.** A
4-D sweep alone cannot tell "a multiplier of 4 is right" from "about 16 members
at four parameters", because every cell holds the parameter count fixed. The
2-D / 4-D / 8-D sweep settles it: 12 seeds, both variants, N over the multiples
of D from 4 to 32, budgets of 240 and 600 with 1200 added at 8-D, the same four
functions. No multiplier holds across dimension — k=4 gives the best cell at
2-D and the worst at 8-D, and k=2 stalls at 2-D and is best at 4-D — while the
member count ranks the same way at every dimension.

A block goes to the N holding the lowest median on the most of the four
functions:

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
— 8-D at 1200 shots, the largest budget in the sweep. **That is where the two
figures come from: eight members for the budgets a lab usually runs, and
sixteen once the budget passes a thousand shots.** Counted by single cells
rather than by blocks, N=8 holds the lowest median in 19 of the 28 (dimension,
budget, function) cells and N=16 in the other 9; neither N=4 nor N=32 is best
in a single cell, in either variant, and at 8-D and 240 shots N=32 is the worst
of the three on every function. The asynchronous variant ranks N the same way,
so this is a property of the budget rather than of the generation barrier.

**The floor is about eight members, and it is a search floor rather than the
mutation floor.** N=4 satisfies `draws + 1` for `best1`, which is all the
constructor requires, and all but stalls: given two and a half times the budget
it improves by under 1% on all eight (dimension, function) cells, where N=8
improves by 10% to 100% on the same cells. Lifting the barrier partly unsticks
it, which says the stall is a collapsed population that a generation of
feedback latency cannot re-spread. Premature convergence is the failure a
default has to avoid, and it sits at N=4, not at N=8.

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

**The caveats these numbers carry.** They are four analytic test functions at
two, four and eight parameters, scored on the median over 12 seeds. A lab
landscape is none of those things: it is noisy, its cost is a measurement, and
its shape is not known in advance. Nothing here says eight is right for a
particular apparatus — it says eight is where a default belongs, and that the
number to change when a run converges early is this one.

This is also the shape the literature uses. Storn and Price's control parameter
is NP, the number of population vectors, given directly, with NP between five
and ten times the parameter count offered as a rule of thumb for *choosing* it
rather than as the parameter's own form.

## The two sets of rows this harness cannot produce

**The former walk.** `benchmarks/results/barrier_4d_former.csv` holds the
learner as it shipped before slice 4, whose walk counted usable records rather
than reading a role off a position, with its `restart_tolerance` on and off. That code was
deleted, and no copy of it is kept here: a verbatim copy of deleted code that
nothing imports is the shape this repository already declines for the M-LOOP
checkout, and git holds it better than a directory does. To run those arms
again, take the file out of git —

```
git show 842c0f6:labscript_optimization/learners/differential_evolution.py
```

— and hand the class to a `Session` the way `de_pipeline.run` hands it the
asynchronous reference. Its constructor takes the same arguments; its three
relative imports need making absolute if it is saved outside the package.

**The withdrawn stopping rule.** `benchmarks/results/truncated_4d.csv` is the
`barrier` sweep run with the defect deliberately reapplied, which is what identifies the
harness rather than the learner as the cause of the withdrawn figures. It is
not a mode of this harness, because a supported way to run the arms on
different numbers of shots is a way to make the mistake again. It is one line:

```diff
-    while not session.stopped:
+    while len(session.proposals) < budget:
```

Apply it, run `run barrier --out benchmarks/results/truncated_4d.csv`, and
revert it. That edit is also the mutation that proves
`tests/test_benchmarks.py`: under it the smoke test fails with
`assert 9 == 12`.
