# Benchmarks

`de_pipeline.py` drives the differential evolution learner that ships through a
real `Session`, over analytic test functions, and writes one row per run. Every
measured figure Issue 2 of `codex_issues_proposal.md` quotes was produced by a
command below, from the rows in `results/`.

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

| table in Issue 2 | command | rows | learner at |
|---|---|---|---|
| The whole 4-D table | `run barrier --out benchmarks/results/barrier_4d.csv` then `report barrier benchmarks/results/barrier_4d.csv benchmarks/results/barrier_4d_former.csv` | `benchmarks/results/barrier_4d.csv`, `benchmarks/results/barrier_4d_former.csv` | `24e7d0a`; former arms `9dd3994` |
| What the barrier costs | as above | as above | as above |
| Rastrigin against the former walk | as above | as above | as above |
| Members against parameters | `run dimension --out benchmarks/results/dimension.csv` then `report dimension benchmarks/results/dimension.csv` | `benchmarks/results/dimension.csv` | `24e7d0a` |
| Which N takes each block | as above | as above | `24e7d0a` |
| The floor at four members | as above | as above | `24e7d0a` |
| The withdrawn stopping rule (Corrections) | `report truncation benchmarks/results/truncated_4d.csv benchmarks/results/barrier_4d.csv` | `benchmarks/results/truncated_4d.csv` | `24e7d0a`, plus the edit below |

Each file in `results/` repeats its date, commit and command in its own header.

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
