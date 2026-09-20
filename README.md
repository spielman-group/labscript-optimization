# labscript-optimization

Machine-learning online optimisation of
[labscript suite](https://github.com/labscript-suite) experiments.

A lyse routine proposes shots, runmanager runs them, and the costs come back
through lyse. It replaces M-LOOP and its lyse plugin
[analysislib-mloop](https://github.com/rpanderson/analysislib-mloop), taking
the algorithms worth keeping and the shape of their TOML configuration, and
depends only on numpy, scipy and scikit-learn. It does not read an
analysislib-mloop file as it stands: see [Using it](#using-it).

## How it works

One lyse routine, one worker process, one session.

```
lyse routine ──observation──▶ worker ──submit_shots──▶ runmanager
     ▲                          │                           │
     │                          └── history, learner,       │
     │                              stop rules              │
     └────────── shot with its cost ◀───────────────────────┘
```

Each shot is identified by the id runmanager mints for its queue row, written
into the shot file and read back by lyse as the `shot_id` column. A cost is
matched to the proposal it answers by that id, so shots can come back in any
order and your own shots can be mixed into the queue. runmanager is never
stopped or waited on.

There is no count of shots in flight. Whether a shot is still coming is
runmanager's answer, asked afresh each time the routine runs, so a shot that is
aborted, cancelled, or deleted stops being waited on instead of holding its
place for ever.

The routine returns immediately. Fitting and submitting happen in the worker,
because lyse runs multishot routines inline and a slow one delays every shot
behind it.

The worker replies before it fits and submits, so an answer can take longer
than the routine is willing to wait for it: a whole generation written through
runmanager is many seconds. Each message the routine sends carries a request
number and each message the worker sends carries the number of the request it
belongs to, so an answer that arrives late is still written onto the shots that
earned it and never onto whichever shots the routine is holding when it comes.

lyse runs a multishot routine once per drained batch of singleshot analyses
rather than once per shot. Where analysis keeps up that is one shot an
invocation; where it does not — a shot arriving while the one before it is
still being analysed, analysis paused and resumed, or lyse started with shots
already in the box — it is several. Each invocation hands over every shot
analysed since the one before it, so a batch costs the optimiser nothing.

## Using it

Add a routine to lyse containing:

```python
import labscript_optimization.routine as optimisation

optimisation.optimise('mloop_config.toml')
```

Adding the routine starts the session; removing it, restarting it, or reaching
the run budget stops it. Progress is saved onto each shot the session proposed,
under the results group `labscript_optimization`, so it comes back as
dataframe columns: `df[('labscript_optimization', 'best_cost')]` is the best
cost so far, beside the parameters and the shot that produced it, which phase
the learner is in, and why the session stopped. Anything the session does not
have yet reads as `NaN`. Shots that are not the optimiser's own — yours, and
runmanager's default shots — are left alone: runmanager mints a shot id for
every queue row it compiles, so carrying one does not make a shot the
optimiser's, and the session writes onto an id it proposed and no other.

The session's own counters are one answer for the whole run rather than
anything about a shot, so they are not written onto every shot of it.
`optimise` returns the whole status, which a routine that wants them prints:

```python
print(optimisation.optimise('mloop_config.toml'))
```

`num_buffered_runs` is usually set above one. BLACS asks for its next shot as
soon as it finishes the last, which is before this optimiser has seen the cost
and proposed a replacement — so a queue holding only one of our shots is empty
at precisely that moment, and runmanager hands BLACS a default shot instead. At
one buffered run roughly every second shot is a default one. The status counts
a `starved` for each time the routine found nothing of its own queued; raise
`num_buffered_runs` if it keeps climbing.

A learner that proposes whole generations sets its own depth and does not take
that setting — `differential_evolution` refuses it, because the depth is the
population size and a second setting for the same number is one that can
disagree with it. Such a learner empties the queue once per generation, by
design, so nothing counts a `starved` there: a number that fires every
generation says nothing about the run.

Those default shots are also what keeps the routine running while the optimiser
waits. They go to BLACS already compiled, so runmanager never writes a shot id
into them, and they deliberately never become the sequence anchor.

You compute the cost yourself, in your own lyse routine, into the column named
by `cost_key`. A shot whose cost is `NaN` is recorded as a bad observation
rather than being waited on.

That column has to be there by the time this routine runs. lyse runs multishot
routines in list order, so the cost has to come from a singleshot routine, or
from a multishot routine above this one in the list. A column that is not there
yet is not waited for either: the shot is recorded as a bad observation, and so
is every other shot of the run, with nothing anywhere saying why.

See [`examples/config_example.toml`](examples/config_example.toml) for the
configuration. It has the shape of an analysislib-mloop file, without the
settings that belong to M-LOOP itself and without that package's alternate
spellings: every key must be one this package knows, and a spelling it does
not stops the load with a message naming it, rather than being accepted and
ignored — a setting nothing reads is one a lab believes is in force when it is
not.

Knowing a key is not the same as acting on it. The learner knobs in `[MLOOP]`
cover every learner between them, and the learner you name is built with the
ones its own constructor takes, so a file can carry the others and go on
working when you switch learners. A `[LEARNER.<name>]` table names its learner,
so every key in it is held to that learner's own constructor; a table written
for a learner you did not name goes unread, exactly as the shared knobs that
learner does not take do.

An `mloop_config` file carried over therefore needs editing before it will
load. Between them, analysislib-mloop's own two example files need the whole
`[COMPILATION]` table deleted, along with `ignore_bad`, `no_delay`,
`visualisations` and the log settings `analysislib_console_log_level`,
`analysislib_file_log_level`, `console_log_level` and `console_log_string`;
`controller_type` is spelt `learner` here and has to be renamed. Parameter
bounds are `min` and `max`, and the long spellings `minimum` and `maximum` are
not read either.

## Learners

| Learner | What it does |
| --- | --- |
| `random` | Uniform draws. The reference the others are measured against. |
| `directed_random` | Draws near a previously seen point, chosen from a band of middling costs rather than from the best one, so it explores rather than refines. |
| `differential_evolution` | Evolves a population, one whole generation at a time: it proposes `population_size` shots together and is not asked again until all of them have been answered for. Good on rough landscapes with no useful gradient. `population_size` is how many members it holds — around eight searches well, and a budget over a thousand shots is worth sixteen — and it is the queue depth too, so `num_buffered_runs` is not accepted beside it. |
| `gaussian_process` | Fits a Gaussian process and searches its posterior. Runs `directed_random` for its training shots first, and falls back to it for any proposal it cannot make. |

A learner is a function from the proposal history to `k` proposals:

```python
def propose(self, history: Sequence[Observation], k: int) -> np.ndarray: ...
```

It is handed the whole history every time, in the order the proposals were
made, and holding every one of them: a shot still running and a shot that will
never report are both in there as a position spent without a usable cost. That
is what keeps the bookkeeping for shots in flight out of the algorithms, and it
means a learner can be used on its own against any cost function:

```python
import numpy as np
from labscript_optimization.learners import DifferentialEvolutionLearner
from labscript_optimization.observations import Observation
from labscript_optimization.space import Parameter, ParameterSpace

space = ParameterSpace([Parameter('x', -5.0, 5.0)])
learner = DifferentialEvolutionLearner(space, np.random.default_rng())

history = []
for step in range(100):
    for params in learner.propose(history, k=4):
        history.append(Observation(str(step), params, float(params[0] ** 2)))
```

### Writing your own

A learner answers `propose` and carries a `last_phase` string and a
`generation`, which is `None` unless the learner proposes only whole groups of
a fixed size and only when none of its proposals is outstanding. Those three
are `Learner`, which everything here inherits — the two-phase wrapper included, so
a session cannot tell a wrapped learner from a plain one. Your own object is
driven by those same three members whether or not it inherits anything.

A learner that wraps another owes an answer for every attribute something
outside a learner reads off it — `last_phase` and `generation`, which the
session reads, and `minimum_observations`, which a wrapper reads to decide
whether the handover it was configured for can happen. Three answers are
honest: answer for itself, where the wrapper's own value is the true one; pass
the wrapped learner's on, where the wrapper can hold what that value promises;
or refuse to be built, where it cannot. Leaving one unanswered is none of the
three — the reader's own default becomes the answer, and the wrapped learner's
declaration is dropped with nothing said. The two-phase wrapper answers
`last_phase` for itself, declares a `minimum_observations` of zero because its
trainer is the fallback for anything its main learner cannot make, and refuses
to wrap a learner declaring a `generation`, since a barrier cannot be held
across two phases only one of which promises it.

Those declarations are facts about an instance. Nothing outside a learner reads
one from a class, a signature, a registry entry or a count — it builds a
learner and asks that. Each of those stands in for the learner and agrees with
it only in the cases that exist on the day it is written: a class attribute
misses the learner that assigns in `__init__`, a signature default misses the
one that derives or clamps what it was given, and a count of proposals is not
the position one was made at. So declare however suits your learner, including
from a value its constructor was handed. Loading a configuration builds the
learner it names for this reason, and nothing is fitted at construction: the
queue depth and the budget are settled against what the object says.

Being named in a configuration asks for more. Inherit `ParameterSpaceLearner`
and add the class to the `LEARNERS` table, both from
`labscript_optimization.learners`: every entry there is built as
`cls(space, rng, **options)`, with the options matched by name against the
signature — so `space` and `rng` come first, in that order, and each of the
learner's own knobs is a keyword argument with a default.

## Failures

Failures stop the session and are reported through lyse's normal error path:
runmanager unreachable, a broken global, a labscript file changed underneath a
running session, or a learner raising. Nothing is retried, and no shot, fit or
submission is put on a clock. The exception is a status that cannot be written
to its shot, which is printed and passed over: a progress report is worth less
than the optimisation that stopping for it would end.

Two waits are bounded, and neither of them is the experiment's. The routine
waits a couple of seconds for the worker's answer, so lyse is held up for that
long and no longer whatever the worker is doing; a worker slower than that is
left to catch up, and its answer lands on the shots that earned it when it
comes. Starting the worker is allowed longer, because a cold runmanager
connection is slower than a shot cycle. A worker whose process has gone is
neither of those, and the routine says so within about a second rather than
reporting it as slow.

One failure stops the optimisation and cannot be reported from here: a shot
that fails to compile stays at the head of runmanager's queue until someone
deletes it, and nothing behind it runs. Nothing then reaches lyse, so the
routine is never called again. The row is red in runmanager and the queue has
visibly halted, which is where that failure belongs — the apparatus cannot
proceed, and an optimiser that stops is behaving correctly.

A cost can arrive after the generation it belonged to, and that is an ordinary
operator path rather than a failure. The shots behind such a row are given up
on, so the search moves on without them; delete the row and they run, and their
costs come back for a generation that has already been replaced. Each of them
is still taken and competes for its own slot and no other, which is what
differential evolution would have done with it had it arrived in time. While
the queue catches up, two generations of the optimiser's shots sit in it and
`awaiting` reads high. Neither that nor the `dropped` those shots were counted
in is a fault to chase.

If the worker dies, the history dies with it. There is no archive and no
persistence layer; a new session starts from nothing.

## Coming from M-LOOP

An existing `mloop_config` file needs a handful of settings cut and one
renamed before it will load, and the two tag globals the old plugin needed can
be deleted from runmanager. [UPGRADING.md](UPGRADING.md) is the step-by-step.

## Installing

Install it into the environment lyse itself runs in, the same way the rest of
the suite is installed there:

```
pip install -e /path/to/labscript-optimization
```

The routine runs inside a lyse analysis subprocess, so an installation
anywhere else is one lyse cannot import.

Needs Python 3.11 or newer, numpy, scipy and scikit-learn. The lyse routine
additionally needs `lyse`, `runmanager` and `labscript_utils`, which a labscript
suite installation already provides. It reads lyse's `shot_id` column and asks
for a bounded number of recent shots rather than the whole dataframe, so it
needs a lyse that has both; an older one refuses the request and says so.

## Tests

```
python -m pytest tests/ -q
```

The tests run the learners against analytic cost functions and the session
against a stand-in for runmanager; none of them needs a lab, a GUI or a
running suite.

## Licence

MIT. Carries code from M-LOOP (MIT, Michael Hush) and analysislib-mloop
(BSD 3-clause); both notices are in [LICENSE](LICENSE).
