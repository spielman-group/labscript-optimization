# labscript-optimization

Machine-learning online optimisation of
[labscript suite](https://github.com/labscript-suite) experiments.

A lyse routine proposes shots, runmanager runs them, and the costs come back
through lyse. It replaces M-LOOP and its lyse plugin
[analysislib-mloop](https://github.com/rpanderson/analysislib-mloop), taking
the algorithms worth keeping and the shape of their TOML configuration, and
depends only on numpy, scipy and scikit-learn. It does not read an
analysislib-mloop file as it stands: see
[Coming from M-LOOP](#coming-from-m-loop).

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

A run is one runmanager sequence. The session's first submission starts it,
and every later one names it and joins it.

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
already in the box — it is several. lyse names them in `lyse.paths`, and
each invocation hands over every one, so a batch costs the optimiser nothing.

## Using it

Add a routine to lyse containing:

```python
import labscript_optimization.routine as optimisation

optimisation.optimise('optimisation_config.toml')
```

Adding the routine starts the session; removing it, restarting it, or reaching
the run budget stops it. Progress is saved onto each shot the session proposed,
under the results group `labscript_optimization`, so it comes back as
dataframe columns: `df[('labscript_optimization', 'best_cost')]` is the best
cost so far, beside the parameters and the shot that produced it, and why the
session stopped. `phase` is the shot's own: what proposed that shot, which is
the source the learner gave it when it proposed it, or `start` for the
configured start, however far the run has moved on since. A value the session
does not have yet is empty — `NaN` for the best cost, an empty string or list
for the rest. Shots that are not the optimiser's own — yours, and
runmanager's default shots — are left alone: runmanager mints a shot id for
every queue row it compiles, so carrying one does not make a shot the
optimiser's, and the session writes onto an id it proposed and no other.

The session's own counters are one answer for the whole run rather than
anything about a shot, so they are not written onto every shot of it.
`optimise` returns the whole status, which a routine that wants them prints:

```python
print(optimisation.optimise('optimisation_config.toml'))
```

`num_buffered_runs` is how many of the session's shots to keep queued, and
defaults to two. BLACS asks for its next shot as soon as it finishes the last,
which is before this optimiser has seen the cost and proposed a replacement —
so a queue holding only one of our shots is empty at precisely that moment,
and runmanager hands BLACS a default shot instead. At one buffered run roughly
every second shot is a default one. The status counts a `starved` for each
time the routine found nothing of its own queued; raise `num_buffered_runs` if
it keeps climbing. The random learners keep exactly that many in flight, so
they refuse zero. For `gaussian_process` it is the number of explorer shots
queued behind each batch, when that is more than `explore_runs`, and zero is
allowed: see [The Gaussian process's cycle](#the-gaussian-processs-cycle).

A learner that proposes whole generations sets its own depth and does not take
that setting — `differential_evolution` refuses it, because the depth is the
population size and a second setting for the same number is one that can
disagree with it. Such a learner empties the queue once per generation, by
design, so nothing counts a `starved` there: a number that fires every
generation says nothing about the run.

Those default shots are also what keeps the routine running while the optimiser
waits. They go to BLACS already compiled, so runmanager never writes a shot id
into them, and the routine passes them over.

`gaussian_process` waits the same way for its batch: the next goes out when
every shot of the last has come back or been given up on. A shot is given up
on when the routine asks runmanager about it, which happens only when a shot
reaches lyse, so a batch whose shots were all deleted is released only by a
shot arriving — an explorer shot still queued, or a default shot. With nothing
of the run's left queued and runmanager sending nothing in its place, the
apparatus idles and the cycle waits with it.

You compute the cost yourself, in your own lyse routine, into the column named
by `cost_key`. A shot whose cost is `NaN` is recorded as a bad observation
rather than being waited on.

That column has to be there by the time this routine runs. lyse runs multishot
routines in list order, so the cost has to come from a singleshot routine, or
from a multishot routine above this one in the list. A column that is not there
yet is not waited for either: the shot is recorded as a bad observation, and so
is every other shot of the run, with nothing anywhere saying why.

See [`examples/config_example.toml`](examples/config_example.toml) for the
configuration. Every key must be one this package knows, and a spelling it does
not stops the load with a message naming it, rather than being accepted and
ignored — a setting nothing reads is one a lab believes is in force when it is
not.

`[GENERAL]` carries the session's own settings — which learner runs, how deep
the queue is, what stops the run. A learner's knobs go in `[LEARNER.<name>]`,
the table of the learner that takes them, and a knob found in `[GENERAL]` is
refused with the table it belongs in named. `gaussian_process` runs an
explorer beside itself, so two learners run whenever it is the one you name,
and a knob both take is written twice, once in each table, and each gets its
own value.

A setting in `[GENERAL]` or in a learner's table is held to the kind it takes:
a quoted `"false"`, a fraction where a whole number belongs, or one number
where a pair belongs is refused, naming the setting, rather than converted into
something nobody wrote.

`[PARAMETERS.<group>.<name>]` is one optimised parameter and
`[RUNMANAGER_GLOBALS.<group>.<name>]` is a runmanager global computed from
several of them; `[ANALYSIS] groups` says which groups take part. A group
defined and not listed there is switched off. A group listed and defined by no
table is refused, naming it: it is a misspelling, and the group it was meant
for would otherwise sit switched off with nothing said.

A parameter's `start` is where the run begins. The session proposes that point
first and once, whichever learner is running: no learner has an opening of its
own, and every one of them meets the start in the history as an ordinary
observation. Its shot reads `start` in the `phase` column, because no learner
proposed it. It is one point over every searched parameter, so write it on all
of them or on none — a start on some is refused, naming which have one and
which do not. A parameter switched off is not searched and needs none.

Knowing a key is not the same as acting on it: a table written for a learner
this file does not build goes unread, so the settings for several learners can
sit side by side and be switched between by changing `[GENERAL] learner`.

## Learners

| Learner | What it does |
| --- | --- |
| `random` | Uniform draws. The reference the others are measured against. |
| `directed_random` | Draws near a previously seen point, chosen from a band of middling costs rather than from the best one, so it explores rather than refines. |
| `differential_evolution` | Evolves a population, one whole generation at a time: it proposes `population_size` shots together and nothing more until all of them have been answered for. Good on rough landscapes with no useful gradient. `population_size` is how many members it holds — around eight searches well and a budget over a thousand shots is worth sixteen, measured over four analytic test functions at two to eight parameters (`benchmarks/README.md`, "What the sweep found", has the tables, the caveats and the harness that produced them) — and it is the queue depth too, so `num_buffered_runs` is not accepted beside it. |
| `gaussian_process` | Fits a Gaussian process and searches its posterior, in batches, with an explorer's shots queued behind each. It warms up on the explorer alone, and `explorer` — `random` or `directed_random`, defaulting to `directed_random` — is its own knob; see below. |

### The Gaussian process's cycle

A Gaussian process has nothing to say until the history holds a spread of
points, and fitting it is slow. It runs its own cycle around both, set in
`[LEARNER.gaussian_process]`:

| Knob | Default | What it does |
| --- | --- | --- |
| `explorer` | `directed_random` | The learner that proposes the warmup and the shots behind each batch, on the knobs of its own `[LEARNER.<name>]` table. `random` or `directed_random`: `differential_evolution` proposes only whole generations and `gaussian_process` only once warmed up, so neither can keep a warmup topped up shot by shot or fill a buffer on demand. |
| `warmup_observations` | max(5, 2 × parameters) | How many usable observations the explorer gathers before the Gaussian process proposes. |
| `batch_size` | 4 | How many points the Gaussian process proposes at a time, each conditioned on the ones before it. |
| `explore_runs` | 1 | How many explorer shots follow each batch at the least. |

**Warmup.** Below `warmup_observations`, the explorer alone keeps
max(`explore_runs`, `num_buffered_runs`, 1) shots queued. It counts
observations a fit can use, not shots: a shot whose cost is `NaN`, or one that
was given up on, does not move it on. Warmup ends at the count, and the
explorer shots already queued at that moment still run, which is the design:
nothing takes a queued shot back, and their costs join the fit when they land.

**The cycle.** After warmup the Gaussian process proposes a batch, and behind
it max(`explore_runs`, `num_buffered_runs`) explorer shots. The next batch
goes out when every shot of this one has come back or been given up on; the
explorer shots never hold it up. The kernel's hyperparameters are refit once
per batch, and the exploration schedule `uncer_bias` is walked from its first
weight at every batch — with the defaults, one pass of 0, 1, 2, 3 across the
four points, opening greedily. The batch does not condition on explorer shots
in flight. A `max_num_runs` with room for less than a whole cycle cuts the
explorer shots first and the batch last.

| `explore_runs` | `num_buffered_runs` | explorer shots behind each batch |
| --- | --- | --- |
| 2 | 0, 1 or 2 | 2 |
| 2 | 5 | 5 |

Computing a batch is slow, so the next cannot go out the moment the last comes
back; the explorer shots queued behind it are what keep the apparatus busy
through that fit, which is what `num_buffered_runs` asks for. Raising
`num_buffered_runs` to cover a slow fit also raises the share of explorer
shots. `explore_runs` is the exploring the Gaussian process does regardless,
and the buffer only ever adds to it. Both at zero is a Gaussian process with
nothing behind its batches: it idles the apparatus during every fit, and
`starved` counts each time.

Each shot says which part of the cycle proposed it in the `phase` column:
`warmup` for the explorer's shots during warmup, `main` for the Gaussian
process's own, `explore` for the explorer's shots behind a batch, and `start`
for the configured start, which takes the first warmup shot's place.

### A cost with noise in it

Shot-to-shot noise is not something these learners quietly absorb. It changes
what the number a run reports means, and it changes which learner to run.

**The `best_cost` a run reports is an extreme-value draw, not an estimate of
the cost at `best_params`.** A run keeps the best cost it happened to measure,
and over a few hundred noisy shots the best one measured is the luckiest one.
At a multiplicative noise of 20% on Rastrigin, a learner reported 1.45 for a
point whose noiseless cost is 3.89 — about a −3σ draw, three standard
deviations being the 3.89 × (1 − 0.6) = 1.56 next to it. The same on Ian's apparatus at 10% noise:
a reported best of 186,531 atoms where those parameters measure 155,290, a
+2.0σ draw. The honest number is a re-measurement at `best_params`, which
costs one shot. Nothing in the reported column is that number.

**`differential_evolution` assumes the cost is deterministic.** Its selection
is elitist on a single draw of each member, so a member that measured lucky is
one nothing honest can displace and the population stalls around it. Measured
at four parameters, a population of eight, 600 shots and 16 seeds, with a
multiplicative noise `f · (1 + σz)`: the true cost at the reported best roughly
doubles on Rosenbrock between σ = 0 and σ = 10–20%, and barely moves on
Rastrigin. It bites where progress depends on chaining small improvements,
which is what Rosenbrock's valley is, and hardly at all where the landscape is
coarse enough for its structure to survive the noise.

**A lab whose cost carries noise should run `gaussian_process` with
`cost_has_noise = true`**, which is its default and is the one setting here
that models exactly this. The white-noise term is what lets the fit attribute
scatter to the measurement rather than to structure, and the posterior mean it
proposes against is an average over the shots near a point rather than any one
of them.

Those figures understate the effect for most apparatus costs, and are worth
reading for their direction rather than their size. The test functions have
their optimum at zero, where a multiplicative noise vanishes — so the region
the search spends its budget in is the quietest part of the landscape. A cost
that is the peak of a positive quantity has the opposite shape: an atom number
carries its largest absolute noise exactly at the top, where the search ends
up.

### Writing your own

A learner is a function from the proposal history and a hint to whatever its
method allows it to propose now:

```python
def propose(
    self, history: Sequence[Observation], hint: int
) -> list[tuple[np.ndarray, str]]: ...
```

It is handed the whole history every time, in the order the proposals were
made, and holding every one of them: a shot still running and a shot that will
never report are both in there as a position spent without a usable cost. That
is what keeps the bookkeeping for shots in flight out of the algorithms. The
shots in flight are the history's pending records, and the hint is
`num_buffered_runs`, how many of them to keep. A learner honours it as far as
its method allows — the random learners keep exactly that many in flight,
`differential_evolution` does not read it, proposing a whole generation when
none of the last is pending and nothing otherwise, and `gaussian_process`
queues it as explorer shots behind each batch — and the session submits what
comes back, in order, cut from the end to what `max_num_runs` has room for. So
a learner can be used on its own against any cost function:

```python
import numpy as np
from labscript_optimization.learners import DifferentialEvolutionLearner
from labscript_optimization.observations import Observation
from labscript_optimization.space import Parameter, ParameterSpace

space = ParameterSpace([Parameter('x', -5.0, 5.0)])
learner = DifferentialEvolutionLearner(space, np.random.default_rng())

history = []
for step in range(100):
    for params, source in learner.propose(history, hint=4):
        history.append(
            Observation(str(len(history)), params, float(params[0] ** 2))
        )
```

The learners here also answer `ask(history, k)`: exactly `k` points from the
same method, with no pacing and no sources.

Each proposal comes back beside its source, a string naming which of the
learner's ways of proposing made it. The session records it when it submits
the shot, and it is what that shot reads in the `phase` column; a learner with
one way of proposing calls it `main`. Nothing supplies one on a learner's
behalf, so a proposal returned without one is refused rather than recorded as
the main learner's. A learner also carries a `generation`, which is `None`
unless it proposes only whole groups of a fixed size and only when none of the
last is pending. The session holds no barrier for it — the barrier is the
learner's own — but leaves the starved refills of such a learner uncounted,
and loading a configuration reads it to refuse a budget under two generations
and a `num_buffered_runs` beside one. Those two members are `Learner`, which
everything here inherits. Your own object is driven by those same two members
whether or not it inherits anything.

A learner that wraps another owes an answer for every attribute something
outside a learner reads off it — `generation`, which the session and the
configuration read. Three answers are honest: answer for itself, where the
wrapper's own value is the true one; pass the wrapped learner's on, where the
wrapper can hold what that value promises; or refuse to be built, where it
cannot. Leaving one unanswered is none of the three — the reader's own default
becomes the answer, and the wrapped learner's declaration is dropped with
nothing said. The Gaussian process, which runs its explorer inside its own
cycle, answers for itself: it declares no generation, because it proposes as
many as its cycle calls for and holds the barrier on its own batch, and so its
starved refills are counted.

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

runmanager refusing to add shots to the run's sequence stops the session
without raising. It remembers a sequence only until it restarts, so this is a
runmanager restarted mid-run. Its reason is the session's `stopped`, and the
routine prints it on every invocation from then on, while lyse goes on
analysing.

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

An existing `mloop_config` file needs its two M-LOOP tables renamed and a
handful of settings cut or renamed before it will load, and the two tag globals
the old plugin needed can be deleted from runmanager.
[UPGRADING.md](UPGRADING.md) is the step-by-step.

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
suite installation already provides. It reads the shots lyse names in
`lyse.paths`, each with its `shot_id`, and submits with `submit_shots`'s
`sequence`, so it needs a lyse and a runmanager that have them; an older one
fails at the first use and says so.

## Tests

```
python -m pytest tests/ -q
```

The tests run the learners against analytic cost functions and the session
against a stand-in for runmanager; none of them needs a lab, a GUI or a
running suite.

## Benchmarks

[`benchmarks/`](benchmarks/) holds the harness behind the measured figures
quoted here, the rows it produced, and a README carrying the tables, their
caveats, and the command and commit behind each one. It runs its test
functions noiseless, so the figures under *A cost with noise in it* are not
among the ones it reproduces; that measurement states its own conditions
where it is quoted. A benchmark measures
search quality, which is a number that moves; the tests prove behaviour. The
suite runs one second of the harness, to keep it from rotting.

## Licence

MIT. Carries code from M-LOOP (MIT, Michael Hush) and analysislib-mloop
(BSD 3-clause); both notices are in [LICENSE](LICENSE).
