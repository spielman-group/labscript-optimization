# labscript-optimization

Machine-learning online optimisation of
[labscript suite](https://github.com/labscript-suite) experiments.

A lyse routine proposes shots, runmanager runs them, and the costs come back
through lyse. It replaces M-LOOP and its lyse plugin
[analysislib-mloop](https://github.com/rpanderson/analysislib-mloop), keeping
their configuration schema and the algorithms worth keeping, and depends only
on numpy, scipy and scikit-learn.

## How it works

One lyse routine, one worker process, one session.

```
lyse routine ──observation──▶ worker ──set_globals + engage──▶ runmanager
     ▲                          │                                   │
     │                          └── history, learner, stop rules    │
     └──────────── shot with its cost ◀────────────────────────────┘
```

Each proposal is stamped with a tag global. A cost is matched to the proposal
it answers by that tag, so shots can come back in any order, your own shots can
be mixed into the queue, and runmanager can mint default shots when the queue
runs dry — none of it needs to be accounted for. runmanager is never stopped or
waited on; a shot is finished when it reaches lyse.

The routine returns immediately. Fitting and submitting happen in the worker,
because lyse runs multishot routines inline and a slow one delays every shot
behind it.

## Using it

Add a routine to lyse containing:

```python
import labscript_optimization.routine as optimisation

optimisation.optimise('mloop_config.toml')
```

Adding the routine starts the session; removing it, restarting it, or reaching
the run budget stops it. Progress comes back as the routine's results: the best
cost so far, how many shots are in flight, and which phase the learner is in.

Your runmanager globals must include `mloop_session` and `mloop_iteration`,
which carry the tag, alongside the globals being optimised.

You compute the cost yourself, in your own lyse routine, into the column named
by `cost_key`. Writing `NaN` means "no cost yet", which is how a routine that
averages several repeats holds the optimiser until it has enough of them.

See [`examples/config_example.toml`](examples/config_example.toml) for the
configuration, which is the schema analysislib-mloop used: an existing
`mloop_config` file loads unchanged.

## Learners

| Learner | What it does |
| --- | --- |
| `random` | Uniform draws. The reference the others are measured against. |
| `directed_random` | Draws near a previously seen point, chosen from a band of middling costs rather than from the best one, so it explores rather than refines. |
| `differential_evolution` | Evolves a population. Good on rough landscapes with no useful gradient. |
| `gaussian_process` | Fits a Gaussian process and searches its posterior. Runs `directed_random` for its training shots first, and falls back to it for any proposal it cannot make. |

A learner is a function from the observation history to `k` proposals:

```python
def propose(self, history: Sequence[Observation], k: int) -> np.ndarray: ...
```

It is handed the whole history every time, in proposal order, containing only
shots that reported a cost. That is what keeps the bookkeeping for shots in
flight out of the algorithms, and it means a learner can be used on its own
against any cost function:

```python
import numpy as np
from labscript_optimization.learners import DifferentialEvolutionLearner
from labscript_optimization.observations import Observation
from labscript_optimization.space import Parameter, ParameterSpace

space = ParameterSpace([Parameter('x', 'g_x', -5.0, 5.0)])
learner = DifferentialEvolutionLearner(space, np.random.default_rng())

history = []
for step in range(100):
    for params in learner.propose(history, k=4):
        history.append(Observation(str(step), params, float(params[0] ** 2)))
```

## Failures

Failures stop the session and are reported through lyse's normal error path:
runmanager unreachable, a missing or broken global, or a learner raising. There
are no retries, timeouts or watchdogs. A shot whose cost never arrives simply
holds its slot.

If the worker dies, the history dies with it. There is no archive and no
persistence layer; a new session starts from nothing.

## Installing

```
pip install -e .
```

Needs Python 3.11 or newer, numpy, scipy and scikit-learn. The lyse routine
additionally needs `lyse`, `runmanager` and `labscript_utils`, which a labscript
suite installation already provides.

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
