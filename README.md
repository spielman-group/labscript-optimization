# the _labscript suite_ » labscript-optimization

### Machine-learning online optimization of experiments

**labscript-optimization** tunes the parameters of an experiment controlled by the [*labscript suite*](https://github.com/labscript-suite/labscript-suite) while it runs. A [**lyse**](https://github.com/labscript-suite/lyse) routine proposes shots, [**runmanager**](https://github.com/labscript-suite/runmanager) runs them, and each shot's cost comes back through lyse to steer the next proposals. You compute the cost in your own lyse routine; this package never does.

It replaces [M-LOOP](https://github.com/michaelhush/M-LOOP) and its lyse plugin [analysislib-mloop](https://github.com/rpanderson/analysislib-mloop), keeping the algorithms worth keeping and the shape of their TOML configuration. It depends only on numpy, scipy and scikit-learn. An existing M-LOOP configuration needs changes before it loads: [UPGRADING.md](UPGRADING.md) lists them.

The optimizer comprises:

- Four learners, chosen in a TOML configuration file: `random`, `directed_random`, `differential_evolution` and `gaussian_process`;
- A window for each routine, with Start, Pause and Reset, a status tab and a plot of the costs;
- Shots matched to their costs by the id runmanager gives each queue row, so shots can return in any order and your own shots can share the queue;
- Progress saved as columns of lyse's dataframe, under `labscript_optimization`;
- A strict configuration: a key this package does not know, or a value of the wrong kind, stops the load with a message naming it.


## Installation

Install labscript-optimization into the Python environment that lyse runs in, alongside the rest of the *labscript suite*. The routine runs inside a lyse analysis subprocess, so an installation anywhere else is one lyse cannot import.

```
pip install "labscript-optimization[lyse]"
```

or, from a clone of this repository:

```
git clone https://github.com/spielman-group/labscript-optimization
pip install -e "./labscript-optimization[lyse]"
```

It needs Python 3.11 or newer. The `lyse` extra adds what the routine needs from the suite: `labscript_utils`, `lyse`, `runmanager`, `qtutils` and `pyqtgraph`. The learners alone, without the extra, need only numpy, scipy and scikit-learn.

The routine needs a lyse and a runmanager that have class routines (`lyse.Routine`), `lyse.data(where=...)`, `save_result`'s `save_to_h5`, and `submit_shots`'s `sequence` and `sequence_index`. An older one fails at the first pass and says so.


## Documentation

The documentation is the Sphinx source in [`docs/`](docs/source). Build it with

```
pip install -e ".[docs]"
sphinx-build -b dirhtml docs/source docs/build
```

and open `docs/build/index.html`.

<!-- Link the hosted Read the Docs documentation here once its address is decided. -->


## Licence

MIT. Carries code from M-LOOP (MIT, Michael Hush) and analysislib-mloop
(BSD 3-clause); both notices are in [LICENSE](LICENSE).
