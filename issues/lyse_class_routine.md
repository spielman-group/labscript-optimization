# The optimizer as a lyse class routine

The optimizer runs as a lyse GUI routine, as specified in lyse's
`issues/class_routines_2026_09_30.md` in the workspace repository. lyse gives
each GUI routine a worker process of its own, a window on the GUI thread and
an analysis thread for `run()`. The optimizer uses that process: the session
runs on a thread started by the routine, and the package starts no process of
its own.

This needs lyse's `ClassRoutines` work. Both repositories carry the effort on
their `ClassRoutines` branches.

## The lab's routine file

A lab adds a three-line class file to lyse's multishot routines:

```python
LYSE_MODE = "gui"
from labscript_optimization.routine import OptimizationRoutine

class Optimization(OptimizationRoutine):
    config_path = "mloop_config.toml"
```

- `OptimizationRoutine` subclasses `lyse.Routine`. lyse counts only classes
  defined in the file, so the lab's subclass is the file's one routine.
- `config_path` is read relative to the routine file's folder, which is the
  worker's working directory.
- The configuration is read once, when the routine starts. Restart the routine
  after editing it.
- `optimize(config_path)`, the classic-script entry point, is removed.

## Threads

| Work | Thread |
| --- | --- |
| Reading the configuration, building the window, `close()`, button slots | GUI main thread |
| Greeting runmanager, recording costs, reconciling, proposing, submitting | Session thread |
| Reading the analysed shots and saving their status columns | lyse's analysis thread, in `run()` |
| A Gaussian-process batch | The learner's background thread, as now |

The session is touched by the session thread alone. Everything else reaches it
by putting a request on one local `queue.Queue`:

- the buttons put `start`, `pause` and `reset`;
- `run()` puts `observe`, or `shot` when it has no costs to report;
- a finished Gaussian-process batch puts `refresh`, as now.

The session thread handles each request as `Worker._run_session` does today,
then hands the window a snapshot through `inmain_later`.

## `__init__`

1. Load `config_path`, keeping its text for the Configuration tab. A file that
   does not load raises. Until construction finishes, output goes to lyse's
   own output box, so the error appears there, and lyse fails later analyses
   until the routine is restarted.
2. Load the window's controls with `self.load_ui(...)`, given the absolute path
   of the package's `window.ui`, a `QWidget` form holding the buttons, the Status and
   Configuration tabs and the plot area. The pyqtgraph plot is inserted into
   the plot area as now. The routine creates no matplotlib figures.
3. Give the routine `optimizer.svg` as its icon, through lyse's support for a
   worker's custom icon, so that the window and the worker's Dock or taskbar
   entry carry it. lyse's spec does not have that support yet; where it is
   missing, the window alone carries the icon.
4. Start the session thread, and put `configure` on its queue.

`__init__` makes no runmanager round trip, so the window appears at once. The
session thread greets runmanager and builds the session; the window shows
"Opening" until it has.

## `run(path, paths)`

`run()` does what `optimize()` does in the routine today, in the same process
as the session:

1. Save the status of any earlier request whose reply has arrived since.
2. Read the rows for `paths` with `lyse.data(where={"filepath": paths})`, and
   extract the shot ids and costs as now.
3. Put one request on the queue, carrying a `concurrent.futures.Future`.
4. Wait up to `REPLY_TIMEOUT` (2 s) for that future. Save the status of each
   shot the session took, with `save_status` as now. A reply that misses the
   deadline is kept, and step 1 of a later `run()` saves it.
5. Raise any error the session thread reported since the last `run()`. lyse
   shows it as that analysis's error.

The future carries what the worker's numbered reply carries now: one verdict
per observation, and the session's status. An error in handling the request
is set on the future and raised by `run()` as the original exception.

A failure in the work after a reply, such as reconciling or submitting, stops
the session as now. Its traceback goes to the Output dock, and the next `run()`
raises it.

## Session lifecycle

- **Opening:** the session thread greets runmanager and builds the session,
  paused. If that fails, for example because runmanager is not running, the
  window shows why, the traceback goes to the Output dock, and the routine
  stays up.
- **Start, Pause, Reset** behave as now. Reset builds a new paused session
  from the configuration already loaded, greeting runmanager again, so it is
  also how a failed opening is retried.
- **Before a session exists,** `run()` hands nothing over, since no shot can be
  the session's yet.
- **A limit** ends the session as now. The reason is shown in the window and
  printed once to the Output dock.

## Window

The window is lyse's routine window, titled with the routine file's name. lyse
shows it once construction finishes, and restores its geometry. Closing it hides it, and lyse's **Show windows** brings
it back. The routine never shows or raises the window itself, so a closed
window stays closed while shots arrive; lyse's spec rules this out for every
routine.

## `close()`

`close()` asks the session thread to quit and does not wait for it. The thread
may be inside a runmanager round trip of up to the client's timeout, and
`close()` must not block the GUI thread. The thread is a daemon, and the
worker's exit ends it.

## What is removed

- `worker.py`, the `Worker` process, its pipes and its request reader.
- From `routine.py`: `optimize`, `start_worker`, `configure_timeout`,
  `CONFIGURE_MARGIN`, `LIVENESS_POLL`, `CONFIGURE_REQUEST`, the numbered
  requests and pending map in `_drain`, `exited_within`, `_stop_worker`,
  `stop_worker`, the `atexit` hook and the use of `lyse.routine_storage`.
- `CHECK_READY_REQUESTS`, which only sized the configure deadline.
  `GREETING_TIMEOUT` stays, so that an absent runmanager is named within
  seconds.
- The `(message, traceback)` error payload. Errors stay exceptions in one
  process.
- `OptimizerWindow`, whose close-to-hide lyse now provides, and the window's
  own `QApplication` setup.

`session.py`, the learners, `config.py` and `runmanager_interface.py`'s
submitting and reconciling are unchanged.

## Documentation and the demo

- The README's "Using it" and UPGRADING's lyse step show the class file. Its
  window opens when the routine is added, not on Run multishot.
- Ian's `optimization_multishot.py` in the userlib becomes the class file.

## Tests

The session and learner tests stand. The routine is tested through lyse's
real host for class routines, with runmanager faked, which is the one thing a
test cannot run. The behaviours covered are:

- a cost handed to `run()` is recorded, and the shot gets its status columns;
- a reply later than the deadline is saved by the next `run()`;
- an error in handling a request is raised by `run()`;
- a failed opening leaves the routine up, and Reset retries it.

`test_worker.py` and the worker parts of `test_routine.py` go with the code
they test.
