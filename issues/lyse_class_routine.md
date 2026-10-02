# The optimizer as a lyse class routine

The optimizer runs as a lyse GUI routine, as specified in lyse's
`issues/class_routines_2026_09_30.md` in the workspace repository. lyse gives
each GUI routine a worker process of its own, a window on the GUI thread and
an analysis thread for `run()`. The optimizer uses that process: the session
runs on a thread started by the routine, and the package starts no process of
its own.

This needs lyse's `ClassRoutines` work. Both repositories carry the effort on
their `ClassRoutines` branches, and this one builds against lyse's committed
branch only.

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
- It is a multishot routine. Added to the singleshot routines, where lyse
  passes `paths=None`, every `run()` raises, saying so.
- `optimize(config_path)`, the classic-script entry point, is removed.

## Structure

- **`worker.py`'s `Worker`** owns the session thread and its queue, and holds
  everything that does not need lyse. It is today's `Worker` without its
  zprocess `Process` base, pipes and Qt setup:
  - constructed with the configuration, the window and an interface factory,
    it starts the session thread and queues `configure`;
  - `hand_over(filepaths, observations)` queues one request carrying a
    `concurrent.futures.Future` and waits up to `REPLY_TIMEOUT` (2 s) for it.
    It returns `(filepath, status)` for every shot the session took, among
    these and any earlier hand-over whose reply has arrived since. It raises
    an error the session thread reported, as the original exception;
  - the window's buttons put `start`, `pause` and `reset` on its queue;
  - `quit()` asks the thread to stop, and does not wait for it.
- **`routine.py`'s `OptimizationRoutine`** is the lyse side alone: it builds the
  window, reads shots out of lyse and saves their status columns into it.

## Threads

| Work | Thread |
| --- | --- |
| Reading the configuration, building the window, `close()`, button slots | GUI main thread |
| Greeting runmanager, recording costs, reconciling, proposing, submitting | Session thread |
| Reading the analysed shots and saving their status columns | lyse's analysis thread, in `run()` |
| A Gaussian-process batch | The learner's background thread, as now |

The session is touched by the session thread alone. It handles each request
as `Worker._run_session` does today, then hands the window a snapshot through
`inmain_later`. A finished Gaussian-process batch puts `refresh` on the queue,
as now.

## `__init__`

1. Load `config_path`, keeping its text for the Configuration tab. A file that
   does not load raises. lyse then shows no window, prints the error in its
   own output box, and reports it again at every analysis until the routine is
   restarted.
2. Load the window's controls with `self.load_ui(...)`, given the absolute path
   of the package's `window.ui`, a `QWidget` form holding the buttons, the
   Status and Configuration tabs and the plot area. The pyqtgraph plot is
   inserted into the plot area as now. The routine creates no matplotlib
   figures.
3. Give the routine `optimizer.svg` as its icon, through lyse's support for a
   worker's custom icon, so that the window and the worker's Dock or taskbar
   entry carry it. Where lyse has no such support, the window alone carries
   the icon.
4. Build the `Worker`, with the `interface_factory` class attribute, which is
   `RunmanagerInterface`.

`__init__` makes no runmanager round trip, so the window appears at once. The
session thread greets runmanager and builds the session, and the window shows
"Opening" until it has. Anything printed during `__init__` goes to lyse's
output box; after construction, output goes to the window's Output dock.

## `run(path, paths)`

1. Raise if `paths` is `None`: the routine is a singleshot one.
2. Read the rows for `paths` with `lyse.data(where={"filepath": paths})`, and
   extract the shot ids and costs as now. An empty pass, from Run multishot
   with nothing analysed, has `paths=[]` and reads nothing.
3. Hand them to the worker, and save each `(filepath, status)` it returns with
   `save_status`, as now.

Status columns are saved only here. lyse resets a routine's results at the
start of each `run()` and replies with them at its end, so a save from any
other thread is lost. A save may name a shot from an earlier pass, for a late
reply: lyse updates every row the reply names, and warns about a shot whose
row has gone.

A failure in the work after a reply, such as reconciling or submitting, stops
the session as now. Its traceback goes to the Output dock, and the next
hand-over raises it, so lyse shows it as that analysis's error.

## Session lifecycle

- **Opening:** the session thread greets runmanager and builds the session,
  paused. If that fails, for example because runmanager is not running, the
  window shows why, the traceback goes to the Output dock, and the routine
  stays up.
- **Start, Pause, Reset** behave as now. Reset builds a new paused session
  from the configuration already loaded, greeting runmanager again, so it is
  also how a failed opening is retried.
- **Before a session exists,** a hand-over is answered with no shots taken,
  since none can be the session's yet.
- **A limit** ends the session as now. The reason is shown in the window and
  printed once to the Output dock.

## Window

The window is lyse's routine window, titled with the routine file's name. lyse
shows it once construction finishes, and restores its geometry. Closing it
hides it, and lyse's **Show windows** brings it back. The routine never shows
or raises the window itself, so a closed window stays closed while shots
arrive; lyse's spec rules this out for every routine.

## `close()`

lyse calls `close()` on the GUI thread once the last `run()` has returned. It
calls `Worker.quit()`, which does not wait: the session thread may be inside a
runmanager round trip of up to the client's timeout, and `close()` must not
block the GUI thread. The thread is a daemon, and the worker's exit ends it.

lyse terminates a worker 2 s after asking it to quit, so `close()` may never
run. Nothing depends on it: shots already queued in runmanager stay queued and
run as ordinary shots, as when a routine is killed today.

## What is removed

- From `worker.py`: the zprocess `Process` base, the pipes, `run()` with its
  `QApplication` setup, `_read_requests`, and the deferred `Session` import.
- From `routine.py`: `optimize`, `analysed`'s use of `lyse.paths`,
  `start_worker`, `configure_timeout`, `CONFIGURE_MARGIN`, `LIVENESS_POLL`,
  `CONFIGURE_REQUEST`, `_drain` with its numbered requests and pending map,
  `exited_within`, `_stop_worker`, `stop_worker`, the `atexit` hook and the use
  of `lyse.routine_storage`. The pending map's job moves into `Worker`.
- `CHECK_READY_REQUESTS`, which only sized the configure deadline.
  `GREETING_TIMEOUT` stays, so that an absent runmanager is named within
  seconds.
- The `(message, traceback)` error payload. Errors stay exceptions in one
  process.
- `OptimizerWindow`, whose close-to-hide lyse now provides.

`session.py`, the learners, `config.py` and `runmanager_interface.py`'s
submitting and reconciling are unchanged.

## Documentation and the demo

- The README's "Using it" and UPGRADING's lyse step show the class file. Its
  window opens when the routine is added, not on Run multishot.
- Ian's `optimization_multishot.py` in the userlib becomes the class file.

## Tests

lyse has no public test API, so the logic lives where it can be tested without
lyse. The session and learner tests stand, and `extract`'s tests stay.

- **`Worker`, in process,** with runmanager faked, which is the one thing a
  test cannot run, and a stand-in window:
  - a cost handed over is recorded, and its shot comes back with its status;
  - a reply later than the deadline comes back from the next hand-over;
  - an error in handling a request is raised by the hand-over;
  - a failed opening leaves the worker up, and Reset retries it.
- **The routine, end to end,** through lyse's real worker subprocess, as lyse's
  own `GuiWorkerTests` drive it: a routine file whose subclass sets
  `interface_factory` to the fake. It constructs, an empty multishot pass
  replies `done`, and a singleshot pass replies `error`. The test removes the
  settings file lyse writes for the routine.

The worker-process tests in `test_worker.py`, and the `optimize` and worker
plumbing tests in `test_routine.py`, go with the code they test.

## Build order

1. **The routine**, in one commit, since the old and new entry points cannot
   stand side by side. It starts once lyse has committed its `GuiWorker` and
   `GuiWorkerTests` on `ClassRoutines`.
   - `Worker` as a plain class, with `hand_over` and futures;
   - `OptimizationRoutine`, with `__init__`, `run` and `close`;
   - `window.ui` with a `QWidget` root, and `WindowController` built on the
     widget `load_ui` returns;
   - the removals listed above;
   - the README, UPGRADING, the package docstrings and `pyproject.toml`'s
     `lyse` extra comment describing the class file;
   - the tests above. The suite passes against lyse's `ClassRoutines`.

   Budget: about 150 lines of code added against about 450 removed, and about
   150 lines of tests against about 1300 removed.
2. **The demo and a live trial.** Ian's `optimization_multishot.py` in the
   userlib becomes the class file. Ian runs it in lyse against runmanager:
   - the window opens when the routine is added;
   - Start submits, and the analysed shots get their status columns;
   - an opening with runmanager absent shows why, and Reset recovers once
     runmanager is up;
   - restarting and removing the routine end it cleanly.
