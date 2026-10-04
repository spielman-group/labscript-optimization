# The optimizer as a lyse class routine

The optimizer runs as a lyse GUI routine, as specified in lyse's
`issues/class_routines_2026_09_30.md` in the workspace repository. lyse gives
each GUI routine a worker process of its own, a window on the GUI thread and
an analysis thread for `run()`. The optimizer uses that process: the session
runs on a thread started by the routine, and the package starts no process of
its own.

This needs lyse's GUI routines, which are on lyse's `Development` branch.

## The lab's routine folder

A lab adds a routine folder, named with a `.lyse` suffix, to lyse's multishot
routines, holding its configuration and a `lyse_routine.py`:

```
optimization.lyse/
    lyse_routine.py
    optimization_config.toml
```

```python
from labscript_optimization.routine import OptimizationRoutine


class Optimization(OptimizationRoutine):
    config_path = "optimization_config.toml"
```

- `OptimizationRoutine` subclasses `lyse.Routine`. lyse counts only classes
  defined in `lyse_routine.py`, so the lab's subclass is the folder's one
  routine.
- The folder's name is the routine's name in lyse and its window's title.
- `config_path` is read relative to the routine folder, which is the worker's
  working directory.
- The configuration is read once, when the routine starts. Restart the routine
  after editing it.
- `OptimizationRoutine` sets `lyse.Routine`'s `icon` class attribute to the
  absolute path of the package's `optimizer.svg`, and lyse shows that icon on
  the window.
- It is a multishot routine. Added to the singleshot routines, where lyse
  sets `self.paths` to `None`, every `run()` raises, saying so.

## Structure

- **`worker.py`'s `Worker`** owns the session thread and holds everything
  that does not need lyse:
  - constructed with the configuration, the window, the command queue and an
    interface factory, it starts the session thread and queues the opening,
    which is a `reset`;
  - `hand_over(filepaths, observations, save)` queues one request carrying a
    `concurrent.futures.Future` and waits up to `REPLY_TIMEOUT` (2 s) for it.
    It calls `save(filepath, status)` for every shot the session took, among
    these and any earlier hand-over whose reply has arrived since. It raises
    an error the session thread reported, as the original exception, and
    raises if the thread itself has stopped, since nothing would answer;
  - the window's buttons put `start`, `pause` and `reset` on its queue;
  - `quit()` asks the thread to stop, and does not wait for it.
- **`routine.py`'s `OptimizationRoutine`** is the lyse side alone: it builds the
  window, reads shots out of lyse and saves their status columns into it. It
  also runs a `RunmanagerStatusMonitor`, from `runmanager_interface.py`, which
  asks runmanager every `POLL_INTERVAL` (2 s) whether it answers, waiting
  `POLL_TIMEOUT` (1 s), and reports each answer to the window's runmanager
  indicator.

## Threads

| Work | Thread |
| --- | --- |
| Reading the configuration, building the window, `close()`, button slots | GUI main thread |
| Recording costs, reconciling, proposing, submitting | Session thread |
| Asking runmanager whether it answers | The monitor's thread |
| Reading the analyzed shots and saving their status columns | lyse's analysis thread, in `run()` |
| A Gaussian-process batch | The learner's background thread |

The session is touched by the session thread alone. It handles each request,
replying before any runmanager round trip, then hands the window a snapshot,
which the window applies on the GUI thread. A finished Gaussian-process batch
puts `refresh` on the queue.

## `__init__`

1. Load `config_path`, and show its text in the Configuration tab. An unset
   `config_path`, or a file that does not load, raises. lyse then shows no
   window, prints the error in its own output box, and reports it again at
   every analysis until the routine is restarted.
2. Load the window's controls with `self.load_ui(...)`, given the absolute path
   of the package's `window.ui`, a `QWidget` form holding the buttons, the
   Status and Configuration tabs and the plot area. The pyqtgraph plot is
   inserted into the plot area. The routine creates no matplotlib
   figures.
3. Build the `Worker`, with the `interface_factory` class attribute, which is
   `RunmanagerInterface`.
4. Start the `RunmanagerStatusMonitor`, reporting to the window.

`__init__` makes no runmanager round trip, so the window appears at once. The
session thread builds the session, and the window shows "Opening" until it
has. Anything printed during `__init__` goes to lyse's output box; after
construction, output goes to the window's Output dock.

## `run()`

lyse sets `self.path` and `self.paths` before each `run()`.

1. Raise if `self.paths` is `None`: the routine is a singleshot one.
2. Read the rows for `self.paths` with `lyse.data(where={"filepath": ...})`,
   and extract the shot ids and costs. An empty pass, from Run multishot with
   nothing analyzed, has `self.paths == []` and reads nothing.
3. Hand them to the worker, which saves each shot's status with
   `save_status`.

Status columns are saved only here. lyse resets a routine's results at the
start of each `run()` and replies with them at its end, so a save from any
other thread is lost. A save may name a shot from an earlier pass, for a late
reply: lyse updates every row the reply names, and warns about a shot whose
row has gone.

A failure in the work after a reply, such as reconciling or submitting, stops
the session. The next hand-over raises it, or this one if it has already
failed by its end, so lyse shows it as that analysis's error.

## Session lifecycle

- **Opening:** the session thread builds the session, paused, without
  contacting runmanager. If that fails, the window shows why, the traceback is
  printed, and the routine stays up.
- **The runmanager indicator** is a row in the window's top bar, right of the
  buttons: runmanager's icon, the label `runmanager` and a status light. The
  light is an hourglass before the first answer, a tick while runmanager
  answers and an exclamation mark while it does not, and its tooltip gives
  the reason. Nothing is printed for runmanager not answering. **Start** is
  enabled only while the session can start and the light shows a tick, and
  enables itself when the tick appears.
- **Start** checks that runmanager's globals evaluate, pins its labscript
  file the first time, and begins submitting. If the check raises, the session
  stays paused with the message as its pause reason, shown as `Paused:
  <reason>`, and nothing is printed or raised; Start again after fixing it.
  **Pause** stops new submissions while shots in flight still report.
  **Reset** builds a new paused session from the configuration already loaded,
  so it is also how a failed opening is retried.
- **runmanager not answering during a run:** a `TimeoutError` from any request
  to runmanager pauses the session with the reason `runmanager is not
  answering`. No traceback is printed, nothing is queued as a failure, and the
  shots in flight are kept. Start resumes the same run once the light shows a
  tick. A runmanager that was restarted no longer knows the run's sequence,
  so the next submission after Start is refused and the session ends with
  runmanager's reason.
- **Before a session exists,** a hand-over is answered with no shots taken,
  since none can be the session's yet.
- **A limit** ends the session. The reason is shown in the window and
  printed once to the Output dock.

## Window

The window is lyse's routine window, titled with the routine folder's name. lyse
shows it once construction finishes if the routine is checked, which an added
routine is, and restores its geometry. Closing it
hides it, and lyse's **Show windows** brings it back. The routine never shows
or raises the window itself, so a closed window stays closed while shots
arrive; lyse's spec rules this out for every routine.

## `close()`

lyse calls `close()` on the GUI thread once the last `run()` has returned. It
shuts down the monitor and calls `Worker.quit()`, neither of which waits: the
session thread may be inside a runmanager round trip of up to the client's
timeout, and `close()` must not block the GUI thread. The threads are daemons,
and the worker's exit ends them.

lyse terminates a worker 2 s after asking it to quit, so `close()` may never
run. Nothing depends on it: shots already queued in runmanager stay queued and
run as ordinary shots, as when any routine is killed.

## Documentation and the demo

- The docs' Getting started page and UPGRADING's lyse step show the routine folder.
  Its window opens when the routine is added, not on Run multishot.
- Ian's demo in the userlib's `example_apparatus` is the routine folder
  `optimization_multishot.lyse/`.

## Tests

lyse has no public test API, so the logic lives where it can be tested without
lyse. The session and learner tests stand, and `extract`'s tests stay.

- **`Worker`, in process,** with runmanager faked, which is the one thing a
  test cannot run, and a stand-in window:
  - a cost handed over is recorded, and its shot comes back with its status;
  - a reply later than the deadline comes back from the next hand-over;
  - an error in handling a request is raised by the hand-over;
  - the session opens paused without contacting runmanager;
  - a Start that runmanager refuses leaves the session paused with the reason,
    and a later Start goes;
  - runmanager not answering during a run pauses the session without raising
    or printing, and Start resumes it.
- **The monitor and the window,** with a fake client: the monitor reports
  whether runmanager answers, with the reason when it does not, and the window
  enables Start only when the session can start and the light shows a tick.
- **The routine, end to end,** through lyse's real worker subprocess, as lyse's
  own `GuiWorkerTests` drive it: a routine folder whose subclass sets
  `interface_factory` to the fake. It constructs, an empty multishot pass
  replies `done`, and a singleshot pass replies `error`. The test removes the
  settings file lyse writes for the routine.

## Remaining work

- **A live trial,** by Ian, of the demo in lyse against runmanager:
  - the window opens when the routine is added;
  - Start submits, and the analyzed shots get their status columns;
  - with runmanager absent the light shows an exclamation mark and Start is
    disabled, and Start enables itself once runmanager is up;
  - a runmanager that stops answering during a run, behind an open dialog,
    pauses the session, and Start resumes it once the dialog is closed;
  - restarting and removing the routine end it cleanly.
