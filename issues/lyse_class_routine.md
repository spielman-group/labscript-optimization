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
- The configuration is read when the routine starts, and again at each Reset,
  so an edit to the file takes effect at the next Reset.
- `OptimizationRoutine` sets `lyse.Routine`'s `icon` class attribute to the
  absolute path of the package's `optimizer.svg`, and lyse shows that icon on
  the window.
- It is a multishot routine. Added to the singleshot routines, where lyse
  sets `self.paths` to `None`, every `run()` raises, saying so.

## Structure

- **`worker.py`'s `Worker`** owns the session thread and holds everything
  that does not need lyse:
  - constructed with a `load` callable, the window, the command queue and an
    interface factory, it starts the session thread and queues the opening,
    which is a `reset`. `load()` returns the configuration and the text of the
    file it was read from. Every `reset`, the opening included, calls it, sets
    the worker's `config` and shows both in the window, and `run()` reads costs
    by that `config`;
  - `hand_over(filepaths, observations, save)` queues one request carrying a
    `concurrent.futures.Future` and waits up to `REPLY_TIMEOUT` (2 s) for it.
    It calls `save(filepath, status)` for every shot the session took, among
    these and any earlier hand-over whose reply has arrived since. It raises
    an error the session thread reported, as the original exception, and
    raises if the thread itself has stopped, since nothing would answer;
  - the window's buttons put `start`, `pause`, `reset`, `set_best` and
    `restore` on its queue, and `start` carries whether the Start from
    runmanager values box was ticked at the click. The runmanager light puts
    `link` on it, carrying whether runmanager answers, for its first answer and
    for each change. The thread handles `link` before anything else, repaints
    nothing for it, and assumes runmanager answers until told;
  - `quit()` asks the thread to stop, and does not wait for it.
- **`routine.py`'s `OptimizationRoutine`** is the lyse side alone: it builds the
  window, reads shots out of lyse and saves their status columns into it. The
  window builds labscript-utils' `LinkIndicator` from runmanager's address,
  which it takes from a `RunmanagerClient`. The indicator says hello to
  runmanager every 2 s through a client of its own, and its answers gate
  Start and are passed to the worker; the routine starts it and shuts it down
  in `close()`.
  The window's Configuration tab has an Edit in text editor button above the
  configuration text, and `edit_config_action`, an action with the same text and
  icon, in the text box's right-click menu after Copy and Select All. Both call
  labscript-utils' `open_in_editor` on the absolute path of the configuration
  file, which opens it in the labconfig's `[programs] text_editor` and does not
  wait. The window is given that path by the routine, which works it out once
  in `__init__`.

## Threads

| Work | Thread |
| --- | --- |
| Reading the configuration at the routine's start, building the window, `close()`, button slots | GUI main thread |
| Reading the configuration at a Reset, recording costs, reconciling, proposing, submitting | Session thread |
| Asking runmanager whether it answers | The indicator's thread |
| Reading the analyzed shots and saving their status columns | lyse's analysis thread, in `run()` |
| A Gaussian-process batch | The learner's background thread |

The session is touched by the session thread alone. It handles each request,
replying before any runmanager round trip, then hands the window a snapshot,
which the window applies on the GUI thread. A finished Gaussian-process batch
puts `refresh` on the queue.

## `__init__`

1. Check that `config_path` is set. An unset `config_path` raises. Work out
   the configuration file's absolute path from it and the folder of the lab's
   subclass, which the window and the reading of the file both use.
2. Load the window's controls with `self.load_ui(...)`, given the absolute path
   of the package's `window.ui`, a `QWidget` form holding the buttons, the
   Status and Configuration tabs and the plot area. The pyqtgraph plot is
   inserted into the plot area. The routine creates no matplotlib
   figures.
3. Register the Start from runmanager values box with `self.saved_widgets(...)`,
   so lyse restores its state when the routine starts and saves it with the
   routine's other settings.
4. Build the `Worker`, with the routine's method that reads `config_path`
   relative to the routine folder and returns the configuration and its text,
   and the `interface_factory` class attribute, which is `RunmanagerInterface`.
   The worker loads the file at once, so a file that does not load raises.
   lyse then shows no window, prints the error in its own output box, and
   reports it again at every analysis until the routine is restarted. The
   session thread loads the file again at the opening, and shows its text in
   the Configuration tab.
5. Start the window's `LinkIndicator`.

`__init__` makes no runmanager round trip, so the window appears at once. The
session thread builds the session, and the window shows "Opening" until it
has. Anything printed during `__init__` goes to lyse's output box; after
construction, output goes to the window's Output dock.

## `run()`

lyse sets `self.path` and `self.paths` before each `run()`.

1. Raise if `self.paths` is `None`: the routine is a singleshot one.
2. Read the rows for `self.paths` with `lyse.data(where={"filepath": ...})`,
   and extract the shot ids and costs, by the worker's current configuration. An empty pass, from Run multishot with
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

- **Opening:** the session thread loads the configuration, shows it in the
  window, and builds the session, paused, without contacting runmanager. If
  that fails, the window shows why, the traceback is printed, and the routine
  stays up.
- **The runmanager indicator** is a row in the window's top bar, right of the
  buttons: runmanager's icon, then the name `runmanager` and a status light,
  with its status under them: `Checking...`, `Responding` or `Not
  responding`. The light is an hourglass before the first answer, a tick while
  runmanager answers and an exclamation mark while it does not, and its tooltip
  names the host and gives the reason. Nothing is printed for runmanager not
  answering. **Start** is enabled only while the session can start and the
  light shows a tick, and enables itself when the tick appears. While the light
  shows runmanager not answering, the worker does not reconcile at a hand-over,
  which would wait out the client's timeout with Reset and Pause queued behind
  it; the hand-over still refills, so a submission that times out pauses the
  session as below.
- **Start** checks that runmanager's globals evaluate and, when the Start from
  runmanager values box was ticked and the session has proposed nothing, reads
  the start from runmanager. Only once every check has passed does it pin the
  labscript file, record the original values of the globals not recorded yet,
  and begin submitting.
  If a check raises, the session stays paused with the message as its pause
  reason, shown as `Paused: <reason>`, and nothing is printed or raised, pinned
  or recorded; Start again after fixing it.
  **Pause** stops new submissions while shots in flight still report.
  **Reset** reads the configuration file again and builds a new paused session
  from it, so it is also how a failed opening is retried. It leaves
  runmanager's values alone, and the worker keeps the original values across
  it.
- **A file that does not load at a Reset** is the user's mistake, not a bug.
  The load is reading the file and `config_module.loads`, and an `OSError` or
  `ValueError` from it ends the Reset: the phase reads `Ended: <message>`, no
  traceback is printed, there is no session, and only Reset is enabled. The
  window goes on showing the configuration it last loaded, and the worker
  keeps that `config` and the original values, so the next Reset after the fix
  builds a session as usual. A Start, Pause, Set best or Restore click queued
  behind that Reset is ignored, so the file's error stays in the window. Any
  other failure of a Reset is handled as an opening that fails.
- **The original values** are the raw Default expression strings, in
  runmanager, of every global a session's configuration has set. The worker
  alone holds them. `check_ready` reads them with one `get_values(raw=True)`
  and returns them until a Start has gone, for the globals of the session's
  configuration; the worker records those it has not recorded yet once every
  check of that Start has passed, and never overwrites one it has. So a refused
  Start records nothing, and a later Start does not record again. The worker's
  dict outlasts the session, so Pause and Reset keep them, a global the new
  configuration no longer sets stays recorded and is restored, and one the new
  configuration adds is recorded at its first Start. A recorded global is
  recorded again only when the routine restarts. They are runmanager's values
  from before the optimizer first ran. The worker gives the window `restorable`,
  whether any are recorded, beside the session's status; the session does not
  own them, so `Session.status()` and the saved results do not carry it.
- **Start from runmanager values** is a box in the window's second row. The
  `start` command carries its state at the click. At a Start that has no
  proposals yet, a ticked box makes the worker set `Session.start_point` to
  `RunmanagerInterface.get_start()`, which replaces the configuration's
  `start` as the point the first refill proposes under the source `start`;
  nothing else about proposing the start changes. `get_start()` reads the
  evaluated values from `get_values()`, takes each enabled parameter from the
  global that takes it directly (the `GlobalMapping` with no `expr` and that
  parameter as its only argument) and returns the vector in parameter order. It
  raises `RuntimeError` for a parameter with no such global, a value that is not
  a real number (a boolean is not one), or a value outside the parameter's
  `min` and `max`, and the worker pauses the session with the message, as it
  does for `check_ready`. A resuming Start does not read again, so the box is
  always enabled. `Session.status()` reports `start_point` as `start`, a list
  or `None`, and the parameters table's Start column shows it once a status
  carries it. A session starts with the configuration's start as its
  `start_point`.
- **Set best values** and **Restore original values** put `set_best` and
  `restore` on the queue. Both are enabled while the session is paused or has
  ended and the light shows a tick. Set best also needs a best cost, and
  Restore needs the original values recorded, which `restorable` says, so it
  works after a Reset. `set_best` calls
  `RunmanagerInterface.set_values(best.params)`, which writes what submitting
  those parameters would leave in runmanager's window; `restore` calls
  `set_values(original, raw=True, skip_missing=True)` with the worker's
  originals, which writes them back as written, so `2*pi*5` comes back as
  written. A global in no active group is skipped and the rest written;
  `set_values` returns the names skipped, and Restore prints one line,
  `Restored runmanager's values, except those in no active group: <names>`,
  and nothing when none is skipped.
- **A write runmanager does not take** is one line in the Output dock, `Could
  not set runmanager's values: <reason>`, with `runmanager is not answering`
  for a `TimeoutError`. No traceback is printed, and the session is left as it
  was: not paused or stopped by the failure.
- **runmanager not answering during a run:** a `TimeoutError` from any request
  to runmanager pauses the session with the reason `runmanager is not
  answering`. No traceback is printed, nothing is queued as a failure, and the
  shots in flight are kept. Start resumes the same run once the light shows a
  tick. A runmanager that was restarted no longer knows the run's sequence,
  so the next submission after Start is refused and the session ends with
  runmanager's reason.
- **Before a session exists,** or after a Reset whose file did not load, a
  hand-over is answered with no shots taken, since none can be the session's,
  and a window command other than Reset is ignored.
- **A session ends,** at a limit or on an error. The reason is shown in the
  window and printed once to the Output dock. The first time the session thread
  sees the session stopped, it also sets the best values in runmanager, if
  there is a best. It engages and submits no shot.

## Window

The window is lyse's routine window, titled with the routine folder's name. lyse
shows it once construction finishes if the routine is checked, which an added
routine is, and restores its geometry. Closing it
hides it, and lyse's **Show windows** brings it back. The routine never shows
or raises the window itself, so a closed window stays closed while shots
arrive; lyse's spec rules this out for every routine.

## `close()`

lyse calls `close()` on the GUI thread once the last `run()` has returned. It
shuts down the indicator and calls `Worker.quit()`, neither of which waits: the
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
    or printing, and Start resumes it;
  - while the light says runmanager does not answer, a hand-over does not ask
    runmanager what became of the shots and the session runs on, and once it
    answers again the hand-over does;
  - a Start with the box ticked makes runmanager's values the first proposal,
    under the source `start`, a refusal leaves the session paused with the
    reason, and a Start that resumes the run does not read again;
  - a session that reaches a limit sets its best values once, `set_best` and
    `restore` call the interface on a paused session, and a write that fails
    prints one line and no traceback and leaves the session as it was, and a
    `restore` that skips a global in no active group names it in one line;
  - a Start that is refused records and pins nothing, and a later Start that
    goes records what runmanager holds then, which `restore` writes;
  - after a Reset, `restore` writes the originals from before the first run,
    and the new session's first Start records only a global the file adds;
  - a Reset after the file changed builds the session from the new
    configuration and shows it, and a Reset whose file does not load shows
    `Ended: <message>`, prints nothing, takes no shots, keeps the originals and
    ignores a Start, Pause, Set best or Restore click queued behind it, and a
    further Reset after the fix opens a session.
- **The interface,** against a fake client: `check_ready` returns the
  configured globals' raw values until `pin_labscript_file` has run and `None`
  after, a labscript file changed after a `check_ready` that was not followed by
  a pin is not refused, a later pin does not move the file,
  `set_values(params)` writes `globals_for(params)` unraw,
  `set_values(originals, raw=True)` writes the dict raw, and with
  `skip_missing=True` it returns the names runmanager skipped. `get_start()`
  returns the direct globals' values in parameter order, and refuses a value
  outside the bounds, a value that is not a real number, and a parameter with
  no direct global.
- **The window,** with an indicator that is never started: it enables Start
  only when the session can start and runmanager answers, and Set best values
  and Restore original values only while the session is paused or has ended,
  runmanager answers, and there is a best cost or `restorable` respectively.
  Shots submitted do not enable Restore, and a Reset session that has submitted
  none does. It puts `link` on the queue for the indicator's first answer,
  whichever it is, and for each change. The Start click carries the Start from
  runmanager values box's state, and the parameters table's Start column shows
  the status's start.
  labscript-utils tests the indicator.
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
  - Start from runmanager values opens the run at runmanager's values of the
    parameters, refuses a value outside the bounds with its reason, and lyse
    remembers the box when the routine restarts;
  - Set best values and Restore original values change runmanager's Default
    values on a paused and on an ended session, and Restore brings back an
    expression such as `2*pi*5` as written, and after a Reset still brings
    back the values from before the first run;
  - a session that reaches a limit leaves runmanager showing its best values,
    and runmanager runs them at once if it is running default shots;
  - editing the configuration and pressing Reset shows the new Method,
    parameters and text and runs the new session, a broken file reads
    `Ended: <message>` with nothing printed, and fixing it and pressing Reset
    opens a session;
  - restarting and removing the routine end it cleanly.
