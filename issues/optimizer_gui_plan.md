# Optimizer window: plan

**Status: planned, not started.** This branch holds only this plan. Every
decision below is Ian's and final. Build it in the slices at the end, after
reading "Before you start".

## Before you start
- **Bring the branch up to date first.** `OptimizerGUI` was cut from
  `Development` at `d1dc586`, before the ServerInterfaces effort merged. Once
  Ian has merged `ServerInterfaces` into `Development`, merge `Development`
  into this branch before writing code. From then on this branch uses
  `runmanager.client`, not `runmanager.remote`. Ask Ian if that merge has not
  happened.
- **Ian's working rules for this repository:**
  - Work on this branch only, and keep it checked out until it merges. Do
    not push or merge without Ian's go.
  - Read and follow two skills, found under the workspace's
    `.claude/skills/`:
    - `labscript-narrow-patch`: declare a patch budget per slice;
    - `labscript-style`: rules 8 and 9 for Qt, rule 13 for tests, and rule
      14's licence banner on new files, from `IDIOMS.md` §1.
  - Ian expects small patches. The package's existing code is verbose, and is
    not a style model.
  - Tests: essential behaviour only, through the public surface, with no
    message-shape tests. Show that each test fails against a mutation of what
    it covers. Run the suite from inside the repository with
    `~/miniforge3/envs/labscript/bin/python -m pytest tests -q`.
  - Work in lyse, runmanager or blacs belongs to those repositories' own
    sessions. This plan needs none.
- **How the code is shaped today:**
  - `routine.py`: `optimise()` is the lyse multishot entry point. Its first
    call starts the worker with `start_worker()`, and every call hands over
    the shots lyse names and waits up to `REPLY_TIMEOUT` = 2 s for a
    numbered reply.
  - `worker.py`: `Worker` is a zprocess `Process`. zprocess calls
    `Worker.run()` in the child's **main thread**
    (zprocess/process_class_wrapper.py). So the Qt event loop must take over
    that thread, and the message loop must move to a thread of its own.
    - `run()` answers each request, then reconciles and refills.
    - On ServerInterfaces, `Session` is imported inside the configure
      branch, on purpose: importing runmanager connects to zlock, which must
      not happen while zprocess is still unpickling `Worker`.
  - `session.py`: `Session` holds the history, the learner and the stop
    reason.
    - `status()` is the dict a snapshot starts from: `submitted`,
      `completed`, `awaiting`, `dropped`, `blocked`, `starved`, `best_cost`,
      `best_params`, `best_shot_id` and `stopped`.
    - `history` gives every proposal with its `source`: `start`, `warmup`,
      `main` or `explore`.
  - `learners/gaussian_process.py`: `GaussianProcessLearner.computation` is
    the `Future` of the batch being computed, or None. It is the "batch
    computing" state, and where Q2's done-callback attaches.
- **Packaging.** `pyproject.toml` has `include-package-data = true` with
  setuptools_scm, so a committed `window.ui` ships with the package.
  pyqtgraph and qtutils are added to its dependencies.
- **Rendering tests.** `tests/conftest.py` does not yet set
  `QT_QPA_PLATFORM`. Add
  `os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')` there, as
  `blacs/tests/conftest.py` does, before any test imports Qt.
- **Trying it live** needs Ian: lyse's **Run multishot** button calls the
  routine once with no shot, which should open the window, paused.

The problem seen in the lab: the user cannot see what the optimizer is doing,
and cannot start, pause or reset it except by running a shot through lyse or
restarting lyse's analysis subprocess.

The answer is a small Qt window, opened by the lyse routine, that shows the
optimizer's state and a plot of cost against shot, with stopwatch controls:
**Start**, **Pause** and **Reset**. This is option (1) of two. It is built so
that option (2), a standalone labscript application with its port in the
labconfig, later means moving where the process is launched, not rewriting the
window.

## Decisions (Ian, 2026-09-27)

- **D1: the window lives in the worker process,** not in lyse's analysis
  subprocess.
  - The worker holds the real state: the history, the learner, whether a
    batch is computing, and the stop reason.
  - The routine blocks the analysis subprocess's Qt main thread for up to
    `REPLY_TIMEOUT` = 2 s on each pass.
  - The controls must act without a lyse pass.
- **D2: opening a session.** The routine's first invocation starts the worker
  and opens its window. A shot reaching lyse triggers it, and so does lyse's
  **Run multishot** button, which runs every multishot routine even when no
  shot is waiting (lyse/filebox.py:800, :935).
  - Nothing of the user's runs when lyse merely loads the routine: its
    analysis subprocess executes the script only when asked to analyse
    (lyse/analysis_subprocess.py:351–450).
- **D3: the session opens paused, with zero shots.** Nothing is submitted
  until **Start**. This is a behaviour change: today the first invocation
  submits the first shots.
- **D4: the configuration still comes from the routine,**
  `optimise(config_path)`, as now. It is loaded once when the session opens;
  Start and Reset do not reread it. Restart the routine after editing the
  file.
- **D5: stopwatch controls.**
  - **Start** resumes submitting.
  - **Pause** stops submitting, and there is no separate Stop. Shots already
    in flight are still taken, and reconciling goes on.
  - **Reset** discards the session and its history and builds a new session,
    paused, from the configuration already loaded.
- **D6: a session a limit has ended is not paused.** Limits are
  `max_num_runs`, the patience limit, a refused join, or an error. The window
  shows the reason. Start is disabled, because the limit would end it again at
  once, and only Reset goes on. Internally, pausing is a flag that can be
  cleared, separate from `Session.stopped`, which stays final and keeps its
  first reason.

## Design

### Worker: one thread owns the session, and the window is a view
The session is touched by exactly one thread, as now. Requests from the
routine and commands from the window meet in one local queue:

- **The zprocess message loop's thread** reads `from_parent` and forwards
  each message into a local `queue.Queue`.
- **The window** puts its commands into the same queue: `start`, `pause`,
  `reset`.
- **A session thread** consumes that queue and does what `Worker.run` does
  today (configure, observe, shot, quit), plus the three commands. It replies
  to the routine exactly as now.
  - After each item it hands the window a snapshot through `inmain_later`:
    the status dict, the paused flag, whether a batch is computing, and the
    costs and sources so far. A batch finishing on the model's thread puts an
    item on the queue too (Q2), so that change is shown when it happens.
- **The Qt event loop runs in the worker's main thread.** The window never
  calls into the session, so a slow submit, a round trip to runmanager,
  never freezes it, and the session never waits on the GUI.

This follows labscript-style rule 8: background threads reach the GUI only
through `inmain`/`inmain_later`, with no QThread subclasses.

### Session
- **A `paused` flag,** True when a session is built.
  - `refill` returns [] while paused.
  - `reconcile` and `record` are unchanged, so in-flight shots are still
    taken.
  - `status()` gains `paused`.
- **`start()` and `pause()`** set the flag. Start is refused, as a no-op
  that reports it, when `stopped` is set.
- **Start submits at once.** The session thread runs a refill right after
  `start`, as it does after a routine message, so shots go out without
  waiting for a lyse pass.

### Window: labscript-style rules 8 and 9
- **Built from a `.ui` file through `UiLoader`,** with a thin window class and
  a plain controller class holding the logic.
- **Qt comes through qtutils**, and the window uses `qtutils.icons` where
  suite windows use icons.
- **It shows:**
  - the phase: paused, warmup, batch computing, batch out, or ended with its
    reason;
  - the counters: submitted, completed, awaiting, dropped, blocked, starved;
  - the best cost and its parameters.
- **A plot of cost against shot order,** with points coloured by source
  (`start`, `warmup`, `main`, `explore`) and a best-so-far line. Maximised
  costs are shown in the lab's own sign, as `best_cost` is. It uses
  pyqtgraph, which runviewer already uses.
- **The controls:** Start, Pause and Reset, enabled according to the state.

### Routine
- **Unchanged calling convention:** `optimise(config_path)`.
- **Its first invocation** starts the worker as now. The worker now opens the
  window and waits paused.
- **Every later request** also shows the window again if it was closed, which
  hides it (Q1). The worker does this when it takes the routine's message; the
  routine sends nothing new.
- **The plot's costs are the session's,** which are sign-flipped under
  `maximize`. The snapshot flips them back, as `Session.status()` does for
  `best_cost`.
- **Stopping the worker** is unchanged: the routine's restart, or lyse's
  quit, stops it, and the window goes with it.

### Optional import
Qt, qtutils and pyqtgraph are imported only in the worker and window modules.
The learners, config and session remain usable on their own, and in tests,
without Qt.

### New files
- `labscript_optimization/window.py`: the window and its controller. It opens
  with the suite banner (labscript-style rule 14).
- `labscript_optimization/window.ui`: the layout, loaded with qtutils'
  `UiLoader`.

## Tests (labscript-style rule 13)
- **Session:** a new session is paused and submits nothing. `start` submits
  up to `num_buffered_runs` at once. `pause` stops submission while costs are
  still recorded. Start after a limit has ended the session submits nothing.
- **Worker, driven headless as now, against fakes:**
  - a `start` command submits without a routine message;
  - `reset` yields a fresh paused session with the loaded configuration;
  - routine requests are answered exactly as before.
- **Window, rendered offscreen:**
  - the snapshot sets the phase and counters shown;
  - each button puts its command on the queue;
  - Start is disabled once the session has ended.
  - The conftest sets `QT_QPA_PLATFORM` to offscreen, per AGENTS.md "Tests
    Must Not Invoke the Application".

Each new test is shown to fail against a mutation of what it covers.

## Docs
- **README:**
  - "Using it": the session opens paused, and Start begins it. Lyse's Run
    multishot opens the window without a shot. Describe the controls and the
    plot.
  - "Failures": a limit-ended session is shown in the window.
- **UPGRADING:** the behaviour change, a paused start.
- **Ian's `optimization_multishot.py`** docstring says adding the routine
  starts a session. That file is outside the repository; tell Ian rather than
  editing it.

## Settled after the first draft (Ian, 2026-09-27)
- **Q1: closing the window hides it.** The session keeps running, and the
  routine's next pass shows the window again.
- **Q2: the window updates as any Qt window does,** when its values are set.
  The one value that changes off the session thread is "batch computing". A
  batch's future gets a done-callback that puts a request on the local queue,
  so the session thread sends the window a fresh snapshot then too. No timer.
- **Q3: the plot uses pyqtgraph,** which needs less code and redraws faster
  than an embedded matplotlib figure. It is already installed with the suite,
  since blacs devices need it, and it is added to pyproject.toml's
  dependencies.

## Slices, each a commit passing the suite with its own docs
1. **Session pause, and the worker's command queue, headless.** The session
   opens paused, and a `start` command submits. No Qt yet.
2. **The window:** state, counters and controls, with offscreen tests.
3. **The plot.**

Size to be declared per slice under labscript-narrow-patch. The window is new
code, so it is budgeted rather than minimised against existing paths.

## Path to option (2)
- Launch the worker process from a desktop entry instead of from the routine,
  with its port in the labconfig.
- The routine becomes a client that forwards its shots, following the
  ServerInterfaces convention: a `labscript_optimization.client` module and a
  server on labscript-utils' extended `ZMQServer`.
- The window, controller, session and command queue carry over unchanged.
