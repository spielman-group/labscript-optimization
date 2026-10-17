"""The optimization session's thread, and the queue everything reaches it by.

The lyse routine hands over the shots lyse has analyzed and returns within
:data:`REPLY_TIMEOUT`, so the fitting and the runmanager traffic happen here,
on a thread of the routine's own. A reply that comes later is saved by the next
hand-over. The thread is purely reactive: it proposes only in response to a
request or a window command.
"""

import queue
import threading
import traceback
from concurrent.futures import Future, wait

from .runmanager_interface import RunmanagerInterface
from .session import Session

#: Seconds a hand-over waits for the session's reply. Generous for an answer
#: that is a dictionary, and short against a shot cycle.
REPLY_TIMEOUT = 2.0


def _set_runmanager_values(session, original, best):
    """Set runmanager's values to the session's best, or restore the originals."""
    # A click queued behind a Reset can find nothing yet to set.
    if session.best is None if best else not original:
        return
    try:
        if best:
            session.interface.set_values(session.best.params)
        else:
            skipped = session.interface.set_values(
                original, raw=True, skip_missing=True
            )
            if skipped:
                print(
                    f"Restored runmanager's values, except those in no active "
                    f"group: {', '.join(skipped)}"
                )
    except Exception as exc:
        reason = (
            "runmanager is not answering"
            if isinstance(exc, TimeoutError)
            else str(exc) or type(exc).__name__
        )
        print(f"Could not set runmanager's values: {reason}")


class Worker:
    """The optimization session, on a thread of its own.

    ``load`` returns the configuration and the text of its file, and is called
    at every Reset, the opening included; a file that does not load raises
    ``OSError`` or ``ValueError``. ``window`` is the view the thread updates
    after each request. ``command_queue`` carries the window's commands and the
    thread's requests, and ``interface_factory`` turns a configuration into a
    runmanager interface.
    """

    def __init__(
        self, load, window, command_queue, interface_factory=RunmanagerInterface
    ):
        self.load = load
        # What a hand-over reads before the opening Reset has loaded the file.
        self.config = load()[0]
        self.window = window
        self.command_queue = command_queue
        self.interface_factory = interface_factory
        # Hand-overs whose status has not been saved yet, oldest first.
        self.unsaved = []
        # Failures in the work after a reply, for the next hand-over to raise.
        self.failures = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run_session, daemon=True)
        self.thread.start()
        self.command_queue.put(("reset", None, None))

    def hand_over(self, filepaths, observations, save):
        """Hand over one pass's observations and save the shots the session took."""
        reply = Future()
        command = "observe" if observations else "shot"
        self.command_queue.put((command, reply, tuple(observations)))
        self.unsaved.append((filepaths, reply))
        wait([reply], timeout=REPLY_TIMEOUT)
        for entry in [entry for entry in self.unsaved if entry[1].done()]:
            self.unsaved.remove(entry)
            filepaths, reply = entry
            recorded, status = reply.result()
            for filepath, source in zip(filepaths, recorded):
                # A source comes back only for a shot the session proposed and had
                # not yet recorded; runmanager mints an id for every row it compiles.
                if source is not None:
                    save(filepath, status | {"phase": source})
        if not self.failures.empty():
            raise self.failures.get()
        if not reply.done() and not self.thread.is_alive():
            # With the thread dead, nothing will complete these.
            self.unsaved = [entry for entry in self.unsaved if entry[1].done()]
            raise RuntimeError(
                "The optimization session's thread has stopped, so nothing will "
                "answer; its traceback is above. Restart the routine."
            )

    def quit(self):
        """Ask the session thread to stop, without waiting for it."""
        self.command_queue.put(("quit", None, None))

    def _run_session(self):
        session = None
        # The raw Default expressions of every global a session has set, as
        # runmanager held them before the first Start that set it. They outlast
        # the session and a Reset whose file does not load.
        original = {}
        printed = None
        watched_future = None
        # Whether runmanager answers, as the window's light last said.
        answering = True
        while True:
            command, reply, payload = self.command_queue.get()
            if command == "quit":
                return
            if command == "link":
                answering = payload
                continue
            if session is None and command in ("start", "pause", "set_best", "restore"):
                # A click queued behind a Reset whose file did not load. The
                # file's error is what the window shows, so it is left there.
                continue
            recorded = ()
            error = None
            try:
                if command == "reset":
                    # Opening is a reset too, so a failed one is retried by Reset.
                    session = None
                    try:
                        self.config, text = self.load()
                    except (OSError, ValueError) as exc:
                        # The file is the user's to fix, and Reset reads it again.
                        error = str(exc) or type(exc).__name__
                        continue
                    self.window.show_config(self.config, text)
                    # An empty status is the window's Opening, until the new
                    # session's own status replaces it.
                    self.window.update({}, False, (), False, False)
                    session = Session(self.config, self.interface_factory(self.config))
                elif command == "start":
                    try:
                        raw = session.interface.check_ready()
                        # The payload says whether the box was ticked; it
                        # applies to the run's first Start only.
                        start = None
                        if payload and not session.proposals:
                            start = session.interface.get_start()
                        session.interface.pin_labscript_file()
                    except RuntimeError as exc:
                        # Start did not go: the session stays paused, with
                        # runmanager's reason in the window, and nothing of
                        # this Start is kept.
                        session.pause(str(exc))
                    else:
                        # A global already recorded keeps what runmanager held
                        # before any run.
                        original = (raw or {}) | original
                        if start is not None:
                            session.start_point = start
                        session.start()
                        session.refill()
                elif command == "pause":
                    session.pause()
                elif command in ("set_best", "restore"):
                    _set_runmanager_values(
                        session, original, best=command == "set_best"
                    )
                elif command == "observe":
                    if session is None:
                        recorded = (None,) * len(payload)
                    else:
                        recorded = tuple(session.record(*o) for o in payload)
                elif command == "refresh":
                    if payload is not session:
                        continue
                    session.refill()
                elif command != "shot":
                    raise ValueError(f"unknown command {command!r}")

                # The reply precedes runmanager round trips.
                if reply is not None:
                    status = {} if session is None else session.status()
                    reply.set_result((recorded, status))

                if session is not None and command in ("observe", "shot"):
                    # A runmanager that is not answering would hold this up for
                    # its client's whole timeout.
                    if answering:
                        session.reconcile()
                    session.refill()
            except Exception as exc:
                if session is not None and isinstance(exc, TimeoutError):
                    # runmanager stopped answering. The run is kept, and Start
                    # resumes it once the window's light shows runmanager again.
                    session.pause("runmanager is not answering")
                    if reply is not None and not reply.done():
                        reply.set_result((recorded, session.status()))
                    continue
                error = str(exc) or type(exc).__name__
                if session is not None:
                    session.stop(error if reply is None else "stopped by an error")
                if reply is None:
                    # A window command has no routine waiting for its error.
                    traceback.print_exc()
                elif not reply.done():
                    reply.set_exception(exc)
                else:
                    # The next hand-over raises this, and may never come.
                    traceback.print_exc()
                    self.failures.put(exc)
            finally:
                if session is not None:
                    ended = session.stopped and printed is not session
                    if ended:
                        print(f"The optimization has stopped: {session.stopped}")
                        printed = session
                    computation = getattr(session.learner, "computation", None)
                    if computation is not None and computation is not watched_future:
                        computation.add_done_callback(
                            lambda future, current=session: self.command_queue.put(
                                ("refresh", None, current)
                            )
                        )
                        watched_future = computation
                    sign = -1 if self.config.maximize else 1
                    self.window.update(
                        # The originals outlast the session, which does not own them.
                        session.status() | {"restorable": bool(original)},
                        computation is not None and not computation.done(),
                        tuple(
                            (
                                observation.source,
                                sign * observation.cost if observation.usable else None,
                            )
                            for observation in session.history
                        ),
                        self.config.learner == "gaussian_process",
                        self.config.maximize,
                    )
                    if ended:
                        # After the window shows the end, so a runmanager that
                        # does not answer cannot hold it up. runmanager
                        # otherwise shows the last proposal made.
                        _set_runmanager_values(session, original, best=True)
                elif error is not None:
                    self.window.update({"stopped": error}, False, (), False, False)
