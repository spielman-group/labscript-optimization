"""The optimization session's thread, and the queue everything reaches it by.

The plugin hands over each shot lyse has analyzed and returns at once, so the
fitting and the runmanager traffic happen here, on a thread of the plugin's own,
and the reply is saved when it comes. The thread is purely reactive: it proposes
only in response to a hand-over or a window command.
"""

import threading
import traceback
from concurrent.futures import Future

from .runmanager_interface import RunmanagerInterface
from .session import Session


def _set_runmanager_values(session, original, best, output):
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
                output(
                    f"Restored runmanager's values, except those in no active "
                    f"group: {', '.join(skipped)}\n"
                )
    except Exception as exc:
        reason = (
            "runmanager is not answering"
            if isinstance(exc, TimeoutError)
            else str(exc) or type(exc).__name__
        )
        output(f"Could not set runmanager's values: {reason}\n", red=True)


class Worker:
    """The optimization session, on a thread of its own."""

    def __init__(
        self, load, window, command_queue, output, interface_factory=RunmanagerInterface
    ):
        # Returns the configuration and the text of its file, and is called at
        # every Reset, the opening included; a file that does not load raises
        # OSError or ValueError.
        self.load = load
        # None until a Reset has loaded the file.
        self.config = None
        self.window = window
        # Carries the window's commands and the thread's own requests.
        self.command_queue = command_queue
        # Takes a message and whether it is an error, as lyse's output box does.
        self.output = output
        self.interface_factory = interface_factory
        self.thread = threading.Thread(target=self._run_session, daemon=True)
        self.thread.start()
        self.command_queue.put(("reset", None, None))

    def hand_over(self, observations):
        """Hand over a shot's observations, without waiting for the session.

        The future gives ``(recorded, status)``: each observation's source, or
        ``None`` for one the session did not take, and the session's status.
        """
        if not self.thread.is_alive():
            raise RuntimeError(
                "The optimization session's thread has stopped, so nothing will "
                "answer. Restart lyse."
            )
        reply = Future()
        command = "observe" if observations else "shot"
        self.command_queue.put((command, reply, tuple(observations)))
        return reply

    def quit(self):
        """Ask the session thread to stop, without waiting for it."""
        self.command_queue.put(("quit", None, None))

    def _run_session(self):
        session = None
        # The raw Default expressions of every global a session has set, as
        # runmanager held them before the first Start that set it. They outlast
        # the session and a Reset whose file does not load.
        original = {}
        reported = None
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
                        session, original, command == "set_best", self.output
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
                self.output(traceback.format_exc(), red=True)
                if reply is not None and not reply.done():
                    reply.set_exception(exc)
            finally:
                if session is not None:
                    ended = session.stopped and reported is not session
                    if ended:
                        self.output(
                            f"The optimization has stopped: {session.stopped}\n"
                        )
                        reported = session
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
                        _set_runmanager_values(session, original, True, self.output)
                elif error is not None:
                    self.window.update({"stopped": error}, False, (), False, False)
