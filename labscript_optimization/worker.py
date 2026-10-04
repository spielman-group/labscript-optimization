"""The optimization session's thread, and the queue everything reaches it by.

The lyse routine hands over the shots lyse has analyzed and returns within
:data:`REPLY_TIMEOUT`, so the fitting and the runmanager traffic happen here,
on a thread of the routine's own. The thread is purely reactive: it proposes
only in response to a request or a window command.
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


class Worker:
    """The optimization session, on a thread of its own.

    Parameters
    ----------
    config : Config
        The session configuration, loaded once.
    window : WindowController
        The view the thread updates after each request.
    command_queue : queue.Queue
        Where the window puts its commands, and the thread takes its requests.
    interface_factory : Callable
        What a configuration is turned into a runmanager interface by.
    """

    def __init__(
        self, config, window, command_queue, interface_factory=RunmanagerInterface
    ):
        self.config = config
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
        """Hand the session one pass's observations, and save what has come back.

        Waits up to :data:`REPLY_TIMEOUT` for this pass's reply. ``save(filepath,
        status)`` is called for each shot the session took, in this pass or in
        an earlier one whose reply has arrived since; ``status`` carries that
        shot's own ``phase``. The session taking a cost is what says the shot
        is one it proposed, since runmanager mints a shot id for every queue
        row it compiles.

        Raises
        ------
        RuntimeError
            If the session thread has stopped, so that nothing will answer.
        Exception
            What handling a request raised, or what failed in the work after an
            earlier reply. runmanager not answering is not raised: it pauses
            the session.
        """
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
        printed = None
        watched_future = None
        while True:
            command, reply, payload = self.command_queue.get()
            if command == "quit":
                return
            recorded = ()
            error = None
            try:
                if command == "reset":
                    # Opening is a reset too, so a failed one is retried by Reset.
                    session = None
                    # An empty status is the window's Opening, until the new
                    # session's own status replaces it.
                    self.window.update({}, False, (), False, False)
                    session = Session(self.config, self.interface_factory(self.config))
                elif command == "start":
                    try:
                        session.interface.check_ready()
                    except RuntimeError as exc:
                        # Start did not go: the session stays paused, with
                        # runmanager's reason in the window.
                        session.pause(str(exc))
                    else:
                        session.start()
                        session.refill()
                elif command == "pause":
                    session.pause()
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
                    if session.stopped and printed is not session:
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
                        session.status(),
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
                elif error is not None:
                    self.window.update({"stopped": error}, False, (), False, False)
