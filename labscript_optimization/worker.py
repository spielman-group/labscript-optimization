"""The persistent optimization process.

Spawned once by the lyse routine and kept until the routine is restarted. The
fitting and the runmanager traffic happen here, so that the routine can hand
over the shots lyse has analysed and return at once.

The worker is purely reactive: it proposes only in response to a request or a
local command. If the routine's process dies the worker goes with it.
"""

import queue
import threading
import traceback
from pathlib import Path

from zprocess import Process

from . import config as config_module
from .runmanager_interface import RunmanagerInterface


class Worker(Process):
    """The optimization session, in a process of its own.

    zprocess enters the child through a wrapper module of its own, imports
    this module by name and calls :meth:`run` with the pipes to the routine
    already attached as ``from_parent`` and ``to_parent``. They and the
    interface are attributes rather than arguments, so the message loop can be
    driven against fakes without a process or a runmanager.

    Args:
        interface_factory: What a configuration is turned into a runmanager
            interface by. The rest are :class:`zprocess.Process`'s own.
    """

    def __init__(self, *args, interface_factory=RunmanagerInterface, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface_factory = interface_factory
        self.command_queue = queue.Queue()

    def run(self) -> None:
        """Run the window and handle messages until told to quit.

        A request is ``(command, number, payload)`` and every message sent
        back is ``(kind, number, payload)`` carrying the number of the request
        it belongs to. The routine numbers its requests and reads the first
        message carrying a request's number as the answer to it, which is what
        lets it tell a reply to the shots it is holding from a reply to the
        ones it handed over two invocations ago.

        A status payload is ``(recorded, status)``. ``recorded`` holds one
        verdict per observation the request carried, in the order it carried
        them: the source of that shot if the session took it, and ``None`` if
        it did not. Taking it is the only thing that says the shot the routine
        is holding is one of this session's -- runmanager mints a shot id for
        every queue row it compiles, so a user's own shots carry one too --
        and the source is what the routine writes onto that shot as its
        ``phase``. It travels with the verdict rather than in the status
        because it is the shot's own: one request can hand over shots that
        different learners proposed.

        Every request is answered with exactly one status, unless handling it
        raised, in which case the error is its reply. The routine waits on
        that: a request answered with nothing would leave it waiting out its
        deadline on a worker that is alive and well. An error payload is
        ``(message, traceback)``, as text, because not every exception keeps
        its message through pickling.

        The reply goes out before the reconciling, proposing and submitting
        that follow it, so the status the routine reads is one step behind:
        the shots this invocation drops and submits are counted in the next
        reply. A failure in that trailing work stops the session and sends an
        error under the number of the request it followed -- a second message
        for a request already answered, which the routine raises when it sees
        it, naming that request.
        """
        request = self.from_parent.get()
        if request[0] == "quit":
            return

        # Session imports runmanager, whose h5_lock connects to zlock. Wait
        # until the parent's first request proves zprocess has connected this
        # child, then import on the main thread before Qt takes it over.
        from .session import Session
        from labscript_utils.splash import get_qapplication
        from qtutils.qt import QtCore, QtGui

        from .window import WindowController

        application = get_qapplication([], "labscript optimizer")
        icon = QtGui.QIcon(str(Path(__file__).with_name("optimizer.svg")))
        application.setWindowIcon(icon)
        application.setQuitOnLastWindowClosed(False)
        window = WindowController(self.command_queue)
        window.ui.show()

        self.command_queue.put(request)
        reader = threading.Thread(target=self._read_requests, daemon=True)
        session = threading.Thread(
            target=self._run_session, args=(Session, window, application), daemon=True
        )
        reader.start()
        QtCore.QTimer.singleShot(0, session.start)
        application.exec()
        self.command_queue.put(("quit", None, None))
        if session.ident is not None:
            session.join()

    def _read_requests(self) -> None:
        while True:
            request = self.from_parent.get()
            self.command_queue.put(request)
            if request[0] == "quit":
                return

    def _run_session(self, session_factory, window, application) -> None:
        from qtutils import inmain_later

        session = None
        config = None
        watched_future = None
        try:
            while True:
                command, number, payload = self.command_queue.get()
                if command == "quit":
                    return
                recorded = ()
                error = None
                try:
                    if command in ("configure", "reset"):
                        if command == "configure":
                            # One read, so the window shows the text that was parsed.
                            text = Path(payload).read_text(encoding="utf-8")
                            replacement_config = config_module.loads(text)
                            window.show_config(replacement_config, text)
                        elif config is None:
                            raise RuntimeError("got reset before being configured")
                        else:
                            replacement_config = config
                        interface = self.interface_factory(replacement_config)
                        interface.check_ready()
                        replacement = session_factory(replacement_config, interface)
                        config, session = replacement_config, replacement
                    elif command == "start":
                        if session is None:
                            raise RuntimeError("got start before being configured")
                        session.start()
                        session.refill()
                    elif command == "pause":
                        if session is None:
                            raise RuntimeError("got pause before being configured")
                        session.pause()
                    elif command == "observe":
                        if session is None:
                            raise RuntimeError(
                                "got an observation before being configured"
                            )
                        recorded = tuple(
                            session.record(*observation) for observation in payload
                        )
                    elif command == "shot":
                        if session is None:
                            self.to_parent.put(("status", number, ((), {})))
                            continue
                    elif command == "refresh":
                        if payload is not session:
                            continue
                        session.refill()
                    else:
                        raise ValueError(f"unknown command {command!r}")

                    # A routine reply precedes runmanager round trips.
                    if number is not None:
                        self.to_parent.put(
                            ("status", number, (recorded, session.status()))
                        )

                    if command not in ("start", "pause", "refresh"):
                        session.reconcile()
                        session.refill()
                except Exception as exc:
                    # A local control has no routine waiting for an error.
                    error = str(exc) or type(exc).__name__
                    if session is not None:
                        session.stop(error if number is None else "stopped by an error")
                    if number is not None:
                        self.to_parent.put(
                            ("error", number, (error, traceback.format_exc()))
                        )
                    else:
                        traceback.print_exc()
                finally:
                    if session is not None:
                        computation = getattr(session.learner, "computation", None)
                        if (
                            computation is not None
                            and computation is not watched_future
                        ):
                            computation.add_done_callback(
                                lambda future, current=session: self.command_queue.put(
                                    ("refresh", None, current)
                                )
                            )
                            watched_future = computation
                        sign = -1 if config.maximize else 1
                        window.update(
                            session.status(),
                            computation is not None and not computation.done(),
                            tuple(
                                (
                                    observation.source,
                                    sign * observation.cost
                                    if observation.usable
                                    else None,
                                )
                                for observation in session.history
                            ),
                            config.learner == "gaussian_process",
                            config.maximize,
                        )
                    elif error is not None:
                        window.update({"stopped": error}, False, (), False, False)
                    if number is not None:
                        inmain_later(window.ui.show)
        finally:
            inmain_later(application.exit, 0)
