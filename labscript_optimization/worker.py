"""The persistent optimisation process.

Spawned once by the lyse routine and kept until the routine is restarted. The
fitting and the runmanager traffic happen here, so that the routine can hand
over one observation and return at once.

The worker is purely reactive: it proposes only in response to a message. If
the routine's process dies no more messages arrive, so the few seconds zprocess
takes to notice a dead parent cannot run away with the queue.
"""

import traceback

from zprocess import Process

from . import config as config_module
from .runmanager_interface import RunmanagerInterface
from .session import Session


class Worker(Process):
    """The optimisation session, in a process of its own.

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

    def run(self) -> None:
        """Handle messages until told to quit. The child's entry point.

        A reply is ``("status", (recorded, status))``. ``recorded`` is whether
        the session took the observation this message carried, and it is the
        only thing that says the shot the routine is holding is one of this
        session's: runmanager mints a shot id for every queue row it compiles,
        so a user's own shots carry one too.

        The reply goes out before the reconciling, proposing and submitting
        that follow it, so the status the routine reads is one step behind:
        the shots this invocation drops and submits are counted in the next
        reply. A failure in that trailing work still stops the session and
        still sends an error, but it may not reach the routine until its next
        invocation.
        """
        session = None
        while True:
            command, payload = self.from_parent.get()
            if command == "quit":
                return
            recorded = False
            try:
                if command == "configure":
                    config = config_module.load(payload)
                    interface = self.interface_factory(config)
                    interface.check_ready()
                    session = Session(config, interface)
                elif command == "observe":
                    if session is None:
                        raise RuntimeError(
                            "got an observation before being configured"
                        )
                    recorded = session.record(*payload)
                elif command == "shot":
                    if session is None:
                        self.to_parent.put(("status", (False, {})))
                        continue
                else:
                    raise ValueError(f"unknown command {command!r}")

                # Nothing slow may come before this line. lyse runs multishot
                # routines inline and one at a time, so the routine blocked on
                # this reply holds up every shot behind it, and both calls
                # below are round trips to runmanager.
                self.to_parent.put(("status", (recorded, session.status())))

                # Every invocation reconciles, not only those that submit: the
                # routine may not be called again for a long time, and a shot
                # that is no longer coming must not hold its place until it is.
                session.reconcile()
                session.refill()
            except Exception:
                # Fail loudly and stop proposing, rather than carry on with a
                # learner or a runmanager that is not doing what it should.
                if session is not None:
                    session.stopped = "stopped by an error"
                self.to_parent.put(("error", traceback.format_exc()))
