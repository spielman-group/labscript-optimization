"""The persistent optimisation process.

Spawned once by the lyse routine and kept until the routine is restarted. The
fitting and the runmanager traffic happen here so that the routine can hand
over one observation and return immediately, which matters because lyse runs
multishot routines inline and a slow one holds up later shots.

The worker is purely reactive: it proposes only in response to a message from
the routine. If the routine's process dies, no more messages arrive and the
worker submits nothing further, so the few seconds zprocess takes to notice a
dead parent and stop the worker cannot run away with the queue.

Run as a script by :func:`labscript_optimization.routine.start_worker`, never
imported by the routine itself.
"""

import os
import sys
import traceback

if __package__ in (None, ""):
    # zprocess runs this file as a script, so there is no package to import
    # relative to. Put the package's parent on the path and import absolutely,
    # which also works when the package is installed normally.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labscript_utils.ls_zprocess import ProcessTree

from labscript_optimization import config as config_module
from labscript_optimization.runmanager_interface import interface_for
from labscript_optimization.session import Session


def serve(from_parent, to_parent, interface_factory=interface_for) -> None:
    """Handle messages until told to quit.

    The reply goes out first, before the reconciling, proposing and submitting
    that follow it, so the routine waiting on the other end is held up by
    neither a fit nor a runmanager that is slow to answer. The status it reads
    is therefore one step behind both the reconciliation and the refill: the
    shots this invocation drops and submits are counted in the next reply,
    which is what a progress report is anyway.

    A failure in that trailing work still stops the session and still sends an
    error, but the routine has already taken this invocation's reply, so the
    error may not reach it until the next one.
    """
    session = None
    while True:
        command, payload = from_parent.get()
        if command == "quit":
            return
        try:
            if command == "configure":
                config = config_module.load(payload)
                interface = interface_factory(config)
                interface.check_ready()
                session = Session(config, interface)
            elif command == "observe":
                if session is None:
                    raise RuntimeError("got an observation before being configured")
                session.record(*payload)
            elif command == "status":
                if session is None:
                    to_parent.put(("status", {}))
                    continue
            else:
                raise ValueError(f"unknown command {command!r}")

            # Reply before anything slow. Reconciling is a blocking round trip
            # to runmanager, which answers within communication_timeout at
            # worst, and the routine is blocked on this reply until it does.
            to_parent.put(("status", session.status()))

            # Every invocation reconciles, not only those that submit. The
            # routine may not be called again for a long time, and a shot that
            # is no longer coming must not go on holding its place until it is.
            session.reconcile()
            session.refill()
        except Exception:
            # Fail loudly and stop proposing, rather than carrying on with a
            # learner or a runmanager that is not doing what it should.
            if session is not None:
                session.stopped = "stopped by an error"
            to_parent.put(("error", traceback.format_exc()))


def main() -> None:
    process_tree = ProcessTree.connect_to_parent()
    serve(process_tree.from_parent, process_tree.to_parent)


if __name__ == "__main__":
    sys.exit(main())
