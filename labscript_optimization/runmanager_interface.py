"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager sends it
to lyse, which is the routine being called on it. Every shot carries the
identifier runmanager minted for its queue row, written into the shot file:
that is what a cost is matched to a proposal by.
"""

from typing import Iterable, Sequence

import numpy as np

#: The state runmanager reports for a shot id it has no row for.
UNKNOWN_SHOT_STATE = "unknown"

#: What runmanager answers for a shot id in that state.
UNKNOWN_SHOT_STATUS = {"pending": False, "state": UNKNOWN_SHOT_STATE}


class RunmanagerInterface:
    """Submits proposals and reports what became of them.

    Args:
        config: The session configuration.
        client: A ``runmanager.remote`` client, or ``None`` to make the default
            one. Injected so the session can be tested without runmanager.
    """

    def __init__(self, config, client=None):
        if client is None:
            from runmanager import remote

            client = remote.Client()
        self.config = config
        self.client = client
        self.labscript_file = None

    def check_ready(self) -> None:
        """Raise if runmanager cannot start a session, and pin its labscript file.

        A global that does not evaluate is a shot that will not compile, and
        every shot this session submits would be one. The file pinned here is
        what :meth:`check_unchanged` compares against for the rest of the
        session.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimisation"
            )

        self.labscript_file = self.client.get_labscript_file()

    def check_unchanged(self) -> None:
        """Raise if the labscript file has changed since the session started."""
        current = self.client.get_labscript_file()
        if current != self.labscript_file:
            raise RuntimeError(
                f"the labscript file changed from {self.labscript_file!r} to "
                f"{current!r} while this session was running; its shots would "
                f"no longer be the experiment it has been optimising"
            )

    def submit(self, proposals: Sequence[Sequence[float]]) -> list[str]:
        """Queue one shot per proposal. Returns their shot ids, in order.

        A refusal means nothing was queued: submit_shots checks every entry --
        that the globals evaluate, and that each produces exactly one shot --
        before submitting any of them, so a raise here leaves nothing behind
        to account for.
        """
        entries = [
            self.config.globals_for(np.asarray(p, dtype=float)) for p in proposals
        ]
        return [d["shot_id"] for d in self.client.submit_shots(entries)]

    def shot_status(self, shot_ids: Iterable[str]) -> dict[str, dict]:
        """What runmanager says about each of these shots, as it says it.

        ``{shot_id: {'pending': bool, 'state': str}}``, one entry per id asked
        about. ``pending`` is whether that shot could still produce a cost.
        ``state`` is the queue row's own state, or ``'submitted'`` for a shot
        runmanager has taken on but has no row for yet, ``'blocked'`` for a row
        sitting behind one an operator has to clear, and ``'unknown'`` for an
        id runmanager does not know -- including one it does not answer for.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return {}
        answer = self.client.shot_status(shot_ids)
        return {
            i: answer[i] if i in answer else dict(UNKNOWN_SHOT_STATUS)
            for i in shot_ids
        }
