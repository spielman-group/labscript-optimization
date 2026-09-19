"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager sends it
to lyse, which is the routine being called on it.

Every shot carries the identifier runmanager minted for its queue row, written
into the shot file. That is what a cost is matched to a proposal by, and what
this module asks about when a cost has not arrived. Nothing here counts shots:
whether a shot is still coming is runmanager's answer, not a number kept on
this side.
"""

from typing import Iterable, Sequence

import numpy as np

#: Attribute runmanager writes into each shot file it queues.
SHOT_ID_ATTR = "shot_id"

#: The state runmanager reports for a shot id it has no row for. It is what a
#: shot that completed and left the queue reads as, as much as one an operator
#: deleted, so it says less on its own than the other not-pending states do.
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
        """Raise unless runmanager can run this session to completion.

        Both checks are about silence rather than error. A queue that empties
        under the 'nothing' policy produces no further shot, so nothing reaches
        lyse, so the routine is never called again and the optimisation stops
        without saying anything. And a labscript file changed underneath a
        running session would optimise a different experiment without a word.

        The policy check earns its place twice over. Under 'default_labscript'
        the gap between submissions is filled by a shot runmanager makes
        itself, which deliberately never becomes the sequence anchor, so a
        whole run stays one sequence with continuing run numbers. Under
        'nothing' the anchor is let go the moment BLACS finds the queue empty.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimisation"
            )

        policy = self.client.get_empty_queue_policy()
        if policy != "default_labscript":
            raise RuntimeError(
                f"runmanager's empty-queue policy is {policy!r}. Set it to "
                f"'default_labscript'. Two things go wrong otherwise, and "
                f"only the first is obvious. Nothing would reach lyse the "
                f"first time the queue emptied, so the routine would never be "
                f"invoked again and the session would stop without saying so. "
                f"And runmanager lets go of the sequence it is continuing as "
                f"soon as BLACS finds the queue empty, so every submission "
                f"would start a sequence of its own and the run would be "
                f"scattered across one sequence per shot."
            )

        self.labscript_file = self.client.get_labscript_file()

    def check_unchanged(self) -> None:
        """Raise if the labscript file has changed since the session started."""
        current = self.client.get_labscript_file()
        if self.labscript_file is not None and current != self.labscript_file:
            raise RuntimeError(
                f"the labscript file changed from {self.labscript_file!r} to "
                f"{current!r} while this session was running; its shots would "
                f"no longer be the experiment it has been optimising"
            )

    def submit(self, proposals: Sequence[Sequence[float]]) -> list[str]:
        """Queue one shot per proposal. Returns their shot ids, in order.

        A refusal means nothing was queued: submit_shots checks every entry --
        that the globals evaluate, and that each produces exactly one shot --
        before submitting any of them. So a raise here leaves nothing behind to
        account for, and the proposals are simply discarded. The learner is a
        function of the history, which this has not touched, so there is no
        state to unwind.
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
        id runmanager does not know. An id it does not answer for at all is
        filled in the same way, because knowing nothing of a shot is what it
        means.

        Passed through rather than reduced to the shots still coming. Whether
        a cost is still on its way is not a property of the queue alone -- a
        shot that completed has left the queue and reads exactly like one that
        was deleted -- so the caller needs the state runmanager gave as well as
        its verdict.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return {}
        answer = self.client.shot_status(shot_ids)
        return {
            i: answer[i] if i in answer else dict(UNKNOWN_SHOT_STATUS)
            for i in shot_ids
        }
