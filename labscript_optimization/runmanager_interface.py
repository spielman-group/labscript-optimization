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
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimisation"
            )

        policy = self.client.get_empty_queue_policy()
        if policy != "default_labscript":
            raise RuntimeError(
                f"runmanager's empty-queue policy is {policy!r}. This session "
                f"would stop silently the first time its queue emptied, "
                f"because nothing would reach lyse to invoke the routine "
                f"again. Set the policy to 'default_labscript' so the "
                f"apparatus keeps running when the optimiser has nothing "
                f"queued."
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

    def pending(self, shot_ids: Iterable[str]) -> set[str]:
        """Which of these shots could still produce a cost.

        An id runmanager no longer knows is not pending, which is how a shot
        deleted by an operator, or lost to a runmanager restart, stops being
        waited on.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return set()
        status = self.client.shot_status(shot_ids)
        return {i for i in shot_ids if status.get(i, {}).get("pending", False)}


class MockInterface:
    """Accepts proposals without a runmanager behind it.

    Selected by ``[COMPILATION] mock = true``. Every shot it accepts stays
    pending until a cost is recorded for it, so a session driven against this
    behaves as though the apparatus never loses one.
    """

    def __init__(self, config):
        self.config = config
        self.submitted: list[tuple[str, dict]] = []
        self.labscript_file = "mock"

    def check_ready(self) -> None:
        pass

    def check_unchanged(self) -> None:
        pass

    def submit(self, proposals) -> list[str]:
        ids = []
        for p in proposals:
            shot_id = f"mock-{len(self.submitted)}"
            values = self.config.globals_for(np.asarray(p, dtype=float))
            self.submitted.append((shot_id, values))
            ids.append(shot_id)
            print(f"mock submit {shot_id}: {values}", flush=True)
        return ids

    def pending(self, shot_ids) -> set[str]:
        return set(shot_ids)


def interface_for(config):
    """The interface a configuration asks for."""
    return MockInterface(config) if config.mock else RunmanagerInterface(config)
