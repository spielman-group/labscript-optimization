"""One optimisation session: the history, the learner, and the stop conditions.

There is no count of shots in flight. How many of this session's shots are
still coming is runmanager's answer, asked for afresh each time it matters,
because a number kept here can only be decremented by a shot coming back --
and a shot that never comes back would hold its place for ever.

History is kept in proposal order. A cost arriving out of order fills the slot
its shot id names, and a shot that is no longer coming is dropped. That is the
whole of the bookkeeping, and it is why the learners never see a shot in
flight.
"""

from typing import Sequence

import numpy as np

from . import learners
from .observations import Observation, usable


class Session:
    """Drives one optimisation.

    Args:
        config: The session configuration.
        interface: Something with ``check_ready()``, ``check_unchanged()``,
            ``submit(proposals)`` and ``pending(shot_ids)``.
        learner: The learner to use, or ``None`` to build the configured one.
    """

    def __init__(self, config, interface, learner=None):
        self.config = config
        self.interface = interface
        self.learner = learners.build(config) if learner is None else learner
        self.proposals: dict[str, np.ndarray] = {}
        self.results: dict[str, tuple[float, float | None, bool]] = {}
        self.dropped: set[str] = set()
        self.stopped: str | None = None

    @property
    def history(self) -> list[Observation]:
        """Completed observations, in the order they were proposed."""
        return [
            Observation(shot_id, params, *self.results[shot_id])
            for shot_id, params in self.proposals.items()
            if shot_id in self.results
        ]

    @property
    def awaiting(self) -> list[str]:
        """Shots submitted that have neither reported a cost nor been dropped."""
        return [
            shot_id
            for shot_id in self.proposals
            if shot_id not in self.results and shot_id not in self.dropped
        ]

    @property
    def best(self) -> Observation | None:
        seen = usable(self.history)
        return min(seen, key=lambda o: o.cost) if seen else None

    def _runs_since_best(self) -> int:
        costs = [o.cost for o in usable(self.history)]
        return 0 if not costs else len(costs) - 1 - int(np.argmin(costs))

    def _check_stop(self) -> None:
        limit = self.config.max_num_runs
        if limit is not None and len(self.results) >= limit:
            self.stopped = f"reached max_num_runs ({limit})"
            return
        patience = self.config.max_num_runs_without_better_params
        if patience is not None and self._runs_since_best() >= patience:
            self.stopped = (
                f"no better parameters in {patience} runs "
                f"(max_num_runs_without_better_params)"
            )

    def record(self, shot_id: str, cost: float, uncer: float | None, bad: bool) -> bool:
        """Take the cost for one shot. Returns whether it was taken.

        A shot this session did not submit is ignored, which is how a user's
        own shots and runmanager's default shots pass through harmlessly. So is
        a second cost for a shot already recorded: a row can be run twice, by a
        retry or by BLACS re-running a file that already held data, and an id
        names a proposal rather than an execution.
        """
        if shot_id not in self.proposals or shot_id in self.results:
            return False
        self.results[shot_id] = (float(cost), uncer, bool(bad))
        self.dropped.discard(shot_id)
        self._check_stop()
        return True

    def reconcile(self) -> list[str]:
        """Ask runmanager which awaited shots are still coming, and drop the rest.

        Returns the shots dropped by this call. A dropped shot is one that will
        never produce a cost -- it was cancelled, it cannot compile, it is held
        waiting on an operator, or runmanager no longer knows it at all.
        """
        awaiting = self.awaiting
        if not awaiting:
            return []
        still_coming = self.interface.pending(awaiting)
        gone = [shot_id for shot_id in awaiting if shot_id not in still_coming]
        self.dropped.update(gone)
        return gone

    def refill(self) -> list[str]:
        """Submit enough proposals to keep the queue topped up.

        Returns the shot ids submitted, which is empty once the session has
        stopped.
        """
        if self.stopped:
            return []
        wanted = self.config.num_buffered_runs - len(self.awaiting)
        if self.config.max_num_runs is not None:
            # Do not queue shots beyond the budget. Dropped shots are not
            # counted against it: they produced nothing, so replacing one is
            # not spending a run.
            room = self.config.max_num_runs - len(self.results) - len(self.awaiting)
            wanted = min(wanted, room)
        if wanted <= 0:
            return []

        self.interface.check_unchanged()
        proposals = np.atleast_2d(self.learner.propose(self.history, wanted))
        shot_ids = self.interface.submit(proposals)
        for shot_id, params in zip(shot_ids, proposals):
            self.proposals[shot_id] = np.asarray(params, dtype=float)
        return shot_ids

    def status(self) -> dict:
        """A snapshot for the routine to write back as lyse results."""
        best = self.best
        return {
            "session": self.config.session,
            "phase": getattr(self.learner, "last_phase", "main"),
            "submitted": len(self.proposals),
            "completed": len(self.results),
            "awaiting": len(self.awaiting),
            "dropped": len(self.dropped),
            "best_cost": None if best is None else best.cost,
            "best_params": None if best is None else best.params.tolist(),
            "best_shot": None if best is None else best.tag,
            "stopped": self.stopped,
        }
