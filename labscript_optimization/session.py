"""One optimisation session: the history, the learner, and the stop conditions.

There is no count of shots in flight. How many of this session's shots are
still coming is runmanager's answer, asked for afresh each time it matters,
because a number kept here can only be decremented by a shot coming back --
and a shot that never comes back would hold its place for ever.

History is kept in proposal order: a cost arriving out of order fills the slot
its shot id names, and a shot that is no longer coming is dropped.
"""

import numpy as np

from . import learners, observations
from .observations import Observation
from .runmanager_interface import BLOCKED_SHOT_STATE, UNKNOWN_SHOT_STATE


class Session:
    """Drives one optimisation.

    Args:
        config: The session configuration.
        interface: Something with ``check_unchanged()``, ``submit(proposals)``
            and ``shot_status(shot_ids)``. Readiness is the worker's to check
            before a session is built.
        learner: The learner to use, or ``None`` to build the configured one.
            It answers ``propose(history, k)`` and carries ``last_phase``.
    """

    def __init__(self, config, interface, learner=None):
        self.config = config
        self.interface = interface
        self.learner = learners.build(config) if learner is None else learner
        self.proposals: dict[str, np.ndarray] = {}
        self.results: dict[str, tuple[float, float | None, bool]] = {}
        self.dropped: set[str] = set()
        self.blocked: set[str] = set()
        # Awaited shots runmanager had no row for at the last reconcile.
        self._unknown: set[str] = set()
        self.starved = 0
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
        """The usable observation with the lowest cost, if there is one."""
        return observations.best(self.history)

    def runs_since_best(self) -> int:
        """How many completed shots came after the one holding the best cost.

        Counted over every completed shot, usable or not; see
        :attr:`~labscript_optimization.config.Config.max_num_runs_without_better_params`.
        """
        history = self.history
        best = observations.best(history)
        if best is None:
            return len(history)
        position = [o.shot_id for o in history].index(best.shot_id)
        return len(history) - 1 - position

    def check_stop(self) -> None:
        limit = self.config.max_num_runs
        if limit is not None and len(self.results) >= limit:
            self.stopped = f"reached max_num_runs ({limit})"
            return
        patience = self.config.max_num_runs_without_better_params
        if patience is not None and self.runs_since_best() >= patience:
            self.stopped = (
                f"no better parameters in {patience} runs "
                f"(max_num_runs_without_better_params)"
            )

    def record(self, shot_id: str, cost: float, uncer: float | None, bad: bool) -> bool:
        """Take the cost for one shot. Returns whether it was taken.

        A shot this session did not submit is ignored, which is how a user's
        own shots and runmanager's defaults pass through harmlessly. So is a
        second cost for a shot already recorded: a row can be run twice, by a
        retry or by BLACS re-running a file that already held data, and an id
        names a proposal rather than an execution.
        """
        if shot_id not in self.proposals or shot_id in self.results:
            return False
        self.results[shot_id] = (float(cost), uncer, bool(bad))
        self.dropped.discard(shot_id)
        self.blocked.discard(shot_id)
        self.check_stop()
        return True

    def reconcile(self) -> list[str]:
        """Ask runmanager what became of the awaited shots, and give some up.

        Returns the shots dropped by this call. Dropping one stops a place in
        the queue being held for it; a cost that turns up afterwards is still
        taken, and the shot stops counting as dropped.

        An ``unknown`` shot is given up on only once it has been unknown across
        two reconciles running. Every other not-pending answer names a reason
        nothing further will happen -- cancelled, will not compile, refused by
        BLACS, or behind a row only an operator can clear -- and goes at once.
        """
        awaiting = self.awaiting
        if not awaiting:
            return []
        answers = self.interface.shot_status(awaiting)
        unknown_before, self._unknown = self._unknown, set()
        gone = []
        for shot_id in awaiting:
            answer = answers[shot_id]
            if answer.get("pending", False):
                continue
            unknown = answer.get("state", UNKNOWN_SHOT_STATE) == UNKNOWN_SHOT_STATE
            # The round of grace is the whole of the difference between a shot
            # that has gone and one that has just run: runmanager says
            # ``unknown`` of both, and a finished shot leaves the queue while
            # its cost is still crossing lyse. Dropping on the first answer
            # counts the ordinary end of every healthy shot as a loss, in the
            # very number a user reads to see whether shots are being lost.
            if unknown and shot_id not in unknown_before:
                self._unknown.add(shot_id)
                continue
            if answer.get("state") == BLOCKED_SHOT_STATE:
                # Behind a row the queue will not hand over. Counted apart
                # from the rest because every other way a shot stops coming
                # is the apparatus getting on with things, and this one is
                # somebody needing to go and look at the queue.
                self.blocked.add(shot_id)
            gone.append(shot_id)
        self.dropped.update(gone)
        return gone

    def refill(self) -> list[str]:
        """Submit enough proposals to keep the queue topped up.

        Returns the shot ids submitted, which is empty once the session has
        stopped. Raises unless the interface answers with one shot id per
        proposal.
        """
        if self.stopped:
            return []
        awaiting = len(self.awaiting)
        if awaiting == 0 and self.proposals:
            # Nothing of ours was queued when this ran, so runmanager gave
            # BLACS a default shot instead: the apparatus staying busy rather
            # than a fault, but a shot the optimiser did not get. A session
            # that starves wants a larger num_buffered_runs.
            self.starved += 1
        wanted = self.config.num_buffered_runs - awaiting
        if self.config.max_num_runs is not None:
            # Dropped shots are not charged against the budget: they produced
            # nothing, so replacing one is not spending a run.
            room = self.config.max_num_runs - len(self.results) - awaiting
            wanted = min(wanted, room)
        if wanted <= 0:
            return []

        self.interface.check_unchanged()
        proposals = np.atleast_2d(self.learner.propose(self.history, wanted))
        shot_ids = self.interface.submit(proposals)
        if len(shot_ids) != len(proposals):
            # Nothing is recorded before the raise: the session is left as it
            # was rather than holding ids that may not name their proposals.
            raise RuntimeError(
                f"submitted {len(proposals)} proposals but got "
                f"{len(shot_ids)} shot ids back; the two cannot be paired, "
                f"and shots may be queued that this session cannot account for"
            )
        for shot_id, params in zip(shot_ids, proposals):
            self.proposals[shot_id] = np.asarray(params, dtype=float)
        return shot_ids

    def status(self) -> dict:
        """A snapshot for the routine to write back as lyse results.

        ``None`` means the session has nothing to report for that key yet.

        ``best_cost`` is in the units and sign of the lab's own cost column,
        beside a ``best_params`` in real units: under ``maximize`` a
        measurement of 7 is reported as 7.
        """
        best = self.best
        # The routine flips a maximised quantity once on the way in, so that
        # everything here minimises. This is the mirror of that flip, and the
        # only place it is undone: minimising is the package's own business
        # and has no place in a column a physicist reads.
        best_cost = None if best is None else best.cost
        if best_cost is not None and self.config.maximize:
            best_cost = -best_cost
        return {
            "session": self.config.session,
            "phase": self.learner.last_phase,
            "submitted": len(self.proposals),
            "completed": len(self.results),
            "awaiting": len(self.awaiting),
            "dropped": len(self.dropped),
            "blocked": len(self.blocked),
            "starved": self.starved,
            "best_cost": best_cost,
            "best_params": None if best is None else best.params.tolist(),
            "best_shot_id": None if best is None else best.shot_id,
            "stopped": self.stopped,
        }
