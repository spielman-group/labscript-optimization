"""One optimisation session: the history, the learner, and the stop conditions.

There is no count of shots in flight. How many of this session's shots are
still coming is runmanager's answer, asked for afresh each time it matters,
because a number kept here can only be decremented by a shot coming back --
and a shot that never comes back would hold its place for ever.

History is kept in proposal order. A cost arriving out of order fills the slot
its shot id names, and a shot that is no longer coming is dropped. That is the
whole of the bookkeeping, and it is why the learners never see a shot in
flight.

The one thing carried between questions is which shots runmanager had no row
for the last time it was asked, because that single answer covers both a shot
that has just finished and a shot that has gone for good. It is a memory of
what was said, not a tally of what is outstanding.
"""

import numpy as np

from . import learners, observations
from .observations import Observation
from .runmanager_interface import UNKNOWN_SHOT_STATE

#: What runmanager answers about a shot it has no queue row for. It says this
#: of a shot that completed and left the queue just as much as of one it never
#: knew or has forgotten, so on its own it is not a reason to give up on a
#: shot; see :meth:`Session.reconcile`.


class Session:
    """Drives one optimisation.

    Args:
        config: The session configuration.
        interface: Something with ``check_unchanged()``, ``submit(proposals)``
            and ``shot_status(shot_ids)``. Readiness is the worker's to check
            before a session is built, and is not asked about again here.
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
        # Awaited shots runmanager had no row for at the last reconcile. An id
        # waits here for one round before being given up on, because the
        # answer that puts it here is also the answer a healthy shot gets the
        # moment it finishes; see reconcile().
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

    def _runs_since_best(self) -> int:
        """How many completed shots came after the one holding the best cost.

        Counted over every completed shot rather than only the usable ones. A
        shot that came back with nothing usable is still a shot spent, and it
        already counts against ``max_num_runs``; leaving it out here is what
        would let a session whose detector has died run for ever on the one
        limit meant to stop it. A history with nothing usable in it has gone
        its whole length without better parameters.
        """
        history = self.history
        best = observations.best(history)
        if best is None:
            return len(history)
        position = [o.shot_id for o in history].index(best.shot_id)
        return len(history) - 1 - position

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
        """Ask runmanager what became of the awaited shots, and give some up.

        ``shot_status(shot_ids)`` hands back runmanager's own answer, one
        ``{'pending': bool, 'state': str}`` per id asked about.

        Returns the shots dropped by this call. Dropping a shot is a decision
        to stop holding a place in the queue for it, not a ruling that its cost
        can never arrive: a cost that turns up afterwards is still taken, and
        the shot stops counting as dropped.

        A shot runmanager has no row for is reported ``unknown``, and that is
        what it says about a shot that completed and left the queue as much as
        about one an operator deleted or a restart lost. The first is the
        ordinary end of every healthy shot, whose cost is on its way through
        lyse, so an unknown shot is given up on only once it has been unknown
        across two reconciles running -- by which time the routine has been
        called for it and its cost has been offered. Dropping on the first
        answer instead would make the normal completion of every shot look
        like a loss, and the count of dropped shots is the number a user reads
        to see whether shots are being lost.

        Every other not-pending answer names a reason nothing further will
        happen: the shot was cancelled, it cannot compile, BLACS refused it, or
        it sits behind a row only an operator can clear. Those are given up on
        at once.
        """
        awaiting = self.awaiting
        if not awaiting:
            self._unknown.clear()
            return []
        answers = self.interface.shot_status(awaiting)
        unknown_before, self._unknown = self._unknown, set()
        gone = []
        for shot_id in awaiting:
            # An id the interface says nothing at all about is read as
            # unknown: no answer is no reason to give up on a shot this round.
            answer = answers.get(shot_id) or {}
            if answer.get("pending", False):
                continue
            unknown = answer.get("state", UNKNOWN_SHOT_STATE) == UNKNOWN_SHOT_STATE
            if unknown and shot_id not in unknown_before:
                self._unknown.add(shot_id)
                continue
            gone.append(shot_id)
        self.dropped.update(gone)
        return gone

    def refill(self) -> list[str]:
        """Submit enough proposals to keep the queue topped up.

        Returns the shot ids submitted, which is empty once the session has
        stopped. Raises if the interface does not answer with one shot id per
        proposal, because a pairing that is not one to one cannot be trusted.
        """
        if self.stopped:
            return []
        awaiting = len(self.awaiting)
        if awaiting == 0 and self.proposals:
            # Nothing of ours was queued when this ran, so runmanager will have
            # given BLACS a default shot instead. That is the apparatus staying
            # busy rather than a fault, but every one is a shot the optimiser
            # did not get, so it is counted: a session that starves often wants
            # a larger num_buffered_runs.
            self.starved += 1
        wanted = self.config.num_buffered_runs - awaiting
        if self.config.max_num_runs is not None:
            # Do not queue shots beyond the budget. Dropped shots are not
            # counted against it: they produced nothing, so replacing one is
            # not spending a run.
            room = self.config.max_num_runs - len(self.results) - awaiting
            wanted = min(wanted, room)
        if wanted <= 0:
            return []

        self.interface.check_unchanged()
        proposals = np.atleast_2d(self.learner.propose(self.history, wanted))
        shot_ids = self.interface.submit(proposals)
        if len(shot_ids) != len(proposals):
            # Pairing what came back against the proposals would either leave
            # out a proposal whose shot is queued and running -- its cost then
            # arrives for an id this session never recorded, and the buffer
            # depth is never reached again -- or attribute a cost to
            # parameters that were not the ones requested. Nothing is recorded
            # here: an id that cannot be trusted to name its proposal is worth
            # no more than one this session did not submit, and a submission
            # that goes wrong leaves the session untouched.
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
        """
        best = self.best
        return {
            "session": self.config.session,
            "phase": self.learner.last_phase,
            "submitted": len(self.proposals),
            "completed": len(self.results),
            "awaiting": len(self.awaiting),
            "dropped": len(self.dropped),
            "starved": self.starved,
            "best_cost": None if best is None else best.cost,
            "best_params": None if best is None else best.params.tolist(),
            "best_shot_id": None if best is None else best.shot_id,
            "stopped": self.stopped,
        }
