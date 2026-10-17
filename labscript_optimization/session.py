"""One optimization session: the history, the learner and the stop conditions.

Nothing here counts shots in flight. Which of the session's shots are still
coming is runmanager's answer, asked for each time it matters, because a count
kept here could only fall when a shot came back, and a shot that never does
would hold its place forever.

The history holds every proposal in the order made, each with its source and
what became of it. A cost arriving out of order fills the slot its shot id
names, and a shot no longer coming is marked dropped, not removed, so a learner
sees a position spent on a shot that produced nothing as spent. The shots the
session awaits are the history's pending records.
"""

import numpy as np
from runmanager.client import (
    BLACS_COMPLETED,
    LYSE_SENT,
    LYSE_WAITING,
    QUEUE_BLOCKED,
    SequenceRefused,
)

from . import learners, observations
from .observations import COMPLETE, DROPPED, PENDING, Observation

#: The source recorded for the configured start, which the session proposes
#: itself whichever learner is running.
START_SOURCE = "start"


class Session:
    """Drives one optimization."""

    def __init__(self, config, interface, learner=None):
        self.config = config
        # The session itself calls only check_unchanged(), submit(proposals) and
        # client.shot_status(shot_ids); the worker checks readiness before each
        # Start.
        self.interface = interface
        # A learner answers propose(history, hint) with (params, source) pairs
        # and carries generation. None builds the configured one.
        self.learner = learners.build(config) if learner is None else learner
        # None opens the run on a draw. The worker replaces it when the first
        # Start asks for runmanager's values.
        self.start_point = config.space.start
        self.proposals: dict[str, np.ndarray] = {}
        self.sources: dict[str, str] = {}
        self.results: dict[str, tuple[float, float | None, bool]] = {}
        self.dropped: set[str] = set()
        self.blocked: set[str] = set()
        self.starved = 0
        self.paused = True
        # Why something other than the user paused the session.
        self.pause_reason: str | None = None
        self._resume_pending = False
        self.stopped: str | None = None

    @property
    def history(self) -> list[Observation]:
        """Every proposal, in the order made, with what became of it."""
        # A waiting proposal and one that will never report stay in: a learner that
        # reads a role off a position needs the positions that produced nothing.
        records = []
        for shot_id, params in self.proposals.items():
            source = self.sources[shot_id]
            if shot_id in self.results:
                records.append(
                    Observation(shot_id, params, *self.results[shot_id], source=source)
                )
                continue
            state = DROPPED if shot_id in self.dropped else PENDING
            records.append(
                Observation(shot_id, params, None, state=state, source=source)
            )
        return records

    @property
    def awaiting(self) -> list[str]:
        """The shots submitted that have neither reported a cost nor been dropped."""
        # The history's pending records, so what the session waits for is what a
        # learner takes to be in flight.
        return [o.shot_id for o in self.history if o.state == PENDING]

    @property
    def best(self) -> Observation | None:
        """The usable observation with the lowest cost, if there is one."""
        return observations.best(self.history)

    def runs_since_best(self) -> int:
        """How many completed shots came after the one holding the best cost."""
        # What max_num_runs_without_better_params limits: every completed shot
        # counts, usable or not, and with no usable cost all of them do.
        completed = [o for o in self.history if o.state == COMPLETE]
        best = observations.best(completed)
        if best is None:
            return len(completed)
        position = [o.shot_id for o in completed].index(best.shot_id)
        return len(completed) - 1 - position

    def stop(self, reason: str) -> None:
        """Stop proposing, for ``reason``, unless the session has stopped already."""
        # The first reason is kept: shots in flight still report after a stop, and
        # each report is checked against the limits again.
        if self.stopped is None:
            self.paused = False
            self.stopped = reason

    def start(self) -> bool:
        """Resume proposing; ``False`` if the session has stopped."""
        if self.stopped is not None:
            return False
        if self.paused:
            self._resume_pending = bool(self.proposals)
        self.paused = False
        self.pause_reason = None
        return True

    def pause(self, reason: str | None = None) -> bool:
        """Pause proposing, showing ``reason`` if any; ``False`` once stopped."""
        if self.stopped is not None:
            return False
        self.paused = True
        self.pause_reason = reason
        return True

    def check_stop(self) -> None:
        """Stop the session if a run limit has been reached."""
        limit = self.config.max_num_runs
        if limit is not None and len(self.results) >= limit:
            self.stop(f"reached max_num_runs ({limit})")
        patience = self.config.max_num_runs_without_better_params
        if patience is not None and self.runs_since_best() >= patience:
            self.stop(
                f"no better parameters in {patience} runs "
                f"(max_num_runs_without_better_params)"
            )

    def record(
        self, shot_id: str, cost: float, uncer: float | None, bad: bool
    ) -> str | None:
        """Take the cost for one shot; return its source, or ``None`` if not taken."""
        # A shot this session did not submit, such as a user's own, is ignored, and
        # so is a second cost for one it did: a row can run twice, by a retry or a
        # BLACS re-run, and an id names a proposal, not an execution.
        if shot_id not in self.proposals or shot_id in self.results:
            return None
        self.results[shot_id] = (float(cost), uncer, bool(bad))
        self.dropped.discard(shot_id)
        self.blocked.discard(shot_id)
        self.check_stop()
        return self.sources[shot_id]

    def reconcile(self) -> list[str]:
        """Ask runmanager what became of the awaited shots; return those given up."""
        awaiting = self.awaiting
        if not awaiting:
            return []
        answers = self.interface.client.shot_status(awaiting)
        gone = []
        # A shot is kept while it may still complete in BLACS, or lyse has its file
        # or will be given it. Any other shot goes at once; a cost that turns up
        # later is still taken by record.
        for shot_id in awaiting:
            record = answers[shot_id]
            if record is not None and record["queue"] == QUEUE_BLOCKED:
                # Behind a queue row only an operator can clear. Counted apart:
                # every other way a shot stops coming is the apparatus getting
                # on, and this one needs someone to look at the queue.
                self.blocked.add(shot_id)
            elif record is not None and (
                record["pending"]
                or (
                    record["blacs"] == BLACS_COMPLETED
                    and record["lyse"] in (LYSE_WAITING, LYSE_SENT)
                )
            ):
                continue
            gone.append(shot_id)
        self.dropped.update(gone)
        return gone

    def refill(self) -> list[str]:
        """Submit what the learner proposes within the budget; return the shot ids."""
        if self.paused or self.stopped:
            return []
        awaiting = len(self.awaiting)
        if (
            awaiting == 0
            and self.proposals
            and self.learner.generation is None
            and not self._resume_pending
        ):
            # Nothing of ours was queued, so BLACS ran a default shot of
            # runmanager's, which a larger num_buffered_runs prevents. A generation
            # learner's queue empties by design, so it is not counted.
            self.starved += 1
        self._resume_pending = False
        room = None
        if self.config.max_num_runs is not None:
            # Dropped shots are not charged to the budget: replacing one spends
            # no run.
            room = self.config.max_num_runs - len(self.results) - awaiting
            if room <= 0:
                return []

        # The start is proposed here, once, at the run's first position,
        # whichever learner runs. The learner sees it as a pending record with
        # shot id None at position 0, so it fills a place in the first batch.
        history, opening = self.history, []
        if not self.proposals and self.start_point is not None:
            start = np.asarray(self.start_point, dtype=float)
            opening.append((start, START_SOURCE))
            history = [
                Observation(None, start, None, state=PENDING, source=START_SOURCE)
            ]
        # max_num_runs caps the whole run, not a batch, so what a learner offers
        # past the room left is cut from the end.
        proposed = [
            *opening,
            *self.learner.propose(history, self.config.num_buffered_runs),
        ][:room]
        if not proposed:
            return []
        for params, source in proposed:
            if not isinstance(source, str):
                raise TypeError(
                    f"{type(self.learner).__name__} proposed {params!r} beside "
                    f"{source!r} rather than a source. A learner returns each "
                    f"proposal as a (params, source) pair, the source naming "
                    f"which of its ways of proposing made it; it is what that "
                    f"shot reads in the phase column, and nothing supplies one "
                    f"on the learner's behalf."
                )

        # A round trip to runmanager, so made only when there is something to
        # submit.
        self.interface.check_unchanged()
        proposals = np.array([params for params, _ in proposed], dtype=float)
        try:
            shot_ids = self.interface.submit(proposals)
        except SequenceRefused as exc:
            # Starting another sequence would split the run in two.
            self.stop(str(exc))
            return []
        if len(shot_ids) != len(proposals):
            # Nothing is recorded before the raise, so the session is left as it was.
            raise RuntimeError(
                f"submitted {len(proposals)} proposals but got "
                f"{len(shot_ids)} shot ids back; the two cannot be paired, "
                f"and shots may be queued that this session cannot account for"
            )
        for shot_id, params, (_, source) in zip(shot_ids, proposals, proposed):
            self.proposals[shot_id] = params
            self.sources[shot_id] = source
        return shot_ids

    def status(self) -> dict:
        """A snapshot for the routine to write back as lyse results."""
        best = self.best
        # The routine flips a maximized cost on the way in; this undoes it, so
        # the status is in the lab's own sign.
        best_cost = None if best is None else best.cost
        if best_cost is not None and self.config.maximize:
            best_cost = -best_cost
        # None marks a key the session has nothing to report for yet.
        return {
            "submitted": len(self.proposals),
            "completed": len(self.results),
            "awaiting": len(self.awaiting),
            "dropped": len(self.dropped),
            "blocked": len(self.blocked),
            "starved": self.starved,
            "best_cost": best_cost,
            "best_params": None if best is None else best.params.tolist(),
            "start": None if self.start_point is None else self.start_point.tolist(),
            "paused": self.paused,
            "pause_reason": self.pause_reason,
            "stopped": self.stopped,
        }
