"""One optimisation session: the history, the learner, and the stop conditions.

There is no count of shots in flight. How many of this session's shots are
still coming is runmanager's answer, asked for afresh each time it matters,
because a number kept here can only be decremented by a shot coming back --
and a shot that never comes back would hold its place for ever.

History holds every proposal the session has made, in the order it made them,
each carrying what proposed it and what became of it: a cost arriving out of
order fills the slot its shot id names, and a shot that is no longer coming is
marked dropped rather than taken out. A learner therefore sees a position
spent on a shot that produced nothing as a position spent, which is what lets
one read a role off a proposal's position. The shots the session is awaiting
are the history's pending records, read off it rather than worked out beside
it.
"""

import numpy as np

from . import learners, observations
from .observations import COMPLETE, DROPPED, PENDING, Observation
from .runmanager_interface import (
    BLOCKED_SHOT_STATE,
    REFUSED_SEQUENCE,
    UNKNOWN_SHOT_STATE,
)

#: The source recorded for the configured start. The session proposes it
#: itself, whichever learner is running, so it carries a name of the session's
#: rather than a learner's.
START_SOURCE = "start"


class Session:
    """Drives one optimisation.

    Args:
        config: The session configuration.
        interface: Something with ``check_unchanged()``, ``submit(proposals)``
            and ``shot_status(shot_ids)``. Readiness is the worker's to check
            before a session is built.
        learner: The learner to use, or ``None`` to build the configured one.
            It answers ``propose(history, hint)`` with ``(params, source)``
            pairs and carries ``generation``.
    """

    def __init__(self, config, interface, learner=None):
        self.config = config
        self.interface = interface
        self.learner = learners.build(config) if learner is None else learner
        self.proposals: dict[str, np.ndarray] = {}
        self.sources: dict[str, str] = {}
        self.results: dict[str, tuple[float, float | None, bool]] = {}
        self.dropped: set[str] = set()
        self.blocked: set[str] = set()
        # Awaited shots runmanager had no row for at the last reconcile.
        self._unknown: set[str] = set()
        self.starved = 0
        self.stopped: str | None = None

    @property
    def history(self) -> list[Observation]:
        """Every proposal, in the order it was made, with what became of it.

        A proposal still waiting and one that will never report are both a
        position spent without a usable cost, and both are here: a learner
        that reads a role off a position needs the positions that produced
        nothing as much as the ones that produced a cost.
        """
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
        """Shots submitted that have neither reported a cost nor been dropped.

        These are the history's pending records and nothing else, so what the
        session waits for and what a learner reading states off the history
        takes to be in flight are one answer. A second account kept beside the
        history would have to be moved in step with it by every drop, late
        cost and repeated cost, and would part company with it the first time
        one was missed.
        """
        return [o.shot_id for o in self.history if o.state == PENDING]

    @property
    def best(self) -> Observation | None:
        """The usable observation with the lowest cost, if there is one."""
        return observations.best(self.history)

    def runs_since_best(self) -> int:
        """How many completed shots came after the one holding the best cost.

        Counted over every completed shot, usable or not; see
        :attr:`~labscript_optimization.config.Config.max_num_runs_without_better_params`.
        """
        completed = [o for o in self.history if o.state == COMPLETE]
        best = observations.best(completed)
        if best is None:
            return len(completed)
        position = [o.shot_id for o in completed].index(best.shot_id)
        return len(completed) - 1 - position

    def stop(self, reason: str) -> None:
        """Stop proposing, for ``reason``, unless the session has stopped already.

        The first reason is the one kept. A session goes on taking costs after
        it stops -- the shots already in flight still report, and each report
        is checked against the limits -- so a later reason would otherwise
        replace the one that stopped it, and the run's recorded cause would be
        whichever limit its last shots happened to reach. The worker stops a
        session through here too, so an error after a session has stopped is
        reported as an error without becoming the reason.
        """
        if self.stopped is None:
            self.stopped = reason

    def check_stop(self) -> None:
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
        """Take the cost for one shot. Returns its source if the cost was taken.

        The source is the one recorded when the shot was proposed, and the
        routine writes it onto the shot as its ``phase``: what proposed that
        shot, whatever has been proposed since. ``None`` says the cost was not
        taken.

        A shot this session did not submit is ignored, which is how a user's
        own shots and runmanager's defaults pass through harmlessly. So is a
        second cost for a shot already recorded: a row can be run twice, by a
        retry or by BLACS re-running a file that already held data, and an id
        names a proposal rather than an execution.
        """
        if shot_id not in self.proposals or shot_id in self.results:
            return None
        self.results[shot_id] = (float(cost), uncer, bool(bad))
        self.dropped.discard(shot_id)
        self.blocked.discard(shot_id)
        self.check_stop()
        return self.sources[shot_id]

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
        """Submit what the learner proposes, as far as the budget has room.

        The learner is handed the history and ``num_buffered_runs`` as a hint,
        and answers with whatever its method allows it to propose now, each
        proposal beside its source: the random learners top the shots in
        flight up to the hint, a learner declaring a generation proposes a
        whole one when nothing of the last is outstanding and nothing
        otherwise, and the Gaussian process proposes a batch with explorer
        shots behind it when nothing of its last batch is outstanding. The
        session holds no barrier of its own. It submits what
        comes back and records each proposal with its source, which nothing
        changes after; a proposal without one is refused rather than recorded
        as anyone's.

        The very first proposal of a run is the space's configured start,
        where the parameters carry one, under :data:`START_SOURCE`, and the
        learner answers around it; see the comment below.

        ``max_num_runs`` is a ceiling on the whole run rather than on a batch,
        so what a learner offers past the room the budget has left is cut, and
        the last generation is whatever that room holds. A generation cut
        short is a generation nothing follows: the walk that rebuilds the
        population reads a proposal's slot off its position, so a short one
        displaces nothing, and each of its trials competes for its own slot as
        it would have in a whole one.

        Returns the shot ids submitted, which is empty once the session has
        stopped. Raises unless the interface answers with one shot id per
        proposal. runmanager refusing to add to the run's sequence stops the
        session, with runmanager's reason; any other refusal is raised.
        """
        if self.stopped:
            return []
        awaiting = len(self.awaiting)
        if awaiting == 0 and self.proposals and self.learner.generation is None:
            # Nothing of ours was queued when this ran, so runmanager gave
            # BLACS a default shot instead: the apparatus staying busy rather
            # than a fault, but a shot the optimiser did not get. A session
            # that starves wants a larger num_buffered_runs. A learner
            # declaring a generation is not counted: the queue emptying is how
            # one generation ends and the next begins, and a counter that fires
            # by design says nothing about the run it is meant to describe.
            self.starved += 1
        room = None
        if self.config.max_num_runs is not None:
            # Dropped shots are not charged against the budget: they produced
            # nothing, so replacing one is not spending a run.
            room = self.config.max_num_runs - len(self.results) - awaiting
            if room <= 0:
                return []

        # Where a run begins is the session's answer, not a learner's. The
        # start is written on the parameters, beside each one's min and max,
        # and is proposed here: once, at the first position of the run,
        # whichever learner is running and whether or not that learner has any
        # idea of an opening. So "proposed exactly once" holds because there
        # is one place that proposes it rather than because a guard somewhere
        # declines to do it again.
        #
        # The learner is asked in the same call, from a history in which the
        # start already holds position 0 as a pending record, so the start
        # takes a place inside the first batch rather than a batch of its own:
        # one of a random learner's places, the first of a Gaussian process's
        # warmup shots, and slot 0 of the first generation, which then still
        # goes out whole and is still waited for as one. A
        # start sent out on its own would be the second route past that
        # barrier. The record's shot id is ``None``, because runmanager has not
        # minted one, and that is how a learner holding a barrier tells a
        # start going out beside its proposals from one it must wait for. The
        # record lives only in this call: what the session keeps is keyed on
        # the ids ``submit`` returns, and ``history`` is rebuilt from those.
        history, opening = self.history, []
        if not self.proposals and self.config.space.start is not None:
            start = np.asarray(self.config.space.start, dtype=float)
            opening.append((start, START_SOURCE))
            history = [
                Observation(None, start, None, state=PENDING, source=START_SOURCE)
            ]
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
        # submit rather than on every refill.
        self.interface.check_unchanged()
        proposals = np.array([params for params, _ in proposed], dtype=float)
        try:
            shot_ids = self.interface.submit(proposals)
        except Exception as exc:
            # Starting another sequence would split the run in two. runmanager's
            # server wraps its reason in a traceback, whose last line it is.
            message = str(exc)
            if REFUSED_SEQUENCE not in message:
                raise
            self.stop(message[message.rfind(REFUSED_SEQUENCE) :].splitlines()[0])
            return []
        if len(shot_ids) != len(proposals):
            # Nothing is recorded before the raise: the session is left as it
            # was rather than holding ids that may not name their proposals.
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
        """A snapshot for the routine to write back as lyse results.

        ``None`` means the session has nothing to report for that key yet.

        ``best_cost`` is in the units and sign of the lab's own cost column,
        beside a ``best_params`` in real units: under ``maximize`` a
        measurement of 7 is reported as 7.

        A shot's ``phase`` is not here. It belongs to the shot rather than to
        the session -- it is what proposed that shot -- and :meth:`record`
        returns it for the shot whose cost it takes.
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
