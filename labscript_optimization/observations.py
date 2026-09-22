"""The proposal record, and the history helpers learners share.

An :class:`Observation` is one proposal and what became of it: the parameters
that were requested, the state that proposal is in, and the cost if one has
come back. Learners are handed every proposal a session has made, in the order
it made them, so a position spent on a shot that is still running or on one
that will never report is visible to them as a position spent.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Submitted, and a cost may still arrive.
PENDING = "pending"

#: No cost will ever arrive: the shot was cancelled, deleted, refused, or is
#: behind a row only an operator can clear.
DROPPED = "dropped"

#: The shot ran and its cost is in the record, whether or not it is usable.
COMPLETE = "complete"


@dataclass(frozen=True)
class Observation:
    """One proposal, and what became of it.

    Args:
        shot_id: The identifier runmanager minted for this shot's queue row,
            written into the shot file as an attribute and read back by the
            routine. Unique within a session, and the key costs are matched to
            proposals by.
        params: The parameter vector that was requested, in real units, in
            :class:`~labscript_optimization.space.ParameterSpace` order.
        cost: The measured cost, or ``None`` while there is none. Lower is
            better; a maximised quantity has already had its sign flipped by
            the time it gets here.
        uncer: Standard error on ``cost``, or ``None`` when the configuration
            names no uncertainty column.
        bad: True when the shot ran but its cost is not usable.
        state: :data:`PENDING`, :data:`DROPPED` or :data:`COMPLETE`. The two
            without a cost are one thing to a learner -- a position spent that
            has produced nothing -- and two to the session, which counts the
            shots a run has lost, so the record keeps them apart rather than
            leaving the difference to be guessed from a missing cost.
    """

    shot_id: str
    params: np.ndarray
    cost: float | None
    uncer: float | None = None
    bad: bool = False
    state: str = COMPLETE

    def __post_init__(self) -> None:
        """Refuse a completed shot with no cost.

        ``state`` defaults to :data:`COMPLETE` because that is what almost
        every record is, and ``cost`` is optional because the two states that
        are not have no cost to carry. The pair a caller reaches by taking
        both defaults at their word -- a completed shot with ``cost=None`` --
        is the one combination that describes nothing: a shot that ran and
        produced nothing usable is bad, and a bad shot carries a cost of NaN
        and says so.
        """
        if self.state == COMPLETE and self.cost is None:
            raise ValueError(
                f"observation {self.shot_id!r} is complete and carries no "
                f"cost. A shot that has run and produced nothing usable is "
                f"bad, not costless: give it a cost of NaN with bad=True. A "
                f"proposal with no cost yet is {PENDING!r}, and one that will "
                f"never report is {DROPPED!r}."
            )

    @property
    def usable(self) -> bool:
        """Whether this observation can inform a fit.

        Only a completed shot has a cost to offer, and only a completed shot
        has one: the constructor refuses the other pairing, so the state is
        the whole of the question and a cost of ``None`` never reaches the
        finiteness test below. A non-finite cost is excluded along with a bad
        one: it carries no gradient information and would poison any statistic
        over the cost range.
        """
        if self.state != COMPLETE or self.bad:
            return False
        return bool(np.isfinite(self.cost))


def usable(history: Sequence[Observation]) -> list[Observation]:
    """The observations a learner may fit to, in proposal order."""
    return [o for o in history if o.usable]


def params_array(history: Sequence[Observation]) -> np.ndarray:
    """Parameters of ``history`` as an ``(n, num_params)`` array.

    An empty history has no parameter count to report, so it gives ``(0, 0)``.
    """
    return np.array([o.params for o in history], dtype=float)


def costs_array(history: Sequence[Observation]) -> np.ndarray:
    """Costs of ``history`` as an ``(n,)`` array."""
    return np.array([o.cost for o in history], dtype=float)


def uncers_array(history: Sequence[Observation]) -> np.ndarray | None:
    """Uncertainties of ``history``, or ``None`` if no observation has one.

    An observation without one gets a zero, so a mixed history claims some of
    its costs were measured exactly; see
    :class:`~labscript_optimization.learners.gaussian_process.GaussianProcessLearner`.
    """
    if not any(o.uncer is not None for o in history):
        return None
    return np.array([0.0 if o.uncer is None else o.uncer for o in history], dtype=float)


def best(history: Sequence[Observation]) -> Observation | None:
    """The usable observation with the lowest cost, or ``None`` if there is none."""
    candidates = usable(history)
    if not candidates:
        return None
    return min(candidates, key=lambda o: o.cost)
