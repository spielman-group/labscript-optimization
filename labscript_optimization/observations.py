"""The proposal record, and the history helpers learners share.

An :class:`Observation` is one proposal and what became of it. Learners are
handed every proposal a session has made, in order, so a position spent on a
shot that is still running or will never report is visible to them as spent.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Submitted, and a cost may still arrive.
PENDING = "pending"

#: No cost will arrive. The shot was canceled, deleted, refused, or is behind a
#: row only an operator can clear.
DROPPED = "dropped"

#: The shot ran and its cost is in the record, whether or not it is usable.
COMPLETE = "complete"


@dataclass(frozen=True)
class Observation:
    """One proposal, and what became of it.

    Parameters
    ----------
    shot_id : str or None
        The identifier runmanager minted for this shot's queue row. Unique in a
        session; costs are matched to proposals by it. ``None`` on a proposal not
        yet submitted: the configured start, in the history handed to the
        learner that places it.
    params : numpy.ndarray
        The parameter vector requested, in real units, in
        :class:`~labscript_optimization.space.ParameterSpace` order.
    cost : float or None
        The measured cost, or ``None`` while there is none. Lower is better; a
        maximized quantity has already had its sign flipped.
    uncer : float, optional
        Standard error on ``cost``; ``None`` when the configuration names no
        uncertainty column.
    bad : bool
        The shot ran but its cost is not usable.
    state : str
        :data:`PENDING`, :data:`DROPPED` or :data:`COMPLETE`. A learner treats
        the two without a cost alike, but the session counts the shots a run has
        lost, so the record keeps them apart.
    source : str, optional
        What proposed this shot, as recorded by the session at submission: the
        source the learner returned beside the proposal, or the session's own
        name for the configured start. The routine writes it onto the shot as
        its ``phase``. ``None`` for a record made outside a session.
    """

    shot_id: str | None
    params: np.ndarray
    cost: float | None
    uncer: float | None = None
    bad: bool = False
    state: str = COMPLETE
    source: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this observation can inform a fit.

        Only a completed shot with a finite cost that is not flagged bad can.
        """
        if self.state != COMPLETE or self.bad:
            return False
        return bool(np.isfinite(self.cost))


def usable(history: Sequence[Observation]) -> list[Observation]:
    """The observations a learner may fit to, in proposal order."""
    return [o for o in history if o.usable]


def params_array(history: Sequence[Observation]) -> np.ndarray:
    """Parameters of ``history`` as an ``(n, num_params)`` array.

    An empty history gives an empty array of shape ``(0,)``.
    """
    return np.array([o.params for o in history], dtype=float)


def costs_array(history: Sequence[Observation]) -> np.ndarray:
    """Costs of ``history`` as an ``(n,)`` array."""
    return np.array([o.cost for o in history], dtype=float)


def uncers_array(history: Sequence[Observation]) -> np.ndarray | None:
    """Uncertainties of ``history``, zero where missing; ``None`` if none has one."""
    if not any(o.uncer is not None for o in history):
        return None
    return np.array([0.0 if o.uncer is None else o.uncer for o in history], dtype=float)


def best(history: Sequence[Observation]) -> Observation | None:
    """The usable observation with the lowest cost, or ``None`` if there is none."""
    candidates = usable(history)
    if not candidates:
        return None
    return min(candidates, key=lambda o: o.cost)
