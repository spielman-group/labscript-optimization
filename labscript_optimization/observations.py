"""The observation record and the history helpers learners share.

An :class:`Observation` is one completed shot: the parameters that were
requested, the cost that came back, and the shot id that ties the two
together.
Learners never see a shot that has not reported a cost yet; the worker holds
those back and keeps the history in the order the proposals were made.
"""

from typing import NamedTuple, Sequence

import numpy as np


class Observation(NamedTuple):
    """One completed shot.

    Args:
        shot_id: The identifier runmanager minted for this shot's queue row,
            written into the shot file as an attribute and read back by the
            routine. Unique within a session, and the key costs are matched to
            proposals by.
        params: The parameter vector that was requested, in real units, in
            :class:`~labscript_optimization.space.ParameterSpace` order.
        cost: The measured cost. Lower is better; a maximised quantity has
            already had its sign flipped by the time it gets here.
        uncer: Standard error on ``cost``, or ``None`` when the configuration
            names no uncertainty column.
        bad: True when the shot ran but its cost is not usable.
    """

    shot_id: str
    params: np.ndarray
    cost: float
    uncer: float | None = None
    bad: bool = False

    @property
    def usable(self) -> bool:
        """Whether this observation can inform a fit.

        A bad shot is excluded by definition. A non-finite cost is excluded
        because it carries no gradient information and, left in, would poison
        any statistic computed over the cost range.
        """
        return not self.bad and bool(np.isfinite(self.cost))


def usable(history: Sequence[Observation]) -> list[Observation]:
    """The observations a learner may fit to, in proposal order."""
    return [o for o in history if o.usable]


def params_array(history: Sequence[Observation]) -> np.ndarray:
    """Parameters of ``history`` as an ``(n, num_params)`` array."""
    if not history:
        return np.empty((0, 0))
    return np.array([o.params for o in history], dtype=float)


def costs_array(history: Sequence[Observation]) -> np.ndarray:
    """Costs of ``history`` as an ``(n,)`` array."""
    return np.array([o.cost for o in history], dtype=float)


def uncers_array(history: Sequence[Observation]) -> np.ndarray | None:
    """Uncertainties of ``history``, or ``None`` if no observation has one.

    A history that mixes observations with and without an uncertainty gets
    zeros for the ones without, which is what the regressor's ``alpha`` means
    for a point measured exactly.
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
