"""What a learner is.

A learner is a function from the observation history to ``k`` proposals. It is
given the whole history every time, in the order the proposals were made, and
containing only shots that have reported a cost. Anything a learner computes
from the history and keeps between calls is a cache: it must hold what an
instance handed the same history for the first time would compute. The one
thing a learner carries that the history does not fix is the position of its
own rng stream, which decides where a draw lands and nothing about what the
learner believes.

That rule is what keeps the out-of-order arrival of costs out of the learners.
The worker holds back shots that have no cost yet and orders the rest; a
learner never reasons about a shot in flight.

Beside the protocol are the pieces the learners have to agree on rather than
each answer for themselves: how a learner says it cannot propose, and where a
run begins.
"""

from typing import Protocol, Sequence

import numpy as np

from ..observations import Observation
from ..space import ParameterSpace


class Learner(Protocol):
    """Proposes the next parameters to try."""

    #: Which of the learner's ways of proposing produced the last batch. The
    #: session reports it, so every learner has to answer: one that proposes
    #: the same way every time says ``"main"`` and never changes it, and a
    #: wrapper says which of the learners it holds was asked.
    last_phase: str = "main"

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """Return ``k`` parameter vectors to run next.

        Args:
            history: Completed observations, in proposal order.
            k: How many proposals to return.

        Returns:
            A ``(k, num_params)`` array, every row inside the bounds.
        """
        ...


class InsufficientData(RuntimeError):
    """A learner cannot propose from the history it has been given.

    Raised rather than returning a fallback so that a wrapper can decide what
    to do; a bare learner reaching this state is a configuration error.
    """


def opening_point(
    space: ParameterSpace, first_params: np.ndarray | None
) -> np.ndarray | None:
    """Resolve where a learner starts, or ``None`` to start from a draw.

    ``first_params`` overrides the space's configured start. Both are checked
    here rather than at the first proposal, because construction is the last
    moment at which a bad starting point costs nothing to fix: the alternative
    is finding out once the apparatus is running.
    """
    if first_params is None:
        first_params = space.start
    if first_params is None:
        return None
    point = np.array(first_params, dtype=float)
    if point.shape != (space.num_params,):
        raise ValueError(
            f"first_params is one point over {space.num_params} parameters, "
            f"so it needs shape ({space.num_params},), got {point.shape}"
        )
    if not space.contains(point).all():
        raise ValueError(f"first_params outside the bounds: {point}")
    return point


def opening_batch(
    space: ParameterSpace,
    rng: np.random.Generator,
    history: Sequence[Observation],
    k: int,
    first_params: np.ndarray | None,
) -> np.ndarray | None:
    """The opening batch, or ``None`` when this is not the opening call.

    A starting point is the first thing proposed and only that; the rest of
    the opening batch is drawn uniformly, since a learner with no history has
    nothing better to go on. Returning ``None`` rather than an empty batch
    keeps the test for "has anything run yet" in one place instead of in every
    learner that can be told where to start.
    """
    if history or first_params is None:
        return None
    proposals = space.uniform(rng, k)
    proposals[0] = first_params
    return proposals
