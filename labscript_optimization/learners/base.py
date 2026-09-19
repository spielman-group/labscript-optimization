"""What a learner is.

A learner is a function from the proposal history to ``k`` proposals. It is
given the whole history every time, in the order the proposals were made, and
holding every one of them: a shot still running and a shot that will never
report are both in there as a position spent without a usable cost. Anything a
learner computes from the history and keeps between calls is a cache: it must
hold what an instance handed the same history for the first time would
compute. The one
thing a learner carries that the history does not fix is the position of its
own rng stream, which decides where a draw lands and nothing about what the
learner believes. That rule is what keeps the out-of-order arrival of costs out
of the learners.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..observations import Observation
from ..space import ParameterSpace


class Learner(ABC):
    """Proposes the next parameters to try.

    This is the whole of what a session drives. A learner that wraps other
    learners is one of these too, so anything holding a learner can hold a
    wrapped one without knowing it.

    Inheriting is not what makes an object usable -- a session proposes from
    anything carrying these three members -- but everything this package calls
    a learner is one.
    """

    #: Which of the learner's ways of proposing produced the last batch, which
    #: the session reports. Declared without a value: a learner that grows
    #: phases and forgets to publish them then fails, rather than being
    #: reported as the main one for a whole run.
    last_phase: str

    #: How many proposals this learner is asked for at a time, or ``None`` for
    #: any number. A learner that declares one proposes only whole groups of
    #: that many, and only when none of its proposals is outstanding, so a
    #: session asks it for a whole generation or for nothing at all. A
    #: declaration of the same kind as :attr:`last_phase`, not a delivery
    #: policy: ``propose`` is unchanged, and whether anything is outstanding
    #: stays runmanager's answer rather than a count kept anywhere here.
    generation: int | None = None

    @abstractmethod
    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """Return ``k`` parameter vectors to run next.

        Args:
            history: Every proposal made so far, in the order they were made,
                each with what became of it.
            k: How many proposals to return.

        Returns:
            A ``(k, num_params)`` array, every row inside the bounds.
        """


class ParameterSpaceLearner(Learner):
    """A learner that searches a :class:`ParameterSpace` of its own.

    Fixes how one is built: ``space`` first and ``rng`` second, positionally,
    then the learner's own knobs as keyword arguments with defaults. Every
    learner named in a configuration is built as ``cls(space, rng, **options)``
    with the options matched by name against the signature -- so the two
    spelled the other way round searches the wrong thing, and knobs collected
    in ``**kwargs`` are matched by nothing.

    Nothing else is shared. A learner keeps its own state, resolves its own
    opening point, and is free not to have one.
    """

    def __init__(self, space: ParameterSpace, rng: np.random.Generator):
        if not isinstance(space, ParameterSpace) or not isinstance(
            rng, np.random.Generator
        ):
            raise TypeError(
                f"a learner takes a ParameterSpace and a Generator, in that "
                f"order; got {type(space).__name__} and {type(rng).__name__}"
            )
        self.space = space
        self.rng = rng


class InsufficientData(RuntimeError):
    """A learner cannot propose from the history it has been given.

    Raised rather than returning a fallback, so that a wrapper decides what to
    do; a bare learner reaching this state is a configuration error.
    """


def opening_point(
    space: ParameterSpace, first_params: np.ndarray | None
) -> np.ndarray | None:
    """Resolve where a learner starts, or ``None`` to start from a draw.

    ``first_params`` overrides the space's configured start. Both are checked
    here rather than at the first proposal, because construction is the last
    moment at which a bad setting costs nothing to fix; the alternative is
    finding out once the apparatus is running.
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

    A starting point is the first thing proposed and only that: the history
    holds a proposal from the moment it is made, so the second call sees the
    first batch whether or not any of it has reported. The rest of the opening
    batch is drawn uniformly, since a learner with no history has nothing
    better to go on.
    """
    if history or first_params is None:
        return None
    proposals = space.uniform(rng, k)
    proposals[0] = first_params
    return proposals
