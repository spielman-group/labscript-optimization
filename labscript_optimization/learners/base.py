"""What a learner is.

A learner is a function from the observation history to ``k`` proposals. It is
given the whole history every time, in the order the proposals were made, and
containing only shots that have reported a cost. Anything a learner remembers
between calls is a cache: it must produce the same proposals as an instance
that had just been handed the same history for the first time.

That rule is what keeps the out-of-order arrival of costs out of the learners.
The worker holds back shots that have no cost yet and orders the rest; a
learner never reasons about a shot in flight.
"""

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..observations import Observation


@runtime_checkable
class Learner(Protocol):
    """Proposes the next parameters to try."""

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
