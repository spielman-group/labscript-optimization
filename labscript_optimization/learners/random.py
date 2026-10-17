"""Random and directed random learners.

:class:`RandomLearner` draws uniformly from the whole space, the reference every
other learner is measured against. :class:`DirectedRandomLearner` draws near a
previously seen point, which biases it towards exploring; it is the explorer the
Gaussian process runs unless told otherwise.

Neither declares a generation, so both keep the hint's number of the run's shots
in flight, each proposing as many as the hint leaves room for beside the
history's pending records, whoever proposed them. They differ only in
:meth:`RandomLearner.ask`.
"""

from collections.abc import Sequence

import numpy as np

from .. import knobs
from ..observations import Observation, costs_array, params_array, usable
from ..space import ParameterSpace
from .base import ParameterSpaceLearner


class RandomLearner(ParameterSpaceLearner):
    """Uniform random draws from the whole space.

    Parameters
    ----------
    space : ParameterSpace
        The parameter space to search.
    rng : numpy.random.Generator
        Source of randomness.
    """

    #: It draws without reading the history.
    history_scope = "none"

    def ask(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """The next ``k`` points as a ``(k, num_params)`` array, without the pacing."""
        return self.space.uniform(self.rng, k)


class DirectedRandomLearner(RandomLearner):
    """Random draws centered on a previously seen point.

    Each proposal is a draw from the whole space with probability
    ``explore_fraction``, otherwise a draw from the trust region centered on one
    of the observations whose cost lies inside ``trust_range``. That range is a
    fraction of the way from the worst cost seen (0) to the best (1), so
    ``[1, 1]`` centers on the best point. The default sits near the worst end,
    which is what makes this learner an explorer.

    Parameters
    ----------
    space : ParameterSpace
        The parameter space to search.
    rng : numpy.random.Generator
        Source of randomness.
    trust_region : float or sequence of float or None
        Largest distance from the center. A float in (0, 1) is a fraction of
        each parameter's range; a sequence is absolute distances. ``None``
        searches the whole space, as :class:`RandomLearner` does.
    trust_range : (float, float)
        Ordered pair of fractions bounding which observations may be the center.
    trust_gaussian : bool
        Draw from a Gaussian of width ``trust_region`` about the center instead
        of uniformly within it.
    explore_fraction : float
        Share of proposals, in [0, 1], that ignore the trust region and draw
        from the whole space.
    """

    #: It centers its draws on the shots seen, whoever proposed them.
    history_scope = "all"

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        trust_region=0.05,
        trust_range: Sequence[float] = (0.1, 0.25),
        trust_gaussian: bool = False,
        explore_fraction: float = 0.0,
    ):
        super().__init__(space, rng)
        self.trust_region = space.absolute_trust_region(trust_region)
        self.trust_gaussian = knobs.boolean("trust_gaussian", trust_gaussian)

        self.explore_fraction = knobs.number("explore_fraction", explore_fraction)
        if not 0 <= self.explore_fraction <= 1:
            raise ValueError(
                f"explore_fraction must be in [0, 1], got {self.explore_fraction}"
            )

        self.trust_range = knobs.pair("trust_range", trust_range)
        # Refused, not sorted: a pair written backwards misreads which end is
        # which, and sorting it would quietly run a search nobody asked for.
        if not 0 <= self.trust_range[0] <= self.trust_range[1] <= 1:
            raise ValueError(
                f"trust_range must be an ordered pair within [0, 1], got "
                f"{trust_range!r}"
            )

    def _center(self, params: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """A point in the cost band to draw around, else the best point."""
        best, worst = costs.min(), costs.max()
        high = worst + (best - worst) * self.trust_range[0]
        low = worst + (best - worst) * self.trust_range[1]
        inside = (low <= costs) & (costs <= high)
        if not inside.any():
            return params[costs.argmin()]
        candidates = params[inside]
        return candidates[self.rng.integers(len(candidates))]

    def _draw_near(self, center: np.ndarray) -> np.ndarray:
        if self.trust_gaussian:
            return self.space.clip(self.rng.normal(center, self.trust_region))
        return self.space.uniform(self.rng, 1, center, self.trust_region)[0]

    def ask(self, history: Sequence[Observation], k: int) -> np.ndarray:
        seen = usable(history)
        if not seen or self.trust_region is None:
            return self.space.uniform(self.rng, k)

        params, costs = params_array(seen), costs_array(seen)
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            if self.rng.uniform() < self.explore_fraction:
                proposals[i] = self.space.uniform(self.rng, 1)[0]
            else:
                proposals[i] = self._draw_near(self._center(params, costs))
        return proposals
