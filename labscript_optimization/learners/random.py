"""Random and directed random learners.

:class:`RandomLearner` draws uniformly from the whole space. It is the simplest
thing that works and the reference every other learner is measured against.

:class:`DirectedRandomLearner` draws near a previously seen point rather than
near the best one, which biases it towards exploring the space instead of
refining a single minimum. It is the trainer the Gaussian process runs first.
"""

from typing import Sequence

import numpy as np

from ..observations import Observation, costs_array, params_array, usable
from ..space import ParameterSpace
from .base import ParameterSpaceLearner, opening_batch, opening_point


class RandomLearner(ParameterSpaceLearner):
    """Uniform random draws from the whole space.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        first_params: A point to return as the very first proposal, or ``None``
            to start from a random draw. Defaults to the space's configured
            start.
    """

    last_phase = "main"

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        first_params: np.ndarray | None = None,
    ):
        super().__init__(space, rng)
        self.first_params = opening_point(space, first_params)

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        opening = opening_batch(
            self.space, self.rng, history, k, self.first_params
        )
        if opening is not None:
            return opening
        return self.space.uniform(self.rng, k)


class DirectedRandomLearner(ParameterSpaceLearner):
    """Random draws centred on a previously seen point.

    Each proposal is either a pure random draw, with probability
    ``explore_fraction``, or a draw from a trust region centred on one of the
    observations whose cost falls inside ``trust_range``.

    ``trust_range`` is measured as a fraction of the way from the worst cost
    seen to the best, so ``[1, 1]`` centres on the best point and values near
    zero centre on poor ones. The default sits near the worst end: spreading
    the search over mediocre points is what makes this learner an explorer.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        trust_region: Maximum distance from the centre point. A float in (0, 1)
            is a fraction of each parameter's range; a sequence is absolute
            distances. ``None`` searches the whole space, which makes this
            learner equivalent to :class:`RandomLearner`.
        trust_range: Two fractions, in order, bounding which observations may
            be chosen as the centre.
        trust_gaussian: Draw from a Gaussian of width ``trust_region`` about
            the centre instead of uniformly within it.
        explore_fraction: Share of proposals that ignore the trust region and
            draw from the whole space.
        first_params: A point to return as the very first proposal.
    """

    last_phase = "main"

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        trust_region=0.05,
        trust_range: Sequence[float] = (0.1, 0.25),
        trust_gaussian: bool = False,
        explore_fraction: float = 0.0,
        first_params: np.ndarray | None = None,
    ):
        super().__init__(space, rng)
        self.trust_region = space.absolute_trust_region(trust_region)
        self.trust_gaussian = bool(trust_gaussian)

        self.explore_fraction = float(explore_fraction)
        if not 0 <= self.explore_fraction <= 1:
            raise ValueError(
                f"explore_fraction must be in [0, 1], got {self.explore_fraction}"
            )

        if len(trust_range) != 2:
            raise ValueError(f"trust_range needs two values, got {trust_range!r}")
        # Refused rather than sorted: a pair written backwards is a
        # misunderstanding of which end is which, and putting it in order
        # quietly runs a search the lab did not ask for.
        if not 0 <= trust_range[0] <= trust_range[1] <= 1:
            raise ValueError(
                f"trust_range must be an ordered pair within [0, 1], got "
                f"{trust_range!r}"
            )
        self.trust_range = tuple(float(t) for t in trust_range)

        self.first_params = opening_point(space, first_params)

    def centre(self, params: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Pick the point to draw around.

        The band is a slice out of the middle of the observed cost range, so a
        history whose costs all lie outside it -- two observations, one at each
        end -- leaves nothing to choose from, and the best point is used. A
        single observation is not that case: it is both best and worst, which
        collapses the band onto it.
        """
        best, worst = costs.min(), costs.max()
        high = worst + (best - worst) * self.trust_range[0]
        low = worst + (best - worst) * self.trust_range[1]
        inside = (low <= costs) & (costs <= high)
        if not inside.any():
            return params[costs.argmin()]
        candidates = params[inside]
        return candidates[self.rng.integers(len(candidates))]

    def draw_near(self, centre: np.ndarray) -> np.ndarray:
        if self.trust_gaussian:
            return self.space.clip(self.rng.normal(centre, self.trust_region))
        return self.space.uniform(self.rng, 1, centre, self.trust_region)[0]

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        opening = opening_batch(
            self.space, self.rng, history, k, self.first_params
        )
        if opening is not None:
            return opening

        seen = usable(history)
        if not seen or self.trust_region is None:
            return self.space.uniform(self.rng, k)

        params, costs = params_array(seen), costs_array(seen)
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            if self.rng.uniform() < self.explore_fraction:
                proposals[i] = self.space.uniform(self.rng, 1)[0]
            else:
                proposals[i] = self.draw_near(self.centre(params, costs))
        return proposals
