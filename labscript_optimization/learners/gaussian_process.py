"""Gaussian process learner.

A scikit-learn :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit
to the scaled history, and a multi-start L-BFGS-B search over its posterior for
the next point.

Exploration comes from the acquisition, which minimises

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

with the weight on the uncertainty stepping 0, 1, 2, ... across successive
proposals and returning to zero every ``generation_size`` of them. One
proposal per generation is therefore purely greedy and the rest trade
predicted cost for a look somewhere less certain. The step is read off the
history, so it advances with the proposals a session makes rather than with a
point's position within a batch.

Refitting the kernel hyperparameters is the expensive part, so it happens once
per ``generation_size`` new observations rather than on every call; the
posterior is refit to all the data every time.
"""

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..observations import (
    Observation,
    best,
    costs_array,
    params_array,
    uncers_array,
    usable,
)
from .base import InsufficientData, Learner
from ..space import ParameterSpace


class GaussianProcessLearner(Learner):
    """Fit a Gaussian process to the history and search its posterior.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        cost_has_noise: Add a white-noise term to the kernel. Leave this on for
            real data; turning it off asserts the cost is measured exactly.
            With it off, a history in which only some observations carry an
            uncertainty gives the rest an ``alpha`` of exactly zero and there
            is no jitter anywhere else, so two shots at the same parameter
            vector make the covariance matrix singular and the fit raises
            ``numpy.linalg.LinAlgError``.
        length_scale_bounds: Bounds on the RBF length scale, in units of the
            unit cube the parameters are scaled onto.
        noise_level_bounds: Bounds on the white-noise level, in units of the
            standardised cost.
        cost_bias: Weight on predicted cost in the acquisition.
        uncer_bias: Weight on predicted uncertainty, one step of the
            exploration schedule described above. Raising it buys a wider look
            without moving each generation's greedy proposal.
        generation_size: How many proposals a generation holds: the period of
            that schedule, and the number of new observations accepted before
            the kernel hyperparameters are refit.
        trust_region: Restrict the search to this distance around the best
            point seen.
        minimum_observations: Refuse to propose until the history holds this
            many usable observations, so a two-phase wrapper keeps using its
            trainer. Defaults to twice the parameter count.
    """

    last_phase = "main"

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        cost_has_noise: bool = True,
        length_scale_bounds: Sequence[float] = (1e-2, 1e2),
        noise_level_bounds: Sequence[float] = (1e-5, 1e1),
        cost_bias: float = 1.0,
        uncer_bias: float = 1.0,
        generation_size: int = 4,
        trust_region=None,
        minimum_observations: int | None = None,
    ):
        super().__init__(space, rng)
        self.cost_has_noise = bool(cost_has_noise)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.noise_level_bounds = tuple(noise_level_bounds)
        self.cost_bias = float(cost_bias)
        self.uncer_bias = float(uncer_bias)
        self.generation_size = int(generation_size)
        if self.generation_size < 1:
            raise ValueError(
                f"generation_size must be at least 1, got {self.generation_size}"
            )
        self.trust_region = space.absolute_trust_region(trust_region)
        self.minimum_observations = (
            2 * space.num_params
            if minimum_observations is None
            else int(minimum_observations)
        )
        self.num_restarts = max(10, space.num_params)

        self._kernel = None
        self._fitted_to = None
        self.regressor = None
        self.cost_scaler = None

    def _new_kernel(self):
        kernel = RBF(
            length_scale=np.ones(self.space.num_params),
            length_scale_bounds=self.length_scale_bounds,
        )
        if self.cost_has_noise:
            kernel = kernel + WhiteKernel(
                noise_level=1.0, noise_level_bounds=self.noise_level_bounds
            )
        return kernel

    def _alpha(self, seen: Sequence[Observation], scaler):
        """Per-point variances for the regressor, in standardised cost units.

        An observation with no uncertainty of its own gets zero here rather
        than the scalar floor, because ``uncers_array`` fills it in as exact.
        """
        uncers = uncers_array(seen)
        if uncers is None:
            return 1e-10
        return (uncers / scaler.scale_[0]) ** 2

    def _fit_hyperparameters(self, prefix: Sequence[Observation]):
        """Fit the cost scaling and the kernel hyperparameters to ``prefix``.

        The scaling belongs with them because it sets the units the noise level
        is measured in: restandardising as each observation arrived would leave
        a cached kernel describing units that had since moved. The restart
        draws are seeded from the length of ``prefix``, not from the learner's
        rng, which would make the search depend on how much this instance had
        already proposed.
        """
        costs = costs_array(prefix).reshape(-1, 1)
        scaler = StandardScaler().fit(costs)
        regressor = GaussianProcessRegressor(
            kernel=self._new_kernel(),
            alpha=self._alpha(prefix, scaler),
            normalize_y=False,
            n_restarts_optimizer=self.num_restarts,
            random_state=len(prefix),
        )
        regressor.fit(
            self.space.scale(params_array(prefix)), scaler.transform(costs).ravel()
        )
        return scaler, regressor.kernel_

    def fit(self, history: Sequence[Observation]) -> bool:
        """Fit the regressor to ``history``. Returns whether a fit was possible."""
        seen = usable(history)
        if len(seen) < self.minimum_observations:
            return False

        # Hyperparameters come from whole generations, so that an instance that
        # has been fitting all session holds the kernel a fresh one handed the
        # same history computes, and what it keeps is a cache. Short of one
        # generation there is nothing to hold back.
        whole = len(seen) - len(seen) % self.generation_size
        prefix = seen[:whole] if whole else seen
        # Keyed on which observations they were fitted to and not how many: a
        # cost arriving late lands in proposal order and rewrites a prefix of
        # unchanged length.
        fitted_to = tuple(o.shot_id for o in prefix)
        if fitted_to != self._fitted_to:
            self.cost_scaler, self._kernel = self._fit_hyperparameters(prefix)
            self._fitted_to = fitted_to

        costs = costs_array(seen).reshape(-1, 1)
        regressor = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._alpha(seen, self.cost_scaler),
            normalize_y=False,
            optimizer=None,
        )
        regressor.fit(
            self.space.scale(params_array(seen)),
            self.cost_scaler.transform(costs).ravel(),
        )
        self.regressor = regressor
        return True

    def predict(self, params: np.ndarray):
        """Predicted cost and standard deviation at ``params``, in real units."""
        if self.regressor is None:
            raise RuntimeError("the learner has not been fit to any history yet")
        params = np.atleast_2d(params)
        mean, std = self.regressor.predict(self.space.scale(params), return_std=True)
        scale = self.cost_scaler.scale_[0]
        return (
            self.cost_scaler.inverse_transform(mean.reshape(-1, 1)).ravel(),
            std * scale,
        )

    def _search_bounds(self, centre: np.ndarray):
        """Scaled bounds for the minimiser, narrowed by the trust region."""
        low, high = self.space.bounds_near(centre, self.trust_region)
        return list(zip(self.space.scale(low), self.space.scale(high)))

    def _minimise_acquisition(
        self, regressor, uncer_bias: float, best: np.ndarray, bounds
    ):
        """Multi-start L-BFGS-B over the acquisition. Returns scaled parameters."""

        def acquisition(u):
            mean, std = regressor.predict(np.atleast_2d(u), return_std=True)
            return self.cost_bias * mean[0] - uncer_bias * std[0]

        lows = np.array([b[0] for b in bounds])
        highs = np.array([b[1] for b in bounds])
        starts = [self.space.scale(best)]
        starts.extend(self.rng.uniform(lows, highs, size=(self.num_restarts, len(bounds))))

        winner, winning_value = None, np.inf
        for start in starts:
            result = minimize(
                acquisition,
                np.clip(start, lows, highs),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if result.fun < winning_value:
                winner, winning_value = result.x, result.fun
        return np.clip(winner, lows, highs)

    def _condition_on(self, regressor, scaled_point: np.ndarray):
        """A copy of ``regressor`` with a point folded in at its predicted cost.

        Posterior variance depends on where a point was measured, not on what
        came back, so adding a proposal at its predicted mean shrinks the
        uncertainty around it exactly as the real shot will. Without this every
        point in a batch with a high uncertainty weight chases the same
        unexplored corner and the batch is spent on one location.

        Returned rather than stored, so that invented points never reach the
        learner and :meth:`predict` describes the measured data however a batch
        turns out.
        """
        mean = regressor.predict(np.atleast_2d(scaled_point))
        alpha = regressor.alpha
        if not np.isscalar(alpha):
            # A fantasy point is as certain as the data it was predicted from.
            alpha = np.append(alpha, np.median(alpha))
        conditioned = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=alpha,
            normalize_y=False,
            optimizer=None,
        )
        conditioned.fit(
            np.vstack([regressor.X_train_, scaled_point]),
            np.append(regressor.y_train_, mean),
        )
        return conditioned

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        if not self.fit(history):
            raise InsufficientData(
                f"the Gaussian process needs {self.minimum_observations} usable "
                f"observations and has {len(usable(history))}"
            )
        seen = usable(history)
        best_params = best(history).params
        bounds = self._search_bounds(best_params)

        # The points of this batch are folded into a fit held here and nowhere
        # else, so the learner goes on describing the measured data.
        regressor = self.regressor
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            # Where the exploration schedule stands, counted off the history
            # and then the points picked so far in this batch. A schedule kept
            # as a counter over the batch would sit at its greedy first step
            # for ever in a session that settles into asking for one point at
            # a time, and uncer_bias would do nothing whatever.
            step = (len(seen) + i) % self.generation_size
            scaled = self._minimise_acquisition(
                regressor, self.uncer_bias * step, best_params, bounds
            )
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                regressor = self._condition_on(regressor, scaled)
        return proposals
