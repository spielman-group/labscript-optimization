"""Gaussian process learner.

A scikit-learn :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit
to the scaled history, and a multi-start L-BFGS-B search over its posterior for
the next point. Carried over from M-LOOP, minus the process and queue
scaffolding that surrounded it there.

Exploration comes from the acquisition, which minimises

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

with ``uncer_bias`` stepping 0, 1, 2, ... across the points of one batch, so a
batch always contains one point that is purely greedy and others that trade
predicted cost for a look somewhere less certain.

Refitting the kernel hyperparameters is the expensive part, so it happens once
per ``generation_size`` new observations rather than on every call. Between
refits the posterior is still refit to all the data; only the hyperparameter
search is skipped. That is a cost control, and the only thing a learner is
permitted to remember between calls.
"""

from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..observations import Observation, costs_array, params_array, uncers_array, usable
from .base import InsufficientData
from ..space import ParameterSpace


class GaussianProcessLearner:
    """Fit a Gaussian process to the history and search its posterior.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        cost_has_noise: Add a white-noise term to the kernel. Leave this on for
            real data; turning it off asserts the cost is measured exactly.
        length_scale_bounds: Bounds on the RBF length scale, in units of the
            unit cube the parameters are scaled onto.
        noise_level_bounds: Bounds on the white-noise level, in units of the
            standardised cost.
        cost_bias: Weight on predicted cost in the acquisition.
        uncer_bias: Weight on predicted uncertainty for the first point of a
            batch; later points step up by one each.
        generation_size: How many new observations to accept before refitting
            the kernel hyperparameters.
        trust_region: Restrict the search to this distance around the best
            point seen.
        minimum_observations: Refuse to propose until the history holds this
            many usable observations, so the two-phase wrapper keeps using its
            trainer. Defaults to twice the parameter count.
    """

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
        self.space = space
        self.rng = rng
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
        self._epoch = None
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

    def fit(self, history: Sequence[Observation]) -> bool:
        """Fit the regressor to ``history``. Returns whether a fit was possible."""
        seen = usable(history)
        if len(seen) < self.minimum_observations:
            return False

        x = self.space.scale(params_array(seen))
        costs = costs_array(seen).reshape(-1, 1)

        self.cost_scaler = StandardScaler().fit(costs)
        y = self.cost_scaler.transform(costs).ravel()

        uncers = uncers_array(seen)
        if uncers is None:
            alpha = 1e-10
        else:
            # Per-point variances, in the same standardised units as y.
            alpha = (uncers / self.cost_scaler.scale_[0]) ** 2

        # Refit hyperparameters once per generation; reuse them in between.
        epoch = len(seen) // self.generation_size
        refit = self._kernel is None or epoch != self._epoch
        regressor = GaussianProcessRegressor(
            kernel=self._new_kernel() if refit else self._kernel,
            alpha=alpha,
            normalize_y=False,
            optimizer="fmin_l_bfgs_b" if refit else None,
            n_restarts_optimizer=self.num_restarts if refit else 0,
            random_state=int(self.rng.integers(2**32)),
        )
        regressor.fit(x, y)
        if refit:
            self._kernel = regressor.kernel_
            self._epoch = epoch
        self.regressor = regressor
        self._x, self._y, self._alpha = x, y, alpha
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
        low, high = self.space.minimum, self.space.maximum
        if self.trust_region is not None:
            low = np.maximum(low, centre - self.trust_region)
            high = np.minimum(high, centre + self.trust_region)
        return list(zip(self.space.scale(low), self.space.scale(high)))

    def _minimise_acquisition(self, uncer_bias: float, best: np.ndarray, bounds):
        """Multi-start L-BFGS-B over the acquisition. Returns scaled parameters."""

        def acquisition(u):
            mean, std = self.regressor.predict(np.atleast_2d(u), return_std=True)
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

    def _condition_on(self, scaled_point: np.ndarray) -> None:
        """Fold a point into the fit at its own predicted cost.

        Posterior variance depends on where a point was measured, not on what
        came back, so adding a proposal at its predicted mean shrinks the
        uncertainty around it exactly as the real shot will. Without this every
        point in a batch with a high uncertainty weight chases the same
        unexplored corner and the batch is spent on one location.
        """
        mean = self.regressor.predict(np.atleast_2d(scaled_point))
        self._x = np.vstack([self._x, scaled_point])
        self._y = np.append(self._y, mean)
        if not np.isscalar(self._alpha):
            # A fantasy point is as certain as the data it was predicted from.
            self._alpha = np.append(self._alpha, np.median(self._alpha))
        regressor = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._alpha,
            normalize_y=False,
            optimizer=None,
        )
        regressor.fit(self._x, self._y)
        self.regressor = regressor

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        if not self.fit(history):
            raise InsufficientData(
                f"the Gaussian process needs {self.minimum_observations} usable "
                f"observations and has {len(usable(history))}"
            )
        seen = usable(history)
        best = params_array(seen)[int(np.argmin(costs_array(seen)))]
        bounds = self._search_bounds(best)
        fitted = self.regressor

        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            scaled = self._minimise_acquisition(self.uncer_bias * i, best, bounds)
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                self._condition_on(scaled)

        # Leave the learner describing the real data, not the fantasy points,
        # so predict() and global_minimum() answer about what was measured.
        self.regressor = fitted
        return proposals

    def global_minimum(self, history: Sequence[Observation]) -> np.ndarray:
        """The posterior's unbiased minimum over the whole space.

        This is the answer to "where does the model think the optimum is",
        which is not the same as the next point to try.
        """
        if not self.fit(history):
            raise InsufficientData("no fit is possible yet")
        seen = usable(history)
        best = params_array(seen)[int(np.argmin(costs_array(seen)))]
        bounds = list(zip(self.space.scale(self.space.minimum), self.space.scale(self.space.maximum)))
        return self.space.clip(self.space.unscale(self._minimise_acquisition(0.0, best, bounds)))
