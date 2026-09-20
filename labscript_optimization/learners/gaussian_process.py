"""Gaussian process learner.

A scikit-learn :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit
to the scaled history, and a multi-start L-BFGS-B search over its posterior for
the next point.

Exploration comes from the acquisition, which minimises

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

with the weight on the uncertainty stepping 0, 1, 2, ... across successive
proposals and returning to zero every ``batch_size`` of them. One proposal in
each batch is therefore purely greedy and the rest trade predicted cost for a
look somewhere less certain. The step is read off the history, so it advances
with the proposals a session makes rather than with a point's position within
the group of them asked for at once.

Refitting the kernel hyperparameters is the expensive part, so it happens once
per ``batch_size`` new observations rather than on every call; the
posterior is refit to all the data every time.

scikit-learn and scipy are imported where they are used rather than at the top
of this module, and nothing here reaches a point of use until a proposal is
asked for. Loading a configuration resolves the class of the learner it names,
to hold the file's knobs to the constructor, and builds it, to ask how many
proposals it makes at a time -- and that happens in the lyse routine's own
process, which never asks the learner for a proposal and should not pay
several seconds to import the scientific stack.
"""

from typing import Sequence

import numpy as np

from ..observations import (
    Observation,
    best,
    costs_array,
    params_array,
    uncers_array,
    usable,
)
from .base import InsufficientData, ParameterSpaceLearner
from ..space import ParameterSpace


class GaussianProcessLearner(ParameterSpaceLearner):
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
            without moving each batch's greedy proposal.
        batch_size: How many proposals a batch holds: the period of that
            schedule, and the number of new observations accepted before the
            kernel hyperparameters are refit.
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
        batch_size: int = 4,
        trust_region=None,
        minimum_observations: int | None = None,
    ):
        super().__init__(space, rng)
        self.cost_has_noise = bool(cost_has_noise)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.noise_level_bounds = tuple(noise_level_bounds)
        self.cost_bias = float(cost_bias)
        self.uncer_bias = float(uncer_bias)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.batch_size}"
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
        self._cost_scaler = None

    def new_kernel(self):
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = RBF(
            length_scale=np.ones(self.space.num_params),
            length_scale_bounds=self.length_scale_bounds,
        )
        if self.cost_has_noise:
            kernel = kernel + WhiteKernel(
                noise_level=1.0, noise_level_bounds=self.noise_level_bounds
            )
        return kernel

    def point_variances(self, seen: Sequence[Observation], scaler):
        """Per-point variances for the regressor, in standardised cost units.

        An observation with no uncertainty of its own gets zero here rather
        than the scalar floor, because ``uncers_array`` fills it in as exact.
        """
        uncers = uncers_array(seen)
        if uncers is None:
            return 1e-10
        return (uncers / scaler.scale_[0]) ** 2

    def fit_hyperparameters(self, prefix: Sequence[Observation]):
        """Fit the cost scaling and the kernel hyperparameters to ``prefix``.

        The scaling belongs with them because it sets the units the noise level
        is measured in: restandardising as each observation arrived would leave
        a cached kernel describing units that had since moved. The restart
        draws are seeded from the length of ``prefix``, not from the learner's
        rng, which would make the search depend on how much this instance had
        already proposed.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler

        costs = costs_array(prefix).reshape(-1, 1)
        scaler = StandardScaler().fit(costs)
        regressor = GaussianProcessRegressor(
            kernel=self.new_kernel(),
            alpha=self.point_variances(prefix, scaler),
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
        from sklearn.gaussian_process import GaussianProcessRegressor

        seen = usable(history)
        if len(seen) < self.minimum_observations:
            return False

        # Hyperparameters come from whole batches, so that an instance that
        # has been fitting all session holds the kernel a fresh one handed the
        # same history computes, and what it keeps is a cache. Short of one
        # batch there is nothing to hold back.
        whole = len(seen) - len(seen) % self.batch_size
        prefix = seen[:whole] if whole else seen
        # Keyed on which observations they were fitted to and not how many: a
        # cost arriving late lands in proposal order and rewrites a prefix of
        # unchanged length.
        fitted_to = tuple(o.shot_id for o in prefix)
        if fitted_to != self._fitted_to:
            self._cost_scaler, self._kernel = self.fit_hyperparameters(prefix)
            self._fitted_to = fitted_to

        costs = costs_array(seen).reshape(-1, 1)
        regressor = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self.point_variances(seen, self._cost_scaler),
            normalize_y=False,
            optimizer=None,
        )
        regressor.fit(
            self.space.scale(params_array(seen)),
            self._cost_scaler.transform(costs).ravel(),
        )
        self.regressor = regressor
        return True

    def predict(self, params: np.ndarray):
        """Predicted cost and standard deviation at ``params``, in real units."""
        if self.regressor is None:
            raise RuntimeError("the learner has not been fit to any history yet")
        params = np.atleast_2d(params)
        mean, std = self.regressor.predict(self.space.scale(params), return_std=True)
        scale = self._cost_scaler.scale_[0]
        return (
            self._cost_scaler.inverse_transform(mean.reshape(-1, 1)).ravel(),
            std * scale,
        )

    def minimise_acquisition(
        self, regressor, uncer_bias: float, best: np.ndarray, lows, highs
    ):
        """Multi-start L-BFGS-B over the acquisition. Returns scaled parameters.

        ``lows`` and ``highs`` are the search bounds, already scaled.
        """
        from scipy.optimize import minimize

        def acquisition(u):
            mean, std = regressor.predict(np.atleast_2d(u), return_std=True)
            return self.cost_bias * mean[0] - uncer_bias * std[0]

        starts = [self.space.scale(best)]
        starts.extend(self.rng.uniform(lows, highs, size=(self.num_restarts, len(lows))))

        winner, winning_value = None, np.inf
        for start in starts:
            result = minimize(
                acquisition,
                np.clip(start, lows, highs),
                method="L-BFGS-B",
                bounds=list(zip(lows, highs)),
            )
            if result.fun < winning_value:
                winner, winning_value = result.x, result.fun
        return np.clip(winner, lows, highs)

    def condition_on(self, regressor, scaled_point: np.ndarray):
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
        from sklearn.gaussian_process import GaussianProcessRegressor

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
        low, high = self.space.bounds_near(best_params, self.trust_region)
        lows, highs = self.space.scale(low), self.space.scale(high)

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
            step = (len(seen) + i) % self.batch_size
            scaled = self.minimise_acquisition(
                regressor, self.uncer_bias * step, best_params, lows, highs
            )
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                regressor = self.condition_on(regressor, scaled)
        return proposals
