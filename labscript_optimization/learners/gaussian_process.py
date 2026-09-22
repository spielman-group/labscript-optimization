"""Gaussian process learner.

A scikit-learn :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit
to the scaled history, and a multi-start L-BFGS-B search over its posterior for
the next point.

Exploration comes from the acquisition, which minimises

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

where ``uncer_bias`` is a list of weights cycled by the number of
observations in hand, so that the schedule's period is the list's length. The
default runs 0, 1, 2, 3: one proposal in four is purely greedy and the rest
trade predicted cost for a look somewhere less certain. The position in the
cycle is read off the history, so it advances with the proposals a session
makes rather than with a point's position within the group of them asked for
at once.

Refitting the kernel hyperparameters is the expensive part, so it happens once
per ``refit_interval`` new observations rather than on every call; the
posterior is refit to all the data every time.

scikit-learn and scipy are imported where they are used rather than at the top
of this module, and nothing here reaches a point of use until a proposal is
asked for. Loading a configuration resolves the class of the learner it names,
to hold the file's knobs to the constructor, and builds it, to ask how many
proposals it makes at a time -- and that happens in the lyse routine's own
process, which never asks the learner for a proposal and should not pay
several seconds to import the scientific stack.
"""

import warnings
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


#: scikit-learn's warning for a fitted length scale sitting at one end of
#: ``length_scale_bounds``, as a regular expression against the start of its
#: message. It is filtered out of the refit and answered instead by
#: :meth:`GaussianProcessLearner.report_length_scale_bounds`. Written narrowly
#: so that nothing else a fit warns about matches it -- and should scikit-learn
#: reword the message, it stops matching and the lab gets the repetition back
#: rather than silence.
LENGTH_SCALE_AT_BOUND = (
    r"The optimal value found for dimension \d+ of parameter \S*length_scale "
)


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
        uncer_bias: The weights on predicted uncertainty the exploration
            schedule above cycles through, one per proposal. A single number
            is a cycle of one step, and so a fixed weight on every proposal.
            A zero in the list is a purely greedy proposal; the larger the
            weight, the wider the look.
        refit_interval: How many new observations are accepted before the
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
        uncer_bias: float | Sequence[float] = (0.0, 1.0, 2.0, 3.0),
        refit_interval: int = 4,
        trust_region=None,
        minimum_observations: int | None = None,
    ):
        super().__init__(space, rng)
        self.cost_has_noise = bool(cost_has_noise)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.noise_level_bounds = tuple(noise_level_bounds)
        self.cost_bias = float(cost_bias)
        # Written out rather than derived from a step and a period, because
        # the schedule a session runs is what a lab wants to read off the
        # file: a list says which proposal explores how far, where a pair of
        # numbers leaves that to be worked out. A single number is a cycle of
        # one step, so a file that writes a weight gets that weight on every
        # proposal.
        schedule = [uncer_bias] if np.ndim(uncer_bias) == 0 else list(uncer_bias)
        self.uncer_bias = tuple(float(weight) for weight in schedule)
        if not self.uncer_bias:
            raise ValueError(
                "uncer_bias is the cycle of weights the exploration schedule "
                "runs through and needs at least one of them; an empty list "
                "leaves no weight to propose at"
            )
        self.refit_interval = int(refit_interval)
        if self.refit_interval < 1:
            raise ValueError(
                f"refit_interval must be at least 1, got {self.refit_interval}"
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
        #: Which parameters the last refit left at an end of
        #: ``length_scale_bounds``, and which end. Empty before the first
        #: refit and after one that leaves every length scale inside them;
        #: see :meth:`report_length_scale_bounds`.
        self.at_length_scale_bounds: dict[str, str] = {}

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

        scikit-learn warns once per length scale left at a bound on every fit,
        which is a real diagnostic said too often to be read. That one message
        is filtered out here and answered by
        :meth:`report_length_scale_bounds`, which names the parameters rather
        than the kernel's numbering and speaks only when the set of them
        changes. :data:`LENGTH_SCALE_AT_BOUND` is what the filter matches, so
        everything else the fit warns about -- the white-noise level reaching
        its own bound, an optimiser that gave up -- reaches the lab untouched.
        """
        from sklearn.exceptions import ConvergenceWarning
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=LENGTH_SCALE_AT_BOUND,
                category=ConvergenceWarning,
            )
            regressor.fit(
                self.space.scale(params_array(prefix)),
                scaler.transform(costs).ravel(),
            )
        self.report_length_scale_bounds(regressor.kernel_)
        return scaler, regressor.kernel_

    def report_length_scale_bounds(self, kernel) -> None:
        """Say which parameters a refit left at an end of the length-scale bounds.

        Recorded on :attr:`at_length_scale_bounds` and warned about only when
        that set changes, so a lab hears once when a parameter reaches a bound
        and once more when it leaves. Reported at every refit instead -- which
        is what scikit-learn does -- a dimension that stays pinned repeats the
        same paragraph for the length of the run, and the one refit where the
        set moves reads like all the others.

        A length scale at the upper end means the fit sees no structure along
        that parameter: the posterior is flat in it and the search is
        effectively over the others. At the lower end it means the opposite, a
        fit that can explain the costs only as varying faster than the shots
        are spaced. Either way the bound rather than the data set the number,
        which is why it is worth saying at all.
        """
        from sklearn.exceptions import ConvergenceWarning

        at_bound = np.isclose(kernel.bounds, np.atleast_2d(kernel.theta).T)
        reached, position = {}, 0
        for hyperparameter in kernel.hyperparameters:
            if hyperparameter.fixed:
                continue
            for dim in range(hyperparameter.n_elements):
                lower, upper = at_bound[position]
                position += 1
                if hyperparameter.name.endswith("length_scale") and (lower or upper):
                    reached[self.space.parameters[dim].name] = (
                        "lower" if lower else "upper"
                    )
        if reached == self.at_length_scale_bounds:
            return
        self.at_length_scale_bounds = reached
        if reached:
            where = ", ".join(
                f"{name} at the {end} end" for name, end in reached.items()
            )
            warnings.warn(
                f"the Gaussian process fit leaves {where} of "
                f"length_scale_bounds {self.length_scale_bounds}. At the upper "
                f"end the fit sees no structure along that parameter; at the "
                f"lower end, structure finer than the shots are spaced. Said "
                f"again only when the set of them changes.",
                ConvergenceWarning,
            )
        else:
            warnings.warn(
                "the Gaussian process fit now leaves every length scale "
                "inside length_scale_bounds",
                ConvergenceWarning,
            )

    def fit(self, history: Sequence[Observation]) -> bool:
        """Fit the regressor to ``history``. Returns whether a fit was possible."""
        from sklearn.gaussian_process import GaussianProcessRegressor

        seen = usable(history)
        if len(seen) < self.minimum_observations:
            return False

        # Hyperparameters come from whole intervals, so that an instance that
        # has been fitting all session holds the kernel a fresh one handed the
        # same history computes, and what it keeps is a cache. Short of one
        # interval there is nothing to hold back.
        whole = len(seen) - len(seen) % self.refit_interval
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
        self, regressor, uncer_weight: float, best: np.ndarray, lows, highs
    ):
        """Multi-start L-BFGS-B over the acquisition. Returns scaled parameters.

        ``uncer_weight`` is one weight, the step of the exploration schedule
        this proposal stands at, rather than the whole of ``uncer_bias``.
        ``lows`` and ``highs`` are the search bounds, already scaled.
        """
        from scipy.optimize import minimize

        def acquisition(u):
            mean, std = regressor.predict(np.atleast_2d(u), return_std=True)
            return self.cost_bias * mean[0] - uncer_weight * std[0]

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
            # and then the points picked so far in this group. A schedule kept
            # as a counter over the group asked for at once would sit at its
            # first weight -- the greedy one, by default -- for ever in a
            # session that settles into asking for one point at a time, and
            # every other weight in the list would go unused.
            weight = self.uncer_bias[(len(seen) + i) % len(self.uncer_bias)]
            scaled = self.minimise_acquisition(
                regressor, weight, best_params, lows, highs
            )
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                regressor = self.condition_on(regressor, scaled)
        return proposals
