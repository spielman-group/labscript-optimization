"""Gaussian process learner.

:class:`GaussianProcess` is the model: a scikit-learn
:class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit to the scaled
history, and a multi-start L-BFGS-B search over its posterior for the next
points. :class:`GaussianProcessLearner` runs it beside an explorer, computing
each batch on a thread of its own so that the apparatus never waits on a fit.

The cycle, which :meth:`GaussianProcessLearner.acquire` is the whole of:

- **Warmup.** Until the history holds ``warmup_observations`` usable
  observations, the explorer alone tops the run's shots in flight up to the
  hint. Usable observations are counted, not shots: a shot whose cost is NaN, or
  one that was dropped, moves it no nearer the end. Explorer shots already
  queued when warmup ends still run and join the fit when their costs land;
  nothing takes a queued shot back.
- **Batches.** After warmup the model computes ``batch_size`` points, each
  conditioned on the ones before it, from the history as it stood when the
  computation began. Every refill tops the shots in flight up to the hint: with
  the batch's points as soon as they are ready, and with explorer shots while it
  computes or once its points have gone out. The next batch is computed once
  every point of the last has completed or been dropped, so it has every answer
  it asked for. Explorer shots never hold it up.
- **Exploring.** A batch's first point waits until ``explore_runs`` explorer
  shots have gone out since the previous batch's last point, wherever in that
  cycle they fell.

The explorer is handed the part of the history its ``history_scope`` names: all
of it, its own shots, or none. The model is always handed all of it.

The hint is the session's ``num_buffered_runs``. It also sets roughly how many
explorer shots a cycle holds: while the last of a batch's points come back, each
shot that returns is replaced from the explorer, so a cycle holds about the
hint's worth even when a fit is instant, and more while it is slow.
``explore_runs`` adds only what that leaves short.

Pending means submitted and not yet completed or dropped, and a shot is known
dropped only once a reconcile has said so, which runs when a shot arrives. So
the next batch waits on a deleted point until some shot reaches lyse: with
nothing of this run's still queued and runmanager sending nothing in its place,
no refill runs and the apparatus idles.

Each proposal carries the source a lab reads in the ``phase`` column:
:data:`WARMUP_SOURCE` for the explorer's shots during warmup,
:data:`BATCH_SOURCE` for the Gaussian process's own, and
:data:`EXPLORE_SOURCE` for the explorer's shots after warmup. The configured
start, which the session proposes itself, reads ``start``. The batch barrier is
read off those sources, so a history whose records carry none, one built outside
a session, holds no barrier.

Exploration within the batch comes from the acquisition, which minimizes

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

where ``uncer_bias`` is a list of weights walked by position within the batch,
so each batch starts from the first weight. The default runs 0, 1, 2, 3: the
first point of a batch is purely greedy and the rest trade predicted cost for a
look somewhere less certain, and the default batch of four is one pass of it.
The Gaussian process does not condition on the explorer's shots in flight: a
random draw is a weaker fantasy than a point of its own, and conditioning on
pending points is measured to make the answer worse.

The kernel hyperparameters are refit once per batch, on the usable observations
in hand when the batch is computed, starting from the last refit's; the
posterior is refit to all of them with those hyperparameters held.

scikit-learn and scipy are imported where they are used, and nothing here
reaches a point of use until a batch is asked for. Loading a configuration
resolves the learner's class and builds it, to read its ``generation``, in
lyse's own process, which should not pay several seconds to import the
scientific stack.
"""

import threading
from collections.abc import Sequence
from concurrent.futures import Future

import numpy as np

from .. import knobs
from ..observations import (
    PENDING,
    Observation,
    best,
    costs_array,
    params_array,
    uncers_array,
    usable,
)
from ..space import ParameterSpace
from .base import ParameterSpaceLearner
from .differential_evolution import DifferentialEvolutionLearner
from .random import DirectedRandomLearner, RandomLearner

#: The learners a Gaussian process may explore with, by the name its ``explorer``
#: knob takes. None fits anything, so an explorer proposes from the first shot of
#: a run and never waits on a fit.
EXPLORERS = {
    "random": RandomLearner,
    "directed_random": DirectedRandomLearner,
    "differential_evolution": DifferentialEvolutionLearner,
}

#: The source of an explorer's shot proposed during warmup.
WARMUP_SOURCE = "warmup"

#: The source of a point of the Gaussian process's own batch.
BATCH_SOURCE = "main"

#: The source of an explorer's shot proposed after warmup.
EXPLORE_SOURCE = "explore"


def in_background(function, *args) -> Future:
    """Run ``function(*args)`` on a daemon thread and return its future."""
    # Not a ThreadPoolExecutor, which joins its threads at exit: a worker
    # quitting mid-fit would wait for the fit. An error goes onto the future,
    # not to threading.excepthook, where only the thread would see it.
    future = Future()

    def run():
        try:
            result = function(*args)
        except BaseException as error:
            future.set_exception(error)
        else:
            future.set_result(result)

    threading.Thread(target=run, daemon=True).start()
    return future


class GaussianProcess:
    """A Gaussian process fit to the history, and a search of its posterior.

    It keeps the kernel of its last refit, which the next refit starts from, so
    its fit depends on the histories handed before as well as the one in hand.
    :class:`GaussianProcessLearner` runs :meth:`ask` on a thread of its own and
    reads nothing else of it. Past ``space`` and ``rng`` it takes, by keyword,
    the knobs of the same names that :class:`GaussianProcessLearner` documents.

    Parameters
    ----------
    space : ParameterSpace
        The parameter space to search.
    rng : numpy.random.Generator
        Source of randomness, drawn on by this model alone.
    """

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        *,
        cost_has_noise: bool,
        length_scale_bounds: Sequence[float],
        noise_level_bounds: Sequence[float],
        cost_bias: float,
        uncer_bias: float | Sequence[float],
        trust_region,
        warmup_observations: int | None,
    ):
        self.space = space
        self.rng = rng
        self.cost_has_noise = knobs.boolean("cost_has_noise", cost_has_noise)
        self.length_scale_bounds = knobs.pair(
            "length_scale_bounds", length_scale_bounds
        )
        self.noise_level_bounds = knobs.pair("noise_level_bounds", noise_level_bounds)
        # scikit-learn rejects a reversed pair only at the first fit, and numpy
        # warns on a zero or negative one.
        for name, (low, high) in (
            ("length_scale_bounds", self.length_scale_bounds),
            ("noise_level_bounds", self.noise_level_bounds),
        ):
            if not 0 < low < high:
                raise ValueError(
                    f"{name} must be an ordered pair of positive numbers, "
                    f"[low, high] with 0 < low < high, got {[low, high]}."
                )
        self.cost_bias = knobs.number("cost_bias", cost_bias)
        # A list, not a step and a period, so a file shows which point of a batch
        # explores how far. A single number is a cycle of one step.
        if knobs.is_number(uncer_bias):
            schedule = [uncer_bias]
        elif knobs.is_numbers(uncer_bias):
            schedule = list(uncer_bias)
        else:
            raise ValueError(
                f"uncer_bias must be written as a number or a list of numbers, "
                f"unquoted, not {uncer_bias!r}."
            )
        self.uncer_bias = tuple(float(weight) for weight in schedule)
        if not self.uncer_bias:
            raise ValueError(
                "uncer_bias needs at least one weight; an empty list leaves "
                "none to propose at"
            )
        self.trust_region = space.absolute_trust_region(trust_region)
        # Scaled with the parameter count; the floor keeps a one-parameter
        # search from fitting on two points.
        self.warmup_observations = (
            max(5, 2 * space.num_params)
            if warmup_observations is None
            else knobs.integer("warmup_observations", warmup_observations)
        )
        if self.warmup_observations < 1:
            raise ValueError(
                f"warmup_observations must be at least 1, got "
                f"{self.warmup_observations}; a fit has nothing to work from "
                f"on an empty history."
            )
        self.num_restarts = max(10, space.num_params)

        self._kernel = None
        self._fitted_to = None
        self.regressor = None
        self._cost_scaler = None

    def new_kernel(self):
        """A fresh RBF kernel, with a white-noise term when ``cost_has_noise``."""
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
        """Per-point variances for the regressor, in standardized cost units."""
        uncers = uncers_array(seen)
        if uncers is None:
            return 1e-10
        if not self.cost_has_noise:
            # Observations lacking an uncertainty get a variance of zero, and
            # with no white-noise term the fit is then singular. Refused here,
            # where it is first known, not as scikit-learn's LinAlgError.
            missing = [o.shot_id for o in seen if o.uncer is None]
            if missing:
                raise ValueError(
                    f"cost_has_noise is off, which says every cost is measured "
                    f"exactly, but {len(missing)} of {len(seen)} observations "
                    f"carry no uncertainty; the first is shot {missing[0]!r}. "
                    f"Turn cost_has_noise on, or write an uncertainty on every "
                    f"shot."
                )
        return (uncers / scaler.scale_[0]) ** 2

    def fit_hyperparameters(self, seen: Sequence[Observation]):
        """Fit the cost scaling and kernel to ``seen``, from the last refit's kernel."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler

        costs = costs_array(seen).reshape(-1, 1)
        # Fitted with the kernel, not per observation, since the noise level is in
        # its units.
        scaler = StandardScaler().fit(costs)
        regressor = GaussianProcessRegressor(
            kernel=self.new_kernel() if self._kernel is None else self._kernel,
            alpha=self.point_variances(seen, scaler),
            normalize_y=False,
        )
        regressor.fit(
            self.space.scale(params_array(seen)),
            scaler.transform(costs).ravel(),
        )
        return scaler, regressor.kernel_

    def fit(self, history: Sequence[Observation]) -> None:
        """Fit the regressor to ``history``; the kernel refits if its data changed."""
        from sklearn.gaussian_process import GaussianProcessRegressor

        seen = usable(history)
        assert len(seen) >= self.warmup_observations

        # Keyed on which observations, not how many: a late cost lands mid-history,
        # so equal counts can hold different observations.
        fitted_to = tuple(o.shot_id for o in seen)
        if fitted_to != self._fitted_to:
            self._cost_scaler, self._kernel = self.fit_hyperparameters(seen)
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

    def predict(self, params: np.ndarray):
        """Predicted cost and standard deviation at ``params``, in real units."""
        if self.regressor is None:
            raise RuntimeError("the model has not been fit to any history yet")
        params = np.atleast_2d(params)
        mean, std = self.regressor.predict(self.space.scale(params), return_std=True)
        scale = self._cost_scaler.scale_[0]
        return (
            self._cost_scaler.inverse_transform(mean.reshape(-1, 1)).ravel(),
            std * scale,
        )

    def acquisition(self, u: np.ndarray, regressor, uncer_weight: float):
        """The acquisition and its gradient at scaled parameters ``u``."""
        from scipy.linalg import solve_triangular

        # Assumes new_kernel's form, with no ConstantKernel, and normalize_y off.
        kernel = regressor.kernel_
        if self.cost_has_noise:
            length_scale, noise = kernel.k1.length_scale, kernel.k2.noise_level
        else:
            length_scale, noise = kernel.length_scale, 0.0
        scaled_distance = (u - regressor.X_train_) / length_scale
        k = np.exp(-0.5 * np.sum(scaled_distance**2, axis=1))
        dk = -k[:, None] * scaled_distance / length_scale
        v = solve_triangular(regressor.L_, k, lower=True, check_finite=False)
        variance = 1.0 + noise - v @ v
        # Rounding takes the variance to zero or below at a training point, so it
        # is floored there and the standard deviation held flat.
        if variance > 1e-12:
            std = np.sqrt(variance)
            k_solved = solve_triangular(
                regressor.L_, v, lower=True, trans="T", check_finite=False
            )
            dstd = -dk.T @ k_solved / std
        else:
            std, dstd = np.sqrt(1e-12), 0.0
        return (
            self.cost_bias * (k @ regressor.alpha_) - uncer_weight * std,
            self.cost_bias * (dk.T @ regressor.alpha_) - uncer_weight * dstd,
        )

    def minimize_acquisition(
        self, regressor, uncer_weight: float, best: np.ndarray, lows, highs
    ):
        """Multi-start L-BFGS-B over the acquisition, in scaled parameters."""
        from scipy.optimize import minimize

        starts = [self.space.scale(best)]
        starts.extend(
            self.rng.uniform(lows, highs, size=(self.num_restarts, len(lows)))
        )

        winner, winning_value = None, np.inf
        for start in starts:
            result = minimize(
                self.acquisition,
                np.clip(start, lows, highs),
                args=(regressor, uncer_weight),
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lows, highs)),
            )
            if result.fun < winning_value:
                winner, winning_value = result.x, result.fun
        if winner is None:
            # Every start was non-finite. np.clip on None would fail with a type
            # error naming neither this learner nor the fit.
            raise RuntimeError(
                f"the acquisition was not finite at any of the "
                f"{len(starts)} starting points, so there is no point to "
                f"propose. Check the costs in the history for a range a fit "
                f"cannot describe, and the length scales of the last refit."
            )
        return np.clip(winner, lows, highs)

    def condition_on(self, regressor, scaled_point: np.ndarray):
        """A copy of ``regressor`` with a point folded in at its predicted cost."""
        from sklearn.gaussian_process import GaussianProcessRegressor

        # Posterior variance depends on where a point was measured, not on what
        # came back, so a point at its predicted mean shrinks the uncertainty
        # around it as the real shot will, and a batch does not chase one corner.
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

    def ask(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """The next ``k`` points, each folded into the fit before the next."""
        self.fit(history)
        best_params = best(history).params
        low, high = self.space.bounds_near(best_params, self.trust_region)
        lows, highs = self.space.scale(low), self.space.scale(high)

        # Batch points are folded into a fit held only here, so ``predict`` goes
        # on describing the measured data.
        regressor = self.regressor
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            # The schedule is walked by position in the batch, so every batch
            # starts from the first weight.
            weight = self.uncer_bias[i % len(self.uncer_bias)]
            scaled = self.minimize_acquisition(
                regressor, weight, best_params, lows, highs
            )
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                regressor = self.condition_on(regressor, scaled)
        return proposals


class GaussianProcessLearner(ParameterSpaceLearner):
    """Run a Gaussian process beside an explorer, keeping the queue topped up.

    Parameters
    ----------
    cost_has_noise : bool
        Add a white-noise term to the kernel. Leave this on for real data: off
        asserts the cost is measured exactly, and a history in which only some
        observations carry an uncertainty is then refused, by
        :meth:`GaussianProcess.point_variances`.
    length_scale_bounds : (float, float)
        Bounds on the RBF length scale, in units of the unit cube the parameters
        are scaled onto. ``0 < low < high``.
    noise_level_bounds : (float, float)
        Bounds on the white-noise level, in units of the standardized cost.
        ``0 < low < high``.
    cost_bias : float
        Weight on predicted cost in the acquisition.
    uncer_bias : float or sequence of float
        The weights on predicted uncertainty the exploration schedule walks
        through, one per point of a batch, starting again from the first at
        every batch. A single number is a fixed weight on every point. A zero
        is a purely greedy point; the larger the weight, the wider the look.
    batch_size : int
        How many points the Gaussian process computes at a time, each
        conditioned on the ones before it. The kernel hyperparameters are refit
        once per batch.
    trust_region : float or sequence of float or None
        Restrict the search to this distance around the best point seen.
    warmup_observations : int or None
        How many usable observations the explorer gathers before the Gaussian
        process computes, counted as observations a fit can use, not shots.
        Defaults to max(5, twice the parameter count). Explorer shots already
        queued when warmup ends still run.
    explorer : str or ParameterSpaceLearner
        The learner whose shots run the warmup and fill the queue after it:
        ``"random"``, ``"directed_random"`` or ``"differential_evolution"``,
        built from its defaults, or an instance of one. A configuration names
        it, and :func:`~labscript_optimization.learners.build` hands over the
        instance built from that learner's own ``[LEARNER.<name>]`` table.
    explore_runs : int
        How many explorer shots each batch cycle holds at the least, from one
        batch's last point to the next batch's first, wherever in the cycle
        they fall. Zero guarantees none, and the explorer only fills the queue.
        Filling alone already puts about ``num_buffered_runs`` in each cycle, so
        a smaller ``explore_runs`` changes nothing.
    """

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        cost_has_noise: bool = True,
        length_scale_bounds: Sequence[float] = (1e-2, 1e2),
        noise_level_bounds: Sequence[float] = (1e-5, 1e1),
        cost_bias: float = 1.0,
        uncer_bias: float | Sequence[float] = (0.0, 1.0, 2.0, 3.0),
        batch_size: int = 4,
        trust_region=None,
        warmup_observations: int | None = None,
        explorer: str | ParameterSpaceLearner = "directed_random",
        explore_runs: int = 1,
    ):
        super().__init__(space, rng)
        # The model runs on a thread of its own, and threads must not share a
        # Generator.
        self.model = GaussianProcess(
            space,
            rng.spawn(1)[0],
            cost_has_noise=cost_has_noise,
            length_scale_bounds=length_scale_bounds,
            noise_level_bounds=noise_level_bounds,
            cost_bias=cost_bias,
            uncer_bias=uncer_bias,
            trust_region=trust_region,
            warmup_observations=warmup_observations,
        )
        self.warmup_observations = self.model.warmup_observations
        self.batch_size = knobs.integer("batch_size", batch_size)
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be at least 1, got {self.batch_size}; a batch "
                f"of none never proposes after warmup."
            )
        if isinstance(explorer, tuple(EXPLORERS.values())):
            self.explorer = explorer
        else:
            self.explorer = EXPLORERS[knobs.choice("explorer", explorer, EXPLORERS)](
                space, rng
            )
        self.explore_runs = knobs.integer("explore_runs", explore_runs)
        if self.explore_runs < 0:
            raise ValueError(
                f"explore_runs cannot be negative, got {self.explore_runs}; "
                f"zero guarantees none, and the explorer only fills the queue."
            )
        #: The batch being computed, as a future, or ``None``.
        self.computation: Future | None = None
        #: Points of the last batch computed that have not gone out yet.
        self.ready: list[np.ndarray] = []

    def acquire(
        self, history: Sequence[Observation], k: int
    ) -> list[tuple[np.ndarray, str]]:
        """The cycle: warmup, then batch points as ready, plus explorer shots."""
        # "mine" is the explorer's own shots, which excludes the configured
        # start and the batches.
        scoped = {
            "all": history,
            "mine": [o for o in history if o.source in (WARMUP_SOURCE, EXPLORE_SOURCE)],
            "none": [],
        }[self.explorer.history_scope]
        if len(usable(history)) < self.warmup_observations:
            return [
                (params, WARMUP_SOURCE) for params in self.explorer.ask(scoped, k)
            ]

        if self.computation is not None and self.computation.done():
            # Read only once done, so a refill never waits on a fit. Whatever the
            # model raised is raised here, on the message loop.
            self.ready.extend(self.computation.result())
            self.computation = None
        if (
            self.computation is None
            and not self.ready
            and not any(
                o.state == PENDING and o.source == BATCH_SOURCE for o in history
            )
        ):
            self.computation = in_background(
                self.model.ask, tuple(history), self.batch_size
            )

        released = []
        if self.ready:
            # Only a whole batch waits, for explore_runs explorer shots since the
            # run began or the last batch's last point. One already begun goes out
            # without waiting.
            mains = [i for i, o in enumerate(history) if o.source == BATCH_SOURCE]
            since = (mains or [0])[-1]
            explored = sum(
                o.source in (WARMUP_SOURCE, EXPLORE_SOURCE) for o in history[since:]
            )
            if len(self.ready) < self.batch_size or explored >= self.explore_runs:
                released, self.ready = self.ready[:k], self.ready[k:]
        # The batch's points first, so that what the run budget has no room for
        # is cut from the explorer's shots.
        return [
            *((params, BATCH_SOURCE) for params in released),
            *(
                (params, EXPLORE_SOURCE)
                for params in self.explorer.ask(scoped, k - len(released))
            ),
        ]
