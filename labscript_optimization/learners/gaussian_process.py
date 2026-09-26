"""Gaussian process learner.

:class:`GaussianProcess` is the model: a scikit-learn
:class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit to the scaled
history, and a multi-start L-BFGS-B search over its posterior for the next
points. :class:`GaussianProcessLearner` runs it beside an explorer, computing
each batch on a thread of its own so that the apparatus never waits on a fit.

The cycle, which :meth:`GaussianProcessLearner.acquire` is the whole of:

- **Warmup.** Until the history holds ``warmup_observations`` usable
  observations, the explorer alone tops the run's shots in flight up to the
  hint. It counts usable observations, not shots: a shot whose cost is NaN, or
  one that was dropped, moves it no nearer the end. Warmup ends at the count,
  and the explorer shots already queued at that moment still run and join the
  fit when their costs land. That is the design, not an overrun: nothing takes
  a queued shot back.
- **Batches.** After warmup the model computes ``batch_size`` points, each
  conditioned on the ones before it, from the history as it stood when the
  computation began. Every refill tops the shots in flight up to the hint:
  with the batch's points as soon as they are ready, and with explorer shots
  while it computes or once its points have gone out. The next batch is
  computed once every point of the last has completed or been dropped, so it
  has every answer it asked for. Explorer shots never hold it up.
- **Exploring.** A batch's first point waits until ``explore_runs`` explorer
  shots have gone out since the previous batch's first point, wherever in that
  cycle they fell.

The explorer is handed the part of the history its ``history_scope`` names:
all of it, its own shots, or none. The model is always handed all of it.

The hint is the session's ``num_buffered_runs``, and it also sets roughly how
many explorer shots a cycle holds: while the last of a batch's points come
back, each shot that returns is replaced from the explorer, so a cycle holds
about the hint's worth even when a fit is instant, and more while it is slow.
``explore_runs`` adds only what that leaves short.

Pending means submitted and not yet completed or dropped, and a shot is known
dropped only once a reconcile has said so, which runs when a shot arrives. So
the next batch waits on a deleted point until some shot reaches lyse: with
nothing of this run's still queued and runmanager sending nothing in its
place, no refill runs and the apparatus idles.

Each proposal carries the source a lab reads in the ``phase`` column:
:data:`WARMUP_SOURCE` for the explorer's shots during warmup,
:data:`BATCH_SOURCE` for the Gaussian process's own, and
:data:`EXPLORE_SOURCE` for the explorer's shots after warmup. The configured
start, which the session proposes itself, reads ``start``. The batch barrier
is read off those sources, so a history whose records carry none -- one built
outside a session -- holds no barrier.

Exploration within the batch comes from the acquisition, which minimises

    cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation

where ``uncer_bias`` is a list of weights walked by position within the batch,
so each batch starts from the first weight. The default runs 0, 1, 2, 3: the
first point of a batch is purely greedy and the rest trade predicted cost for a
look somewhere less certain, and the default batch of four is one pass of it.
The Gaussian process does not condition on the explorer's shots in flight: a
random draw is a weaker thing to fold in as a fantasy than a point of its own,
and conditioning on pending points is measured to make the answer worse.

Refitting the kernel hyperparameters is the expensive part, so it happens once
per batch, on the usable observations in hand when the batch is computed; the
posterior is refit to all of them with those hyperparameters held.

scikit-learn and scipy are imported where they are used rather than at the top
of this module, and nothing here reaches a point of use until a batch is
asked for. Loading a configuration resolves the class of the learner it names,
to hold the file's knobs to the constructor, and builds it and asks it to open
a run, to learn what it does with the queue depth -- and that happens in the
lyse routine's own process, which should not pay several seconds to import the
scientific stack. Opening a run is warmup, which is the explorer's.
"""

import threading
import warnings
from concurrent.futures import Future
from typing import Sequence

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
from .base import InsufficientData, ParameterSpaceLearner
from .differential_evolution import DifferentialEvolutionLearner
from .random import DirectedRandomLearner, RandomLearner
from ..space import ParameterSpace


#: scikit-learn's warning for a fitted length scale sitting at one end of
#: ``length_scale_bounds``, as a regular expression against the start of its
#: message. It is filtered out of every refit and answered instead by
#: :meth:`GaussianProcess.report_length_scale_bounds`. Written narrowly so
#: that nothing else a fit warns about matches it -- and should scikit-learn
#: reword the message, it stops matching and the lab gets the repetition back
#: rather than silence.
LENGTH_SCALE_AT_BOUND = (
    r"The optimal value found for dimension \d+ of parameter \S*length_scale "
)

#: The learners a Gaussian process may explore with, by the name its
#: ``explorer`` knob takes. None of them fits anything, so an explorer proposes
#: from the first shot of a run and never waits on a fit.
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
    """Run ``function(*args)`` on a daemon thread, and return its future.

    Whatever it raises is set on the future, to be raised by ``result()``:
    left to the thread it would reach ``threading.excepthook`` and nobody
    else. Not a ``ThreadPoolExecutor``, which joins its threads at exit, so a
    worker quitting mid-fit would wait for the fit to end.
    """
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

    A function of the history it is handed, apart from its own rng: what it
    keeps between calls is a cache of the kernel. :class:`GaussianProcessLearner`
    runs :meth:`ask` on a thread of its own and reads nothing else of it. Past
    ``space`` and ``rng`` it takes, by keyword, the knobs of the same names
    that :class:`GaussianProcessLearner` documents.

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
        self.cost_bias = knobs.number("cost_bias", cost_bias)
        # Written out rather than derived from a step and a period, because
        # the schedule a session runs is what a lab wants to read off the
        # file: a list says which point of a batch explores how far, where a
        # pair of numbers leaves that to be worked out. A single number is a
        # cycle of one step, so a file that writes a weight gets that weight
        # on every point.
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
                "uncer_bias is the cycle of weights the exploration schedule "
                "runs through and needs at least one of them; an empty list "
                "leaves no weight to propose at"
            )
        self.trust_region = space.absolute_trust_region(trust_region)
        # Scaled with the search, because a constant warmup is too short for a
        # search over enough parameters and too long for one over few; the
        # floor keeps a one-parameter search from fitting on two points.
        self.warmup_observations = (
            max(5, 2 * space.num_params)
            if warmup_observations is None
            else knobs.integer("warmup_observations", warmup_observations)
        )
        if self.warmup_observations < 1:
            # A fit needs something to fit to. At zero the guard in ``fit``
            # passes on an empty history, and the refusal a file gets is
            # scikit-learn's, from inside a scaler, naming neither this
            # setting nor the learner.
            raise ValueError(
                f"warmup_observations is how many usable observations the "
                f"Gaussian process gathers before it will fit, so it must be "
                f"at least 1, got {self.warmup_observations}. A fit has "
                f"nothing to work from on an empty history."
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

        # Process-wide rather than around each refit: catch_warnings is not
        # thread-safe, and on this model's thread it would filter, and on exit
        # reset, warnings raised on the main thread. Here rather than at
        # import, where a catch_warnings around the import would drop it.
        warnings.filterwarnings(
            "ignore", message=LENGTH_SCALE_AT_BOUND, category=UserWarning
        )

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

        With ``cost_has_noise`` off there is no jitter anywhere else, so those
        zeros are the whole of what keeps the covariance matrix conditioned
        and a history carrying only some uncertainties is refused here. It is
        refused at the first moment it is knowable: whether a lab's
        uncertainty column is written on every shot is not something a
        configuration can say, and the alternative is the ``LinAlgError`` the
        fit raises from inside scikit-learn two calls later.
        """
        uncers = uncers_array(seen)
        if uncers is None:
            return 1e-10
        if not self.cost_has_noise:
            missing = [o.shot_id for o in seen if o.uncer is None]
            if missing:
                raise ValueError(
                    f"cost_has_noise is off, which says every cost is measured "
                    f"exactly, but {len(missing)} of {len(seen)} observations "
                    f"carry no uncertainty -- the first is shot "
                    f"{missing[0]!r}. Those are fitted with a variance of "
                    f"exactly zero and there is no white-noise term to "
                    f"condition the covariance matrix, so the fit is singular. "
                    f"Turn cost_has_noise on, or write an uncertainty on every "
                    f"shot."
                )
        return (uncers / scaler.scale_[0]) ** 2

    def fit_hyperparameters(self, seen: Sequence[Observation]):
        """Fit the cost scaling and the kernel hyperparameters to ``seen``.

        The scaling belongs with them because it sets the units the noise level
        is measured in: restandardising as each observation arrived would leave
        a cached kernel describing units that had since moved. The restart
        draws are seeded from the number of observations in ``seen``, not from
        the learner's rng, which would make the search depend on how much this
        instance had already proposed.

        scikit-learn warns once per length scale left at a bound on every fit,
        which is a real diagnostic said too often to be read. That one message
        is filtered out by the constructor and answered by
        :meth:`report_length_scale_bounds`, which names the parameters rather
        than the kernel's numbering and speaks only when the set of them
        changes. :data:`LENGTH_SCALE_AT_BOUND` is what the filter matches, so
        everything else the fit warns about -- the white-noise level reaching
        its own bound, an optimiser that gave up -- reaches the lab untouched.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.preprocessing import StandardScaler

        costs = costs_array(seen).reshape(-1, 1)
        scaler = StandardScaler().fit(costs)
        regressor = GaussianProcessRegressor(
            kernel=self.new_kernel(),
            alpha=self.point_variances(seen, scaler),
            normalize_y=False,
            n_restarts_optimizer=self.num_restarts,
            random_state=len(seen),
        )
        regressor.fit(
            self.space.scale(params_array(seen)),
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
        """Fit the regressor to ``history``. Returns whether a fit was possible.

        The hyperparameters are fitted to the usable observations in hand and
        kept for as long as those are the observations handed over, which
        within a session is the one call per batch that computes it. What an
        instance keeps is a cache: it holds the kernel a fresh instance handed
        the same history computes, however much this one has fitted before.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor

        seen = usable(history)
        if len(seen) < self.warmup_observations:
            return False

        # Keyed on which observations they were fitted to and not how many: a
        # cost arriving late lands in proposal order, in the middle, so two
        # histories holding the same number of usable observations can hold
        # different ones.
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
        return True

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
        if winner is None:
            # Every start came back non-finite, so no comparison above was
            # ever true. Raised rather than clipped: ``np.clip`` on nothing
            # gives a type error out of numpy, which names neither this
            # learner nor the fit it came from.
            raise RuntimeError(
                f"the acquisition was not finite at any of the "
                f"{len(starts)} starting points, so there is no point to "
                f"propose. The posterior this was searched over is not "
                f"usable: check the costs in the history for a range a fit "
                f"cannot describe, and the length scales the last refit "
                f"reported."
            )
        return np.clip(winner, lows, highs)

    def condition_on(self, regressor, scaled_point: np.ndarray):
        """A copy of ``regressor`` with a point folded in at its predicted cost.

        Posterior variance depends on where a point was measured, not on what
        came back, so adding a proposal at its predicted mean shrinks the
        uncertainty around it exactly as the real shot will. Without this every
        point in a batch with a high uncertainty weight chases the same
        unexplored corner and the batch is spent on one location.

        Returned rather than stored, so that invented points never reach the
        model and :meth:`predict` describes the measured data however a batch
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

    def ask(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """The next ``k`` points, each folded into the fit before the next.

        The search without the cycle, as a ``(k, num_params)`` array: what
        :class:`GaussianProcessLearner` computes each batch with. The
        exploration schedule is walked from its first weight by position among
        these ``k``, and the hyperparameters are refit here if the usable
        observations differ from the ones they were last fitted to.
        """
        if not self.fit(history):
            raise InsufficientData(
                f"the Gaussian process needs {self.warmup_observations} usable "
                f"observations and has {len(usable(history))}"
            )
        best_params = best(history).params
        low, high = self.space.bounds_near(best_params, self.trust_region)
        lows, highs = self.space.scale(low), self.space.scale(high)

        # The points of this batch are folded into a fit held here and nowhere
        # else, so the model goes on describing the measured data.
        regressor = self.regressor
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            # Where the exploration schedule stands is the point's position in
            # the batch, so every batch starts from the first weight -- the
            # greedy one, by default -- and a batch as long as the schedule
            # spends one point at each weight.
            weight = self.uncer_bias[i % len(self.uncer_bias)]
            scaled = self.minimise_acquisition(
                regressor, weight, best_params, lows, highs
            )
            proposals[i] = self.space.clip(self.space.unscale(scaled))
            if i + 1 < k:
                regressor = self.condition_on(regressor, scaled)
        return proposals


class GaussianProcessLearner(ParameterSpaceLearner):
    """Run a Gaussian process beside an explorer, keeping the queue topped up
    while each batch is computed.

    Parameters
    ----------
    space : ParameterSpace
        The parameter space to search.
    rng : numpy.random.Generator
        Source of randomness. The model draws on a generator spawned from it,
        because it draws on a thread of its own.
    cost_has_noise : bool
        Add a white-noise term to the kernel. Leave this on for real data;
        turning it off asserts the cost is measured exactly. With it off, a
        history in which only some observations carry an uncertainty is
        refused: the rest would be fitted with an ``alpha`` of exactly zero
        and there is no jitter anywhere else, so the covariance matrix is
        singular. The refusal comes from
        :meth:`GaussianProcess.point_variances`, the first place that can see
        the history is mixed.
    length_scale_bounds : (float, float)
        Bounds on the RBF length scale, in units of the unit cube the
        parameters are scaled onto.
    noise_level_bounds : (float, float)
        Bounds on the white-noise level, in units of the standardised cost.
    cost_bias : float
        Weight on predicted cost in the acquisition.
    uncer_bias : float or sequence of float
        The weights on predicted uncertainty the exploration schedule walks
        through, one per point of a batch, starting again from the first at
        every batch. A single number is a cycle of one step, and so a fixed
        weight on every point. A zero in the list is a purely greedy point;
        the larger the weight, the wider the look.
    batch_size : int
        How many points the Gaussian process computes at a time, each
        conditioned on the ones before it. The kernel hyperparameters are
        refit once per batch.
    trust_region : float or sequence of float, optional
        Restrict the search to this distance around the best point seen.
    warmup_observations : int, optional
        How many usable observations the explorer gathers before the Gaussian
        process computes, counted as observations a fit can use and not as
        shots. Defaults to max(5, twice the parameter count). Warmup ends at
        the count, and explorer shots already queued then still run.
    explorer : str or learner
        The learner whose shots run the warmup and fill the queue after it:
        ``"random"``, ``"directed_random"`` or ``"differential_evolution"``,
        built from its defaults, or an instance of one. A configuration names
        it, and :func:`~labscript_optimization.learners.build` hands over the
        instance built from that learner's own ``[LEARNER.<name>]`` table.
    explore_runs : int
        How many explorer shots each batch cycle holds at the least, from one
        batch's first point to the next's, wherever in the cycle they fall.
        Zero guarantees none, and the explorer only fills the queue. Filling
        alone already puts about ``num_buffered_runs`` in each cycle, so a
        smaller ``explore_runs`` changes nothing.
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
        # The model draws on a thread of its own, and two threads must not
        # share a Generator.
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
                f"batch_size is how many points the Gaussian process proposes "
                f"at a time, so it must be at least 1, got {self.batch_size}. "
                f"A batch of none is a learner that never proposes after "
                f"warmup."
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
                f"explore_runs is how many explorer shots each batch cycle "
                f"holds at the least, so it cannot be negative, got "
                f"{self.explore_runs}. Zero guarantees none: the explorer then "
                f"only fills the queue."
            )
        #: The batch being computed, as a future, or ``None``.
        self.computation: Future | None = None
        #: Points of the last batch computed that have not gone out yet.
        self.ready: list[np.ndarray] = []

    def acquire(
        self, history: Sequence[Observation], k: int
    ) -> list[tuple[np.ndarray, str]]:
        """The cycle: warmup, then batch points as they are ready, topped up
        with explorer shots.

        The model is read only through the future of its computation, and
        only once that is done, so a refill never waits on a fit.
        """
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
            # Raises here, on the message loop, whatever the model raised.
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
            # Only a whole batch waits, for explore_runs explorer shots since
            # the first point of the batch before it, or since the run began.
            mains = [i for i, o in enumerate(history) if o.source == BATCH_SOURCE]
            since = (mains[-self.batch_size :] or [0])[0]
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
