"""Differential evolution, in ask/tell form.

The algorithm is written here rather than called from scipy, whose optimizers
are callback-driven and want to own the loop, while a lab optimizer hands out a
point now and receives its cost hours later, possibly out of order.

This is textbook generational differential evolution, scipy's deferred updating:
a whole population is proposed at once and no trial is judged until the
generation is complete, so the incumbent a trial competes against is the one it
was bred from. The barrier is the one
:meth:`~labscript_optimization.learners.base.Learner.propose` holds for every
learner declaring a generation.

The population is not carried between calls but rebuilt by walking the history,
where a proposal's position is its role: the first ``population_size``
proposals found the population, one to a slot, and each later block of
``population_size`` is a generation of trials, again one to a slot. A slot's
member is the best usable result that slot has produced, so a cost arriving
after its generation still competes for its own slot and disturbs no other.

As the Gaussian process's explorer it is asked for a point or two at a time over
its own shots alone, with no barrier: asynchronous differential evolution. Its
generation, and the budget that asks for, apply only when it is the selected
learner.
"""

from collections.abc import Sequence

import numpy as np

from .. import knobs
from ..observations import Observation
from ..space import ParameterSpace
from .base import ParameterSpaceLearner

#: The mutation strategies, each with how many other population members it draws
#: on. The mutation and the check on ``population_size`` both read the counts.
STRATEGIES = {"best1": 2, "best2": 4, "rand1": 3, "rand2": 5}


class DifferentialEvolutionLearner(ParameterSpaceLearner):
    """Evolve a population of parameter vectors, a generation at a time.

    Parameters
    ----------
    space : ParameterSpace
        The parameter space to search.
    rng : numpy.random.Generator
        Source of randomness.
    population_size : int
        Members in the population (the literature's NP), and so proposals per
        generation. :data:`STRATEGIES` sets the floor this refuses below. About
        eight members suit the budgets a lab usually runs, and sixteen suit
        budgets past a thousand shots.
    evolution_strategy : str
        The mutation, one of :data:`STRATEGIES`.
    mutation_scale : (float, float)
        ``(low, high)`` bounds on the differential weight, drawn once per
        ``ask`` call and shared by every trial it makes: once per generation
        when this learner is selected, once per trial or two as the Gaussian
        process's explorer, the variant known as dither.
    cross_over_probability : float
        Chance that a given coordinate comes from the mutant rather than the
        incumbent.
    trust_region : float or sequence of float or None
        Restrict sampling to this distance around the best member.
    """

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        population_size: int = 8,
        evolution_strategy: str = "best1",
        mutation_scale: Sequence[float] = (0.5, 1.0),
        cross_over_probability: float = 0.7,
        trust_region=None,
    ):
        super().__init__(space, rng)
        self.evolution_strategy = knobs.choice(
            "evolution_strategy", evolution_strategy, STRATEGIES
        )
        self.population_size = knobs.integer("population_size", population_size)
        # A mutation draws distinct members from the population minus the slot
        # it replaces, so it needs one member more than it draws on.
        draws = STRATEGIES[evolution_strategy]
        if self.population_size < draws + 1:
            raise ValueError(
                f"evolution_strategy {evolution_strategy!r} draws on {draws} "
                f"other members, so it needs a population of at least "
                f"{draws + 1}; population_size is {self.population_size}"
            )
        self.mutation_scale = knobs.pair("mutation_scale", mutation_scale)
        if not 0 <= self.mutation_scale[0] <= self.mutation_scale[1]:
            raise ValueError(
                f"mutation_scale must be an ordered non-negative pair, got "
                f"{mutation_scale!r}"
            )
        self.cross_over_probability = knobs.number(
            "cross_over_probability", cross_over_probability
        )
        if not 0 <= self.cross_over_probability <= 1:
            raise ValueError(
                f"cross_over_probability must be in [0, 1], got "
                f"{self.cross_over_probability}"
            )
        self.trust_region = space.absolute_trust_region(trust_region)

    #: A proposal's role is its position among this learner's own proposals.
    history_scope = "mine"

    @property
    def generation(self) -> int:
        """One whole population, judged as a whole: its members go out together."""
        return self.population_size

    def _replay(self, history: Sequence[Observation]):
        """The members' parameters and costs per slot, ``nan`` where vacant.

        A slot is vacant until one of its own proposals has a usable cost.
        """
        params = np.full((self.population_size, self.space.num_params), np.nan)
        costs = np.full(self.population_size, np.nan)
        for position, record in enumerate(history):
            if not record.usable:
                continue
            slot = position % self.population_size
            # Selection keeps the better of incumbent and trial: over a slot's
            # run, its cheapest usable result.
            if np.isnan(costs[slot]) or record.cost < costs[slot]:
                params[slot] = record.params
                costs[slot] = record.cost
        return params, costs

    def _sample_new_member(self, params: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """A founder's point: near the best member once one has a cost, else uniform."""
        if np.isnan(costs).all():
            return self.space.uniform(self.rng, 1)[0]
        best = params[int(np.nanargmin(costs))]
        return self.space.uniform(self.rng, 1, best, self.trust_region)[0]

    def _mutant(
        self, slot: int, params: np.ndarray, costs: np.ndarray, scale: float
    ) -> np.ndarray:
        occupied = np.flatnonzero(~np.isnan(costs))
        best = params[int(np.nanargmin(costs))]
        others = occupied[occupied != slot]
        drawn = params[
            self.rng.choice(
                others, size=STRATEGIES[self.evolution_strategy], replace=False
            )
        ]
        if self.evolution_strategy == "best1":
            return best + scale * (drawn[0] - drawn[1])
        if self.evolution_strategy == "rand1":
            return drawn[0] + scale * (drawn[1] - drawn[2])
        if self.evolution_strategy == "best2":
            return best + scale * (drawn[0] + drawn[1] - drawn[2] - drawn[3])
        return drawn[0] + scale * (drawn[1] + drawn[2] - drawn[3] - drawn[4])

    def _trial(
        self, slot: int, params: np.ndarray, costs: np.ndarray, scale: float
    ) -> np.ndarray:
        """A point for ``slot``: its mutant crossed over with its member."""
        draws = STRATEGIES[self.evolution_strategy]
        if int((~np.isnan(costs)).sum()) < draws + 1:
            # Too few members to draw distinct ones around this slot, so the
            # point is drawn as a founder's is.
            return self._sample_new_member(params, costs)

        mutant = self._mutant(slot, params, costs, scale)

        crossovers = (
            self.rng.random(self.space.num_params) < self.cross_over_probability
        )
        # At least one coordinate comes from the mutant, or the trial would copy
        # the incumbent and the generation would stall.
        crossovers[self.rng.integers(self.space.num_params)] = True
        trial = np.where(crossovers, mutant, params[slot])

        # A coordinate pushed out of bounds is resampled rather than clipped,
        # which would pile members onto the boundary.
        fallback = self.space.uniform(
            self.rng, 1, params[int(np.nanargmin(costs))], self.trust_region
        )[0]
        outside = (trial < self.space.minimum) | (trial > self.space.maximum)
        return np.where(outside, fallback, trial)

    def ask(self, history: Sequence[Observation], k: int) -> np.ndarray:
        """The next ``k`` proposals as a ``(k, num_params)`` array, without the barrier.

        Each is for the slot its position names.
        """
        params, costs = self._replay(history)
        # One differential weight per call, shared by every trial it makes.
        scale = self.rng.uniform(*self.mutation_scale)
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            slot = (len(history) + i) % self.population_size
            if np.isnan(costs[slot]):
                # An empty slot has no incumbent to cross over with, so it is
                # drawn as a founder is: the whole founding generation, and
                # later a slot whose founder produced no cost.
                proposals[i] = self._sample_new_member(params, costs)
            else:
                proposals[i] = self._trial(slot, params, costs, scale)
        return proposals
