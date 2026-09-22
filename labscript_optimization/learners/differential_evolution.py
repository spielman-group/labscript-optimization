"""Differential evolution, in ask/tell form.

The algorithm is written out here rather than called from scipy, whose
optimisers are callback-driven and want to own the loop, while a lab optimiser
has to hand out a point now and receive its cost hours later, possibly out of
order.

This is textbook generational differential evolution -- scipy's deferred
updating. A whole population is proposed at once and none of its trials is
judged until the generation is complete, so the incumbent a trial competes
against is the one it was bred from.

The population is not carried between calls. It is rebuilt by walking the
history, where a proposal's position is its role: the first ``population_size``
proposals found the population, one to a slot, and every block of
``population_size`` after them is a generation of trials, again one to a slot.
A slot's member is the best usable result that slot has produced, so a cost
arriving after the generation that would have used it still competes for its
own slot and disturbs no other -- which is what textbook differential
evolution would have done with it had it arrived in time.
"""

from typing import Sequence

import numpy as np

from ..observations import Observation
from ..space import ParameterSpace
from .base import ParameterSpaceLearner

#: The mutation strategies, and how many other population members each one
#: draws on. The counts are read by :meth:`DifferentialEvolutionLearner.mutant`
#: and by the population guard, so neither can drift from the other.
STRATEGIES = {"best1": 2, "best2": 4, "rand1": 3, "rand2": 5}


class DifferentialEvolutionLearner(ParameterSpaceLearner):
    """Evolve a population of parameter vectors, a generation at a time.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        population_size: How many members the population holds, which is the
            literature's NP given directly, and so also how many proposals a
            generation carries. How few will do depends on the strategy: see
            :data:`STRATEGIES`, which sets the floor this refuses below. That
            floor is a long way under a population that searches well: around
            eight members is where one stops converging prematurely, and a
            budget over a thousand shots is worth sixteen. Both numbers are
            read off a sweep over four analytic test functions at two to eight
            parameters, recorded in ``codex_issues_proposal.md`` with its
            caveats. Rules of thumb scaling it with the parameter count are
            for choosing a number, not the shape of the setting.
        evolution_strategy: Which mutation to use, one of :data:`STRATEGIES`.
        mutation_scale: ``(low, high)`` bounds on the differential weight,
            redrawn each generation.
        cross_over_probability: Chance that a given coordinate comes from the
            mutant rather than the incumbent.
        trust_region: Restrict sampling to this distance around the best member.
    """

    last_phase = "main"

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
        if evolution_strategy not in STRATEGIES:
            raise ValueError(
                f"evolution_strategy must be one of {tuple(STRATEGIES)}, got "
                f"{evolution_strategy!r}"
            )
        self.evolution_strategy = evolution_strategy
        self.population_size = int(population_size)
        # A mutation draws distinct members from the population minus the slot
        # it is replacing, so it needs one member more than it draws on.
        draws = STRATEGIES[evolution_strategy]
        if self.population_size < draws + 1:
            raise ValueError(
                f"evolution_strategy {evolution_strategy!r} draws on {draws} "
                f"other members, so it needs a population of at least "
                f"{draws + 1}; population_size is {self.population_size}"
            )
        self.mutation_scale = tuple(float(m) for m in mutation_scale)
        if len(self.mutation_scale) != 2 or not 0 <= self.mutation_scale[0] <= self.mutation_scale[1]:
            raise ValueError(
                f"mutation_scale must be an ordered non-negative pair, got "
                f"{mutation_scale!r}"
            )
        self.cross_over_probability = float(cross_over_probability)
        if not 0 <= self.cross_over_probability <= 1:
            raise ValueError(
                f"cross_over_probability must be in [0, 1], got "
                f"{self.cross_over_probability}"
            )
        self.trust_region = space.absolute_trust_region(trust_region)

    @property
    def generation(self) -> int:
        """One whole population.

        A generation is judged as a whole, so its members go out together and
        nothing is proposed until all of them have been answered for.
        """
        return self.population_size

    def replay(self, history: Sequence[Observation]):
        """The population as the history so far leaves it.

        Returns the members' parameters and costs, one row and one cost per
        slot, with ``nan`` for a slot that is vacant. A slot is vacant until
        one of its own proposals comes back with a usable cost: its founder
        may have been dropped or measured nothing, and no other slot's result
        stands in for it.
        """
        params = np.full((self.population_size, self.space.num_params), np.nan)
        costs = np.full(self.population_size, np.nan)
        for position, record in enumerate(history):
            if not record.usable:
                continue
            slot = position % self.population_size
            # Selection keeps the better of the incumbent and the trial, which
            # over a slot's whole run is the cheapest usable result it has had.
            if np.isnan(costs[slot]) or record.cost < costs[slot]:
                params[slot] = record.params
                costs[slot] = record.cost
        return params, costs

    def sample_new_member(self, params: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Draw a point for a slot that has nothing to evolve."""
        if np.isnan(costs).all():
            return self.space.uniform(self.rng, 1)[0]
        best = params[int(np.nanargmin(costs))]
        return self.space.uniform(self.rng, 1, best, self.trust_region)[0]

    def mutant(
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

    def trial(self, slot: int, params: np.ndarray, costs: np.ndarray) -> np.ndarray:
        draws = STRATEGIES[self.evolution_strategy]
        if int((~np.isnan(costs)).sum()) < draws + 1:
            # Too few slots hold a member for the mutation to draw distinct
            # ones from around this slot, so there is no population to breed
            # from and the point is drawn the way a founder is.
            return self.sample_new_member(params, costs)

        scale = self.rng.uniform(*self.mutation_scale)
        mutant = self.mutant(slot, params, costs, scale)

        crossovers = self.rng.random(self.space.num_params) < self.cross_over_probability
        # At least one coordinate must come from the mutant, or the trial would
        # be a copy of the incumbent and the generation would stall.
        crossovers[self.rng.integers(self.space.num_params)] = True
        trial = np.where(crossovers, mutant, params[slot])

        # A coordinate pushed out of bounds is resampled rather than clipped,
        # which would pile members onto the boundary.
        fallback = self.space.uniform(
            self.rng, 1, params[int(np.nanargmin(costs))], self.trust_region
        )[0]
        outside = (trial < self.space.minimum) | (trial > self.space.maximum)
        return np.where(outside, fallback, trial)

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        params, costs = self.replay(history)
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            slot = (len(history) + i) % self.population_size
            if np.isnan(costs[slot]):
                # An empty slot has no incumbent to cross over with, so its
                # proposal is drawn founder-style. That is the whole of the
                # founding generation, where every slot is empty, and later on
                # a slot whose founder produced no cost, which stays empty
                # until one of its trials lands.
                proposals[i] = self.sample_new_member(params, costs)
            else:
                proposals[i] = self.trial(slot, params, costs)
        return proposals
