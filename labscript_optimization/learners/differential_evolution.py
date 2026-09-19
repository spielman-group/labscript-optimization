"""Differential evolution, in ask/tell form.

Carried over from M-LOOP, which hand-codes the algorithm rather than calling
scipy. That is the right shape here: scipy's optimisers are callback-driven and
want to own the loop, while a lab optimiser has to hand out a point now and
receive its cost hours later, possibly out of order.

The population is not carried between calls. It is rebuilt by walking the
history in proposal order, which makes the learner a function of that history
and keeps the bookkeeping for shots in flight out of the algorithm: a trial
that never reported simply never displaced anything.
"""

from typing import Sequence

import numpy as np

from ..observations import Observation, usable
from ..space import ParameterSpace

#: The mutation strategies, and how many other population members each one
#: draws on. The counts are read by :meth:`DifferentialEvolutionLearner._mutant`
#: and by the population guard, so neither can drift from the other.
STRATEGIES = {"best1": 2, "best2": 4, "rand1": 3, "rand2": 5}


class DifferentialEvolutionLearner:
    """Evolve a population of parameter vectors.

    Args:
        space: The parameter space to search.
        rng: Source of randomness.
        population_size: Multiplier on the parameter count; the population
            holds ``population_size * num_params`` members. How few members
            will do depends on the strategy: see :data:`STRATEGIES`.
        evolution_strategy: Which mutation to use, one of :data:`STRATEGIES`.
        mutation_scale: ``(low, high)`` bounds on the differential weight,
            redrawn each generation.
        cross_over_probability: Chance that a given coordinate comes from the
            mutant rather than the incumbent.
        restart_tolerance: Restart the population once the spread of its costs
            falls below this fraction of the spread it started with.
        trust_region: Restrict sampling to this distance around the best member.
        first_params: A point to return as the very first proposal.
    """

    def __init__(
        self,
        space: ParameterSpace,
        rng: np.random.Generator,
        population_size: int = 15,
        evolution_strategy: str = "best1",
        mutation_scale: Sequence[float] = (0.5, 1.0),
        cross_over_probability: float = 0.7,
        restart_tolerance: float = 0.01,
        trust_region=None,
        first_params: np.ndarray | None = None,
    ):
        self.space = space
        self.rng = rng
        if evolution_strategy not in STRATEGIES:
            raise ValueError(
                f"evolution_strategy must be one of {tuple(STRATEGIES)}, got "
                f"{evolution_strategy!r}"
            )
        self.evolution_strategy = evolution_strategy
        self.num_members = int(population_size) * space.num_params
        # A mutation draws distinct members from the population minus the slot
        # it is replacing, so it needs one member more than it draws on.
        draws = STRATEGIES[evolution_strategy]
        if self.num_members < draws + 1:
            raise ValueError(
                f"evolution_strategy {evolution_strategy!r} draws on {draws} "
                f"other members, so it needs a population of at least "
                f"{draws + 1}; population_size {population_size} over "
                f"{space.num_params} parameters gives {self.num_members}"
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
        self.restart_tolerance = float(restart_tolerance)
        self.trust_region = space.absolute_trust_region(trust_region)

        if first_params is None:
            first_params = space.start
        self.first_params = (
            None if first_params is None else np.array(first_params, dtype=float)
        )
        if self.first_params is not None and not self.space.contains(self.first_params):
            raise ValueError(f"first_params outside the bounds: {self.first_params}")

    def _replay(self, history: Sequence[Observation]):
        """Rebuild the population by walking the history in proposal order.

        Returns the population parameters, their costs, and the slot the next
        trial targets. While the population is still being filled the returned
        arrays are short and the slot is meaningless.
        """
        params: list[np.ndarray] = []
        costs: list[float] = []
        init_spread = None
        slot = 0

        for obs in usable(history):
            if len(costs) < self.num_members:
                params.append(np.asarray(obs.params, dtype=float))
                costs.append(float(obs.cost))
                if len(costs) == self.num_members:
                    init_spread = float(np.std(costs))
                    slot = 0
                continue

            if obs.cost < costs[slot]:
                params[slot] = np.asarray(obs.params, dtype=float)
                costs[slot] = float(obs.cost)
            slot += 1
            if slot == self.num_members:
                slot = 0
                # A population whose costs have collapsed together has found a
                # minimum and stopped exploring; start again elsewhere.
                if init_spread and np.std(costs) < self.restart_tolerance * init_spread:
                    params, costs, init_spread = [], [], None

        return params, costs, slot

    def _sample_new_member(self, params: list, costs: list) -> np.ndarray:
        """Draw a point while the population is still being filled."""
        if self.trust_region is None or not costs:
            return self.space.uniform(self.rng, 1)[0]
        best = params[int(np.argmin(costs))]
        low = np.maximum(self.space.minimum, best - self.trust_region)
        high = np.minimum(self.space.maximum, best + self.trust_region)
        return self.rng.uniform(low, high)

    def _distinct_indices(self, exclude: int, count: int) -> np.ndarray:
        choices = np.delete(np.arange(self.num_members), exclude)
        return self.rng.choice(choices, size=count, replace=False)

    def _mutant(self, slot: int, population: np.ndarray, best: int, scale: float):
        # How many members to draw comes from STRATEGIES, which is also what
        # sets the minimum population, so a strategy can never ask for more
        # points than the constructor guaranteed it.
        drawn = population[
            self._distinct_indices(slot, STRATEGIES[self.evolution_strategy])
        ]
        if self.evolution_strategy == "best1":
            return population[best] + scale * (drawn[0] - drawn[1])
        if self.evolution_strategy == "rand1":
            return drawn[0] + scale * (drawn[1] - drawn[2])
        if self.evolution_strategy == "best2":
            return population[best] + scale * (
                drawn[0] + drawn[1] - drawn[2] - drawn[3]
            )
        return drawn[0] + scale * (drawn[1] + drawn[2] - drawn[3] - drawn[4])

    def _trial(self, slot: int, population: np.ndarray, costs: np.ndarray) -> np.ndarray:
        best = int(np.argmin(costs))
        scale = self.rng.uniform(*self.mutation_scale)
        mutant = self._mutant(slot, population, best, scale)

        crossovers = self.rng.random(self.space.num_params) < self.cross_over_probability
        # At least one coordinate must come from the mutant, or the trial would
        # be a copy of the incumbent and the generation would stall.
        crossovers[self.rng.integers(self.space.num_params)] = True
        trial = np.where(crossovers, mutant, population[slot])

        # A coordinate pushed out of bounds is resampled rather than clipped,
        # which would pile members onto the boundary.
        if self.trust_region is None:
            fallback = self.space.uniform(self.rng, 1)[0]
        else:
            low = np.maximum(self.space.minimum, population[best] - self.trust_region)
            high = np.minimum(self.space.maximum, population[best] + self.trust_region)
            fallback = self.rng.uniform(low, high)
        outside = (trial < self.space.minimum) | (trial > self.space.maximum)
        return np.where(outside, fallback, trial)

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        if not history and self.first_params is not None:
            proposals = self.space.uniform(self.rng, k)
            proposals[0] = self.first_params
            return proposals

        params, costs, slot = self._replay(history)
        proposals = np.empty((k, self.space.num_params))
        for i in range(k):
            if len(costs) < self.num_members:
                # Still filling: each proposal is another founding member. The
                # ones already proposed in this batch are not yet members, so
                # they cannot be drawn around, which only costs some locality.
                proposals[i] = self._sample_new_member(params, costs)
            else:
                proposals[i] = self._trial(
                    (slot + i) % self.num_members,
                    np.array(params),
                    np.array(costs),
                )
        return proposals
