"""Learner behaviour, against analytic cost functions.

Every learner is exercised through the one method the protocol defines, so
these tests survive any rewrite that keeps the protocol.
"""

import numpy as np
import pytest

from labscript_optimization.learners import (
    DifferentialEvolutionLearner,
    DirectedRandomLearner,
    GaussianProcessLearner,
    InsufficientData,
    RandomLearner,
    TwoPhaseLearner,
)

from conftest import observe, run_loop


def sphere(params):
    """Minimum 0 at the origin."""
    return float(np.sum(np.asarray(params) ** 2))


def offset_sphere(params):
    """Minimum 0 at (1.3, -2.1)."""
    return float((params[0] - 1.3) ** 2 + (params[1] + 2.1) ** 2)


def every_learner(space, rng):
    return [
        RandomLearner(space, rng),
        DirectedRandomLearner(space, rng, trust_region=0.1),
        DifferentialEvolutionLearner(space, rng, population_size=3),
        TwoPhaseLearner(RandomLearner(space, rng), RandomLearner(space, rng), 3),
    ]


@pytest.mark.parametrize('k', [1, 3, 7])
def test_proposals_have_the_requested_shape_and_stay_in_bounds(space, rng, k):
    history = [observe(i, p, sphere(p)) for i, p in enumerate(space.uniform(rng, 20))]
    for learner in every_learner(space, rng):
        proposals = np.atleast_2d(learner.propose(history, k))
        assert proposals.shape == (k, space.num_params), type(learner).__name__
        assert space.contains(proposals).all(), type(learner).__name__


def test_learners_propose_from_an_empty_history(space, rng):
    for learner in every_learner(space, rng):
        proposals = np.atleast_2d(learner.propose([], 2))
        assert space.contains(proposals).all(), type(learner).__name__


# --- directed random -------------------------------------------------------


def test_directed_random_searches_near_points_it_has_seen(space, rng):
    """The trust region has to actually engage once there is history.

    A guard that sends every draw to the whole space regardless of the history
    turns this learner into a plain random one, silently.
    """
    seen = np.array([[1.0, 1.0], [1.2, 0.8], [-3.0, 4.0], [0.0, -1.0]])
    history = [observe(i, p, sphere(p)) for i, p in enumerate(seen)]

    learner = DirectedRandomLearner(space, rng, trust_region=0.05)
    proposals = learner.propose(history, 300)

    # 5% of a range of 10 is 0.5 in each direction.
    distances = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2)
    assert (distances.min(axis=1) <= 0.5 + 1e-9).all()


def test_directed_random_is_not_derailed_by_a_bad_run(space, rng):
    """An infinite cost must not disable the trust region.

    The band is measured across the range of observed costs, so an infinite
    worst cost makes its bounds undefined and every point falls outside it.
    """
    seen = np.array([[1.0, 1.0], [1.2, 0.8], [-3.0, 4.0]])
    history = [observe(i, p, sphere(p)) for i, p in enumerate(seen)]
    history.append(observe('bad', [4.9, 4.9], float('inf')))

    learner = DirectedRandomLearner(space, rng, trust_region=0.05)
    proposals = learner.propose(history, 300)

    distances = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2)
    assert (distances.min(axis=1) <= 0.5 + 1e-9).all()
    # And it must not centre on the point that produced the bad cost.
    assert (np.abs(proposals - np.array([4.9, 4.9])).max(axis=1) > 0.5).all()


def test_directed_random_explores_the_whole_space_when_told_to(space, rng):
    history = [observe(0, [1.0, 1.0], 1.0), observe(1, [1.1, 1.1], 2.0)]
    learner = DirectedRandomLearner(
        space, rng, trust_region=0.01, explore_fraction=1.0
    )
    proposals = learner.propose(history, 400)
    assert proposals.min() < -4.0 and proposals.max() > 4.0


def test_directed_random_without_a_trust_region_is_a_random_learner(space, rng):
    history = [observe(0, [1.0, 1.0], 1.0), observe(1, [1.1, 1.1], 2.0)]
    learner = DirectedRandomLearner(space, rng, trust_region=None)
    proposals = learner.propose(history, 400)
    assert proposals.min() < -4.0 and proposals.max() > 4.0


def test_directed_random_prefers_mediocre_points_over_the_best_one(space, rng):
    """The default band sits away from the best point, on purpose.

    Centring the search on middling results is what makes this learner explore
    rather than refine, which is the reason it exists.
    """
    seen = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    costs = [0.0, 10.0, 25.0, 30.0]
    history = [observe(i, p, c) for i, (p, c) in enumerate(zip(seen, costs))]

    learner = DirectedRandomLearner(
        space, rng, trust_region=0.02, trust_range=(0.1, 0.25)
    )
    proposals = learner.propose(history, 300)
    nearest = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2).argmin(axis=1)
    # The band runs from 10% to 25% of the way from the worst cost (30) towards
    # the best (0), that is [22.5, 27], which picks out the point costing 25
    # and never the best one.
    assert set(np.unique(nearest)) == {2}


def test_directed_random_falls_back_to_the_best_point_when_the_band_is_empty(
    space, rng
):
    """With no observation in the band there is still somewhere to search."""
    seen = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    costs = [0.0, 10.0, 20.0, 30.0]  # nothing lands in [22.5, 27]
    history = [observe(i, p, c) for i, (p, c) in enumerate(zip(seen, costs))]

    learner = DirectedRandomLearner(
        space, rng, trust_region=0.02, trust_range=(0.1, 0.25)
    )
    proposals = learner.propose(history, 200)
    nearest = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2).argmin(axis=1)
    assert set(np.unique(nearest)) == {0}


# --- differential evolution ------------------------------------------------


@pytest.mark.parametrize('strategy', ['best1', 'best2', 'rand1', 'rand2'])
def test_differential_evolution_finds_the_minimum(space, rng, strategy):
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=5, evolution_strategy=strategy
    )
    history = run_loop(learner, space, sphere, batches=50, k=4, rng=rng)
    assert min(o.cost for o in history) < 0.05


def test_differential_evolution_state_depends_only_on_the_history(space, rng):
    """Two learners given the same history must be in the same state.

    The population is not carried between calls; it is rebuilt from the
    history, which is what keeps costs arriving out of order from mattering.
    """
    history = [observe(i, p, sphere(p)) for i, p in enumerate(space.uniform(rng, 40))]
    first = DifferentialEvolutionLearner(
        space, np.random.default_rng(1), population_size=3
    )._replay(history)
    second = DifferentialEvolutionLearner(
        space, np.random.default_rng(2), population_size=3
    )._replay(history)
    np.testing.assert_allclose(first[1], second[1])
    assert first[2] == second[2]


@pytest.mark.parametrize(
    'kwargs, message',
    [
        (dict(population_size=1), 'at least 5 members'),
        (dict(evolution_strategy='nope'), 'evolution_strategy'),
        (dict(cross_over_probability=2.0), 'cross_over_probability'),
        (dict(mutation_scale=(1.0, 0.5)), 'mutation_scale'),
    ],
)
def test_impossible_differential_evolution_settings_are_rejected(
    space, rng, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        DifferentialEvolutionLearner(space, rng, **{'population_size': 5, **kwargs})


# --- gaussian process ------------------------------------------------------


def test_gaussian_process_refuses_before_it_has_enough_data(space, rng):
    learner = GaussianProcessLearner(space, rng, minimum_observations=6)
    with pytest.raises(InsufficientData):
        learner.propose([observe(0, [0.0, 0.0], 1.0)], 1)


def test_gaussian_process_finds_the_minimum(space, rng):
    learner = GaussianProcessLearner(space, rng)
    seed = [
        observe(i, p, offset_sphere(p))
        for i, p in enumerate(space.uniform(np.random.default_rng(4), 12))
    ]
    history = run_loop(
        learner, space, offset_sphere, batches=12, k=4, rng=rng, history=seed
    )
    best = min(history, key=lambda o: o.cost)
    assert best.cost < 0.05
    np.testing.assert_allclose(best.params, [1.3, -2.1], atol=0.3)


def test_a_gaussian_process_batch_does_not_repeat_itself(space, rng):
    """Every point of a batch must be somewhere new.

    Points chosen for their uncertainty all chase the same unexplored corner
    unless each one is folded into the fit before the next is chosen, and a
    repeated proposal is a wasted shot.
    """
    learner = GaussianProcessLearner(space, rng)
    history = [
        observe(i, p, offset_sphere(p))
        for i, p in enumerate(space.uniform(np.random.default_rng(5), 12))
    ]
    proposals = learner.propose(history, 6)
    separations = np.linalg.norm(
        proposals[:, None, :] - proposals[None, :, :], axis=2
    )
    off_diagonal = separations[~np.eye(len(proposals), dtype=bool)]
    assert off_diagonal.min() > 1e-6


def test_a_gaussian_process_describes_the_real_data_after_proposing(space, rng):
    """Proposing must not leave the model believing its own guesses."""
    learner = GaussianProcessLearner(space, rng)
    history = [
        observe(i, p, offset_sphere(p))
        for i, p in enumerate(space.uniform(np.random.default_rng(6), 12))
    ]
    learner.propose(history, 1)
    before = learner.predict(history[0].params)[0]
    learner.propose(history, 5)
    after = learner.predict(history[0].params)[0]
    np.testing.assert_allclose(before, after)


def test_gaussian_process_uses_per_point_uncertainties(space, rng):
    """A noisy point should be trusted less than an exact one."""
    points = space.uniform(np.random.default_rng(7), 12)
    exact = [observe(i, p, offset_sphere(p), uncer=0.0) for i, p in enumerate(points)]
    noisy = [observe(i, p, offset_sphere(p), uncer=5.0) for i, p in enumerate(points)]

    tight = GaussianProcessLearner(space, rng)
    tight.fit(exact)
    loose = GaussianProcessLearner(space, rng)
    loose.fit(noisy)

    probe = np.array([0.0, 0.0])
    assert loose.predict(probe)[1][0] > tight.predict(probe)[1][0]


# --- two phase -------------------------------------------------------------


class Ready:
    def propose(self, history, k):
        return np.zeros((k, 2))


class NeverReady:
    def propose(self, history, k):
        raise InsufficientData('not yet')


def test_two_phase_trains_first_then_hands_over(space, rng):
    learner = TwoPhaseLearner(RandomLearner(space, rng), Ready(), num_training=3)
    history = []
    phases = []
    for i in range(6):
        proposal = learner.propose(history, 1)
        phases.append(learner.last_phase)
        history.append(observe(i, proposal[0], float(i)))
    assert phases == ['training'] * 3 + ['main'] * 3


def test_two_phase_keeps_training_when_the_main_learner_cannot_propose(space, rng):
    learner = TwoPhaseLearner(RandomLearner(space, rng), NeverReady(), num_training=1)
    history = [observe(i, [0.0, 0.0], float(i)) for i in range(5)]
    proposals = learner.propose(history, 2)
    assert learner.last_phase == 'training (fallback)'
    assert space.contains(proposals).all()


def test_shots_without_a_usable_cost_do_not_count_as_training(space, rng):
    learner = TwoPhaseLearner(RandomLearner(space, rng), Ready(), num_training=2)
    history = [
        observe('a', [0.0, 0.0], 1.0, bad=True),
        observe('b', [0.0, 0.0], float('inf')),
        observe('c', [0.0, 0.0], float('nan')),
    ]
    learner.propose(history, 1)
    assert learner.last_phase == 'training'
