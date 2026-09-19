"""Learner behaviour, against analytic cost functions.

Every learner is exercised through the one method the protocol defines, so
these tests survive any rewrite that keeps the protocol.
"""

import numpy as np
import pytest

from labscript_optimization import learners
from labscript_optimization.config import Config
from labscript_optimization.learners import (
    DifferentialEvolutionLearner,
    DirectedRandomLearner,
    GaussianProcessLearner,
    InsufficientData,
    RandomLearner,
    TwoPhaseLearner,
    build,
)
from labscript_optimization.space import Parameter, ParameterSpace

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


def test_a_learner_without_phases_of_its_own_still_reports_one(space, rng):
    """The session reads the phase off whatever learner it was handed.

    Reading it with a default would paper over the opposite case as well: a
    learner that grows phases and forgets to publish them would be reported
    as the main one throughout.
    """
    for learner in [
        RandomLearner(space, rng),
        DirectedRandomLearner(space, rng, trust_region=0.1),
        DifferentialEvolutionLearner(space, rng, population_size=3),
        GaussianProcessLearner(space, rng),
    ]:
        assert learner.last_phase == 'main', type(learner).__name__


# --- the opening point -----------------------------------------------------


def learners_starting_at(space, rng, first_params):
    """Every learner that can be told where to begin."""
    return [
        RandomLearner(space, rng, first_params=first_params),
        DirectedRandomLearner(space, rng, trust_region=0.1, first_params=first_params),
        DifferentialEvolutionLearner(
            space, rng, population_size=3, first_params=first_params
        ),
    ]


def test_a_learner_told_where_to_start_proposes_that_point_first(space, rng):
    """A run begins from the settings the lab already had.

    Only the first proposal, though: the rest of the opening batch has
    nothing to go on and explores.
    """
    start = np.array([1.5, -2.5])
    for learner in learners_starting_at(space, rng, start):
        proposals = np.atleast_2d(learner.propose([], 3))
        np.testing.assert_allclose(proposals[0], start, err_msg=type(learner).__name__)
        assert not np.allclose(proposals[1], start), type(learner).__name__


def test_the_opening_point_defaults_to_the_configured_start(rng):
    started = ParameterSpace(
        [
            Parameter('x', 'g_x', -5.0, 5.0, start=1.5),
            Parameter('y', 'g_y', -5.0, 5.0, start=-2.5),
        ]
    )
    for learner in learners_starting_at(started, rng, None):
        np.testing.assert_allclose(
            learner.propose([], 1)[0], [1.5, -2.5], err_msg=type(learner).__name__
        )


def test_the_opening_point_is_offered_only_while_nothing_has_run(space, rng):
    history = [observe(0, [0.0, 0.0], 1.0)]
    start = np.array([1.5, -2.5])
    for learner in learners_starting_at(space, rng, start):
        proposal = np.atleast_2d(learner.propose(history, 1))[0]
        assert not np.allclose(proposal, start), type(learner).__name__


@pytest.mark.parametrize(
    'first_params, message',
    [
        ([9.0, 0.0], 'outside the bounds'),
        ([0.0], 'shape'),
        ([[0.0, 0.0], [1.0, 1.0]], 'shape'),
    ],
)
def test_an_impossible_opening_point_is_refused_at_construction(
    space, rng, first_params, message
):
    """Construction is the last moment at which it costs nothing to fix.

    A point of the wrong shape is the one that gets through unnoticed: a
    single coordinate broadcasts over the whole vector, and several points
    make the bounds check answer once per row, so neither is caught by asking
    only whether the values are in range.
    """
    for build_one in (
        RandomLearner,
        DirectedRandomLearner,
        DifferentialEvolutionLearner,
    ):
        with pytest.raises(ValueError, match=message):
            build_one(space, rng, first_params=first_params)


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


@pytest.mark.parametrize(
    'trust_range', [(0.25, 0.1), (0.1,), (0.1, 1.5), (-0.1, 0.25)]
)
def test_an_impossible_trust_range_is_refused(space, rng, trust_range):
    """Including one written backwards, which is not silently put in order.

    A backwards pair asks for a band running from the best cost towards the
    worst, which is nothing the learner can honour; sorting it would run a
    search the lab did not ask for and never say so.
    """
    with pytest.raises(ValueError, match='trust_range'):
        DirectedRandomLearner(space, rng, trust_region=0.05, trust_range=trust_range)


# --- differential evolution ------------------------------------------------


#: The smallest population each strategy can mutate: it draws distinct members
#: from the population minus the slot it is replacing, so it needs one more
#: member than it draws on.
SMALLEST_POPULATIONS = [('best1', 3), ('rand1', 4), ('best2', 5), ('rand2', 6)]


def one_parameter_space():
    """A single parameter, so a population of any size is reachable."""
    return ParameterSpace([Parameter('x', 'g_x', -5.0, 5.0)])


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


@pytest.mark.parametrize('strategy, smallest', SMALLEST_POPULATIONS)
def test_every_strategy_evolves_at_its_own_smallest_population(rng, strategy, smallest):
    """A population the constructor accepted has to survive the mutation.

    The strategies draw different numbers of distinct members, so one minimum
    does not serve all four. A population that passes construction but is too
    small to mutate gets through the filling phase and then fails on the first
    trial, which is mid-session, with the apparatus running.
    """
    space = one_parameter_space()
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=smallest, evolution_strategy=strategy
    )
    history = run_loop(learner, space, sphere, batches=8, k=2, rng=rng)
    # Long past the filling phase, so mutation did the bulk of the proposing.
    assert len(history) > 2 * learner.num_members
    assert space.contains(np.array([o.params for o in history])).all()


@pytest.mark.parametrize('strategy, smallest', SMALLEST_POPULATIONS)
def test_a_population_too_small_for_its_strategy_is_refused(rng, strategy, smallest):
    """And a population below that minimum is refused at construction.

    Construction is the last moment at which the configuration can be fixed
    for free; the alternative is finding out on the first trial.
    """
    with pytest.raises(ValueError, match=strategy) as refusal:
        DifferentialEvolutionLearner(
            one_parameter_space(),
            rng,
            population_size=smallest - 1,
            evolution_strategy=strategy,
        )
    # Naming the strategy alone does not say what to do about it: the message
    # has to carry the population it got and the one that strategy needs.
    assert str(smallest) in str(refusal.value)
    assert str(smallest - 1) in str(refusal.value)


@pytest.mark.parametrize(
    'kwargs, message',
    [
        (dict(population_size=1), 'at least 3'),
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


def gaussian_process_history(space, seed, count=12):
    """A spread of observations wide enough for the posterior to mean something."""
    points = space.uniform(np.random.default_rng(seed), count)
    return [observe(i, p, offset_sphere(p)) for i, p in enumerate(points)]


def test_gaussian_process_refuses_before_it_has_enough_data(space, rng):
    learner = GaussianProcessLearner(space, rng, minimum_observations=6)
    with pytest.raises(InsufficientData):
        learner.propose([observe(0, [0.0, 0.0], 1.0)], 1)


def test_gaussian_process_finds_the_minimum(space, rng):
    learner = GaussianProcessLearner(space, rng)
    history = run_loop(
        learner,
        space,
        offset_sphere,
        batches=12,
        k=4,
        rng=rng,
        history=gaussian_process_history(space, 4),
    )
    best = min(history, key=lambda o: o.cost)
    assert best.cost < 0.05
    np.testing.assert_allclose(best.params, [1.3, -2.1], atol=0.3)


@pytest.mark.parametrize('generation_size, carried, count', [(4, 12, 15), (8, 6, 7)])
def test_gaussian_process_state_depends_only_on_the_history(
    space, generation_size, carried, count
):
    """Two learners given the same history must hold the same model.

    The kernel hyperparameters are cached between calls, so they have to be a
    function of the history alone: an instance that has been fitting all
    session must arrive at what a fresh one computes, not at a kernel fitted to
    however much it happened to hold when the cache was last filled. The second
    case is a history short of one full generation, where there is no whole
    generation to fit to and the cache has to give way on every arrival.
    """
    history = gaussian_process_history(space, 9, count=count)
    all_session = GaussianProcessLearner(
        space, np.random.default_rng(1), generation_size=generation_size
    )
    all_session.fit(history[:carried])
    all_session.fit(history)

    fresh = GaussianProcessLearner(
        space, np.random.default_rng(2), generation_size=generation_size
    )
    fresh.fit(history)

    np.testing.assert_allclose(all_session._kernel.theta, fresh._kernel.theta)
    # And so the same proposals, once the two stand at the same point in their
    # own rng streams: that position is the one thing the history does not fix.
    all_session.rng = np.random.default_rng(3)
    fresh.rng = np.random.default_rng(3)
    np.testing.assert_allclose(
        all_session.propose(history, 1), fresh.propose(history, 1)
    )


#: The generation the exploration tests below configure, and so the number of
#: proposals the schedule takes to come back round to its greedy step.
GENERATION = 4


def exploring_and_greedy(space, count):
    """The same proposal made with the exploration weight up and turned off.

    The two learners share a seed and see the same history, so the acquisition
    weight is the only thing that differs between them.
    """
    history = gaussian_process_history(space, 5, count=count)
    greedy = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=0.0, generation_size=GENERATION
    )
    explorer = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=50.0, generation_size=GENERATION
    )
    return float(
        np.linalg.norm(explorer.propose(history, 1)[0] - greedy.propose(history, 1)[0])
    )


def test_the_exploration_weight_reaches_a_proposal_asked_for_on_its_own(space):
    """A session running one shot at a time still has to explore.

    It asks for a full batch once and then for a single point per completed
    shot, so a weight that stepped with the position within a batch would
    stand at its greedy first step for the whole run and uncer_bias would do
    nothing whatever in the lab.
    """
    assert exploring_and_greedy(space, count=13) > 0.1


@pytest.mark.parametrize('count', [12, 13, 14, 15, 16])
def test_the_exploration_schedule_advances_as_observations_arrive(space, count):
    """The weight steps once per proposal and cycles over a generation.

    One proposal in each generation is purely greedy and the rest look
    progressively further afield, which is how M-LOOP spends a generation.
    Reading the position off the history rather than a counter is what keeps
    a learner handed the same history proposing the same thing.
    """
    apart = exploring_and_greedy(space, count=count)
    if count % GENERATION:
        assert apart > 0.1
    else:
        # A weight of zero times anything is the greedy proposal itself.
        assert apart == 0.0


def test_a_gaussian_process_batch_does_not_repeat_itself(space, rng):
    """Two picks made at the same exploration weight must land apart.

    Points chosen for their uncertainty chase the same unexplored corner unless
    each one is folded into the fit before the next is chosen, and a repeated
    proposal is a wasted shot. The greedy pick is the exception and is left out
    below: at a weight of zero the acquisition is the posterior mean, and
    folding a point in at its own predicted cost leaves that mean where it was,
    so a batch spanning two generations asks for the same greedy point twice.
    """
    learner = GaussianProcessLearner(space, rng, generation_size=GENERATION)
    history = gaussian_process_history(space, 5)
    proposals = learner.propose(history, GENERATION + 2)
    # Twelve observations in hand, so the weights run 0, 1, 2, 3, 0, 1 and the
    # sixth pick repeats the weight of the second. Nothing but the fold-in
    # keeps it off that point: without it the two land 4e-6 apart.
    assert np.linalg.norm(proposals[5] - proposals[1]) > 1e-3


def test_a_gaussian_process_describes_the_real_data_after_proposing(space, rng):
    """Proposing must not leave the model believing its own guesses."""
    learner = GaussianProcessLearner(space, rng)
    history = gaussian_process_history(space, 6)
    learner.propose(history, 1)
    before = learner.predict(history[0].params)[0]
    learner.propose(history, 5)
    after = learner.predict(history[0].params)[0]
    np.testing.assert_allclose(before, after)


def test_a_gaussian_process_describes_the_real_data_after_a_proposal_fails(
    space, rng
):
    """Even a batch that dies partway must leave no invented points behind.

    Folding a batch's own picks into the fit is what stops them all chasing one
    corner, but those picks are guesses at what the apparatus will report. If
    the search then raises -- a minimiser giving up, a prediction on a
    degenerate kernel -- anything reading the model next, for a prediction or
    for where it thinks the optimum is, would be reading those guesses back as
    measurements.
    """
    learner = GaussianProcessLearner(space, rng)
    history = gaussian_process_history(space, 8)
    # The first pick of a batch, and so where its first invented point lands.
    probe = learner.propose(history, 1)[0]
    before = learner.predict(probe)

    searches = []
    search = learner._minimise_acquisition

    def give_up_after_the_first(*args, **kwargs):
        searches.append(1)
        if len(searches) > 1:
            raise RuntimeError('the minimiser gave up')
        return search(*args, **kwargs)

    learner._minimise_acquisition = give_up_after_the_first
    with pytest.raises(RuntimeError, match='gave up'):
        learner.propose(history, 4)

    np.testing.assert_allclose(learner.predict(probe), before)


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


# --- building from a configuration -----------------------------------------


def a_config(space, learner, options):
    """A configuration whose only interesting part is the shared learner table."""
    return Config(
        space=space,
        globals=(),
        cost_key=('routine', 'cost'),
        learner=learner,
        learner_options={'shared': options},
        seed=11,
    )


def test_a_learner_is_built_from_a_table_holding_other_learners_knobs(space):
    """One [MLOOP] table serves every learner.

    A lab's knobs are written once and shared, so most of them mean nothing to
    whichever learner is selected. Building takes the ones this learner accepts
    and passes over the rest in silence; rejecting them would mean a file that
    works for one learner breaks the moment another is chosen.
    """
    config = a_config(
        space,
        'differential_evolution',
        {
            'population_size': 4,
            'cost_has_noise': False,
            'length_scale_bounds': (1e-3, 1e3),
        },
    )
    assert build(config).num_members == 4 * space.num_params


def test_a_shared_knob_is_matched_against_arguments_not_constructor_locals(
    space, monkeypatch
):
    """What a constructor accepts is its arguments, and nothing else.

    A name a constructor happens to use as a scratch variable is not a knob it
    takes. Matching against anything wider than the signature lets such a key
    through on the strength of that name, and the TypeError blames the shared
    table for a collision the user cannot see.
    """

    class Scratch:
        def __init__(self, space, rng, population_size=3):
            cost_has_noise = population_size  # a local, not an argument
            self.population_size = cost_has_noise

    monkeypatch.setitem(learners.LEARNERS, 'scratch', Scratch)
    config = a_config(space, 'scratch', {'cost_has_noise': True, 'population_size': 4})
    assert build(config).population_size == 4


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
