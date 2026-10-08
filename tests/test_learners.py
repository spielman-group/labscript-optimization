"""Learner behavior, against analytic cost functions.

How many a learner proposes, and under what source, is exercised through the
one method the base class declares, ``propose``. What a shipped learner's
method makes of a history is exercised through its own ``ask``, which proposes
a given number of points without pacing them, so that a test of the algorithm
can ask for the points it needs at any history length.
"""

import threading
import warnings

import numpy as np
import pytest

from labscript_optimization import learners
from labscript_optimization.config import Config
from labscript_optimization.observations import COMPLETE, DROPPED, PENDING, Observation
from labscript_optimization.session import Session
from labscript_optimization.learners import (
    DifferentialEvolutionLearner,
    DirectedRandomLearner,
    GaussianProcessLearner,
    RandomLearner,
)
from labscript_optimization.space import Parameter, ParameterSpace

from conftest import FakeRunmanager, observe, run_loop, settle


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
    ]


@pytest.mark.parametrize('seen', [0, 20])
@pytest.mark.parametrize('k', [1, 3, 7])
def test_proposals_have_the_requested_shape_and_stay_in_bounds(space, rng, k, seen):
    """With nothing in flight, the hint is what a learner is asked for -- by
    every learner but one declaring a generation, which proposes the rest of
    the block of positions its next proposal falls in, whatever the hint.
    A learner with one way of proposing calls it ``main``.
    """
    history = [
        observe(i, p, sphere(p)) for i, p in enumerate(space.uniform(rng, seen))
    ]
    for learner in every_learner(space, rng):
        generation = learner.generation
        wanted = k if generation is None else generation - len(history) % generation
        proposed = learner.propose(history, k)
        proposals = np.array([params for params, _ in proposed])
        assert proposals.shape == (wanted, space.num_params), type(learner).__name__
        assert space.contains(proposals).all(), type(learner).__name__
        assert {source for _, source in proposed} == {'main'}, type(learner).__name__


@pytest.mark.parametrize(
    'learner',
    [
        RandomLearner,
        DirectedRandomLearner,
        lambda space, rng: GaussianProcessLearner(
            space, rng, warmup_observations=10, explore_runs=0
        ),
    ],
    ids=['random', 'directed_random', 'gaussian_process warming up'],
)
def test_learners_filling_to_the_hint_keep_exactly_that_many_in_flight(
    space, rng, learner
):
    """Every pending record takes one of the hint's places, whoever proposed
    it, and a shot that will never report takes none. The configured start
    going out beside this call's proposals is in flight all the same.
    """
    learner = learner(space, rng)
    history = [
        Observation(None, np.zeros(2), None, state=PENDING, source='start'),
        observe(1, [1.0, 1.0], 2.0),
        observe(2, [1.0, -1.0], 3.0),
        observe(3, [2.0, 2.0], None, state=DROPPED),
        observe(4, [-1.0, 1.0], None, state=PENDING),
        observe(5, [-2.0, 2.0], 4.0),
    ]

    assert len(learner.propose(history, 5)) == 3
    assert learner.propose(history, 2) == []
    assert learner.propose(history, 1) == []
    assert learner.propose(history, 0) == []


@pytest.mark.parametrize(
    'name, options',
    [
        # Out of the range the knob may take.
        ('directed_random', dict(trust_range=(0.25, 0.1))),
        ('directed_random', dict(trust_range=(0.1, 1.5))),
        ('differential_evolution', dict(population_size=1)),
        ('differential_evolution', dict(evolution_strategy='nope')),
        ('differential_evolution', dict(cross_over_probability=2.0)),
        ('differential_evolution', dict(mutation_scale=(1.0, 0.5))),
        ('gaussian_process', dict(warmup_observations=0)),
        ('gaussian_process', dict(batch_size=0)),
        ('gaussian_process', dict(explore_runs=-1)),
        ('gaussian_process', dict(uncer_bias=[])),
        ('gaussian_process', dict(explorer='gaussian_process')),
        ('gaussian_process', dict(length_scale_bounds=(1e2, 1e-2))),
        ('gaussian_process', dict(noise_level_bounds=(0.0, 1.0))),
        # Written as the wrong kind of thing, which is refused and not converted.
        ('directed_random', dict(trust_gaussian='false')),
        ('directed_random', dict(explore_fraction='0.5')),
        ('directed_random', dict(trust_range=0.5)),
        ('differential_evolution', dict(population_size=8.9)),
        ('differential_evolution', dict(mutation_scale=0.8)),
        ('differential_evolution', dict(evolution_strategy=['best1'])),
        ('gaussian_process', dict(cost_has_noise=0)),
        ('gaussian_process', dict(batch_size=True)),
        ('gaussian_process', dict(cost_bias=True)),
        ('gaussian_process', dict(uncer_bias=['0', '1'])),
        ('gaussian_process', dict(length_scale_bounds=5)),
        ('gaussian_process', dict(trust_region='0.1')),
    ],
)
def test_a_knob_that_cannot_mean_what_it_says_is_refused_naming_it(
    space, rng, name, options
):
    """In the constructor, so a learner built in Python is held to its knobs
    exactly as one built from a file."""
    with pytest.raises(ValueError, match=next(iter(options))):
        learners.LEARNERS[name](space, rng, **options)


# --- directed random -------------------------------------------------------


def nearest(proposals, seen):
    """The index of the seen point nearest each proposal, and how far it is."""
    distances = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2)
    return distances.argmin(axis=1), distances.min(axis=1)


def test_directed_random_searches_near_the_usable_points_it_has_seen(space, rng):
    """An infinite cost is no centre, and must not disable the trust region."""
    seen = np.array([[1.0, 1.0], [1.2, 0.8], [-3.0, 4.0]])
    history = [observe(i, p, sphere(p)) for i, p in enumerate(seen)]
    history.append(observe('bad', [4.9, 4.9], float('inf')))

    proposals = DirectedRandomLearner(space, rng, trust_region=0.05).ask(history, 300)

    # 5% of a range of 10 is 0.5 in each direction.
    assert (nearest(proposals, seen)[1] <= 0.5 + 1e-9).all()
    assert (np.abs(proposals - [4.9, 4.9]).max(axis=1) > 0.5).all()


def test_directed_random_explores_the_whole_space_when_told_to(space, rng):
    history = [observe(0, [1.0, 1.0], 1.0), observe(1, [1.1, 1.1], 2.0)]
    learner = DirectedRandomLearner(
        space, rng, trust_region=0.01, explore_fraction=1.0
    )
    proposals = learner.ask(history, 400)
    assert proposals.min() < -4.0 and proposals.max() > 4.0


def test_directed_random_prefers_mediocre_points_over_the_best_one(space, rng):
    """The default band sits away from the best point, on purpose."""
    seen = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    costs = [0.0, 10.0, 25.0, 30.0]
    history = [observe(i, p, c) for i, (p, c) in enumerate(zip(seen, costs))]

    learner = DirectedRandomLearner(
        space, rng, trust_region=0.02, trust_range=(0.1, 0.25)
    )
    # The band runs from 10% to 25% of the way from the worst cost (30) towards
    # the best (0), that is [22.5, 27], which picks out the point costing 25.
    assert set(np.unique(nearest(learner.ask(history, 300), seen)[0])) == {2}


def test_directed_random_falls_back_to_the_best_point_when_the_band_is_empty(
    space, rng
):
    seen = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # Nothing lands in [22.5, 27], and the best point is not the first one.
    costs = [30.0, 10.0, 20.0, 0.0]
    history = [observe(i, p, c) for i, (p, c) in enumerate(zip(seen, costs))]

    learner = DirectedRandomLearner(
        space, rng, trust_region=0.02, trust_range=(0.1, 0.25)
    )
    assert set(np.unique(nearest(learner.ask(history, 200), seen)[0])) == {3}


# --- differential evolution ------------------------------------------------


@pytest.mark.parametrize('strategy', ['best1', 'best2', 'rand1', 'rand2'])
def test_differential_evolution_finds_the_minimum(space, rng, strategy):
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=8, evolution_strategy=strategy
    )
    # Twenty-five generations of eight: two hundred shots.
    history = run_loop(learner, space, sphere, batches=25, k=4, rng=rng)
    assert len(history) == 200
    assert min(o.cost for o in history) < 0.05


# The smallest population each strategy can mutate: it draws distinct members
# from the population minus the slot it is replacing, so it needs one more
# member than it draws on.
@pytest.mark.parametrize(
    'strategy, smallest', [('best1', 3), ('rand1', 4), ('best2', 5), ('rand2', 6)]
)
def test_every_strategy_runs_at_its_smallest_population_and_refuses_less(
    space, rng, strategy, smallest
):
    with pytest.raises(ValueError, match=strategy):
        DifferentialEvolutionLearner(
            space, rng, population_size=smallest - 1, evolution_strategy=strategy
        )
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=smallest, evolution_strategy=strategy
    )
    history = run_loop(learner, space, sphere, batches=8, k=2, rng=rng)
    # Long past the founding generation, so mutation did the bulk of the proposing.
    assert len(history) > 2 * smallest
    assert space.contains(np.array([o.params for o in history])).all()


def walk_space():
    """Four parameters, so a trial sharing all but one coordinate with a
    member picks that member out and no other.
    """
    return ParameterSpace([Parameter(name, -5.0, 5.0) for name in 'wxyz'])


def walk_history(space, rng, *blocks):
    """A history of whole blocks of four, one proposal per slot.

    A cost of ``None`` is a proposal that produced nothing, and ``nan`` is a
    shot that ran and measured nothing usable.
    """
    records = []
    for block in blocks:
        for cost in block:
            records.append(
                observe(
                    f'p{len(records)}',
                    space.uniform(rng, 1)[0],
                    cost,
                    state=DROPPED if cost is None else COMPLETE,
                )
            )
    return records


@pytest.mark.parametrize(
    'blocks, seen, held',
    [
        # A founder with no cost leaves its slot vacant, and no other slot's
        # result stands in for it: that slot's proposal is a fresh point.
        ([[None, 10.0, 20.0, 30.0]], 4, [None, 1, 2, 3]),
        # The first trial is then the slot's member, and no other slot moves.
        ([[None, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]], 8, [4, 5, 6, 7]),
        # A founder's cost arriving late still competes for its own slot.
        ([[0.5, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]], 8, [0, 5, 6, 7]),
        # A trial with no usable cost displaces nothing.
        ([[10.0, 20.0, 30.0, 40.0], [float('nan'), 2.0, 3.0, 4.0]], 8, [0, 5, 6, 7]),
        # Part way through a block, a proposal's slot is the position it is
        # made at, not its place in the batch asked for.
        ([[None, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]], 6, [4, 5, 2, 3]),
    ],
)
def test_a_proposal_is_bred_from_the_member_of_the_slot_its_position_names(
    rng, blocks, seen, held
):
    """``held[slot]`` is the position of the record holding that slot. With
    crossover off, a trial shares all but one coordinate with the member it was
    bred from, which names the slot it was drawn for.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=4, cross_over_probability=0.0
    )
    history = walk_history(space, rng, *blocks)[:seen]

    for offset, proposal in enumerate(learner.ask(history, 4)):
        position = held[(len(history) + offset) % 4]
        shared = [int(np.isclose(proposal, o.params).sum()) for o in history]
        if position is None:
            assert max(shared) < space.num_params - 1, offset
        else:
            assert shared[position] == space.num_params - 1, offset


def test_mutation_scale_sets_the_differential_weight(rng):
    """Every coordinate taken from the mutant, at a weight of zero, breeds each
    trial as a copy of the best member."""
    space = walk_space()
    learner = DifferentialEvolutionLearner(
        space,
        rng,
        population_size=5,
        mutation_scale=(0.0, 0.0),
        cross_over_probability=1.0,
    )
    founders = np.random.default_rng(8).uniform(-0.5, 0.5, (5, space.num_params))
    history = [observe(slot, p, float(slot)) for slot, p in enumerate(founders)]

    np.testing.assert_allclose(learner.ask(history, 5), np.tile(founders[0], (5, 1)))


# --- gaussian process ------------------------------------------------------


def gaussian_process_history(space, seed, count=12):
    """A spread of observations wide enough for the posterior to mean something."""
    points = space.uniform(np.random.default_rng(seed), count)
    return [observe(i, p, offset_sphere(p)) for i, p in enumerate(points)]


def test_gaussian_process_finds_the_minimum(space, rng):
    model = GaussianProcessLearner(space, rng).model
    history = gaussian_process_history(space, 4)
    for batch in range(12):
        for params in model.ask(history, 4):
            history.append(
                observe(f'{batch}-{len(history)}', params, offset_sphere(params))
            )
    best = min(history, key=lambda o: o.cost)
    assert best.cost < 0.05
    np.testing.assert_allclose(best.params, [1.3, -2.1], atol=0.3)


def test_each_batch_walks_the_exploration_schedule_from_its_first_weight(space):
    """The weight is the point's position in the batch, whatever the number of
    observations in hand, and a single number is that weight on every point.
    """

    def batch(uncer_bias):
        model = GaussianProcessLearner(
            space, np.random.default_rng(3), uncer_bias=uncer_bias
        ).model
        # Thirteen observations, which is not a multiple of the four weights.
        return model.ask(gaussian_process_history(space, 5, count=13), 4)

    greedy, walking, fixed = batch(0.0), batch([0.0, 50.0, 100.0, 150.0]), batch(50.0)
    apart = np.linalg.norm(walking - greedy, axis=1)
    # A weight of zero times anything is the greedy proposal itself.
    assert apart[0] == 0.0
    assert (apart[1:] > 0.1).all()
    assert np.linalg.norm(fixed[0] - greedy[0]) > 0.1


def test_a_gaussian_process_batch_does_not_repeat_itself(space, rng):
    """Two picks made at the same exploration weight must land apart, because
    each point is folded into the fit before the next is chosen."""
    model = GaussianProcessLearner(space, rng).model
    proposals = model.ask(gaussian_process_history(space, 5), 6)
    # The weights run 0, 1, 2, 3, 0, 1 by position, so the sixth pick repeats
    # the weight of the second.
    assert np.linalg.norm(proposals[5] - proposals[1]) > 1e-3


def flat_in_y_history(space, count):
    """Observations of a cost that varies in x alone, so the fit finds no
    structure along y and leaves its length scale at the upper bound."""
    points = space.uniform(np.random.default_rng(5), count)
    return [observe(i, p, float((p[0] - 1.3) ** 2)) for i, p in enumerate(points)]


def test_a_length_scale_at_a_bound_is_reported_when_the_set_of_them_changes(
    space, rng
):
    """Once when a parameter reaches a bound, once when it leaves, and not at
    every refit in between."""
    from sklearn.exceptions import ConvergenceWarning

    model = GaussianProcessLearner(space, rng).model

    with warnings.catch_warnings(record=True) as first:
        model.fit(flat_in_y_history(space, 12))
    assert [w for w in first if 'y at the upper end' in str(w.message)]

    with warnings.catch_warnings(record=True) as again:
        model.fit(flat_in_y_history(space, 13))
    assert not [w for w in again if 'length_scale' in str(w.message)]

    with pytest.warns(ConvergenceWarning, match='inside length_scale_bounds'):
        model.fit(gaussian_process_history(space, 5))


def test_gaussian_process_uses_per_point_uncertainties(space, rng):
    """A noisy point is trusted less than an exact one, and a batch can still
    be proposed from points that carry one."""
    points = space.uniform(np.random.default_rng(7), 12)
    exact = [observe(i, p, offset_sphere(p), uncer=0.0) for i, p in enumerate(points)]
    noisy = [observe(i, p, offset_sphere(p), uncer=5.0) for i, p in enumerate(points)]

    tight = GaussianProcessLearner(space, rng).model
    tight.fit(exact)
    loose = GaussianProcessLearner(space, rng).model
    loose.fit(noisy)

    probe = np.array([0.0, 0.0])
    assert loose.predict(probe)[1][0] > tight.predict(probe)[1][0]
    assert space.contains(loose.ask(noisy, 3)).all()


@pytest.mark.parametrize(
    'cost_has_noise, uncertainties, fits',
    [
        (False, (0.1, 0.2), True),
        (False, (None, None), True),
        # Every cost exact, but one with no uncertainty to say so: singular.
        (False, (0.1, None), False),
        (True, (0.1, None), True),
    ],
)
def test_a_history_with_only_some_uncertainties_is_refused_without_noise(
    space, rng, cost_has_noise, uncertainties, fits
):
    model = GaussianProcessLearner(
        space, rng, cost_has_noise=cost_has_noise, warmup_observations=2
    ).model
    history = [
        observe('a', space.minimum, 1.0, uncer=uncertainties[0]),
        observe('b', space.maximum, 2.0, uncer=uncertainties[1]),
    ]
    if fits:
        assert space.contains(model.ask(history, 1)).all()
    else:
        with pytest.raises(ValueError, match='cost_has_noise'):
            model.ask(history, 1)


# --- the Gaussian process's cycle ------------------------------------------


class Model:
    """Stands in for the Gaussian process's model, so a test decides when a
    batch is ready.

    A batch is constant points, finished once ``ready`` is set, and every
    history the model is handed is kept. It runs on the learner's own thread,
    as the real one does.
    """

    def __init__(self, space, ready=True):
        self.space = space
        self.ready = threading.Event()
        if ready:
            self.ready.set()
        self.error = None
        self.handed = []

    def ask(self, history, k):
        self.handed.append(history)
        self.ready.wait(10)
        if self.error is not None:
            raise self.error
        return np.full((k, self.space.num_params), 0.5)


def driven(space, model, hint=2, **knobs):
    """A session running a Gaussian process with ``model`` swapped in, and a
    warmup of three."""
    learner = GaussianProcessLearner(
        space, np.random.default_rng(5), warmup_observations=3, **knobs
    )
    learner.model = model
    config = Config(
        space=space, globals=(), cost_key=('r', 'c'), num_buffered_runs=hint
    )
    session = Session(config, FakeRunmanager(), learner)
    session.start()
    return session


def step(session, back=1):
    """The oldest ``back`` shots in flight report and the session refills, as
    the routine drives it. Returns the source of each shot submitted, once any
    batch a ready model was set computing has finished."""
    for shot_id in session.awaiting[:back]:
        session.record(shot_id, offset_sphere(session.proposals[shot_id]), None, False)
    submitted = session.refill()
    if session.learner.model.ready.is_set():
        settle(session.learner)
    return [session.sources[shot_id] for shot_id in submitted]


def past_warmup(session):
    """Step ``session`` from its opening until a refill proposes no warmup
    shot, and return that refill's sources."""
    sources = step(session, back=0)
    while 'warmup' in sources:
        sources = step(session)
    return sources


def test_while_a_batch_computes_the_queue_is_topped_up_with_explorer_shots(space):
    """The apparatus does not wait on a fit."""
    model = Model(space, ready=False)
    session = driven(space, model, hint=3)
    assert past_warmup(session) == ['explore']
    for _ in range(5):
        assert step(session) == ['explore']
        assert len(session.awaiting) == 3
    model.ready.set()


def test_the_next_batch_is_computed_once_none_of_the_last_is_pending(space):
    """So the model has every answer it asked for. Explorer shots in flight do
    not hold it up, and a point that will never report is as back as it will
    ever be."""
    model = Model(space)
    session = driven(space, model, hint=2, batch_size=2)
    past_warmup(session)
    assert step(session) == ['main']
    assert step(session) == ['main']
    assert step(session) == ['explore']
    assert len(model.handed) == 1

    (last,) = [s for s in session.awaiting if session.sources[s] == 'main']
    session.interface.lose(last)
    session.reconcile()
    assert step(session, back=0) == ['explore']
    assert len(model.handed) == 2
    assert {session.sources[s] for s in session.awaiting} == {'explore'}


def test_a_model_that_raises_stops_the_session_at_the_next_refill(space):
    """Its error comes out of the refill that finds the computation over,
    where the worker stops the session with it. Raised on the model's own
    thread it would reach nobody."""
    model = Model(space)
    model.error = RuntimeError('the fit gave up')
    session = driven(space, model)
    past_warmup(session)
    with pytest.raises(RuntimeError, match='the fit gave up'):
        step(session)


def test_each_batch_cycle_holds_at_least_explore_runs_explorer_shots(space):
    """Counted from one batch's last point to the next batch's first, and every
    explorer shot in that span counts, the ones that kept the queue topped up
    while the batch computed among them."""
    model = Model(space)
    session = driven(space, model, hint=1, batch_size=2, explore_runs=3)
    sources = past_warmup(session)
    for _ in range(20):
        sources += step(session)
    firsts = [i for i, source in enumerate(sources) if source == 'main'][::2]
    between = [sources[a:b].count('explore') for a, b in zip(firsts, firsts[1:])]
    assert len(between) >= 3
    assert between == [3] * len(between)
    assert len(model.handed) == len(firsts) + 1


def test_with_explore_runs_of_zero_the_explorer_only_fills(space):
    """A model ready at once leaves explorer shots only where no point of a
    batch is ready, which at a depth of one is the refill that starts each
    batch computing."""
    session = driven(space, Model(space), hint=1, batch_size=2, explore_runs=0)
    sources = past_warmup(session)
    for _ in range(11):
        sources += step(session)
    assert sources == ['explore', 'main', 'main'] * 4


def test_a_batch_is_computed_from_the_history_as_its_computation_began(space):
    """Not as it stands when its points go out: the shots proposed while it
    computed are not in it, and neither is anything that came back since."""
    model = Model(space, ready=False)
    session = driven(space, model)
    sources = past_warmup(session)
    began = [o.shot_id for o in session.history][: -len(sources)]
    step(session)
    step(session)
    model.ready.set()
    settle(session.learner)
    assert step(session) == ['main']
    assert [o.shot_id for o in model.handed[0]] == began


def test_warmup_ends_at_a_count_of_usable_observations_and_not_of_shots(space, rng):
    """A shot whose cost is NaN, a bad one and a dropped one each spend a
    position and give the fit nothing, so none moves warmup on."""
    learner = GaussianProcessLearner(space, rng, warmup_observations=5)
    learner.model = Model(space)
    points = space.uniform(np.random.default_rng(8), 8)
    history = [observe(i, p, offset_sphere(p)) for i, p in enumerate(points[:4])]
    history += [
        observe('nan', points[4], float('nan')),
        observe('bad', points[5], 1.0, bad=True),
        observe('gone', points[6], None, state=DROPPED),
    ]
    assert [source for _, source in learner.propose(history, 2)] == ['warmup'] * 2

    history.append(observe(4, points[7], offset_sphere(points[7])))
    assert [source for _, source in learner.propose(history, 2)] == ['explore'] * 2
    settle(learner)
    assert len(learner.model.handed) == 1


@pytest.mark.parametrize('num_params, warmup', [(1, 5), (5, 10)])
def test_the_warmup_scales_with_the_search_above_a_floor_of_five(num_params, warmup):
    """max(5, 2 × num_params): a constant is too short for a wide search, and
    twice the parameters alone fits a one-parameter search on two points."""
    space = ParameterSpace([Parameter(f'p{i}', 0.0, 1.0) for i in range(num_params)])
    learner = GaussianProcessLearner(space, np.random.default_rng(1))
    assert learner.warmup_observations == warmup


def test_the_explorer_proposes_the_warmup_and_the_shots_that_fill_the_queue(space):
    class Marked(RandomLearner):
        """An explorer whose draws cannot be mistaken for anything else's."""

        def ask(self, history, k):
            return np.full((k, self.space.num_params), 4.5)

    marked = Marked(space, np.random.default_rng(1))
    session = driven(space, Model(space), explorer=marked)
    past_warmup(session)
    for _ in range(8):
        step(session)
    by_source = {}
    for shot_id, params in session.proposals.items():
        by_source.setdefault(session.sources[shot_id], []).append(params)
    assert set(by_source) == {'warmup', 'main', 'explore'}
    assert all((p == 4.5).all() for p in by_source['warmup'] + by_source['explore'])
    assert not any((p == 4.5).all() for p in by_source['main'])


class Recording(DifferentialEvolutionLearner):
    """Differential evolution, keeping every history it is handed."""

    def ask(self, history, k):
        self.handed.append(list(history))
        return super().ask(history, k)


def test_differential_evolution_explores_over_its_own_shots_alone():
    """A proposal's role is its position among its own proposals, so it is
    handed those and nothing else: not the configured start, and not the
    batches. The model is handed every record."""
    space = ParameterSpace(
        [Parameter('x', -5.0, 5.0, start=1.0), Parameter('y', -5.0, 5.0, start=1.0)]
    )
    explorer = Recording(space, np.random.default_rng(2), population_size=4)
    explorer.handed = []
    model = Model(space)
    session = driven(space, model, explorer=explorer)
    past_warmup(session)
    for _ in range(8):
        step(session)
    handed = {o.source for history in explorer.handed for o in history}
    assert handed == {'warmup', 'explore'}
    assert {o.source for o in model.handed[-1]} == {
        'start', 'warmup', 'main', 'explore'
    }
