"""Learner behaviour, against analytic cost functions.

Every learner is exercised through the one method the base class declares, so
these tests survive any rewrite that keeps the interface.
"""

import inspect
import warnings

import numpy as np
import pytest

from labscript_optimization import learners
from labscript_optimization.config import Config
from labscript_optimization.learners.differential_evolution import STRATEGIES
from labscript_optimization.observations import COMPLETE, DROPPED
from labscript_optimization.session import Session
from labscript_optimization.learners import (
    DifferentialEvolutionLearner,
    DirectedRandomLearner,
    GaussianProcessLearner,
    InsufficientData,
    Learner,
    ParameterSpaceLearner,
    RandomLearner,
    TwoPhaseLearner,
    build,
)
from labscript_optimization.space import Parameter, ParameterSpace

from conftest import FakeRunmanager, observe, run_loop


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


# --- the interface ---------------------------------------------------------


def test_every_learner_is_a_learner_including_the_wrapper(space, rng):
    """The wrapper is what ``build`` returns for the default configuration.

    A session holds it and proposes from it exactly as it does from the
    learners it wraps, so it has to be substitutable for one. An interface the
    most-used learner in the package cannot satisfy describes the wrong thing.
    """
    for cls in [*learners.LEARNERS.values(), TwoPhaseLearner]:
        assert issubclass(cls, Learner), cls.__name__
    for learner in every_learner(space, rng):
        assert isinstance(learner, Learner), type(learner).__name__


@pytest.mark.parametrize('name, cls', sorted(learners.LEARNERS.items()))
def test_a_learner_named_in_a_configuration_takes_the_space_first(name, cls):
    """Building one is ``cls(space, rng, **options)``, the options matched by
    name against the signature. A learner spelling the first two the other way
    round constructs happily and searches the wrong thing, and a knob with no
    default cannot be left out of a shared table that serves every learner.
    """
    assert issubclass(cls, ParameterSpaceLearner)
    params = list(inspect.signature(cls).parameters.values())
    assert [p.name for p in params[:2]] == ['space', 'rng']
    for knob in params[2:]:
        assert knob.default is not knob.empty, knob.name


def test_a_learner_handed_its_two_arguments_backwards_is_refused(space, rng):
    for cls in learners.LEARNERS.values():
        with pytest.raises(TypeError, match='in that order'):
            cls(rng, space)


def test_a_learner_that_does_not_propose_cannot_be_built(space, rng):
    """At either level, and when it is built rather than mid-experiment."""

    class Forgetful(Learner):
        last_phase = 'main'

    class ForgetfulOverASpace(ParameterSpaceLearner):
        last_phase = 'main'

    with pytest.raises(TypeError, match='propose'):
        Forgetful()
    with pytest.raises(TypeError, match='propose'):
        ForgetfulOverASpace(space, rng)


def test_there_is_no_last_phase_to_inherit(space, rng):
    """The other half of reporting the phase off the learner itself.

    A learner that grows phases and forgets to publish them has to fail, so
    the attribute is declared without a value. Read through the deeper class,
    which finds a default given at either level.
    """

    class Silent(ParameterSpaceLearner):
        def propose(self, history, k):
            return self.space.uniform(self.rng, k)

    with pytest.raises(AttributeError, match='last_phase'):
        Silent(space, rng).last_phase


# --- the opening point -----------------------------------------------------


@pytest.fixture
def started_space():
    """A space in which every parameter has a configured start."""
    return ParameterSpace(
        [
            Parameter('x', -5.0, 5.0, start=1.5),
            Parameter('y', -5.0, 5.0, start=-2.5),
        ]
    )


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


def test_the_opening_point_defaults_to_the_configured_start(started_space, rng):
    for learner in learners_starting_at(started_space, rng, None):
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
    return ParameterSpace([Parameter('x', -5.0, 5.0)])


@pytest.mark.parametrize('strategy', ['best1', 'best2', 'rand1', 'rand2'])
def test_differential_evolution_finds_the_minimum(space, rng, strategy):
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=8, evolution_strategy=strategy
    )
    history = run_loop(learner, space, sphere, batches=50, k=4, rng=rng)
    assert min(o.cost for o in history) < 0.05


def test_differential_evolution_state_depends_only_on_the_history(space, rng):
    """The population is not carried between calls; it is rebuilt every time.

    A learner that has been proposing all session therefore holds nothing a
    fresh one handed the same history would not, which is what keeps costs
    arriving out of order out of the algorithm.
    """
    history = [observe(i, p, sphere(p)) for i, p in enumerate(space.uniform(rng, 40))]
    driven = DifferentialEvolutionLearner(
        space, np.random.default_rng(1), population_size=3
    )
    driven.propose(history, 3)
    fresh = DifferentialEvolutionLearner(
        space, np.random.default_rng(2), population_size=3
    )

    earlier = history[:7]
    np.testing.assert_allclose(driven.replay(earlier)[0], fresh.replay(earlier)[0])
    np.testing.assert_allclose(driven.replay(earlier)[1], fresh.replay(earlier)[1])


# --- differential evolution: position in the block is the role --------------


def walk_space():
    """Four parameters, so a trial sharing all but one coordinate with a
    member picks that member out and no other.
    """
    return ParameterSpace([Parameter(name, -5.0, 5.0) for name in 'wxyz'])


def walk_history(space, rng, *blocks):
    """A history of whole blocks of four, one proposal per slot.

    A cost of ``None`` is a proposal that produced nothing -- dropped here,
    and the learner cannot tell that from one still running -- and ``nan`` is
    a shot that ran and measured nothing usable.
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


def held_by(learner, history, positions):
    """Check the population is the records at ``positions``, slot by slot."""
    params, costs = learner.replay(history)
    for slot, position in enumerate(positions):
        np.testing.assert_allclose(params[slot], history[position].params, err_msg=slot)
        assert costs[slot] == history[position].cost, slot


def test_a_dropped_founder_leaves_its_slot_vacant_and_shifts_no_other_slot(rng):
    """Founding from the first four *usable* records instead promotes the
    first trial to founder and re-indexes every role after it: the population
    comes out holding p5, p6, p7 and p4, and the next proposal is read as a
    trial for slot 3 rather than slot 0.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(space, rng, population_size=4)
    history = walk_history(
        space, rng, [None, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]
    )

    vacant = learner.replay(history[:4])[1]
    assert np.isnan(vacant[0])
    np.testing.assert_allclose(vacant[1:], [10.0, 20.0, 30.0])

    held_by(learner, history, [4, 5, 6, 7])


def test_a_late_returning_founder_changes_no_other_proposals_role(rng):
    """It competes for its own slot, and for no other.

    Founding by usable count re-reads every role the moment it lands, because
    the founding block then ends one record earlier than it did; the test is
    what the other slots held before it arrived, not what they hold after.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(space, rng, population_size=4)
    history = walk_history(
        space, rng, [None, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]
    )
    before = learner.replay(history)

    history[0] = history[0]._replace(cost=0.5, state=COMPLETE)
    after = learner.replay(history)

    np.testing.assert_allclose(after[0][1:], before[0][1:])
    np.testing.assert_allclose(after[1][1:], before[1][1:])
    held_by(learner, history, [0, 5, 6, 7])


def test_a_trial_with_no_usable_cost_leaves_its_slots_member_alone(rng):
    """A bad trial spends its slot's turn and displaces nothing. Walking the
    usable records instead closes the gap it left and hands every trial after
    it to the wrong slot.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(space, rng, population_size=4)
    history = walk_history(
        space, rng, [10.0, 20.0, 30.0, 40.0], [float('nan'), 2.0, 3.0, 4.0]
    )

    held_by(learner, history, [0, 5, 6, 7])


def test_a_trial_is_bred_from_the_member_holding_its_own_block_position(rng):
    """With crossover off, one coordinate of a trial comes from the mutant and
    the rest from the incumbent, so the trial names the slot it was drawn for.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=4, cross_over_probability=0.0
    )
    history = walk_history(
        space, rng, [None, 10.0, 20.0, 30.0], [1.0, 2.0, 3.0, 4.0]
    )
    members = learner.replay(history)[0]

    for slot, proposal in enumerate(learner.propose(history, 4)):
        shared = int(np.isclose(proposal, members[slot]).sum())
        assert shared == space.num_params - 1, slot


def test_every_proposal_keeps_the_role_its_position_gave_it(rng):
    """Roles assigned at proposal are the roles read at replay, however the
    costs come back: out of order, not at all, without a usable value, or
    after the generation that would have used them.

    Checked at every refill and after every arrival, not once at the end. A
    final replay is order-independent by construction and reports nothing
    wrong however broken the walk was on the way there, so what is inspected
    is the decision the learner makes right then: with crossover off a trial
    shares all but one coordinate with the member it was bred from, which
    names the slot it was drawn for.
    """
    space = walk_space()
    size = 4
    learner = DifferentialEvolutionLearner(
        space, np.random.default_rng(3), population_size=size,
        cross_over_probability=0.0,
    )
    runmanager = FakeRunmanager()
    session = Session(
        Config(space=space, globals=(), cost_key=('r', 'c')), runmanager, learner
    )

    def population(history):
        """What holds each slot: the cheapest usable result at its position."""
        members = [None] * size
        for position, record in enumerate(history):
            slot = position % size
            if record.usable and (
                members[slot] is None or record.cost < members[slot].cost
            ):
                members[slot] = record
        return members

    def bred_from(proposal, members):
        """The one slot whose member this proposal was crossed with."""
        named = [
            slot
            for slot, record in enumerate(members)
            if record is not None
            and int(np.isclose(proposal, record.params).sum()) == space.num_params - 1
        ]
        return named[0] if len(named) == 1 else None

    def check(proposal, position, members):
        slot = position % size
        held = sum(record is not None for record in members)
        if members[slot] is None or held <= STRATEGIES['best1']:
            # No incumbent, or too little population to breed from: the
            # proposal is drawn founder-style and names no slot.
            return
        assert bred_from(proposal, members) == slot, position

    def probe():
        """The role the learner would give its very next proposal."""
        history = session.history
        members = population(history)
        check(np.atleast_2d(learner.propose(history, 1))[0], len(history), members)

    late, lost, unusable = [], 0, 0
    for _ in range(12):
        first = len(session.history)
        members = population(session.history)
        submitted = session.refill()
        assert len(submitted) == size
        for offset, shot_id in enumerate(submitted):
            check(session.proposals[shot_id], first + offset, members)

        # The operator cleared the red row: the shots behind it have run, and
        # their costs land now, after this generation was built without them.
        for shot_id in late:
            session.record(shot_id, float(rng.uniform(0, 10)), None, False)
            probe()

        late = []
        arriving = list(submitted)
        rng.shuffle(arriving)
        for shot_id in arriving:
            roll = rng.random()
            if roll < 0.2:
                runmanager.lose(shot_id)
                session.reconcile()
                late.append(shot_id)
                lost += 1
            elif roll < 0.35:
                session.record(shot_id, float('nan'), None, False)
                unusable += 1
            else:
                session.record(shot_id, float(rng.uniform(0, 10)), None, False)
            probe()

    # The run has to have taken all four paths, or it proved only the easy one.
    assert lost and unusable
    assert session.status()['completed'] == len(session.proposals) - len(late)


def test_a_slot_whose_founder_produced_nothing_is_drawn_founder_style(rng):
    """There is no incumbent in an empty slot, so there is nothing to cross
    over with: the proposal for it is a fresh point rather than a trial.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(
        space, rng, population_size=4, cross_over_probability=0.0
    )
    history = walk_history(space, rng, [None, 10.0, 20.0, 30.0])
    members = learner.replay(history)[0]

    proposals = learner.propose(history, 4)
    assert space.contains(proposals).all()
    for member in members[1:]:
        assert int(np.isclose(proposals[0], member).sum()) < space.num_params - 1
    for slot in (1, 2, 3):
        shared = int(np.isclose(proposals[slot], members[slot]).sum())
        assert shared == space.num_params - 1, slot


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
    assert len(history) > 2 * learner.population_size
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


def test_a_gaussian_process_has_no_opening_point(started_space, rng):
    """An empty history is refused even where every parameter has a start.

    The other learners fall back to that start when told nothing. This one
    must not: :func:`build` always wraps it, so an opening point here is
    unreachable through a configuration, and bare it would answer from no
    data instead of saying it cannot.
    """
    with pytest.raises(InsufficientData):
        GaussianProcessLearner(started_space, rng).propose([], 1)


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


@pytest.mark.parametrize('batch_size, carried, count', [(4, 12, 15), (8, 6, 7)])
def test_gaussian_process_state_depends_only_on_the_history(
    space, batch_size, carried, count
):
    """Two learners given the same history must hold the same model.

    The kernel hyperparameters are cached between calls, so they have to be a
    function of the history alone: an instance that has been fitting all
    session must arrive at what a fresh one computes, not at a kernel fitted to
    however much it happened to hold when the cache was last filled. The second
    case is a history short of one full batch, where there is no whole batch
    to fit to and the cache has to give way on every arrival.
    """
    history = gaussian_process_history(space, 9, count=count)
    all_session = GaussianProcessLearner(
        space, np.random.default_rng(1), batch_size=batch_size
    )
    all_session.fit(history[:carried])
    all_session.fit(history)

    fresh = GaussianProcessLearner(
        space, np.random.default_rng(2), batch_size=batch_size
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


#: The batch the exploration tests below configure, and so the number of
#: proposals the schedule takes to come back round to its greedy step.
BATCH = 4


def exploring_and_greedy(space, count):
    """The same proposal made with the exploration weight up and turned off.

    The two learners share a seed and see the same history, so the acquisition
    weight is the only thing that differs between them.
    """
    history = gaussian_process_history(space, 5, count=count)
    greedy = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=0.0, batch_size=BATCH
    )
    explorer = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=50.0, batch_size=BATCH
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
    """The weight steps once per proposal and cycles over a batch.

    One proposal in each batch is purely greedy and the rest look
    progressively further afield, which is how the schedule spends a batch.
    Reading the position off the history rather than a counter is what keeps
    a learner handed the same history proposing the same thing.
    """
    apart = exploring_and_greedy(space, count=count)
    if count % BATCH:
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
    so a run of proposals spanning two batches asks for the same greedy point
    twice.
    """
    learner = GaussianProcessLearner(space, rng, batch_size=BATCH)
    history = gaussian_process_history(space, 5)
    proposals = learner.propose(history, BATCH + 2)
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
    search = learner.minimise_acquisition

    def give_up_after_the_first(*args, **kwargs):
        searches.append(1)
        if len(searches) > 1:
            raise RuntimeError('the minimiser gave up')
        return search(*args, **kwargs)

    learner.minimise_acquisition = give_up_after_the_first
    with pytest.raises(RuntimeError, match='gave up'):
        learner.propose(history, 4)

    np.testing.assert_allclose(learner.predict(probe), before)


def test_a_gaussian_process_waits_for_twice_as_many_points_as_parameters(space, rng):
    """Its default patience, and what a two-phase wrapper trains through."""
    history = gaussian_process_history(space, 3, count=2 * space.num_params)
    learner = GaussianProcessLearner(space, rng)
    with pytest.raises(InsufficientData):
        learner.propose(history[:-1], 1)
    assert space.contains(learner.propose(history, 1)).all()


def test_a_batch_is_proposed_from_a_history_whose_points_carry_uncertainties(
    space, rng
):
    """Each point of the batch is folded into the fit before the next is
    picked, and a fit holding a variance per point needs one for the invented
    point too, or it is handed one fewer variance than it has points.
    """
    points = space.uniform(np.random.default_rng(7), 12)
    history = [observe(i, p, offset_sphere(p), uncer=0.1) for i, p in enumerate(points)]
    proposals = GaussianProcessLearner(space, rng).propose(history, 3)
    assert space.contains(proposals).all()


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
        shared_learner_options=options,
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
            'population_size': 15,
            'cost_has_noise': False,
            'length_scale_bounds': (1e-3, 1e3),
        },
    )
    # And population_size is the number of members, not a multiplier on the
    # parameter count: fifteen here, over however many parameters.
    assert build(config).population_size == 15


def test_the_default_learner_comes_back_wrapped_in_its_training_phase(space):
    """A Gaussian process has nothing to say until it has a spread of points,
    so the learner a file gets by saying nothing runs directed_random first.
    """
    config = Config(
        space=space,
        globals=(),
        cost_key=('routine', 'cost'),
        num_training_runs=7,
        seed=11,
    )
    learner = build(config)
    assert isinstance(learner, TwoPhaseLearner)
    assert isinstance(learner.trainer, DirectedRandomLearner)
    assert isinstance(learner.main, GaussianProcessLearner)
    assert learner.num_training == 7


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


def test_a_learner_that_hides_its_knobs_in_kwargs_is_refused(space, monkeypatch):
    """Matching against the signature is what makes the filtering silent here.

    ``**kwargs`` names nothing, so every knob in the table is passed over and
    the learner is built entirely from its defaults, with no error and a
    search the lab did not configure.
    """

    class Swallower(ParameterSpaceLearner):
        last_phase = 'main'

        def __init__(self, space, rng, **kwargs):
            super().__init__(space, rng)
            self.population_size = kwargs.get('population_size', 3)

        def propose(self, history, k):
            return self.space.uniform(self.rng, k)

    monkeypatch.setitem(learners.LEARNERS, 'swallower', Swallower)
    with pytest.raises(TypeError, match='kwargs'):
        build(a_config(space, 'swallower', {'population_size': 4}))


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


def test_a_handover_that_cannot_happen_yet_is_warned_about(space, rng):
    """The fallback covers it, but the user configured a handover that will
    not happen when they expect, and neither number is in any document."""
    main = GaussianProcessLearner(space, rng, minimum_observations=6)
    with pytest.warns(UserWarning, match='needs 6 usable observations'):
        TwoPhaseLearner(RandomLearner(space, rng), main, num_training=5)


def test_a_handover_that_works_is_not_warned_about(space, rng):
    main = GaussianProcessLearner(space, rng, minimum_observations=4)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        TwoPhaseLearner(RandomLearner(space, rng), main, num_training=4)


def test_a_two_phase_learner_reports_a_phase_before_it_has_proposed(space, rng):
    """A session may report its status before it first refills the queue."""
    learner = TwoPhaseLearner(RandomLearner(space, rng), Ready(), num_training=3)
    assert learner.last_phase == 'training'


def test_a_two_phase_learner_answers_for_the_observations_it_needs(space, rng):
    """It never refuses to propose -- the trainer is the fallback for anything
    the main learner cannot make -- so it absorbs its main learner's
    requirement rather than passing it on. Declared, because an attribute read
    off a learner with a default is the reader's answer and not the learner's.
    """
    main = GaussianProcessLearner(space, rng, minimum_observations=6)
    learner = TwoPhaseLearner(RandomLearner(space, rng), main, num_training=6)
    assert learner.minimum_observations == 0


def test_a_generational_learner_cannot_be_put_behind_a_trainer(space, rng):
    """Refused at construction, because the wrapper cannot hold the barrier.

    A two-phase learner declares no generation of its own, so a session tops
    its queue up whenever there is room: the population would be proposed in
    pieces and judged before the generation was complete.
    """
    main = DifferentialEvolutionLearner(space, rng, population_size=4)
    with pytest.raises(ValueError, match='whole generations of 4'):
        TwoPhaseLearner(RandomLearner(space, rng), main, num_training=8)


def test_a_generational_learner_cannot_be_the_trainer_either(space, rng):
    """The same barrier reaches a session by the same route. Which phase the
    learner proposes in changes nothing about what its declaration promises.
    """
    trainer = DifferentialEvolutionLearner(space, rng, population_size=4)
    with pytest.raises(ValueError, match='whole generations of 4'):
        TwoPhaseLearner(trainer, RandomLearner(space, rng), num_training=8)


def test_a_learner_that_declares_no_barrier_is_wrapped(space, rng):
    """What is refused is a barrier the wrapper cannot hold, not wrapping. The
    wrapper's own ``generation`` is true of it because of that refusal.
    """
    learner = TwoPhaseLearner(
        DirectedRandomLearner(space, rng), GaussianProcessLearner(space, rng), 8
    )
    assert learner.generation is None
