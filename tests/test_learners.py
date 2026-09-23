"""Learner behaviour, against analytic cost functions.

How many a learner proposes, and under what source, is exercised through the
one method the base class declares, ``propose``, so those tests survive any
rewrite that keeps the interface. What a shipped learner's method makes of a
history is exercised through its own ``ask``, which proposes a given number of
points without pacing them, so that a test of the algorithm can ask for the
points it needs at any history length.
"""

import inspect
import itertools
import numbers
import warnings

import dataclasses

import numpy as np
import pytest

from labscript_optimization import learners
from labscript_optimization.config import Config
from labscript_optimization.learners.differential_evolution import STRATEGIES
from labscript_optimization.observations import (
    COMPLETE,
    DROPPED,
    PENDING,
    Observation,
    best,
    usable,
)
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
    """With nothing in flight, the hint is what a learner is asked for -- by
    every learner but one declaring a generation, which proposes the rest of
    the block of positions its next proposal falls in, whatever the hint.
    """
    history = [observe(i, p, sphere(p)) for i, p in enumerate(space.uniform(rng, 20))]
    for learner in every_learner(space, rng):
        generation = learner.generation
        wanted = k if generation is None else generation - len(history) % generation
        proposals = np.array([params for params, _ in learner.propose(history, k)])
        assert proposals.shape == (wanted, space.num_params), type(learner).__name__
        assert space.contains(proposals).all(), type(learner).__name__


def test_learners_propose_from_an_empty_history(space, rng):
    for learner in every_learner(space, rng):
        proposals = np.array([params for params, _ in learner.propose([], 2)])
        assert space.contains(proposals).all(), type(learner).__name__


def test_a_learner_without_phases_of_its_own_still_names_its_source(space, rng):
    """Every proposal comes back beside the source the session records for it,
    and a learner with one way of proposing calls it ``main``.

    Nothing supplies a source on a learner's behalf, so a default would paper
    over the opposite case as well: a learner that grows phases and forgets to
    say which one proposed would be reported as the main one throughout.
    """
    for learner, history in [
        (RandomLearner(space, rng), []),
        (DirectedRandomLearner(space, rng, trust_region=0.1), []),
        (DifferentialEvolutionLearner(space, rng, population_size=3), []),
        (GaussianProcessLearner(space, rng), gaussian_process_history(space, 4)),
    ]:
        proposed = learner.propose(history, 2)
        assert proposed, type(learner).__name__
        assert {source for _, source in proposed} == {'main'}, type(learner).__name__


# --- the interface ---------------------------------------------------------


@pytest.mark.parametrize('name, cls', sorted(learners.LEARNERS.items()))
def test_a_learner_named_in_a_configuration_takes_the_space_first(name, cls):
    """Building one is ``cls(space, rng, **options)``, the options matched by
    name against the signature. A learner spelling the first two the other way
    round constructs happily and searches the wrong thing, and a knob with no
    default is one no file may leave out of that learner's table.
    """
    assert issubclass(cls, ParameterSpaceLearner)
    params = list(inspect.signature(cls).parameters.values())
    assert [p.name for p in params[:2]] == ['space', 'rng']
    for knob in params[2:]:
        assert knob.default is not knob.empty, knob.name


@pytest.mark.parametrize('name, cls', sorted(learners.LEARNERS.items()))
def test_a_learner_named_in_a_configuration_declares_what_it_is_read_for(name, cls):
    """The generation the budget and the starvation count are measured by.

    It is read off an instance, so an instance is where it has to be true,
    and a learner may declare it however it likes: a class attribute, an
    assignment in ``__init__``, a property over its own settings. This holds
    a learner added later to the declaration whichever way it makes it. Built
    from its defaults over the smallest space there is, because a learner
    that only declares once it has been configured a particular way has not
    declared.
    """
    one_parameter = ParameterSpace([Parameter('x', 0.0, 1.0)])
    learner = cls(one_parameter, np.random.default_rng(4))

    generation = learner.generation
    # Zero is not "any number at a time", it is a session that proposes
    # nothing; a bool is an int that says true or false, not how many.
    assert generation is None or (
        isinstance(generation, numbers.Integral)
        and not isinstance(generation, bool)
        and generation > 0
    ), f'{name} declares generation {generation!r}'


def test_a_learner_handed_its_two_arguments_backwards_is_refused(space, rng):
    for cls in learners.LEARNERS.values():
        with pytest.raises(TypeError, match='in that order'):
            cls(rng, space)


def test_a_learner_that_does_not_propose_cannot_be_built(space, rng):
    """At either level, and when it is built rather than mid-experiment."""

    class Forgetful(Learner):
        pass

    class ForgetfulOverASpace(ParameterSpaceLearner):
        pass

    with pytest.raises(TypeError, match='propose'):
        Forgetful()
    with pytest.raises(TypeError, match='propose'):
        ForgetfulOverASpace(space, rng)


def test_a_proposal_without_a_source_is_refused_rather_than_recorded(space, rng):
    """The other half of taking the source from the learner itself.

    A learner that forgets to say where its proposals came from has to fail
    rather than have its shots recorded under some name nobody gave them.
    Over two parameters a bare row of proposals unpacks as a pair of numbers,
    so the refusal cannot be left to the unpacking.
    """

    class Silent(ParameterSpaceLearner):
        def propose(self, history, hint):
            return self.space.uniform(self.rng, hint)

    runmanager = FakeRunmanager()
    session = Session(
        Config(space=space, globals=(), cost_key=('r', 'c')),
        runmanager,
        Silent(space, rng),
    )
    with pytest.raises(TypeError, match='rather than a source'):
        session.refill()
    assert session.proposals == {}
    assert runmanager.submitted == []


@pytest.mark.parametrize(
    'learner',
    [
        RandomLearner,
        DirectedRandomLearner,
        lambda space, rng: GaussianProcessLearner(
            space, rng, minimum_observations=3
        ),
    ],
    ids=['random', 'directed_random', 'gaussian_process'],
)
def test_learners_filling_to_the_hint_keep_exactly_that_many_in_flight(
    space, rng, learner
):
    """Every pending record takes one of the hint's places, whoever proposed
    it, and a shot that will never report takes none.

    The start going out beside this call's proposals has not been submitted
    yet and is in flight all the same, so it takes a place too.
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


# --- directed random -------------------------------------------------------


def test_directed_random_searches_near_points_it_has_seen(space, rng):
    """The trust region has to actually engage once there is history.

    A guard that sends every draw to the whole space regardless of the history
    turns this learner into a plain random one, silently.
    """
    seen = np.array([[1.0, 1.0], [1.2, 0.8], [-3.0, 4.0], [0.0, -1.0]])
    history = [observe(i, p, sphere(p)) for i, p in enumerate(seen)]

    learner = DirectedRandomLearner(space, rng, trust_region=0.05)
    proposals = learner.ask(history, 300)

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
    proposals = learner.ask(history, 300)

    distances = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2)
    assert (distances.min(axis=1) <= 0.5 + 1e-9).all()
    # And it must not centre on the point that produced the bad cost.
    assert (np.abs(proposals - np.array([4.9, 4.9])).max(axis=1) > 0.5).all()


def test_directed_random_explores_the_whole_space_when_told_to(space, rng):
    history = [observe(0, [1.0, 1.0], 1.0), observe(1, [1.1, 1.1], 2.0)]
    learner = DirectedRandomLearner(
        space, rng, trust_region=0.01, explore_fraction=1.0
    )
    proposals = learner.ask(history, 400)
    assert proposals.min() < -4.0 and proposals.max() > 4.0


def test_directed_random_without_a_trust_region_is_a_random_learner(space, rng):
    history = [observe(0, [1.0, 1.0], 1.0), observe(1, [1.1, 1.1], 2.0)]
    learner = DirectedRandomLearner(space, rng, trust_region=None)
    proposals = learner.ask(history, 400)
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
    proposals = learner.ask(history, 300)
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
    # Nothing lands in [22.5, 27], and the best point is not the first one, so
    # falling back to the best is distinguishable from falling back to
    # whichever observation happens to head the history.
    costs = [30.0, 10.0, 20.0, 0.0]
    history = [observe(i, p, c) for i, (p, c) in enumerate(zip(seen, costs))]

    learner = DirectedRandomLearner(
        space, rng, trust_region=0.02, trust_range=(0.1, 0.25)
    )
    proposals = learner.ask(history, 200)
    nearest = np.abs(proposals[:, None, :] - seen[None, :, :]).max(axis=2).argmin(axis=1)
    assert set(np.unique(nearest)) == {3}


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
    # Twenty-five generations of eight: two hundred shots.
    history = run_loop(learner, space, sphere, batches=25, k=4, rng=rng)
    assert len(history) == 200
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

    history[0] = dataclasses.replace(history[0], cost=0.5, state=COMPLETE)
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

    for slot, proposal in enumerate(learner.ask(history, 4)):
        shared = int(np.isclose(proposal, members[slot]).sum())
        assert shared == space.num_params - 1, slot

    # And from part-way through a block, which is where the position the
    # proposal is made at stops agreeing with its place in the batch asked
    # for. ``propose`` reaches it after the run budget cuts a generation
    # short, and finishes the block from there; ``ask`` reaches it at any
    # history length, for as many as it is asked, across the end of the block.
    part_way = history[:6]
    members = learner.replay(part_way)[0]
    for offset, proposal in enumerate(learner.ask(part_way, 4)):
        slot = (len(part_way) + offset) % 4
        shared = int(np.isclose(proposal, members[slot]).sum())
        assert shared == space.num_params - 1, offset


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
        check(learner.ask(history, 1)[0], len(history), members)

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

    proposals = learner.ask(history, 4)
    assert space.contains(proposals).all()
    for member in members[1:]:
        assert int(np.isclose(proposals[0], member).sum()) < space.num_params - 1
    for slot in (1, 2, 3):
        shared = int(np.isclose(proposals[slot], members[slot]).sum())
        assert shared == space.num_params - 1, slot


def test_differential_evolution_proposes_a_generation_only_when_none_is_pending(rng):
    """A whole generation when nothing submitted is pending, and nothing while
    anything is, whatever the hint.

    The configured start is waited for like the founder it is, once it has
    been submitted. On the call that places it, it carries no shot id and goes
    out beside the rest of its generation, so that call proposes the other
    slots of the first block around it.
    """
    space = walk_space()
    learner = DifferentialEvolutionLearner(space, rng, population_size=4)
    founders = walk_history(space, rng, [10.0, 20.0, 30.0, 40.0])

    generation = learner.propose(founders, 1)
    assert len(generation) == 4
    assert {source for _, source in generation} == {'main'}

    trials = [observe(f't{slot}', params, 1.0) for slot, (params, _) in enumerate(generation)]
    trials[3] = dataclasses.replace(trials[3], cost=None, state=PENDING)
    assert learner.propose(founders + trials, 8) == []

    placing = Observation(None, space.minimum, None, state=PENDING, source='start')
    assert len(learner.propose([placing], 1)) == 3

    submitted = dataclasses.replace(placing, shot_id='shot-0')
    assert learner.propose([submitted] + founders[1:], 1) == []


def weight_bred_with(trial, best, members, slot):
    """The differential weight a ``best1`` trial was bred with, read off it.

    With every coordinate taken from the mutant and none resampled at a bound,
    such a trial is the best member plus the weight times the difference of two
    members other than its own slot's. Of all those pairs, the one the step
    from the best lies along gives the weight as the ratio; in four dimensions
    no other pair lines up with it by chance.
    """
    step = trial - best
    others = [member for i, member in enumerate(members) if i != slot]
    fits = []
    for a, b in itertools.combinations(others, 2):
        difference = a - b
        weight = step @ difference / (difference @ difference)
        fits.append((np.linalg.norm(step - weight * difference), abs(weight)))
    residual, weight = min(fits)
    assert residual < 1e-9
    return weight


def test_every_trial_in_a_generation_shares_one_differential_weight():
    """Drawn once per generation, as textbook differential evolution and scipy
    draw it, rather than once per trial.

    The founders sit in a small cluster in the middle of a wide space and every
    coordinate is taken from the mutant, so no trial reaches a bound and each
    is exactly the best member plus a weight times a difference of two others.
    Each generation's trials cost more than every founder, so the population
    they are bred from is the founders throughout, and the weight can be read
    back off every trial of three successive generations.
    """
    space = walk_space()
    size = 5
    learner = DifferentialEvolutionLearner(
        space,
        np.random.default_rng(7),
        population_size=size,
        cross_over_probability=1.0,
    )
    founders = np.random.default_rng(8).uniform(-0.5, 0.5, (size, space.num_params))
    history = [observe(slot, point, float(slot)) for slot, point in enumerate(founders)]

    generations = []
    for generation in range(3):
        trials = [params for params, _ in learner.propose(history, 1)]
        generations.append(
            [
                weight_bred_with(trial, founders[0], founders, slot)
                for slot, trial in enumerate(trials)
            ]
        )
        history += [
            observe(f'{generation}-{slot}', trial, 100.0)
            for slot, trial in enumerate(trials)
        ]

    for weights in generations:
        np.testing.assert_allclose(weights, weights[0])
        assert 0.5 <= weights[0] <= 1.0
    firsts = [weights[0] for weights in generations]
    assert len({round(weight, 9) for weight in firsts}) == 3, firsts


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
    """Including from nothing at all, which is not a case of its own.

    An empty history is a history too short, answered the same way. Nothing
    stands in for the data this learner does not have: where a run begins is
    the session's to propose, so a learner asked from nothing says it cannot
    rather than answering from nothing.
    """
    learner = GaussianProcessLearner(space, rng, minimum_observations=6)
    with pytest.raises(InsufficientData):
        learner.propose([observe(0, [0.0, 0.0], 1.0)], 1)

    with pytest.raises(InsufficientData):
        GaussianProcessLearner(space, rng).propose([], 1)


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


@pytest.mark.parametrize('refit_interval, carried, count', [(4, 8, 15), (8, 6, 7)])
def test_gaussian_process_state_depends_only_on_the_history(
    space, refit_interval, carried, count
):
    """Two learners given the same history must hold the same model.

    The kernel hyperparameters are cached between calls, so they have to be a
    function of the history alone: an instance that has been fitting all
    session must arrive at what a fresh one computes, not at a kernel fitted to
    however much it happened to hold when the cache was last filled.

    Both cases are ones where the cache has to give way between the two fits,
    because a case where it does not cannot tell the two learners apart
    whatever the caching does. The first crosses a refit boundary -- the kernel
    is owed to eight observations and then to twelve -- and the second is a
    history short of one whole interval, where there is no whole interval to
    fit to and the cache gives way on every arrival.
    """
    history = gaussian_process_history(space, 9, count=count)
    all_session = GaussianProcessLearner(
        space, np.random.default_rng(1), refit_interval=refit_interval
    )
    all_session.fit(history[:carried])
    all_session.fit(history)

    fresh = GaussianProcessLearner(
        space, np.random.default_rng(2), refit_interval=refit_interval
    )
    fresh.fit(history)

    # The cache is observable through the posterior it produces: two learners
    # that hold the same kernel predict the same thing everywhere, which is
    # what "a cache holds what a fresh instance would compute" means.
    grid = space.uniform(np.random.default_rng(4), 5)
    carried_mean, carried_std = all_session.predict(grid)
    fresh_mean, fresh_std = fresh.predict(grid)
    np.testing.assert_allclose(carried_mean, fresh_mean)
    np.testing.assert_allclose(carried_std, fresh_std)

    # And so the same proposals, once the two stand at the same point in their
    # own rng streams: that position is the one thing the history does not fix.
    all_session.rng = np.random.default_rng(3)
    fresh.rng = np.random.default_rng(3)
    np.testing.assert_allclose(all_session.ask(history, 1), fresh.ask(history, 1))


#: The schedule the exploration tests below configure: four weights, the
#: first of them greedy, so it takes four proposals to come back round.
SCHEDULE = [0.0, 50.0, 100.0, 150.0]


def exploring_and_greedy(space, count, uncer_bias=SCHEDULE):
    """The same proposal made under ``uncer_bias`` and with exploration off.

    The two learners share a seed and see the same history, so the acquisition
    weight is the only thing that differs between them.
    """
    history = gaussian_process_history(space, 5, count=count)
    greedy = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=0.0
    )
    explorer = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=uncer_bias
    )
    return float(
        np.linalg.norm(explorer.ask(history, 1)[0] - greedy.ask(history, 1)[0])
    )


@pytest.mark.parametrize('count', [12, 13, 14, 15])
def test_the_schedule_is_the_list_of_weights_it_was_handed(space, count):
    """``[0.0, 5.0]`` proposes greedily and then widely, by turns.

    The list is the schedule itself: its entries are the weights and its
    length is the period. Asked for two proposals, a learner running it spends
    one of them at each weight, and which one explores is decided by the
    parity of the history -- so the wider of the two picks changes places as
    the session goes on.
    """
    learner = GaussianProcessLearner(
        space, np.random.default_rng(3), uncer_bias=[0.0, 5.0]
    )
    history = gaussian_process_history(space, 5, count=count)
    away = np.linalg.norm(
        learner.ask(history, 2) - best(history).params, axis=1
    )
    # The greedy weight comes first from an even history and second from an
    # odd one, and the exploring proposal is the one that leaves the incumbent.
    assert (away[0] < away[1]) == (count % 2 == 0)


@pytest.mark.parametrize('count', [12, 13, 14, 15])
def test_a_single_weight_is_a_fixed_one_and_not_a_first_step(space, count):
    """A number written where a schedule goes weights every proposal.

    A cycle of one step has no greedy proposal in it unless the weight itself
    is zero, so a fixed weight explores at every count -- including the ones a
    four-step schedule spends on its greedy step.
    """
    assert exploring_and_greedy(space, count=count, uncer_bias=50.0) > 0.1


def test_a_schedule_of_no_weights_is_refused(space, rng):
    """There is no weight to propose at, and the cycle has no period."""
    with pytest.raises(ValueError, match='uncer_bias'):
        GaussianProcessLearner(space, rng, uncer_bias=[])


def test_the_exploration_weight_reaches_a_proposal_asked_for_on_its_own(space):
    """A session running one shot at a time still has to explore.

    It asks for several points once and then for a single point per completed
    shot, so a schedule that advanced with the position within the group asked
    for would stand at its first weight for the whole run and every other
    weight in the list would go unused.
    """
    assert exploring_and_greedy(space, count=13) > 0.1


@pytest.mark.parametrize('count', [12, 13, 14, 15, 16])
def test_the_exploration_schedule_advances_as_observations_arrive(space, count):
    """The weight moves on once per proposal and cycles over the list.

    The first weight of this schedule is zero, so one proposal in four is
    purely greedy and the rest look progressively further afield. Reading the
    position off the history rather than a counter is what keeps a learner
    handed the same history proposing the same thing.
    """
    apart = exploring_and_greedy(space, count=count)
    if count % len(SCHEDULE):
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
    learner = GaussianProcessLearner(space, rng)
    history = gaussian_process_history(space, 5)
    proposals = learner.ask(history, 6)
    # Twelve observations in hand, so the weights run 0, 1, 2, 3, 0, 1 and the
    # sixth pick repeats the weight of the second. Nothing but the fold-in
    # keeps it off that point: without it the two land 4e-6 apart.
    assert np.linalg.norm(proposals[5] - proposals[1]) > 1e-3


def flat_in_y_history(space, count):
    """Observations of a cost that varies in x alone.

    The fit finds no structure along y, so its length scale there runs to the
    upper end of ``length_scale_bounds`` -- which is the fit scikit-learn
    warns about, once per dimension, on every refit. The same fit drives the
    white-noise level to its own lower bound and is warned about for that too.
    """
    points = space.uniform(np.random.default_rng(5), count)
    return [
        observe(i, p, float((p[0] - 1.3) ** 2)) for i, p in enumerate(points)
    ]


def test_a_length_scale_at_a_bound_is_reported_when_the_set_of_them_changes(
    space, rng
):
    """Once when a parameter reaches a bound, once when it leaves.

    A dimension the apparatus keeps pinned is warned about by scikit-learn on
    every refit, which for a run of several hundred shots buries the
    diagnostic in copies of itself. Said on the change instead, the one thing
    a lab has to act on -- that the set moved -- is the only thing it reads.
    """
    from sklearn.exceptions import ConvergenceWarning

    learner = GaussianProcessLearner(space, rng, refit_interval=1)

    with pytest.warns(ConvergenceWarning, match='y at the upper end') as first:
        learner.fit(flat_in_y_history(space, 12))
    assert learner.at_length_scale_bounds == {'y': 'upper'}
    # And scikit-learn's own copy of it is not there beside ours. Its wording
    # is what tells it apart from the white-noise level's bound, which is the
    # same message about a different hyperparameter and is left alone.
    assert not [
        w
        for w in first
        if 'close to the specified' in str(w.message)
        and 'length_scale' in str(w.message)
    ]

    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter('always')
        learner.fit(flat_in_y_history(space, 13))
    assert learner.at_length_scale_bounds == {'y': 'upper'}
    assert not [w for w in again if 'length_scale' in str(w.message)]

    # The set moving back to empty is a change like any other: silence here
    # would read the same as the refit above, where it had not moved.
    with pytest.warns(ConvergenceWarning, match='inside length_scale_bounds'):
        learner.fit(gaussian_process_history(space, 5))
    assert learner.at_length_scale_bounds == {}


def test_the_rest_of_what_a_refit_warns_about_still_reaches_the_lab(space, rng):
    """Only the length-scale message is held back.

    The fit that pins a length scale drives the white-noise level to its own
    bound as well, and scikit-learn warns about that in the same category. A
    filter written against the category rather than the message would take it
    too, and a lab would lose a diagnostic that was never repeating.
    """
    from sklearn.exceptions import ConvergenceWarning

    learner = GaussianProcessLearner(space, rng, refit_interval=1)
    with pytest.warns(ConvergenceWarning, match='noise_level'):
        learner.fit(flat_in_y_history(space, 12))


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
    probe = learner.ask(history, 1)[0]
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
        learner.ask(history[:-1], 1)
    assert space.contains(learner.ask(history, 1)).all()


def test_a_batch_is_proposed_from_a_history_whose_points_carry_uncertainties(
    space, rng
):
    """Each point of the batch is folded into the fit before the next is
    picked, and a fit holding a variance per point needs one for the invented
    point too, or it is handed one fewer variance than it has points.
    """
    points = space.uniform(np.random.default_rng(7), 12)
    history = [observe(i, p, offset_sphere(p), uncer=0.1) for i, p in enumerate(points)]
    proposals = GaussianProcessLearner(space, rng).ask(history, 3)
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


def a_config(space, learner, options, **tables):
    """A configuration whose only interesting part is the learner's own table."""
    return Config(
        space=space,
        globals=(),
        cost_key=('routine', 'cost'),
        learner=learner,
        learner_options={learner: options, **tables},
        seed=11,
    )


def test_a_learner_takes_the_knobs_in_its_own_table_and_no_others(space):
    """A knob reaches the learner whose table it is written in and no other.

    That is what lets one file carry the settings for several learners and be
    switched between them: the tables for the learners this session does not
    build are simply not read.
    """
    config = a_config(
        space,
        'differential_evolution',
        {'population_size': 15},
        gaussian_process={'trust_region': 0.5, 'cost_has_noise': False},
    )
    built = build(config)
    # population_size is the number of members, not a multiplier on the
    # parameter count: fifteen here, over however many parameters.
    assert built.population_size == 15
    # trust_region is a knob this learner takes as well, written in a table
    # that is not its own. It reaches the Gaussian process and nothing else,
    # so this learner is left with its own default of the whole space.
    assert built.trust_region is None


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


def test_a_knob_is_a_constructor_argument_and_not_a_constructor_local(
    space, monkeypatch
):
    """What a constructor accepts is its arguments, and nothing else.

    A name a constructor happens to use as a scratch variable is not a knob it
    takes. Held to anything wider than the signature, such a key is welcomed
    into the learner's table and then raises a TypeError out of the
    constructor, blaming the file for a collision the user cannot see.
    """

    class Scratch(ParameterSpaceLearner):
        def __init__(self, space, rng, population_size=3):
            super().__init__(space, rng)
            cost_has_noise = population_size  # a local, not an argument
            self.population_size = cost_has_noise

        def propose(self, history, hint):
            return [(p, 'main') for p in self.space.uniform(self.rng, hint)]

    monkeypatch.setitem(learners.LEARNERS, 'scratch', Scratch)
    with pytest.raises(ValueError, match=r'\[LEARNER\.scratch\].*cost_has_noise'):
        build(a_config(space, 'scratch', {'cost_has_noise': True}))
    sized = build(a_config(space, 'scratch', {'population_size': 4}))
    assert sized.population_size == 4


def test_a_learner_that_hides_its_knobs_in_kwargs_is_refused(space, monkeypatch):
    """``**kwargs`` names nothing, so a table written for such a learner is
    refused key by key with nothing saying why -- and were the table empty,
    the learner would be built from its defaults alone.
    """

    class Swallower(ParameterSpaceLearner):
        def __init__(self, space, rng, **kwargs):
            super().__init__(space, rng)
            self.population_size = kwargs.get('population_size', 3)

        def propose(self, history, hint):
            return [(p, 'main') for p in self.space.uniform(self.rng, hint)]

    monkeypatch.setitem(learners.LEARNERS, 'swallower', Swallower)
    with pytest.raises(TypeError, match='kwargs'):
        build(a_config(space, 'swallower', {'population_size': 4}))


# --- two phase -------------------------------------------------------------


class Ready(Learner):
    """A main learner that always proposes, as many as it is hinted.

    A :class:`Learner`, and so carrying the declarations a wrapper reads off
    what it wraps: a double that answered for fewer of them than the learners
    that ship would let the wrapper read one with a default and still pass.
    """

    def propose(self, history, hint):
        return [(np.zeros(2), 'main') for _ in range(hint)]


class NeverReady(Learner):
    """A main learner that can never propose from the history it is given."""

    def propose(self, history, hint):
        raise InsufficientData('not yet')


class Marked(Learner):
    """A trainer whose proposals cannot be mistaken for the main learner's.

    ``Ready`` proposes the origin, so a batch of sevens came from here and a
    batch of zeros did not: which learner made a batch is then read off the
    batch rather than off the phase the wrapper labels it with.
    """

    def propose(self, history, hint):
        return [(np.full(2, 7.0), 'main') for _ in range(hint)]


def phase_of(proposed):
    """The one phase a batch from a two-phase learner is labelled with."""
    (phase,) = {source for _, source in proposed}
    return phase


def test_two_phase_trains_first_then_hands_over(space, rng):
    learner = TwoPhaseLearner(RandomLearner(space, rng), Ready(), num_training=3)
    history = []
    phases = []
    for i in range(6):
        [(params, phase)] = learner.propose(history, 1)
        phases.append(phase)
        history.append(observe(i, params, float(i)))
    assert phases == ['training'] * 3 + ['main'] * 3


def test_a_main_learner_that_cannot_propose_is_not_caught(space, rng):
    """There is nothing left to catch it.

    The wrapper used to fall back to the trainer here. It cannot be reached
    from a configuration any more -- the training phase is refused unless it
    covers what the main learner needs -- so a main learner refusing to
    propose after that is a broken learner, and the run stops with that
    learner's own message rather than going on under a phase reading "main".
    """
    learner = TwoPhaseLearner(RandomLearner(space, rng), NeverReady(), num_training=1)
    history = [observe(i, [0.0, 0.0], float(i)) for i in range(5)]
    with pytest.raises(InsufficientData, match='not yet'):
        learner.propose(history, 2)


def test_shots_without_a_usable_cost_do_not_count_as_training(space, rng):
    learner = TwoPhaseLearner(RandomLearner(space, rng), Ready(), num_training=2)
    history = [
        observe('a', [0.0, 0.0], 1.0, bad=True),
        observe('b', [0.0, 0.0], float('inf')),
        observe('c', [0.0, 0.0], float('nan')),
    ]
    assert phase_of(learner.propose(history, 1)) == 'training'


def test_a_warmup_shorter_than_the_main_learner_needs_is_refused(space, rng):
    """A number that does not mean what it says.

    With a warmup of five in front of a learner that will not fit below six,
    the handover happens at six: the sixth shot comes from the trainer under a
    setting that says the fifth was the last of them. Both numbers are named,
    because a message giving one leaves the reader to go and find the other.
    """
    main = GaussianProcessLearner(space, rng, minimum_observations=6)
    with pytest.raises(ValueError) as raised:
        TwoPhaseLearner(RandomLearner(space, rng), main, num_training=5)
    assert 'num_training_runs is 5' in str(raised.value)
    assert 'holds 6 usable observations' in str(raised.value)


def test_a_warmup_exactly_as_long_as_the_main_learner_needs_is_accepted(space, rng):
    """The other arm of the comparison, and the one that matters.

    At six the main learner can propose on the very shot the handover is
    configured for, so this is the shortest warmup that means what it says --
    and a refusal written one number wide would take it too.
    """
    main = GaussianProcessLearner(space, rng, minimum_observations=6)
    learner = TwoPhaseLearner(RandomLearner(space, rng), main, num_training=6)
    assert learner.num_training == 6


def test_a_two_phase_learner_answers_for_the_observations_it_needs(space, rng):
    """It never refuses to propose -- each of its phases is held to a learner
    that can propose throughout it -- so it absorbs its main learner's
    requirement rather than passing it on. Declared, because an attribute read
    off a learner with a default is the reader's answer and not the learner's.
    """
    main = GaussianProcessLearner(space, rng, minimum_observations=6)
    learner = TwoPhaseLearner(RandomLearner(space, rng), main, num_training=6)
    assert learner.minimum_observations == 0


def test_a_learner_that_declares_nothing_is_not_one_that_declares_no_barrier(
    space, rng
):
    """A wrapper reads the declaration off what it wraps and supplies none.

    Filled in with a default here, a learner that had stopped declaring its
    generation would be wrapped without a word and its declaration lost behind
    the wrapper, with the budget refusal and the starvation count taking it
    for a learner that proposes any number and everything still running.
    """

    class Undeclared:
        def propose(self, history, hint):
            return [(np.zeros(2), 'main') for _ in range(hint)]

    with pytest.raises(AttributeError, match='generation'):
        TwoPhaseLearner(RandomLearner(space, rng), Undeclared(), num_training=3)


def test_a_generational_learner_cannot_be_put_behind_a_trainer(space, rng):
    """Refused at construction, because the wrapper cannot answer for it.

    Behind a trainer the history the learner reads its roles off opens with
    positions it never proposed, and a two-phase learner declares no
    generation of its own for the budget and the starvation count to read.
    """
    main = DifferentialEvolutionLearner(space, rng, population_size=4)
    with pytest.raises(ValueError, match='whole generations of 4'):
        TwoPhaseLearner(RandomLearner(space, rng), main, num_training=8)


def test_a_generational_learner_cannot_be_the_trainer_either(space, rng):
    """The same declaration is lost by the same route. Which phase the learner
    proposes in changes nothing about what its declaration promises.
    """
    trainer = DifferentialEvolutionLearner(space, rng, population_size=4)
    with pytest.raises(ValueError, match='whole generations of 4'):
        TwoPhaseLearner(trainer, RandomLearner(space, rng), num_training=8)


def test_a_trainer_that_withholds_its_proposals_is_refused(space, rng):
    """Nothing stands behind the trainer.

    It proposes the first shot of the run, from an empty history, and every
    periodic run after training. One that refuses to propose until the history
    holds points raises out through the session at the first shot, and the
    wrapper's own ``minimum_observations`` of zero -- which a wrapper of this
    reads to size its own training phase -- is a promise it could not keep.
    """
    trainer = GaussianProcessLearner(space, rng)
    assert trainer.minimum_observations > 0
    with pytest.raises(ValueError, match='will not propose until'):
        TwoPhaseLearner(trainer, Ready(), num_training=8)


def test_a_learner_that_declares_no_barrier_is_wrapped(space, rng):
    """What is refused is a barrier the wrapper cannot hold, not wrapping. The
    wrapper's own ``generation`` is true of it because of that refusal.
    """
    learner = TwoPhaseLearner(
        DirectedRandomLearner(space, rng), GaussianProcessLearner(space, rng), 8
    )
    assert learner.generation is None


# --- the trainer's periodic return -----------------------------------------


def phases_over(learner, count, k=1):
    """Drive ``learner`` one batch at a time and collect the phase of each."""
    history = []
    phases = []
    for i in range(count):
        phases.append(phase_of(learner.propose(history, k)))
        history.append(observe(i, [0.0, 0.0], float(i)))
    return phases


def test_the_trainer_does_not_come_back_unless_a_period_is_asked_for():
    """The default leaves every existing file proposing what it proposed.

    Switching this on by default would change the shots of every run already
    configured, with the file that configured it unchanged and saying nothing.
    """
    learner = TwoPhaseLearner(Marked(), Ready(), num_training=2)
    assert learner.num_runs_between_trainer_runs is None
    assert phases_over(learner, 10) == ['training'] * 2 + ['main'] * 8


def test_the_trainer_takes_one_proposal_in_every_cycle():
    """Three from the main learner, then one from the trainer, and round again.

    The period is how many come from the main learner between one of the
    trainer's and the next, so the cycle is one longer than the number.
    """
    learner = TwoPhaseLearner(
        Marked(), Ready(), num_training=2, num_runs_between_trainer_runs=3
    )
    assert phases_over(learner, 12) == (
        ['training'] * 2
        + ['main', 'main', 'main', 'periodic trainer'] * 2
        + ['main', 'main']
    )


def test_the_periodic_proposal_is_the_trainers_own():
    """The routing, not only the label the wrapper puts on it."""
    learner = TwoPhaseLearner(
        Marked(), Ready(), num_training=0, num_runs_between_trainer_runs=2
    )
    history = [observe(i, [0.0, 0.0], float(i)) for i in range(2)]
    [(params, phase)] = learner.propose(history, 1)
    assert (params == 7.0).all()
    assert phase == 'periodic trainer'
    # And the position before it, which is the main learner's.
    [(params, phase)] = learner.propose(history[:1], 1)
    assert (params == 0.0).all()
    assert phase == 'main'


def test_a_position_that_produced_no_observation_does_not_move_the_cycle_on():
    """Read off the history like the handover, and not off a count of calls.

    A dropped shot spends a position and produces nothing, so a learner
    counting its own proposals would take the trainer's turn a shot early and
    stay a shot out for the rest of the run. Handed the same usable
    observations, this answers the same thing.
    """
    learner = TwoPhaseLearner(
        Marked(), Ready(), num_training=0, num_runs_between_trainer_runs=2
    )
    seen = [observe(i, [0.0, 0.0], float(i)) for i in range(2)]
    assert phase_of(learner.propose(seen, 1)) == 'periodic trainer'
    spent = seen + [observe('gone', [0.0, 0.0], None, state=DROPPED)]
    assert phase_of(learner.propose(spent, 1)) == 'periodic trainer'
    # A third usable observation is what moves it on.
    moved_on = seen + [observe(2, [0.0, 0.0], 2.0)]
    assert phase_of(learner.propose(moved_on, 1)) == 'main'


def test_a_batch_is_made_whole_by_the_learner_whose_turn_it_opens_on():
    """The turn is taken at the batch boundary rather than splitting the batch.

    A split would have to say which position it had reached part way through a
    batch, and the only count available for that is one observation per
    proposal -- untrue of every shot that is dropped or comes back unusable.
    """
    learner = TwoPhaseLearner(
        Marked(), Ready(), num_training=0, num_runs_between_trainer_runs=2
    )
    opening_on_the_trainer = [observe(i, [0.0, 0.0], float(i)) for i in range(2)]
    proposed = learner.propose(opening_on_the_trainer, 3)
    assert len(proposed) == 3
    assert all((params == 7.0).all() for params, _ in proposed)
    assert phase_of(proposed) == 'periodic trainer'
    # One position earlier the trainer's turn falls inside the batch, and the
    # main learner still makes the whole of it.
    proposed = learner.propose(opening_on_the_trainer[:1], 3)
    assert len(proposed) == 3
    assert all((params == 0.0).all() for params, _ in proposed)
    assert phase_of(proposed) == 'main'


def test_a_period_leaving_the_main_learner_nothing_to_propose_is_refused():
    """Zero is every proposal the trainer's, which is the named learner never
    running at all. Leaving the setting out is how it is switched off.
    """
    with pytest.raises(ValueError, match='at least 1'):
        TwoPhaseLearner(
            Marked(), Ready(), num_training=0, num_runs_between_trainer_runs=0
        )
    # And the floor itself, which alternates the two.
    learner = TwoPhaseLearner(
        Marked(), Ready(), num_training=0, num_runs_between_trainer_runs=1
    )
    assert phases_over(learner, 4) == ['main', 'periodic trainer'] * 2


def test_a_gaussian_process_that_would_fit_nothing_is_refused(space, rng):
    """At zero the guard in ``fit`` passes on an empty history and the refusal
    a file gets comes from inside a scikit-learn scaler.
    """
    with pytest.raises(ValueError, match='minimum_observations.*at least 1'):
        GaussianProcessLearner(space, rng, minimum_observations=0)


def test_a_gaussian_process_needing_one_observation_is_accepted(space, rng):
    learner = GaussianProcessLearner(space, rng, minimum_observations=1)
    assert learner.minimum_observations == 1


def test_an_acquisition_that_is_never_finite_says_so(space, rng):
    """Every start comes back NaN, so no comparison in the search is ever
    true and there is no winner to clip.
    """
    class NotFinite:
        def predict(self, u, return_std=False):
            rows = np.atleast_2d(u).shape[0]
            if return_std:
                return np.full(rows, np.nan), np.full(rows, np.nan)
            return np.full(rows, np.nan)

    learner = GaussianProcessLearner(space, rng)
    lows = np.zeros(space.num_params)
    highs = np.ones(space.num_params)
    with pytest.raises(RuntimeError, match='not finite at any'):
        learner.minimise_acquisition(
            NotFinite(), 1.0, space.minimum, lows, highs
        )


def test_an_exact_cost_beside_one_with_no_uncertainty_is_refused(space, rng):
    """cost_has_noise off says every cost is measured exactly. A history where
    only some shots carry an uncertainty gives the rest a variance of exactly
    zero with no white-noise term anywhere, and the fit is singular.
    """
    learner = GaussianProcessLearner(
        space, rng, cost_has_noise=False, minimum_observations=2
    )
    history = [
        observe('a', space.minimum, 1.0, uncer=0.1),
        observe('b', space.maximum, 2.0),
    ]
    with pytest.raises(ValueError, match="'b'"):
        learner.fit(history)


def test_costs_that_all_carry_an_uncertainty_fit_without_noise(space, rng):
    learner = GaussianProcessLearner(
        space, rng, cost_has_noise=False, minimum_observations=2
    )
    history = [
        observe('a', space.minimum, 1.0, uncer=0.1),
        observe('b', space.maximum, 2.0, uncer=0.2),
    ]
    assert learner.fit(history)


def test_costs_that_carry_no_uncertainty_at_all_fit_without_noise(space, rng):
    learner = GaussianProcessLearner(
        space, rng, cost_has_noise=False, minimum_observations=2
    )
    history = [
        observe('a', space.minimum, 1.0),
        observe('b', space.maximum, 2.0),
    ]
    assert learner.fit(history)


def test_a_mixed_history_still_fits_when_the_cost_has_noise(space, rng):
    """The ordinary case: a lab whose uncertainty fit fails on some shots."""
    learner = GaussianProcessLearner(
        space, rng, cost_has_noise=True, minimum_observations=2
    )
    history = [
        observe('a', space.minimum, 1.0, uncer=0.1),
        observe('b', space.maximum, 2.0),
    ]
    assert learner.fit(history)


def test_a_late_cost_refits_a_prefix_of_unchanged_length(space):
    """The hyperparameter cache is keyed on *which* observations it was fitted
    to, not on how many.

    Costs arrive out of order and ``usable`` returns them in proposal order, so
    a cost that turns up late is inserted in the middle rather than appended.
    With nine proposals and one of them still pending, the prefix a refit uses
    is eight long; when that pending cost lands the prefix is eight long still,
    and holds a different eight. Keyed on a count, the cache would answer with
    the kernel it fitted to the other set.
    """
    history = gaussian_process_history(space, 11, count=9)
    waiting = list(history)
    waiting[2] = dataclasses.replace(
        waiting[2], cost=None, uncer=None, state=PENDING
    )

    carried = GaussianProcessLearner(
        space, np.random.default_rng(1), refit_interval=4, minimum_observations=4
    )
    carried.fit(waiting)
    assert len(usable(waiting)) == 8
    carried.fit(history)

    fresh = GaussianProcessLearner(
        space, np.random.default_rng(1), refit_interval=4, minimum_observations=4
    )
    fresh.fit(history)

    grid = space.uniform(np.random.default_rng(5), 5)
    np.testing.assert_allclose(carried.predict(grid)[0], fresh.predict(grid)[0])
    np.testing.assert_allclose(carried.predict(grid)[1], fresh.predict(grid)[1])


# --- what a knob may be written as ----------------------------------------


#: One value of the wrong kind for every knob of every learner a configuration
#: can name, and the whole of the refusal it gets. Each is something a
#: conversion would have taken: a quoted boolean is a non-empty string and so
#: true, a fraction truncates, a boolean is 1, a quoted number reads as the
#: number, and a single number where a pair belongs dies inside ``tuple`` in
#: Python's words rather than this package's.
REGION = (
    "trust_region must be written as a number in (0, 1), a fraction of each "
    "parameter's range, or as a list of numbers, one distance per parameter, "
    "not {}."
)
MISWRITTEN = [
    (
        'directed_random',
        'trust_gaussian',
        'false',
        "trust_gaussian must be written as true or false, unquoted, not 'false'.",
    ),
    (
        'gaussian_process',
        'cost_has_noise',
        'false',
        "cost_has_noise must be written as true or false, unquoted, not 'false'.",
    ),
    (
        'gaussian_process',
        'cost_has_noise',
        0,
        "cost_has_noise must be written as true or false, unquoted, not 0.",
    ),
    (
        'differential_evolution',
        'population_size',
        8.9,
        "population_size must be written as a whole number, got 8.9.",
    ),
    (
        'differential_evolution',
        'population_size',
        True,
        "population_size must be written as a whole number, got True.",
    ),
    (
        'gaussian_process',
        'refit_interval',
        True,
        "refit_interval must be written as a whole number, got True.",
    ),
    (
        'gaussian_process',
        'refit_interval',
        4.5,
        "refit_interval must be written as a whole number, got 4.5.",
    ),
    (
        'gaussian_process',
        'minimum_observations',
        6.5,
        "minimum_observations must be written as a whole number, got 6.5.",
    ),
    (
        'gaussian_process',
        'minimum_observations',
        True,
        "minimum_observations must be written as a whole number, got True.",
    ),
    (
        'directed_random',
        'explore_fraction',
        '0.5',
        "explore_fraction must be written as a number, unquoted, not '0.5'.",
    ),
    (
        'directed_random',
        'explore_fraction',
        True,
        "explore_fraction must be written as a number, unquoted, not True.",
    ),
    (
        'differential_evolution',
        'cross_over_probability',
        '0.7',
        "cross_over_probability must be written as a number, unquoted, not '0.7'.",
    ),
    (
        'differential_evolution',
        'cross_over_probability',
        True,
        "cross_over_probability must be written as a number, unquoted, not True.",
    ),
    (
        'gaussian_process',
        'cost_bias',
        '1.0',
        "cost_bias must be written as a number, unquoted, not '1.0'.",
    ),
    (
        'gaussian_process',
        'cost_bias',
        True,
        "cost_bias must be written as a number, unquoted, not True.",
    ),
    (
        'gaussian_process',
        'uncer_bias',
        '1.0',
        "uncer_bias must be written as a number or a list of numbers, "
        "unquoted, not '1.0'.",
    ),
    (
        'gaussian_process',
        'uncer_bias',
        True,
        "uncer_bias must be written as a number or a list of numbers, "
        "unquoted, not True.",
    ),
    (
        'gaussian_process',
        'uncer_bias',
        ['0', '1'],
        "uncer_bias must be written as a number or a list of numbers, "
        "unquoted, not ['0', '1'].",
    ),
    (
        'directed_random',
        'trust_range',
        0.5,
        "trust_range must be written as a pair of numbers, [low, high], not 0.5.",
    ),
    (
        'directed_random',
        'trust_range',
        '0.1, 0.25',
        "trust_range must be written as a pair of numbers, [low, high], not "
        "'0.1, 0.25'.",
    ),
    (
        'directed_random',
        'trust_range',
        ['0.1', '0.25'],
        "trust_range must be written as a pair of numbers, [low, high], not "
        "['0.1', '0.25'].",
    ),
    (
        'directed_random',
        'trust_range',
        [0.1, 0.2, 0.25],
        "trust_range must be written as a pair of numbers, [low, high], not "
        "[0.1, 0.2, 0.25].",
    ),
    (
        'gaussian_process',
        'length_scale_bounds',
        5,
        "length_scale_bounds must be written as a pair of numbers, [low, "
        "high], not 5.",
    ),
    (
        'gaussian_process',
        'length_scale_bounds',
        np.array(5.0),
        "length_scale_bounds must be written as a pair of numbers, [low, "
        "high], not array(5.).",
    ),
    (
        'gaussian_process',
        'length_scale_bounds',
        [1e-2, [1e2]],
        "length_scale_bounds must be written as a pair of numbers, [low, "
        "high], not [0.01, [100.0]].",
    ),
    (
        'gaussian_process',
        'noise_level_bounds',
        'fixed',
        "noise_level_bounds must be written as a pair of numbers, [low, "
        "high], not 'fixed'.",
    ),
    (
        'gaussian_process',
        'noise_level_bounds',
        [1e-5, True],
        "noise_level_bounds must be written as a pair of numbers, [low, "
        "high], not [1e-05, True].",
    ),
    (
        'differential_evolution',
        'mutation_scale',
        0.8,
        "mutation_scale is the range the differential weight is drawn from, "
        "[low, high], not one weight: for a fixed weight of 0.8, write "
        "[0.8, 0.8].",
    ),
    (
        'differential_evolution',
        'mutation_scale',
        [0.5],
        "mutation_scale must be written as a pair of numbers, [low, high], "
        "not [0.5].",
    ),
    (
        'differential_evolution',
        'mutation_scale',
        ['0.5', '1.0'],
        "mutation_scale must be written as a pair of numbers, [low, high], "
        "not ['0.5', '1.0'].",
    ),
    (
        'differential_evolution',
        'evolution_strategy',
        ['best1'],
        "evolution_strategy must be one of ('best1', 'best2', 'rand1', "
        "'rand2'), got ['best1']",
    ),
    ('directed_random', 'trust_region', '0.1', REGION.format("'0.1'")),
    ('differential_evolution', 'trust_region', '0.1', REGION.format("'0.1'")),
    ('gaussian_process', 'trust_region', '0.1', REGION.format("'0.1'")),
    ('directed_random', 'trust_region', True, REGION.format('True')),
    ('directed_random', 'trust_region', ['1', '1'], REGION.format("['1', '1']")),
]


@pytest.mark.parametrize('name, knob, written, message', MISWRITTEN)
def test_a_knob_of_the_wrong_kind_is_refused_naming_it(
    space, rng, name, knob, written, message
):
    """Checked in the constructor, so a learner built in Python is held to its
    kinds exactly as one built from a file.
    """
    with pytest.raises(ValueError) as refusal:
        learners.LEARNERS[name](space, rng, **{knob: written})
    assert str(refusal.value) == message


def test_every_knob_of_every_learner_is_held_to_its_kind():
    """Read off the constructors, so a knob added to one without a case above
    fails here rather than going unchecked.
    """
    held = {(name, knob) for name, knob, _, _ in MISWRITTEN}
    taken = {
        (name, knob)
        for knob, names in learners.knobs_by_learner().items()
        for name in names
    }
    assert held == taken


def test_a_knob_takes_the_kinds_python_and_numpy_hand_it(space, rng):
    """A list is what TOML gives. Tuples, numpy arrays and numpy scalars are
    what code building a learner directly passes, and each is the kind of
    thing its knob asks for.
    """
    directed = DirectedRandomLearner(
        space,
        rng,
        trust_region=np.array([1.0, 2.0]),
        trust_range=np.array([0.1, 0.25]),
        trust_gaussian=True,
        explore_fraction=np.float64(0.25),
    )
    np.testing.assert_array_equal(directed.trust_region, [1.0, 2.0])
    assert directed.trust_range == (0.1, 0.25)
    assert directed.trust_gaussian is True
    assert directed.explore_fraction == 0.25

    evolving = DifferentialEvolutionLearner(
        space,
        rng,
        population_size=np.int64(5),
        mutation_scale=[0.5, 1],
        cross_over_probability=np.float32(0.5),
        trust_region=np.float64(0.1),
    )
    assert evolving.population_size == 5
    assert evolving.mutation_scale == (0.5, 1.0)
    assert evolving.cross_over_probability == 0.5
    np.testing.assert_allclose(evolving.trust_region, [1.0, 1.0])

    fitting = GaussianProcessLearner(
        space,
        rng,
        cost_has_noise=False,
        length_scale_bounds=np.array([1e-2, 1e2]),
        noise_level_bounds=(1e-5, 10),
        cost_bias=np.int64(2),
        uncer_bias=np.array([0.0, 1.5]),
        refit_interval=np.int64(3),
        minimum_observations=np.int64(4),
        trust_region=[1, 2],
    )
    assert fitting.cost_has_noise is False
    assert fitting.length_scale_bounds == (1e-2, 1e2)
    assert fitting.noise_level_bounds == (1e-5, 10.0)
    assert fitting.cost_bias == 2.0
    assert fitting.uncer_bias == (0.0, 1.5)
    assert fitting.refit_interval == 3
    assert fitting.minimum_observations == 4
    np.testing.assert_array_equal(fitting.trust_region, [1.0, 2.0])
