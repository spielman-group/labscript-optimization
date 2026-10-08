"""Session behavior: matching costs to shots, and never waiting on a lost one."""

import math

import pytest
from runmanager.client import (
    LYSE_NOT_SENT,
    LYSE_REJECTED,
    LYSE_WAITING,
    SequenceRefused,
)

from labscript_optimization import config as config_module
from labscript_optimization.observations import COMPLETE, DROPPED, PENDING, usable
from labscript_optimization.session import Session

from conftest import settle


def make_config(learner='random', table='', maximize=False, start=None, **general):
    """A one-parameter configuration.

    ``general`` is written into ``[GENERAL]``, ``table`` into the learner's
    table, and ``start`` onto the parameter.
    """
    general = ''.join(f'{key} = {value}\n' for key, value in general.items())
    table = f'[LEARNER.{learner}]\n{table}' if table else ''
    start = '' if start is None else f'start = {start}'
    return config_module.loads(
        f"""
[ANALYSIS]
cost_key = ["r", "c"]
maximize = {str(maximize).lower()}
groups = ["G"]
[GENERAL]
learner = "{learner}"
{general}{table}
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
{start}
"""
    )


def running_session(config, runmanager):
    session = Session(config, runmanager)
    session.start()
    return session


@pytest.fixture
def session(runmanager):
    return running_session(make_config(num_buffered_runs=3), runmanager)


def test_a_new_session_waits_for_start(runmanager):
    session = Session(make_config(num_buffered_runs=3), runmanager)

    assert session.status()['paused'] is True
    assert session.refill() == []
    assert runmanager.submitted == []

    assert session.start() is True
    assert session.refill() == ['shot-0', 'shot-1', 'shot-2']
    assert session.status()['paused'] is False


def test_pause_keeps_taking_costs_but_submits_nothing(runmanager):
    session = running_session(make_config(num_buffered_runs=2), runmanager)
    first, second = session.refill()

    assert session.pause() is True
    assert session.record(first, 1.0, None, False) == 'main'
    assert session.refill() == []
    assert session.status()['completed'] == 1
    assert session.status()['paused'] is True
    session.record(second, 2.0, None, False)
    session.start()
    assert len(session.refill()) == 2
    assert session.status()['starved'] == 0


def test_the_queue_is_kept_filled_to_the_buffer_depth(session):
    assert session.refill() == ['shot-0', 'shot-1', 'shot-2']
    assert session.refill() == []
    session.record('shot-1', 4.0, None, False)
    assert session.refill() == ['shot-3']


def test_costs_arriving_out_of_order_land_in_proposal_order(session):
    """The history is the proposals, in the order they were made, so a cost
    that arrives late fills its own place rather than being appended to the
    end of one.
    """
    session.refill()
    session.record('shot-2', 3.0, None, False)
    session.record('shot-0', 9.0, None, False)
    assert [o.shot_id for o in session.history] == ['shot-0', 'shot-1', 'shot-2']
    with_a_cost = [o.shot_id for o in session.history if o.cost is not None]
    assert with_a_cost == ['shot-0', 'shot-2']


def test_only_the_first_cost_of_a_shot_this_session_submitted_is_taken(session):
    """Shots of a user's own and runmanager's defaults pass through harmlessly,
    and a row can run twice, by a retry or by BLACS re-running a file with data.
    """
    session.refill()
    assert session.record('someone-elses-shot', 1.0, None, False) is None
    assert session.record('shot-0', 5.0, None, False) is not None
    assert session.record('shot-0', 99.0, None, False) is None
    assert session.status()['completed'] == 1
    assert session.best.cost == 5.0


def test_the_history_tells_a_shot_still_coming_from_one_that_never_will(
    session, runmanager
):
    """``dropped`` is the number a user reads to see whether shots are being
    lost, so the record keeps the two apart instead of leaving them to be told
    from a missing cost.
    """
    session.refill()
    runmanager.lose('shot-0')
    session.reconcile()
    session.record('shot-1', 1.0, None, False)

    assert {o.shot_id: o.state for o in session.history} == {
        'shot-0': DROPPED,
        'shot-1': COMPLETE,
        'shot-2': PENDING,
    }
    assert [o.shot_id for o in usable(session.history)] == ['shot-1']


def test_a_shot_that_will_never_arrive_stops_being_waited_on(session, runmanager):
    """A lost shot must free its place, or the session stalls for ever: nothing
    will report its cost.
    """
    session.refill()
    runmanager.lose('shot-0', 'shot-1', 'shot-2')

    assert session.reconcile() == ['shot-0', 'shot-1', 'shot-2']
    assert session.refill() == ['shot-3', 'shot-4', 'shot-5']


def test_a_shot_whose_cost_cannot_come_is_dropped_at_the_first_reconcile(runmanager):
    """Refused by BLACS, not handed to lyse, rejected by lyse, or forgotten by
    runmanager as it is across a restart: nothing further will happen to any of
    them, so there is no second round to wait for.
    """
    session = running_session(make_config(num_buffered_runs=4), runmanager)
    session.refill()
    runmanager.lose('shot-0')
    runmanager.finish('shot-1', lyse=LYSE_NOT_SENT)
    runmanager.finish('shot-2', lyse=LYSE_REJECTED)
    runmanager.submitted.remove('shot-3')

    assert session.reconcile() == ['shot-0', 'shot-1', 'shot-2', 'shot-3']
    assert session.status()['awaiting'] == 0


def test_a_shot_that_has_only_just_run_is_not_treated_as_lost(session, runmanager):
    """A completed shot leaves runmanager's queue at once, while its cost is
    still on its way through lyse, so giving up on it would count the ordinary
    end of every healthy shot as a loss. So would giving up on one not yet
    handed to lyse.
    """
    session.refill()
    runmanager.finish('shot-0')
    runmanager.finish('shot-1', lyse=LYSE_WAITING)

    assert session.reconcile() == []
    assert session.status()['dropped'] == 0


def test_a_cost_arriving_after_a_shot_was_given_up_on_is_still_taken(
    session, runmanager
):
    """Giving up is a decision to stop waiting, not a refusal to listen, and a
    shot that did report must not stay counted as dropped or blocked.
    """
    session.refill()
    runmanager.lose('shot-0')
    runmanager.blocked.add('shot-1')
    session.reconcile()
    assert session.status()['dropped'] == 2

    session.record('shot-0', 2.0, None, False)
    session.record('shot-1', 3.0, None, False)
    status = session.status()
    assert (status['completed'], status['dropped'], status['blocked']) == (2, 0, 0)
    assert status['best_shot_id'] == 'shot-0'


def test_status_counts_every_shot_by_what_became_of_it(runmanager):
    session = running_session(make_config(num_buffered_runs=4), runmanager)
    session.refill()
    session.record('shot-0', 2.0, None, False)
    runmanager.lose('shot-1')
    runmanager.blocked.add('shot-2')
    session.reconcile()

    status = session.status()
    assert (status['submitted'], status['completed'], status['awaiting']) == (4, 1, 1)
    # A shot behind a row only an operator can clear is dropped as well.
    assert (status['dropped'], status['blocked']) == (2, 1)
    assert (status['best_cost'], status['best_shot_id']) == (2.0, 'shot-0')
    assert status['stopped'] is None


@pytest.mark.parametrize(
    'maximize, costs, best',
    [
        # The routine flips a maximized cost once on the way in, so the session
        # minimizes: these are measurements of 3.0 and 7.0.
        (True, (-3.0, -7.0), (7.0, 'shot-1')),
        (False, (3.0, 7.0), (3.0, 'shot-0')),
    ],
)
def test_the_best_cost_is_reported_in_the_labs_own_sign(
    runmanager, maximize, costs, best
):
    session = running_session(make_config(maximize=maximize), runmanager)
    session.refill()
    for shot_id, cost in zip(['shot-0', 'shot-1'], costs):
        session.record(shot_id, cost, None, False)

    status = session.status()
    assert (status['best_cost'], status['best_shot_id']) == best


def test_a_session_stops_at_the_run_budget_and_stays_stopped(runmanager):
    session = running_session(
        make_config(num_buffered_runs=2, max_num_runs=4), runmanager
    )
    session.refill()
    while session.awaiting:
        session.record(session.awaiting[0], 1.0, None, False)
        session.refill()

    assert session.stopped == 'reached max_num_runs (4)'
    assert len(runmanager.submitted) == 4
    assert session.refill() == []
    assert session.start() is False


def test_a_lost_shot_is_not_charged_against_the_run_budget(runmanager):
    """A shot that produced nothing has not spent one of the runs."""
    session = running_session(
        make_config(num_buffered_runs=1, max_num_runs=3), runmanager
    )
    for _ in range(4):
        submitted = session.refill()
        runmanager.lose(*submitted)
        session.reconcile()
    assert session.stopped is None

    for _ in range(3):
        for shot_id in session.refill():
            session.record(shot_id, 1.0, None, False)
    assert session.stopped == 'reached max_num_runs (3)'


@pytest.mark.parametrize(
    'costs, best_cost, stopped',
    [
        # An improvement resets the count.
        ([5.0, 6.0, 7.0, 4.0, 8.0, 9.0], 4.0, False),
        # Unusable costs count too, or a dead detector would run for ever.
        ([1.0, math.nan, math.inf, 2.0], 1.0, True),
        # With no best to count from, the whole run has been without one.
        ([math.nan] * 5, None, True),
    ],
)
def test_a_session_gives_up_after_runs_that_bring_nothing_better(
    runmanager, costs, best_cost, stopped
):
    session = running_session(
        make_config(num_buffered_runs=1, max_num_runs_without_better_params=3),
        runmanager,
    )
    for cost in costs:
        submitted = session.refill()
        if not submitted:
            break
        session.record(submitted[0], cost, None, not math.isfinite(cost))

    assert (session.stopped is not None) == stopped
    if stopped:
        assert 'no better parameters in 3 runs' in session.stopped
    assert (None if session.best is None else session.best.cost) == best_cost


def test_a_session_out_of_patience_keeps_that_reason_as_its_last_shots_land(
    runmanager,
):
    """The shots in flight when a session stops go on reporting, and the budget
    they reach is not why it stopped. The last of them is the best cost yet, so
    the patience limit no longer holds when the budget is reached.
    """
    session = running_session(
        make_config(
            num_buffered_runs=3, max_num_runs=3, max_num_runs_without_better_params=1
        ),
        runmanager,
    )
    first, second, third = session.refill()
    session.record(first, 1.0, None, False)
    assert session.stopped is None
    session.record(second, 2.0, None, False)
    patience = 'no better parameters in 1 runs (max_num_runs_without_better_params)'
    assert session.stopped == patience

    session.record(third, 0.5, None, False)
    assert session.status()['completed'] == 3
    assert session.stopped == patience


def test_a_changed_labscript_file_is_refused_before_anything_is_submitted(
    session, runmanager
):
    runmanager.labscript_changed = True
    with pytest.raises(RuntimeError):
        session.refill()
    assert runmanager.submitted == []


def test_the_configured_start_is_the_first_proposal_and_the_only_one(runmanager):
    """Every shot of the opening batch is lost, so not one cost has come back
    when the session refills, and the run still does not begin over from the
    point the file named.
    """
    session = running_session(
        make_config(num_buffered_runs=2, start=0.25), runmanager
    )
    submitted = session.refill()
    runmanager.lose(*submitted)
    session.reconcile()
    for _ in range(3):
        for shot_id in session.refill():
            submitted.append(shot_id)
            session.record(shot_id, 1.0, None, False)

    started = [session.proposals[shot_id][0] for shot_id in submitted]
    assert len(started) == 8
    assert started[0] == 0.25
    assert started.count(0.25) == 1


def test_a_generation_opening_on_the_configured_start_is_still_whole(runmanager):
    """The start takes the first place in the batch rather than a batch of its
    own, and the next generation waits for it as for the rest: bred without it,
    the population would be a slot short.
    """
    session = running_session(
        make_config('differential_evolution', 'population_size = 4', start=0.25),
        runmanager,
    )

    opening = session.refill()
    assert len(opening) == 4
    assert session.proposals[opening[0]][0] == 0.25
    assert session.refill() == []

    start, *founders = opening
    for shot_id in founders:
        session.record(shot_id, 1.0, None, False)
    assert session.refill() == []

    assert session.record(start, 1.0, None, False) == 'start'
    second = session.refill()
    assert len(second) == 4
    started = [session.proposals[shot_id][0] for shot_id in opening + second]
    assert started.count(0.25) == 1


@pytest.mark.parametrize(
    'config, starved',
    [
        # The queue is empty each time the session refills.
        (make_config(num_buffered_runs=1), 5),
        (make_config(num_buffered_runs=3), 0),
        # A generation empties the queue by design.
        (make_config('differential_evolution', 'population_size = 4'), 0),
    ],
    ids=['emptied', 'kept-topped-up', 'generational'],
)
def test_a_session_counts_the_refills_that_found_its_queue_empty(
    runmanager, config, starved
):
    """Every time nothing of ours is queued, BLACS ran a default shot instead:
    the apparatus staying busy, but a shot the optimizer did not get.
    """
    session = running_session(config, runmanager)
    session.refill()
    for _ in range(5):
        session.record(session.awaiting[0], 1.0, None, False)
        session.refill()
    assert session.status()['starved'] == starved


def test_a_generation_goes_out_whole_and_waits_to_be_answered_for(runmanager):
    session = running_session(
        make_config('differential_evolution', 'population_size = 4'), runmanager
    )

    assert len(session.refill()) == 4
    assert session.refill() == []
    for shot_id in session.awaiting[:3]:
        session.record(shot_id, 1.0, None, False)
    assert session.refill() == []

    session.record(session.awaiting[0], 1.0, None, False)
    assert len(session.refill()) == 4


@pytest.mark.parametrize(
    'explorer', ['random', 'directed_random', 'differential_evolution']
)
def test_a_gaussian_process_session_runs_end_to_end_with_each_explorer(
    runmanager, explorer
):
    """Warmup, then the model's points and explorer shots, every one of them
    costed, driven as the routine drives a session. Each batch is waited for
    before the next shot reports, so the run reaches the model's points however
    long a fit takes.
    """
    session = running_session(
        make_config(
            'gaussian_process',
            f'explorer = "{explorer}"\nwarmup_observations = 4\nbatch_size = 2',
            max_num_runs=16,
            seed=20260923,
        ),
        runmanager,
    )
    proposed_by, written = {}, {}
    while True:
        session.refill()
        settle(session.learner)
        for o in session.history:
            proposed_by.setdefault(o.shot_id, o.source)
        if not session.awaiting:
            break
        shot_id = session.awaiting[0]
        x = session.proposals[shot_id][0]
        written[shot_id] = session.record(shot_id, float((x - 0.3) ** 2), None, False)

    assert all(o.state == COMPLETE for o in session.history)
    assert session.stopped == 'reached max_num_runs (16)'
    # What is written onto a shot is what proposed it, whatever was proposed
    # since.
    assert set(written.values()) == {'warmup', 'main', 'explore'}
    assert written == proposed_by


def test_the_budget_may_cut_the_last_generation_short(runmanager):
    """``max_num_runs`` is a ceiling on the run, not on a batch. Nothing reads
    the population after the last generation, so the shots the budget has left
    are spent rather than withheld to keep the generation whole.
    """
    session = running_session(
        make_config(
            'differential_evolution', 'population_size = 5', max_num_runs=13
        ),
        runmanager,
    )

    sizes = []
    while not session.stopped:
        submitted = session.refill()
        if not submitted:
            break
        sizes.append(len(submitted))
        for position, shot_id in enumerate(submitted):
            session.record(shot_id, float(position), None, False)

    assert sizes == [5, 5, 3]


def test_a_refused_submission_is_raised_and_leaves_the_session_untouched(
    session, runmanager
):
    def refuse(proposals):
        raise RuntimeError('refused')

    runmanager.submit = refuse
    with pytest.raises(RuntimeError, match='refused'):
        session.refill()
    assert session.history == []


def test_a_sequence_runmanager_cannot_add_to_ends_the_session(session, runmanager):
    """runmanager forgets a run's sequence when it restarts, and refuses to add
    to it rather than split the run in two."""
    reason = 'runmanager has no record of the sequence'

    def refuse(proposals):
        raise SequenceRefused(reason)

    runmanager.submit = refuse
    assert session.refill() == []
    assert session.stopped == reason
    assert session.history == []
