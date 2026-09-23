"""Session behaviour: matching costs to shots, and never waiting on a lost one."""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.observations import COMPLETE, DROPPED, PENDING, usable
from labscript_optimization.session import Session

BASE = """
[ANALYSIS]
cost_key = ["r", "c"]
maximize = {maximize}
groups = ["G"]
[GENERAL]
learner = "random"
num_buffered_runs = {buffered}
{extra}
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


GENERATIONAL = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "differential_evolution"
[LEARNER.differential_evolution]
population_size = 4
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


STARTED = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "random"
num_buffered_runs = 2
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
start = 0.25
"""


GENERATIONAL_STARTED = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "differential_evolution"
[LEARNER.differential_evolution]
population_size = 4
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
start = 0.25
"""


CYCLED = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "gaussian_process"
num_buffered_runs = 5
max_num_runs = 24
seed = 20260923
[LEARNER.gaussian_process]
warmup_observations = 4
batch_size = 3
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


def gaussian_process_config(general='', table=''):
    """A one-parameter Gaussian process, with whatever else a test writes into
    ``[GENERAL]`` and its own table."""
    return config_module.loads(
        f"""
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
seed = 20260923
{general}
[LEARNER.gaussian_process]
{table}
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""
    )


def make_config(buffered=3, maximize=False, **extra):
    lines = '\n'.join(f'{k} = {v}' for k, v in extra.items())
    return config_module.loads(
        BASE.format(
            buffered=buffered, extra=lines, maximize='true' if maximize else 'false'
        )
    )


@pytest.fixture
def session(runmanager):
    return Session(make_config(), runmanager)


def test_the_queue_is_filled_to_the_buffer_depth(session):
    assert session.refill() == ['shot-0', 'shot-1', 'shot-2']
    assert len(session.awaiting) == 3


def test_a_full_queue_is_not_topped_up(session):
    session.refill()
    assert session.refill() == []


def test_a_cost_frees_a_slot(session):
    session.refill()
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


def test_a_shot_this_session_did_not_submit_is_ignored(session):
    session.refill()
    assert session.record('someone-elses-shot', 1.0, None, False) is None
    assert [o.shot_id for o in session.history] == ['shot-0', 'shot-1', 'shot-2']
    assert all(o.cost is None for o in session.history)


def test_the_history_tells_a_shot_still_coming_from_one_that_never_will(
    session, runmanager
):
    """Both are a position spent that has produced nothing, which is all a
    learner needs of them. The session needs more: ``dropped`` is the number a
    user reads to see whether shots are being lost, so the record keeps the
    two apart instead of leaving them to be told from a missing cost.
    """
    session.refill()
    runmanager.lose('shot-0')
    session.reconcile()

    assert {o.shot_id: o.state for o in session.history} == {
        'shot-0': DROPPED,
        'shot-1': PENDING,
        'shot-2': PENDING,
    }
    assert usable(session.history) == []


def test_a_proposal_still_waiting_is_a_position_spent_and_nothing_more(runmanager):
    """It is in the history, because a learner reading a role off a position
    has to see the positions that produced nothing. It is not an observation:
    nothing fits to it, it is not a completed run, and the patience limit does
    not count it -- counting it would stop a session for the shots it is
    waiting on.
    """
    session = Session(
        make_config(buffered=3, max_num_runs_without_better_params=2), runmanager
    )
    session.refill()
    session.record('shot-0', 1.0, None, False)

    waiting = [o for o in session.history if o.state != COMPLETE]
    assert [o.shot_id for o in waiting] == ['shot-1', 'shot-2']
    assert usable(waiting) == []
    assert session.status()['completed'] == 1
    assert session.runs_since_best() == 0
    assert session.stopped is None


def test_a_second_cost_for_one_shot_is_ignored(session):
    """A row can run twice: a retry, or BLACS re-running a file with data."""
    session.refill()
    assert session.record('shot-0', 5.0, None, False) is not None
    assert session.record('shot-0', 99.0, None, False) is None
    assert session.best.cost == 5.0


# --- the part that must never regress ---------------------------------------


def test_a_shot_that_will_never_arrive_stops_being_waited_on(session, runmanager):
    """A lost shot must free its slot, or the session stalls for ever.

    Aborted, cancelled and compile-failed shots never reach lyse, so nothing
    will ever report their cost. Counting submissions against costs received
    would leave their places held permanently.
    """
    session.refill()
    runmanager.lose('shot-0', 'shot-1', 'shot-2')

    assert sorted(session.reconcile()) == ['shot-0', 'shot-1', 'shot-2']
    assert session.awaiting == []
    assert session.refill() == ['shot-3', 'shot-4', 'shot-5']


def test_total_attrition_does_not_end_the_session(session, runmanager):
    """Losing every shot, repeatedly, must not stop the optimisation."""
    for _ in range(5):
        submitted = session.refill()
        runmanager.lose(*submitted)
        session.reconcile()
    assert session.stopped is None
    assert len(runmanager.submitted) == 15


def test_a_lost_shot_is_not_charged_against_the_run_budget(runmanager):
    """A shot that produced nothing has not spent one of the runs."""
    session = Session(make_config(buffered=1, max_num_runs=3), runmanager)
    for _ in range(4):
        submitted = session.refill()
        if submitted:
            runmanager.lose(*submitted)
            session.reconcile()
    assert session.stopped is None

    for _ in range(3):
        for shot_id in session.refill():
            session.record(shot_id, 1.0, None, False)
    assert session.stopped == 'reached max_num_runs (3)'


def test_a_cost_arriving_after_a_shot_was_dropped_is_still_taken(session, runmanager):
    """Being dropped is a decision to stop waiting, not a refusal to listen."""
    session.refill()
    runmanager.lose('shot-0')
    session.reconcile()
    assert session.record('shot-0', 2.0, None, False) is not None
    assert session.best.shot_id == 'shot-0'


def test_a_late_cost_stops_its_shot_counting_as_dropped(session, runmanager):
    """``dropped`` is what a user reads to see whether shots are being lost, so
    a shot that did report must not be left standing in it.
    """
    session.refill()
    runmanager.lose('shot-0')
    session.reconcile()
    assert session.status()['dropped'] == 1

    session.record('shot-0', 2.0, None, False)
    assert session.status()['dropped'] == 0


def test_reconciling_asks_only_about_shots_still_awaited(session, runmanager):
    session.refill()
    session.record('shot-0', 1.0, None, False)
    asked = []
    still_coming = {'pending': True, 'state': 'running'}
    runmanager.shot_status = lambda ids: (
        asked.extend(ids) or {i: dict(still_coming) for i in ids}
    )
    session.reconcile()
    assert sorted(asked) == ['shot-1', 'shot-2']


def test_nothing_is_asked_when_nothing_is_awaited(session, runmanager):
    runmanager.shot_status = lambda ids: pytest.fail('asked with nothing awaited')
    assert session.reconcile() == []


def test_a_shot_that_has_only_just_run_is_not_treated_as_lost(session, runmanager):
    """A completed shot leaves runmanager's queue at once, while its cost is
    still on its way through lyse. runmanager reports it exactly as it reports
    an id it has never heard of, so giving up on that answer alone would make
    the ordinary end of every healthy shot count as a loss -- and the number of
    dropped shots is what a user reads to see whether shots are being lost.
    """
    session.refill()
    runmanager.finish('shot-0')

    assert session.reconcile() == []
    assert session.status()['dropped'] == 0

    assert session.record('shot-0', 1.0, None, False) is not None
    assert session.reconcile() == []
    assert session.status()['dropped'] == 0


def test_a_shot_still_unknown_at_the_next_reconcile_is_dropped(session, runmanager):
    """Staying unknown with no cost is how a shot that has really gone is told
    from one that has just finished. An operator's deletion and a runmanager
    restart both leave an id nothing will ever answer for, and its place must
    not be held for the rest of the session.
    """
    session.refill()
    runmanager.finish('shot-0')

    assert session.reconcile() == []
    assert session.reconcile() == ['shot-0']
    assert session.awaiting == ['shot-1', 'shot-2']
    assert session.refill() == ['shot-3']


def test_a_shot_with_a_reason_is_dropped_at_the_first_reconcile(session, runmanager):
    """An answer that names a state names a reason nothing further will happen,
    so there is nothing to wait a second round for."""
    session.refill()
    runmanager.lose('shot-0')
    assert session.reconcile() == ['shot-0']


def test_what_the_session_awaits_is_the_historys_pending_records(
    session, runmanager
):
    """A learner reads which proposals are in flight off the history, and the
    session reads the same thing to decide how many to submit, what to ask
    runmanager about and what to report. An account kept beside the history
    has to be moved in step with it by every way a shot stops being awaited,
    and a cost for a shot already given up on, or a second cost for one
    already taken, is where a count kept by hand moves twice.
    """

    def pending():
        return [o.shot_id for o in session.history if o.state == PENDING]

    session.refill()
    runmanager.lose('shot-0')
    session.reconcile()
    assert session.awaiting == pending() == ['shot-1', 'shot-2']

    # A late cost for the shot just dropped, and a repeated one.
    session.record('shot-0', 1.0, None, False)
    session.record('shot-1', 2.0, None, False)
    session.record('shot-1', 3.0, None, False)
    assert session.awaiting == pending() == ['shot-2']
    assert session.status()['awaiting'] == 1

    assert session.refill() == ['shot-3', 'shot-4']
    assert session.awaiting == pending() == ['shot-2', 'shot-3', 'shot-4']


# --- stopping ---------------------------------------------------------------


def test_a_session_stops_at_the_run_budget(runmanager):
    session = Session(make_config(buffered=2, max_num_runs=4), runmanager)
    for _ in range(10):
        for shot_id in session.refill():
            session.record(shot_id, 1.0, None, False)
    assert session.stopped == 'reached max_num_runs (4)'
    assert len(runmanager.submitted) == 4


def test_a_stopped_session_proposes_nothing_further(runmanager):
    session = Session(make_config(buffered=1, max_num_runs=1), runmanager)
    session.record(session.refill()[0], 1.0, None, False)
    assert session.stopped
    assert session.refill() == []


def test_a_session_out_of_patience_keeps_that_reason_as_its_last_shots_land(
    runmanager,
):
    """The shots already in flight when a session stops go on reporting, and
    the budget they reach is not why it stopped. The last of them here is the
    best cost yet, so the patience limit no longer holds when the budget is
    reached: the reason kept is the one that stopped the run, not whichever
    limit happens to hold at the last report.
    """
    session = Session(
        make_config(
            buffered=3, max_num_runs=3, max_num_runs_without_better_params=1
        ),
        runmanager,
    )
    first, second, third = session.refill()
    session.record(first, 1.0, None, False)
    session.record(second, 2.0, None, False)
    patience = 'no better parameters in 1 runs (max_num_runs_without_better_params)'
    assert session.stopped == patience

    session.record(third, 0.5, None, False)
    assert len(session.results) == 3
    assert session.runs_since_best() == 0
    assert session.stopped == patience


def test_a_session_gives_up_when_nothing_improves(runmanager):
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=3), runmanager
    )
    for cost in [1.0, 2.0, 3.0, 4.0, 5.0]:
        submitted = session.refill()
        if not submitted:
            break
        session.record(submitted[0], cost, None, False)
    assert 'no better parameters in 3 runs' in session.stopped


def test_improvement_resets_the_patience(runmanager):
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=3), runmanager
    )
    for cost in [5.0, 6.0, 7.0, 4.0, 8.0, 9.0]:
        submitted = session.refill()
        assert submitted, 'gave up despite an improvement'
        session.record(submitted[0], cost, None, False)
    assert session.stopped is None


def test_shots_with_no_usable_cost_still_exhaust_the_patience(runmanager):
    """A detector that has died returns costs, none of them usable.

    Those shots count against max_num_runs, so they must count here too. A
    patience limit that ignored them would be the one setting that runs for
    ever on a broken apparatus -- which is the failure it exists to catch.
    """
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=3), runmanager
    )
    for cost, bad in [
        (1.0, False),
        (float('nan'), False),
        (2.0, True),
        (float('inf'), False),
        (0.5, False),
    ]:
        submitted = session.refill()
        if not submitted:
            break
        session.record(submitted[0], cost, None, bad)
    assert 'no better parameters in 3 runs' in session.stopped
    assert session.best.cost == 1.0


def test_a_session_that_never_gets_a_usable_cost_gives_up(runmanager):
    """With no best to count from, the whole history has been without one."""
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=2), runmanager
    )
    for _ in range(5):
        submitted = session.refill()
        if not submitted:
            break
        session.record(submitted[0], float('nan'), None, False)
    assert 'no better parameters in 2 runs' in session.stopped
    assert session.best is None


def test_a_changed_labscript_file_stops_the_session(session, runmanager):
    runmanager.labscript_changed = True
    with pytest.raises(RuntimeError, match='labscript file changed'):
        session.refill()


def test_status_reports_progress(session, runmanager):
    session.refill()
    session.record('shot-0', 2.0, None, False)
    runmanager.lose('shot-1')
    session.reconcile()

    status = session.status()
    assert status['submitted'] == 3
    assert status['completed'] == 1
    assert status['awaiting'] == 1
    assert status['dropped'] == 1
    assert status['best_cost'] == 2.0
    assert status['best_shot_id'] == 'shot-0'
    assert status['stopped'] is None


def test_a_maximised_best_cost_is_reported_in_the_labs_own_sign(runmanager):
    """A lab maximising a figure of merit reads back what it measured.

    Two measurements rather than one, because one would prove only that a
    number was negated somewhere. The larger of the two says the session
    minimised its way to the right shot, and the sign it comes back in says
    the flip the routine made on the way in was undone on the way out.
    """
    session = Session(make_config(maximize=True), runmanager)
    session.refill()
    # The costs the routine hands over, flipped once so the session
    # minimises: measurements of 3.0 and 7.0.
    session.record('shot-0', -3.0, None, False)
    session.record('shot-1', -7.0, None, False)

    status = session.status()
    assert (status['best_cost'], status['best_shot_id']) == (7.0, 'shot-1')


def test_a_minimised_best_cost_is_reported_as_it_was_measured(runmanager):
    """The mirror, which a session negating unconditionally would fail."""
    session = Session(make_config(), runmanager)
    session.refill()
    session.record('shot-0', 3.0, None, False)
    session.record('shot-1', 7.0, None, False)

    status = session.status()
    assert (status['best_cost'], status['best_shot_id']) == (3.0, 'shot-0')


def test_the_configured_start_is_the_first_proposal_and_the_only_one(runmanager):
    """Where a run begins is written on the parameters, and the session says it.

    Once, and by construction: one place proposes it, so nothing has to
    decline to propose it again. Every shot of the opening batch is lost
    here, so not one cost has come back when the session refills -- and the
    run still does not begin over from the point the file named.
    """
    session = Session(config_module.loads(STARTED), runmanager)
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
    """The start takes the first place in the batch, not a batch of its own.

    A generational learner proposes a whole generation and only when none of
    the last is outstanding. A start sent out on its own would be a
    second route past that barrier: one shot, then a generation with the
    queue drained between them, and a population founded a slot short of the
    generation it is bred from.
    """
    session = Session(config_module.loads(GENERATIONAL_STARTED), runmanager)

    opening = session.refill()
    assert len(opening) == 4
    assert session.proposals[opening[0]][0] == 0.25
    assert session.refill() == []

    for shot_id in opening:
        session.record(shot_id, 1.0, None, False)
    second = session.refill()
    assert len(second) == 4

    started = [session.proposals[shot_id][0] for shot_id in opening + second]
    assert started.count(0.25) == 1


def test_a_configured_start_still_out_holds_the_next_generation_back(runmanager):
    """Once submitted, the start is a founder like the rest of its generation,
    and the next generation waits for it as it waits for them. Bred without
    it, the population would be a slot short, and the start's cost would land
    on a generation that had already gone.
    """
    session = Session(config_module.loads(GENERATIONAL_STARTED), runmanager)

    start, *founders = session.refill()
    for shot_id in founders:
        session.record(shot_id, 1.0, None, False)
    assert session.refill() == []

    session.record(start, 1.0, None, False)
    assert len(session.refill()) == 4


def test_the_configured_start_carries_a_source_of_its_own(runmanager):
    """The session proposes the start itself, whichever learner is running, so
    it is not any learner's phase: the shot it goes out in says ``start``, and
    the learner's shots beside it say what the learner said of them.
    """
    session = Session(config_module.loads(STARTED), runmanager)
    session.refill()
    assert [o.source for o in session.history] == ['start', 'main']
    assert session.record('shot-0', 1.0, None, False) == 'start'


# --- what proposed each shot ------------------------------------------------


def test_each_shot_carries_the_phase_of_the_learner_that_proposed_it(runmanager):
    """The routine's order of events, one shot at a time: the oldest shot
    outstanding reports its cost, the reply carries what is written onto it,
    and the refill comes after. Warmup hands over to the Gaussian process part
    way through the run, each batch goes out with explorer shots behind it,
    and the budget ends the run with shots in flight.

    What is written onto a shot is the phase the learner gave it when it
    proposed it. The phase of the latest proposal is another shot's whenever
    more than one is in flight: it names the explorer for the batch's own
    shots queued ahead of it, the Gaussian process for warmup shots still
    queued at the handover, and once the budget is spent it stays at whatever
    was proposed last. The history holds the same answer, fixed when each shot
    was proposed and unchanged by everything proposed since.
    """
    session = Session(config_module.loads(CYCLED), runmanager)
    proposed_by, latest, written = {}, {}, {}
    # What the learner says of each proposal, heard as it says it: its answer
    # to the latest call, and the source of the newest proposal it has made.
    answer, newest = [], [None]
    propose = session.learner.propose

    def listen(history, hint):
        answer[:] = propose(history, hint)
        if answer:
            newest[0] = answer[-1][1]
        return list(answer)

    session.learner.propose = listen

    def refill():
        # The session submits the answer in order, cut to the budget's room.
        for shot_id, (_, source) in zip(session.refill(), answer):
            proposed_by[shot_id] = source

    refill()
    while session.awaiting:
        shot_id = session.awaiting[0]
        cost = float((session.proposals[shot_id][0] - 0.3) ** 2)
        written[shot_id] = session.record(shot_id, cost, None, False)
        latest[shot_id] = newest[0]
        refill()

    assert len(written) == 24
    # All three of the Gaussian process's sources reach the phase column.
    assert set(written.values()) == {'warmup', 'main', 'explore'}
    # The run is one where the latest proposal's phase is the wrong answer
    # for some of its shots; otherwise nothing here tells the two apart.
    assert latest != proposed_by
    assert written == proposed_by
    assert {o.shot_id: o.source for o in session.history} == proposed_by


def test_an_empty_queue_at_refill_is_counted(runmanager):
    """Every time nothing of ours is queued, BLACS ran a default shot instead.

    That is the apparatus staying busy rather than a fault, but it is a shot
    the optimiser did not get, so it is worth telling the user about.
    """
    session = Session(make_config(buffered=1), runmanager)
    for _ in range(3):
        for shot_id in session.refill():
            session.record(shot_id, 1.0, None, False)
    assert session.status()['starved'] == 2


def test_a_generation_goes_out_whole_and_waits_to_be_answered_for(runmanager):
    """The learner proposes nothing while any of its proposals is outstanding,
    which is what stops a trial being bred against a half-built population.
    """
    session = Session(config_module.loads(GENERATIONAL), runmanager)

    assert len(session.refill()) == 4
    assert session.refill() == []
    for shot_id in list(session.awaiting)[:3]:
        session.record(shot_id, 1.0, None, False)
    assert session.refill() == []

    session.record(session.awaiting[0], 1.0, None, False)
    assert len(session.refill()) == 4


def test_the_session_submits_what_its_learner_proposes_and_holds_no_barrier(
    runmanager,
):
    """Pacing is the learner's. A learner declaring a generation is still
    handed the hint on every refill, and whatever it answers goes out, with
    shots of its own in flight or not.
    """

    class OneAtATime:
        generation = 4

        def propose(self, history, hint):
            return [(np.array([0.5]), 'main')]

    session = Session(make_config(), runmanager, OneAtATime())
    assert [len(session.refill()) for _ in range(3)] == [1, 1, 1]
    assert len(session.awaiting) == 3


def test_the_budget_keeps_the_first_proposals_a_learner_offers(runmanager):
    """A learner orders what it offers, and the budget cuts from the end.

    Differential evolution computes each trial for the slot its position will
    give it, so a generation cut short by the budget must keep its first
    trials: kept from the other end, every surviving trial would land in a
    slot it was not bred for and compete against the wrong incumbent.
    """

    class Labelled:
        generation = None

        def propose(self, history, hint):
            return [(np.array([x]), 'main') for x in (0.1, 0.2, 0.3, 0.4)]

    session = Session(make_config(max_num_runs=2), runmanager, Labelled())
    session.refill()
    assert [p[0] for p in session.proposals.values()] == [0.1, 0.2]


def test_the_budget_cuts_explorer_shots_before_any_of_the_batch(runmanager):
    """The Gaussian process offers its batch first and the explorer shots
    behind it, and the budget cuts from the end, so a budget with room for
    less than both spends it on the batch.

    Offered the other way round, the budget would cut into the batch, and the
    points the Gaussian process chose each conditioned on the ones before it
    would go out without the ones they were chosen beside.
    """
    session = Session(
        gaussian_process_config('max_num_runs = 9', 'warmup_observations = 3'),
        runmanager,
    )
    # Warmup at the default depth of two, until three usable observations are
    # in hand with one warmup shot still queued.
    for shot_id in session.refill():
        session.record(shot_id, 1.0, None, False)
    first, _ = session.refill()
    session.record(first, 2.0, None, False)
    # Three completed and one awaited leave room for five of the six offered:
    # a batch of four and two explorer shots behind it.
    submitted = session.refill()
    assert [session.sources[shot_id] for shot_id in submitted] == (
        ['main'] * 4 + ['explore']
    )


def test_a_gaussian_process_with_no_buffer_counts_its_starvation(runmanager):
    """A pure Gaussian process -- no buffer and no explorer shots -- leaves the
    queue empty while it waits for a shot, and while each batch is fitted, and
    every refill that finds it so is counted. It declares no generation, so
    nothing exempts it: the queue running dry is a cost of the settings, not
    of the method, and ``starved`` is how a lab sees it.
    """
    session = Session(
        gaussian_process_config('num_buffered_runs = 0', 'explore_runs = 0'),
        runmanager,
    )
    assert session.learner.generation is None
    sizes = []
    for _ in range(7):
        submitted = session.refill()
        sizes.append(len(submitted))
        for shot_id in submitted:
            session.record(shot_id, float(len(runmanager.submitted)), None, False)
    # Warmup one shot at a time until its five, then batches of four with
    # nothing behind them.
    assert sizes == [1, 1, 1, 1, 1, 4, 4]
    # Every refill but the first found nothing of ours queued.
    assert session.status()['starved'] == 6


def test_a_generational_run_counts_no_starvation(runmanager):
    """A generational learner empties the queue once per generation, by
    design, and a counter that fires by design is noise in the one number the
    documentation tells a lab to watch.
    """
    session = Session(config_module.loads(GENERATIONAL), runmanager)
    for _ in range(5):
        for shot_id in session.refill():
            session.record(shot_id, 1.0, None, False)

    assert len(runmanager.submitted) == 20
    assert session.status()['starved'] == 0


def test_a_queue_kept_topped_up_does_not_starve(runmanager):
    session = Session(make_config(buffered=3), runmanager)
    session.refill()
    for _ in range(5):
        session.record(session.awaiting[0], 1.0, None, False)
        session.refill()
    assert session.status()['starved'] == 0


def test_a_refused_submission_leaves_the_session_untouched(session, runmanager):
    """submit_shots checks every entry before submitting any, so a refusal
    means nothing was queued. Nothing here may be left thinking otherwise."""
    def refuse(proposals):
        raise RuntimeError(
            'Cannot submit {...} as one shot: the globals as they stand '
            'produce 3. Expanded by: height, width.'
        )

    runmanager.submit = refuse
    with pytest.raises(RuntimeError, match='as one shot'):
        session.refill()
    assert session.proposals == {}
    assert session.awaiting == []
    assert session.history == []


def test_a_submission_that_returns_too_few_ids_fails_loudly(session, runmanager):
    """Ids that cannot be paired with the proposals must stop the session.

    Pairing them anyway would leave a proposal out of the record while its
    shot is queued and running: the cost comes back for an id the session
    never recorded, record() ignores it, and the session waits for a buffer
    depth it can no longer reach. Guessing the pairing instead would charge a
    cost to parameters that were never the ones requested.
    """
    runmanager.submit = lambda proposals: ['shot-0']

    with pytest.raises(RuntimeError, match='3 proposals but got 1'):
        session.refill()
    assert session.proposals == {}
    assert session.awaiting == []


def test_a_shot_behind_a_row_nobody_can_move_is_counted_apart(session, runmanager):
    """Every other way a shot stops coming is the apparatus getting on with
    things. This one is somebody needing to go and look at the queue, and
    counting the two together makes a jammed queue read like a finished run."""
    session.refill()
    runmanager.blocked.add('shot-0')
    runmanager.cancelled.add('shot-1')
    session.reconcile()

    status = session.status()
    assert status['blocked'] == 1
    assert status['dropped'] == 2


def test_a_cost_arriving_for_a_blocked_shot_clears_it(session, runmanager):
    session.refill()
    runmanager.blocked.add('shot-0')
    session.reconcile()
    session.record('shot-0', 1.0, None, False)
    assert session.status()['blocked'] == 0


def de_config(population_size, max_num_runs):
    """A differential-evolution configuration with a chosen budget."""
    return config_module.loads(
        f"""
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[GENERAL]
learner = "differential_evolution"
max_num_runs = {max_num_runs}
[LEARNER.differential_evolution]
population_size = {population_size}
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""
    )


def test_the_budget_may_cut_the_last_generation_short(runmanager):
    """``max_num_runs`` is a ceiling on the run, not on a batch. Nothing reads
    the population after the last generation, so the shots the budget has left
    are spent rather than withheld to keep the generation whole.
    """
    config = de_config(population_size=5, max_num_runs=13)
    session = Session(config, runmanager)

    sizes = []
    while not session.stopped:
        submitted = session.refill()
        if not submitted:
            break
        sizes.append(len(submitted))
        for position, shot_id in enumerate(submitted):
            session.record(shot_id, float(position), None, False)

    assert sizes == [5, 5, 3]


def test_a_short_last_generation_evolves_the_slots_it_reaches(runmanager):
    """Its trials compete for their own slots like any other generation's.

    The founding generation carries one shot whose cost is not usable, so a
    proposal's position and the number of usable costs before it part company:
    the slot each of the last three trials competes for is the one its
    position names, not the one a running count would name.
    """
    config = de_config(population_size=5, max_num_runs=13)
    session = Session(config, runmanager)
    learner = session.learner

    generations = (
        [9.0, float('nan'), 9.0, 9.0, 9.0],
        [9.0] * 5,
        [0.0] * 3,
    )
    for costs in generations:
        submitted = session.refill()
        for shot_id, cost in zip(submitted, costs):
            session.record(shot_id, cost, None, not np.isfinite(cost))

    _, costs = learner.replay(session.history)
    # The third generation opened at position 10, which is slot 0, so it
    # reached slots 0, 1 and 2 and improved each. Slots 3 and 4 were never
    # offered a trial and keep the members the second generation left them.
    assert list(costs) == [0.0, 0.0, 0.0, 9.0, 9.0]


def test_a_budget_under_two_generations_is_still_refused():
    with pytest.raises(ValueError, match='less than two whole generations'):
        de_config(population_size=5, max_num_runs=9)


def test_the_budget_refusal_does_not_claim_a_generation_is_never_short():
    with pytest.raises(ValueError) as raised:
        de_config(population_size=5, max_num_runs=9)
    message = str(raised.value)
    assert 'never evolves anything at all' in message
    assert 'need not be a whole number of generations' in message
