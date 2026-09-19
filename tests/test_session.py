"""Session behaviour: matching costs to shots, and never waiting on a lost one."""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.session import Session

BASE = """
[ANALYSIS]
cost_key = ["r", "c"]
maximize = {maximize}
groups = ["G"]
[MLOOP]
session = "s"
learner = "random"
num_buffered_runs = {buffered}
{extra}
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


def make_config(buffered=3, maximize=False, **extra):
    lines = '\n'.join(f'{k} = {v}' for k, v in extra.items())
    return config_module.loads(
        BASE.format(
            buffered=buffered, extra=lines, maximize='true' if maximize else 'false'
        )
    )


class FakeRunmanager:
    """Stands in for runmanager, and decides what is still coming.

    It answers as runmanager does: one ``{'pending', 'state'}`` per id asked
    about. A cancelled shot keeps its row and says so, while a shot that has
    run leaves the queue and becomes indistinguishable from an id runmanager
    never had -- both are ``unknown``.
    """

    def __init__(self):
        self.submitted: list[str] = []
        self.cancelled: set[str] = set()
        self.blocked: set[str] = set()
        self.finished: set[str] = set()
        self.labscript_changed = False

    def check_ready(self):
        pass

    def check_unchanged(self):
        if self.labscript_changed:
            raise RuntimeError('the labscript file changed')

    def submit(self, proposals):
        ids = [f'shot-{len(self.submitted) + i}' for i in range(len(proposals))]
        self.submitted.extend(ids)
        return ids

    def shot_status(self, shot_ids):
        answers = {}
        for shot_id in shot_ids:
            if shot_id in self.blocked:
                answers[shot_id] = {'pending': False, 'state': 'blocked'}
            elif shot_id in self.cancelled:
                answers[shot_id] = {'pending': False, 'state': 'cancelled'}
            elif shot_id in self.finished or shot_id not in self.submitted:
                answers[shot_id] = {'pending': False, 'state': 'unknown'}
            else:
                answers[shot_id] = {'pending': True, 'state': 'running'}
        return answers

    def lose(self, *shot_ids):
        """An operator disposes of these shots, so they will never run."""
        self.cancelled.update(shot_ids)

    def finish(self, *shot_ids):
        """These shots run and leave the queue, as every healthy shot does."""
        self.finished.update(shot_ids)


@pytest.fixture
def runmanager():
    return FakeRunmanager()


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
    session.refill()
    session.record('shot-2', 3.0, None, False)
    session.record('shot-0', 9.0, None, False)
    assert [o.shot_id for o in session.history] == ['shot-0', 'shot-2']


def test_a_shot_this_session_did_not_submit_is_ignored(session):
    session.refill()
    assert session.record('someone-elses-shot', 1.0, None, False) is False
    assert session.history == []


def test_a_second_cost_for_one_shot_is_ignored(session):
    """A row can run twice: a retry, or BLACS re-running a file with data."""
    session.refill()
    assert session.record('shot-0', 5.0, None, False) is True
    assert session.record('shot-0', 99.0, None, False) is False
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
    assert session.record('shot-0', 2.0, None, False) is True
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

    assert session.record('shot-0', 1.0, None, False) is True
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


@pytest.mark.parametrize(
    'refused, setting',
    [({'buffered': 0}, 'num_buffered_runs'), ({'max_num_runs': 0}, 'max_num_runs')],
    ids=['num_buffered_runs', 'max_num_runs'],
)
def test_a_setting_that_makes_a_silent_session_is_refused(refused, setting):
    """And this is the session it is refused on behalf of.

    Nothing is ever submitted, so nothing is ever recorded; check_stop is
    reached only from record, so there is no stop reason either, and starved
    counts nothing because the queue was never the optimiser's to lose. The
    run looks like a session that is thinking about it.
    """
    with pytest.raises(ValueError, match=f'{setting} must be at least 1'):
        make_config(**refused)

    config = make_config()
    setattr(config, setting, 0)
    session = Session(config, FakeRunmanager())
    assert [session.refill() for _ in range(3)] == [[], [], []]
    assert session.starved == 0
    assert session.stopped is None
