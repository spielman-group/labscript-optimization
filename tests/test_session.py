"""Session behaviour: matching costs to shots, and never waiting on a lost one."""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.session import Session

BASE = """
[ANALYSIS]
cost_key = ["r", "c"]
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


def make_config(buffered=3, **extra):
    lines = '\n'.join(f'{k} = {v}' for k, v in extra.items())
    return config_module.loads(BASE.format(buffered=buffered, extra=lines))


class FakeRunmanager:
    """Stands in for runmanager, and decides what is still coming."""

    def __init__(self):
        self.submitted: list[str] = []
        self.gone: set[str] = set()
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

    def pending(self, shot_ids):
        return {i for i in shot_ids if i not in self.gone}

    def lose(self, *shot_ids):
        """The apparatus or an operator disposes of these shots."""
        self.gone.update(shot_ids)


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
    assert [o.tag for o in session.history] == ['shot-0', 'shot-2']


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
    assert session.best.tag == 'shot-0'


def test_reconciling_asks_only_about_shots_still_awaited(session, runmanager):
    session.refill()
    session.record('shot-0', 1.0, None, False)
    asked = []
    runmanager.pending = lambda ids: (asked.extend(ids) or set(ids))
    session.reconcile()
    assert sorted(asked) == ['shot-1', 'shot-2']


def test_nothing_is_asked_when_nothing_is_awaited(session, runmanager):
    runmanager.pending = lambda ids: pytest.fail('asked with nothing awaited')
    assert session.reconcile() == []


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
    assert status['best_shot'] == 'shot-0'
    assert status['stopped'] is None


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
