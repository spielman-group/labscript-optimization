"""Session behaviour: matching costs to proposals, and knowing when to stop."""

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
    def __init__(self):
        self.submitted = []

    def check_ready(self):
        pass

    def submit(self, tag, params):
        self.submitted.append((tag, float(np.asarray(params).ravel()[0])))


@pytest.fixture
def session():
    return Session(make_config(), FakeRunmanager())


def test_the_queue_is_filled_to_the_buffer_depth(session):
    assert session.refill() == ['s:0', 's:1', 's:2']
    assert session.outstanding == 3


def test_a_full_queue_is_not_topped_up(session):
    session.refill()
    assert session.refill() == []


def test_a_cost_frees_a_slot_for_another_proposal(session):
    session.refill()
    session.record('s:1', 4.0, None, False)
    assert session.refill() == ['s:3']


def test_costs_arriving_out_of_order_land_in_proposal_order(session):
    session.refill()
    session.record('s:2', 3.0, None, False)
    session.record('s:0', 9.0, None, False)
    assert [o.tag for o in session.history] == ['s:0', 's:2']


def test_a_shot_this_session_did_not_propose_is_ignored(session):
    session.refill()
    assert session.record('somebody-elses-shot', 1.0, None, False) is False
    assert session.history == []


def test_a_repeated_cost_for_one_shot_is_ignored(session):
    """Several shots can share a tag when a routine averages repeats."""
    session.refill()
    assert session.record('s:0', 5.0, None, False) is True
    assert session.record('s:0', 99.0, None, False) is False
    assert session.best.cost == 5.0


def test_the_best_result_ignores_unusable_shots(session):
    session.refill()
    session.record('s:0', 5.0, None, False)
    session.record('s:1', 1.0, None, True)
    session.record('s:2', float('inf'), None, False)
    assert session.best.tag == 's:0'


def test_a_session_stops_at_the_run_budget():
    session = Session(make_config(buffered=2, max_num_runs=4), FakeRunmanager())
    for _ in range(10):
        for tag in session.refill():
            session.record(tag, 1.0, None, False)
    assert session.stopped == 'reached max_num_runs (4)'
    assert len(session.interface.submitted) == 4


def test_a_stopped_session_proposes_nothing_further():
    session = Session(make_config(buffered=1, max_num_runs=1), FakeRunmanager())
    session.record(session.refill()[0], 1.0, None, False)
    assert session.stopped
    assert session.refill() == []


def test_a_session_gives_up_when_nothing_improves():
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=3),
        FakeRunmanager(),
    )
    costs = [1.0, 2.0, 3.0, 4.0, 5.0]
    for cost in costs:
        tags = session.refill()
        if not tags:
            break
        session.record(tags[0], cost, None, False)
    assert 'no better parameters in 3 runs' in session.stopped


def test_improvement_resets_the_patience():
    session = Session(
        make_config(buffered=1, max_num_runs_without_better_params=3),
        FakeRunmanager(),
    )
    for cost in [5.0, 6.0, 7.0, 4.0, 8.0, 9.0]:
        tags = session.refill()
        assert tags, 'gave up despite an improvement'
        session.record(tags[0], cost, None, False)
    assert session.stopped is None


def test_status_reports_progress(session):
    session.refill()
    session.record('s:0', 2.0, None, False)
    status = session.status()
    assert status['session'] == 's'
    assert status['proposed'] == 3
    assert status['completed'] == 1
    assert status['outstanding'] == 2
    assert status['best_cost'] == 2.0
    assert status['best_tag'] == 's:0'
    assert status['stopped'] is None


def test_a_shot_whose_cost_never_arrives_holds_its_slot(session):
    """No timeouts, no retries: the slot simply stays taken."""
    session.refill()
    session.record('s:0', 1.0, None, False)
    session.record('s:2', 1.0, None, False)
    session.refill()
    assert session.outstanding == 3
    assert 's:1' not in {o.tag for o in session.history}
