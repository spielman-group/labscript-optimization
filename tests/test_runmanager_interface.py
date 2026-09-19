"""The seam against runmanager.remote.

These use a fake client returning the shapes the real one documents, so a
change to those shapes shows up here rather than in the lab.
"""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.runmanager_interface import RunmanagerInterface

CONFIG = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP]
session = "s"
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 10.0
[MLOOP_PARAMS.G.y]
min = 0.0
max = 10.0
[RUNMANAGER_GLOBALS.G.gy_doubled]
expr = "lambda v: 2 * v"
args = ["y"]
"""


class FakeClient:
    def __init__(self, policy='default_labscript', labscript='/lab/expt.py'):
        self.policy = policy
        self.labscript = labscript
        self.entries = []
        self.states = {}
        self.refuse = None

    def error_in_globals(self):
        return False

    def get_empty_queue_policy(self):
        return self.policy

    def get_labscript_file(self):
        return self.labscript

    def submit_shots(self, entries):
        if self.refuse:
            raise RuntimeError(self.refuse)
        self.entries.extend(entries)
        return [
            {
                'shot_id': f'id-{len(self.entries) - len(entries) + i}',
                'sequence_id': '20260918T120000_expt',
                'run_number': len(self.entries) - len(entries) + i,
                'path': f'/data/shot{i}.h5',
            }
            for i in range(len(entries))
        ]

    def shot_status(self, shot_ids):
        return {
            i: {'pending': self.states.get(i, True), 'state': 'running'}
            for i in shot_ids
        }


@pytest.fixture
def config():
    return config_module.loads(CONFIG)


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def interface(config, client):
    return RunmanagerInterface(config, client)


def test_a_session_starts_when_runmanager_can_sustain_it(interface):
    interface.check_ready()
    interface.check_unchanged()


def test_a_policy_that_would_let_the_session_die_silently_is_refused(config):
    interface = RunmanagerInterface(config, FakeClient(policy='nothing'))
    with pytest.raises(RuntimeError, match="sequence of its own"):
        interface.check_ready()


def test_a_labscript_file_changed_mid_session_is_refused(interface, client):
    interface.check_ready()
    client.labscript = '/lab/something_else.py'
    with pytest.raises(RuntimeError, match='labscript file changed'):
        interface.check_unchanged()


def test_submitting_sends_one_entry_of_globals_per_proposal(interface, client):
    ids = interface.submit([[1.0, 2.0], [3.0, 4.0]])
    assert ids == ['id-0', 'id-1']
    assert client.entries == [
        {'gx': 1.0, 'gy_doubled': 4.0},
        {'gx': 3.0, 'gy_doubled': 8.0},
    ]


def test_a_refusal_reaches_the_caller_unchanged(interface, client):
    client.refuse = 'Cannot submit {...} as one shot: Expanded by: gx.'
    with pytest.raises(RuntimeError, match='Expanded by: gx'):
        interface.submit([[1.0, 2.0]])


def test_pending_reports_only_the_shots_still_coming(interface, client):
    client.states = {'a': True, 'b': False, 'c': True}
    assert interface.pending(['a', 'b', 'c']) == {'a', 'c'}


def test_an_id_runmanager_does_not_answer_for_is_not_pending(interface, client):
    client.shot_status = lambda ids: {}
    assert interface.pending(['a', 'b']) == set()


def test_nothing_is_asked_about_an_empty_list(interface, client):
    client.shot_status = lambda ids: pytest.fail('asked about nothing')
    assert interface.pending([]) == set()
