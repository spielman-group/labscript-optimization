"""The seam against runmanager.client.

These use a fake client returning the shapes the real one documents, so a
change to those shapes shows up here rather than in the lab.
"""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.runmanager_interface import (
    RunmanagerInterface,
    RunmanagerStatusMonitor,
    runmanager_link_display,
)

CONFIG = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 10.0
[PARAMETERS.G.y]
min = 0.0
max = 10.0
[RUNMANAGER_GLOBALS.G.gy_doubled]
expr = "lambda v: 2 * v"
args = ["y"]
"""


class FakeClient:
    """The shape of ``runmanager.client.RunmanagerClient``, answers and all."""

    def __init__(self, labscript='/lab/expt.py'):
        self.labscript = labscript
        self.broken_globals = False
        self.entries = []
        self.sequences = []
        self.states = {}
        self.refuse = None
        self.silent = False
        self.scan_enabled = {}
        self.jit_enabled = {}

    def say_hello(self, timeout=None):
        if self.silent:
            raise TimeoutError('no response from server')
        return 'hello'

    def error_in_globals(self):
        return self.broken_globals

    def get_labscript_file(self):
        return self.labscript

    def get_scan_enabled(self):
        return self.scan_enabled

    def get_jit_enabled(self):
        return self.jit_enabled

    def submit_shots(self, entries, sequence=None, sequence_index=None):
        """Starts a sequence for each submission that names none."""
        if self.refuse:
            raise RuntimeError(self.refuse)
        self.sequences.append((sequence, sequence_index))
        self.entries.extend(entries)
        if sequence is None:
            sequence = f'20260918T12000{len(self.sequences)}_expt'
            sequence_index = len(self.sequences)
        return [
            {
                'shot_id': f'id-{len(self.entries) - len(entries) + i}',
                'sequence_id': sequence,
                'sequence_index': sequence_index,
                'run_number': len(self.entries) - len(entries) + i,
                'path': f'/data/shot{i}.h5',
            }
            for i in range(len(entries))
        ]

    def shot_status(self, shot_ids):
        """One entry per id asked about, as runmanager documents it."""
        return {
            i: self.states.get(i, {'pending': True, 'state': 'running'})
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


def test_the_monitor_reports_whether_runmanager_answers_and_why_not(client):
    seen = []
    monitor = RunmanagerStatusMonitor(seen.append, client)
    monitor.poll()
    client.silent = True
    monitor.poll()
    assert seen == [
        {'reachable': True},
        {'reachable': False, 'reason': 'no response from server'},
    ]


def test_the_light_is_checking_until_asked_then_online_or_offline():
    assert runmanager_link_display(None)[0] == 'checking'
    assert runmanager_link_display({'reachable': True})[0] == 'online'
    state, tooltip = runmanager_link_display({'reachable': False, 'reason': 'why'})
    assert state == 'offline' and 'why' in tooltip


def test_a_runmanager_whose_globals_do_not_evaluate_is_refused(interface, client):
    """Every shot the session went on to submit would fail to compile."""
    client.broken_globals = True
    with pytest.raises(RuntimeError, match='error in its globals'):
        interface.check_ready()


def test_a_labscript_file_changed_mid_session_is_refused(interface, client):
    interface.check_ready()
    client.labscript = '/lab/something_else.py'
    # A Start that resumes the run does not move the file it is compared with.
    interface.check_ready()
    with pytest.raises(RuntimeError, match='labscript file changed'):
        interface.check_unchanged()


@pytest.mark.parametrize('box', ['scan_enabled', 'jit_enabled'])
def test_a_global_with_scan_or_jit_ticked_is_refused_before_submitting(
    interface, client, box
):
    setattr(client, box, {'gx': False, 'gy_doubled': True})
    with pytest.raises(RuntimeError):
        interface.submit(np.array([[1.0, 2.0]]))
    assert client.entries == []


def test_submitting_sends_one_entry_of_globals_per_proposal(interface, client):
    ids = interface.submit([[1.0, 2.0], [3.0, 4.0]])
    assert ids == ['id-0', 'id-1']
    assert client.entries == [
        {'gx': 1.0, 'gy_doubled': 4.0},
        {'gx': 3.0, 'gy_doubled': 8.0},
    ]


def test_every_submission_after_the_first_joins_its_sequence(interface, client):
    """A run is one runmanager sequence, whatever an operator engages in
    between."""
    for _ in range(3):
        interface.submit([[1.0, 2.0]])
    joined = ('20260918T120001_expt', 1)
    assert client.sequences == [(None, None), joined, joined]


def holds_only_python_values(value):
    """Whether ``value`` is built of Python's own types, however it nests.

    Compared by ``type`` rather than ``isinstance``, because numpy's
    ``float64`` subclasses ``float`` and would pass for one.
    """
    if type(value) in (tuple, list):
        return all(holds_only_python_values(item) for item in value)
    return type(value) in (bool, int, float, str)


@pytest.mark.parametrize(
    'name, submitted',
    [
        ('gx', [1.0, 3.0]),
        ('shim', [(1.0, 2.0), (3.0, 4.0)]),
        ('decay', [float(np.exp(-1.0)), float(np.exp(-3.0))]),
    ],
    ids=['a global_name mapping', 'an expr building a tuple', 'an expr using exp'],
)
def test_a_global_is_submitted_as_a_python_value(client, name, submitted):
    """runmanager writes a submitted value into its global's expression as the
    value's repr, and a numpy scalar's repr names numpy -- ``np.float64(1.0)``
    in the operator's runmanager and in the shot file, evaluating only because
    runmanager's namespace happens to carry numpy. Proposals arrive as a numpy
    array, so every value is checked, down to the members of a tuple.
    """
    config = config_module.loads(
        """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 10.0
[PARAMETERS.G.y]
min = 0.0
max = 10.0
[RUNMANAGER_GLOBALS.G.shim]
expr = "lambda a, b: (a, b)"
args = ["x", "y"]
[RUNMANAGER_GLOBALS.G.decay]
expr = "lambda a: exp(-a)"
args = ["x"]
"""
    )
    RunmanagerInterface(config, client).submit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    values = [entry[name] for entry in client.entries]
    assert values == submitted
    for value in values:
        assert eval(repr(value), {'__builtins__': {}}) == value
        # And by type as well as by repr, which a numpy scalar printed the way
        # numpy printed one before version 2 would pass.
        assert holds_only_python_values(value), repr(value)


def test_a_refusal_reaches_the_caller_unchanged(interface, client):
    client.refuse = 'Cannot submit {...} as one shot: Expanded by: gx.'
    with pytest.raises(RuntimeError, match='Expanded by: gx'):
        interface.submit([[1.0, 2.0]])


def test_runmanagers_verdict_and_its_reason_both_reach_the_caller(interface, client):
    """A shot stops being pending for several different reasons, and they are
    not the same news. One that has left the queue may still have a cost on its
    way through lyse; one an operator has to unblock will not move until they
    do. Reducing the answer to the ids still coming throws that away.
    """
    client.states = {
        'b': {'pending': False, 'state': 'unknown'},
        'c': {'pending': False, 'state': 'blocked'},
    }
    assert interface.shot_status(['a', 'b', 'c']) == {
        'a': {'pending': True, 'state': 'running'},
        'b': {'pending': False, 'state': 'unknown'},
        'c': {'pending': False, 'state': 'blocked'},
    }



def test_nothing_is_asked_about_an_empty_list(interface, client):
    client.shot_status = lambda ids: pytest.fail('asked about nothing')
    assert interface.shot_status([]) == {}
