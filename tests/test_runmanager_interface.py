"""The interface a session reaches runmanager through, over a fake client."""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.runmanager_interface import RunmanagerInterface

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
    """What ``runmanager.client.RunmanagerClient`` answers, for the calls made."""

    def __init__(self, labscript='/lab/expt.py'):
        self.labscript = labscript
        self.broken_globals = False
        self.entries = []
        self.sequences = []
        self.values = {'gx': '2*pi*5', 'gy_doubled': '3', 'other': '7'}
        self.written = []
        self.missing = []

    def error_in_globals(self):
        return self.broken_globals

    def get_labscript_file(self):
        return self.labscript

    def get_values(self, raw=False):
        if raw:
            return dict(self.values)
        return {name: eval(value, {'pi': np.pi}) for name, value in self.values.items()}

    def set_values(self, globals, raw=False, skip_missing=False):
        self.written.append((globals, raw, skip_missing))
        return self.missing if skip_missing else []

    def submit_shots(self, entries, sequence=None, sequence_index=None):
        """Starts a sequence for each submission that names none."""
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
            }
            for i in range(len(entries))
        ]


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def interface(client):
    return RunmanagerInterface(config_module.loads(CONFIG), client)


def test_a_start_is_refused_while_runmanager_cannot_sustain_a_session(
    interface, client
):
    """Every shot the session went on to submit would fail to compile, if its
    globals do not evaluate, one of the session's is in no active group, or
    no labscript file is selected.
    """
    client.broken_globals = True
    with pytest.raises(RuntimeError, match='error in its globals'):
        interface.check_ready()
    client.broken_globals = False
    del client.values['gx']
    with pytest.raises(RuntimeError, match='gx not found in any active group'):
        interface.check_ready()
    client.labscript = ''
    with pytest.raises(RuntimeError, match='select a labscript file'):
        interface.pin_labscript_file()


def test_a_labscript_file_changed_mid_session_is_refused(interface, client):
    interface.check_ready()
    # A Start refused after its checks has pinned nothing.
    client.labscript = '/lab/another.py'
    interface.pin_labscript_file()
    interface.check_unchanged()
    client.labscript = '/lab/something_else.py'
    # A Start that resumes the run does not move the file it is compared with.
    interface.pin_labscript_file()
    with pytest.raises(RuntimeError, match='labscript file changed'):
        interface.check_unchanged()


def test_the_original_values_are_read_until_a_start_goes(interface, client):
    originals = {'gx': '2*pi*5', 'gy_doubled': '3'}
    assert interface.check_ready() == originals
    # A Start refused after its checks has kept nothing, so the next reads again.
    client.values = {'gx': '1', 'gy_doubled': '2', 'other': '3'}
    assert interface.check_ready() == {'gx': '1', 'gy_doubled': '2'}
    interface.pin_labscript_file()
    # A Start that resumes the run reads nothing.
    assert interface.check_ready() is None


def test_values_are_set_from_a_point_or_as_originally_written(interface, client):
    originals = {'gx': '2*pi*5', 'gy_doubled': '3'}
    interface.set_values([1.0, 2.0])
    interface.set_values(originals, raw=True)
    assert client.written == [
        ({'gx': 1.0, 'gy_doubled': 4.0}, False, False),
        (originals, True, False),
    ]
    # The names runmanager skips are handed back.
    client.missing = ['gx']
    assert interface.set_values(originals, raw=True, skip_missing=True) == ['gx']


def test_runmanagers_values_are_read_back_as_the_start(interface, client):
    client.values.update(gx='2.5', gy='1.5')
    # y reaches runmanager only through an expr.
    with pytest.raises(RuntimeError, match='y reaches runmanager only through'):
        interface.get_start()
    interface.config = config_module.loads(
        CONFIG.replace('[PARAMETERS.G.y]', '[PARAMETERS.G.y]\nglobal_name = "gy"')
    )
    assert interface.get_start().tolist() == [2.5, 1.5]
    for value, reason in [
        ('2*pi*5', 'gx is 31.4159, outside'),
        ('True', 'is True, not a number'),
        ('"fast"', "is 'fast', not a number"),
    ]:
        client.values['gx'] = value
        with pytest.raises(RuntimeError, match=reason):
            interface.get_start()


def test_every_submission_after_the_first_joins_its_sequence(interface, client):
    """A run is one runmanager sequence, whatever an operator engages in
    between."""
    assert interface.submit([[1.0, 2.0], [3.0, 4.0]]) == ['id-0', 'id-1']
    for _ in range(2):
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


def test_every_global_is_submitted_as_a_python_value(client):
    """runmanager writes a submitted value into its global's expression as the
    value's repr, and a numpy scalar's repr names numpy -- ``np.float64(1.0)``
    in the operator's runmanager and in the shot file. Proposals arrive as a
    numpy array, so every value is checked, down to the members of a tuple.
    """
    config = config_module.loads(
        CONFIG
        + '[RUNMANAGER_GLOBALS.G.pair]\nexpr = "lambda a, b: (a, b)"\n'
        + 'args = ["x", "y"]\n'
        + '[RUNMANAGER_GLOBALS.G.decay]\nexpr = "lambda a: exp(-a)"\nargs = ["x"]\n'
    )
    RunmanagerInterface(config, client).submit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert len(client.entries) == 2
    for entry in client.entries:
        for name, value in entry.items():
            assert holds_only_python_values(value), (name, repr(value))
