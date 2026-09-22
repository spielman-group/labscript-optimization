"""The seam against runmanager.remote.

These use a fake client returning the shapes the real one documents, so a
change to those shapes shows up here rather than in the lab.
"""

import pytest

from labscript_optimization import config as config_module
from labscript_optimization.runmanager_interface import (
    CHECK_READY_REQUESTS,
    RunmanagerInterface,
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
    """The shape of ``runmanager.remote.Client``, answers and all.

    ``timeout`` is the real client's own: every request waits that long for an
    answer, and it comes from labconfig's ``communication_timeout``. Each call
    is recorded with the timeout in force when it was made, because how long a
    question is allowed to go unanswered is part of what this seam promises.
    """

    def __init__(self, labscript='/lab/expt.py'):
        self.labscript = labscript
        self.broken_globals = False
        self.entries = []
        self.states = {}
        self.refuse = None
        self.timeout = 60.0
        self.silent = False
        self.asked = []

    def say_hello(self):
        self.asked.append(('say_hello', self.timeout))
        if self.silent:
            raise TimeoutError('no response from server')
        return 'hello'

    def error_in_globals(self):
        self.asked.append(('error_in_globals', self.timeout))
        return self.broken_globals

    def get_labscript_file(self):
        self.asked.append(('get_labscript_file', self.timeout))
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


def test_a_runmanager_that_does_not_answer_is_reported_as_the_cause(interface, client):
    """The worker has one startup allowance, and everything it does with
    runmanager happens inside it. A runmanager that is not running answers
    nothing, so without being named here the lab waits out the client's own
    deadline and then reads whichever question went unanswered -- true, and
    no help at all.
    """
    client.silent = True
    with pytest.raises(RuntimeError, match='runmanager did not answer'):
        interface.check_ready()
    assert [call for call, _ in client.asked] == ['say_hello']


def test_the_greeting_is_asked_first_and_is_the_only_short_wait(interface, client):
    """A runmanager that is not there is found out by the greeting rather than
    by whichever question happened to be asked first, and found out in seconds
    rather than in the minute the client would spend on a question.

    Only the greeting is shortened. A submission compiles shots, which takes
    as long as it takes, and holding it to a few seconds would end a healthy
    session; the routine's allowance for configuring is summed to cover the
    questions at their full length.
    """
    interface.check_ready()
    assert client.asked == [
        ('say_hello', 5.0),
        ('error_in_globals', 60.0),
        ('get_labscript_file', 60.0),
    ]


def test_the_deadline_is_summed_over_as_many_requests_as_are_made(
    interface, client
):
    """The routine's allowance for configuring is the greeting's deadline plus
    one of the client's per question asked after it. That count lives beside
    the questions, so a question added below is a visible reason to change it;
    this is what says the two agree.
    """
    interface.check_ready()
    assert len(client.asked) == 1 + CHECK_READY_REQUESTS


def test_a_runmanager_whose_globals_do_not_evaluate_is_refused(interface, client):
    """Every shot the session went on to submit would fail to compile."""
    client.broken_globals = True
    with pytest.raises(RuntimeError, match='error in its globals'):
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
