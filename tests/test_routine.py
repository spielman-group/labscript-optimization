"""Reading a cost out of the lyse dataframe.

lyse labels its columns with a MultiIndex: a runmanager global named ``x`` is
the column ``('x', '')`` and an analysis result is ``('routine', 'result')``.
These tests use that real shape.
"""

import numpy as np
import pandas as pd
import pytest

from labscript_optimization import config as config_module
from labscript_optimization.routine import extract

CONFIG = """
[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
groups = ["G"]
[MLOOP]
session = "run-a"
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""


@pytest.fixture
def config():
    return config_module.loads(CONFIG)


def frame(rows, multiindex=True):
    """Build a dataframe shaped the way lyse shapes one."""
    columns = list(rows[0])
    if multiindex:
        columns = pd.MultiIndex.from_tuples(
            [c if isinstance(c, tuple) else (c, '') for c in columns]
        )
        rows = [{(k if isinstance(k, tuple) else (k, '')): v for k, v in r.items()}
                for r in rows]
    return pd.DataFrame(rows, columns=columns)


def shot(session='run-a', iteration=3, cost=7.0, uncer=None):
    row = {'mloop_session': session, 'mloop_iteration': iteration, ('zTOF', 'Nb'): cost}
    if uncer is not None:
        row[('zTOF', 'u_Nb')] = uncer
    return row


def test_the_tag_identifies_which_proposal_the_shot_answers(config):
    tag, cost, uncer, bad = extract(frame([shot()]), config)
    assert tag == 'run-a:3'


def test_a_maximised_quantity_has_its_sign_flipped(config):
    _, cost, _, bad = extract(frame([shot(cost=7.0)]), config)
    assert cost == -7.0 and not bad


def test_a_minimised_quantity_is_passed_through():
    config = config_module.loads(CONFIG.replace('maximize = true', 'maximize = false'))
    _, cost, _, _ = extract(frame([shot(cost=7.0)]), config)
    assert cost == 7.0


def test_an_uncertainty_column_is_picked_up_when_present(config):
    _, _, uncer, _ = extract(frame([shot(uncer=0.5)]), config)
    assert uncer == 0.5


def test_a_missing_uncertainty_is_reported_as_absent_not_as_zero(config):
    _, _, uncer, _ = extract(frame([shot()]), config)
    assert uncer is None


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_shot_with_no_usable_cost_is_bad(config, value):
    _, _, _, bad = extract(frame([shot(cost=value)]), config)
    assert bad


def test_the_most_recent_shot_is_the_one_read(config):
    rows = [shot(iteration=1, cost=1.0), shot(iteration=2, cost=2.0)]
    tag, cost, _, _ = extract(frame(rows), config)
    assert tag == 'run-a:2' and cost == -2.0


def test_a_shot_from_outside_the_session_is_ignored(config):
    """A user's own shot carries no tag, so there is nothing to attribute."""
    rows = [{('zTOF', 'Nb'): 5.0}]
    assert extract(frame(rows), config) is None


def test_a_shot_whose_tag_is_blank_is_ignored(config):
    rows = [shot(iteration=float('nan'))]
    assert extract(frame(rows), config) is None


def test_an_empty_dataframe_yields_nothing(config):
    assert extract(frame([shot()]).iloc[0:0], config) is None


def test_a_cost_column_that_does_not_exist_yet_reads_as_bad(config):
    rows = [{'mloop_session': 'run-a', 'mloop_iteration': 3}]
    tag, _, _, bad = extract(frame(rows), config)
    assert tag == 'run-a:3' and bad


def test_flat_columns_work_too(config):
    """The same reader has to work against a plain dataframe in a test."""
    rows = [{'mloop_session': 'run-a', 'mloop_iteration': 3, ('zTOF', 'Nb'): 7.0}]
    tag, cost, _, _ = extract(frame(rows, multiindex=False), config)
    assert tag == 'run-a:3' and cost == -7.0
