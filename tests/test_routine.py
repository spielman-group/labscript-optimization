"""Reading a cost out of the lyse dataframe.

lyse labels its columns with a MultiIndex, so an analysis result is the column
``('routine', 'result')``. The shot's identifier is not a column at all: lyse
does not carry it into the dataframe, so it is read from the shot file that
``filepath`` names.
"""

import h5py
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
        rows = [
            {(k if isinstance(k, tuple) else (k, '')): v for k, v in r.items()}
            for r in rows
        ]
    return pd.DataFrame(rows, columns=columns)


@pytest.fixture
def shot(tmp_path):
    """Make a dataframe row with a real shot file behind it."""
    made = []

    def build(shot_id='row-3', cost=7.0, uncer=None, with_cost=True):
        path = tmp_path / f'shot{len(made)}.h5'
        made.append(path)
        with h5py.File(path, 'w') as f:
            if shot_id is not None:
                f.attrs['shot_id'] = shot_id
        row = {'filepath': str(path)}
        if with_cost:
            row[('zTOF', 'Nb')] = cost
        if uncer is not None:
            row[('zTOF', 'u_Nb')] = uncer
        return row

    return build


def test_the_shot_id_says_which_proposal_the_shot_answers(config, shot):
    shot_id, _, _, _ = extract(frame([shot()]), config)
    assert shot_id == 'row-3'


def test_a_maximised_quantity_has_its_sign_flipped(config, shot):
    _, cost, _, bad = extract(frame([shot(cost=7.0)]), config)
    assert cost == -7.0 and not bad


def test_a_minimised_quantity_is_passed_through(shot):
    config = config_module.loads(CONFIG.replace('maximize = true', 'maximize = false'))
    _, cost, _, _ = extract(frame([shot(cost=7.0)]), config)
    assert cost == 7.0


def test_an_uncertainty_column_is_picked_up_when_present(config, shot):
    _, _, uncer, _ = extract(frame([shot(uncer=0.5)]), config)
    assert uncer == 0.5


def test_a_missing_uncertainty_is_absent_rather_than_zero(config, shot):
    _, _, uncer, _ = extract(frame([shot()]), config)
    assert uncer is None


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_shot_with_no_usable_cost_is_bad(config, shot, value):
    _, _, _, bad = extract(frame([shot(cost=value)]), config)
    assert bad


def test_the_most_recent_shot_is_the_one_read(config, shot):
    rows = [shot(shot_id='row-1', cost=1.0), shot(shot_id='row-2', cost=2.0)]
    shot_id, cost, _, _ = extract(frame(rows), config)
    assert shot_id == 'row-2' and cost == -2.0


def test_a_shot_carrying_no_identifier_is_not_ours(config, shot):
    """A user's own shot, or one of runmanager's defaults, has no shot_id."""
    assert extract(frame([shot(shot_id=None)]), config) is None


def test_a_shot_whose_file_has_gone_is_not_ours(config, shot, tmp_path):
    row = shot()
    (tmp_path / 'shot0.h5').unlink()
    assert extract(frame([row]), config) is None


def test_a_dataframe_with_no_filepath_yields_nothing(config):
    assert extract(frame([{('zTOF', 'Nb'): 5.0}]), config) is None


def test_an_empty_dataframe_yields_nothing(config, shot):
    assert extract(frame([shot()]).iloc[0:0], config) is None


def test_a_cost_column_that_does_not_exist_yet_reads_as_bad(config, shot):
    shot_id, _, _, bad = extract(frame([shot(with_cost=False)]), config)
    assert shot_id == 'row-3' and bad


def test_flat_columns_work_too(config, shot):
    """The same reader has to work against a plain dataframe in a test."""
    shot_id, cost, _, _ = extract(frame([shot()], multiindex=False), config)
    assert shot_id == 'row-3' and cost == -7.0
