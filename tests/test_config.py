"""The configuration schema, which existing lab files must keep loading."""

import pytest

from labscript_optimization import config as config_module

FULL = """
[COMPILATION]
mock = false

[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
ignore_bad = true
groups = ["CMOT", "SHIMS"]

[MLOOP]
session = "run-a"
num_buffered_runs = 3
num_training_runs = 20
max_num_runs = 400
trust_region = 0.05
cost_has_noise = true
controller_type = "gaussian_process"
no_delay = true
visualisations = false

[MLOOP_PARAMS.CMOT.width]
global_name = "CMOTCaptureWidth"
min = 0.01
max = 0.5
start = 0.05

[MLOOP_PARAMS.CMOT.disabled_one]
global_name = "NotUsed"
enable = false
min = 0.0
max = 1.0

[MLOOP_PARAMS.INACTIVE.elsewhere]
global_name = "AlsoNotUsed"
min = 0.0
max = 1.0

[MLOOP_PARAMS.SHIMS.bx]
min = -1.0
max = 1.0

[MLOOP_PARAMS.SHIMS.by]
min = -1.0
max = 1.0

[RUNMANAGER_GLOBALS.SHIMS.ShimVector]
expr = "lambda a, b: (a, b)"
args = ["bx", "by"]
"""


@pytest.fixture
def config():
    return config_module.loads(FULL)


def test_only_parameters_in_active_enabled_groups_are_searched(config):
    assert [p.name for p in config.space.parameters] == ['width', 'bx', 'by']


def test_a_disabled_parameter_is_remembered_but_not_searched(config):
    assert [p.name for p in config.space.disabled] == ['disabled_one']


def test_global_name_is_shorthand_for_a_direct_mapping(config):
    direct = next(g for g in config.globals if g.name == 'CMOTCaptureWidth')
    assert direct.expr is None and direct.args == ('width',)
    assert direct.evaluate({'width': 0.25}) == 0.25


def test_several_parameters_can_feed_one_global(config):
    combined = next(g for g in config.globals if g.name == 'ShimVector')
    assert combined.args == ('bx', 'by')
    assert combined.evaluate({'bx': 0.1, 'by': 0.2}) == (0.1, 0.2)


def test_the_globals_for_a_proposal_cover_every_mapping(config):
    values = config.globals_for([0.25, 0.1, 0.2])
    assert values == {'CMOTCaptureWidth': 0.25, 'ShimVector': (0.1, 0.2)}


def test_the_uncertainty_column_is_the_cost_column_prefixed(config):
    assert config.cost_key == ('zTOF', 'Nb')
    assert config.uncertainty_key == ('zTOF', 'u_Nb')


def test_keys_that_only_m_loop_needed_are_ignored_rather_than_rejected(config):
    assert config.learner == 'gaussian_process'
    assert 'no_delay' not in config.options_for('gaussian_process')
    assert 'visualisations' not in config.options_for('gaussian_process')


def test_learner_knobs_in_the_mloop_table_are_shared_defaults(config):
    options = config.options_for('gaussian_process')
    assert options['trust_region'] == 0.05
    assert options['cost_has_noise'] is True


def test_a_per_learner_table_overrides_the_shared_defaults():
    config = config_module.loads(
        FULL + '\n[LEARNER.gaussian_process]\ntrust_region = 0.2\n'
    )
    assert config.options_for('gaussian_process')['trust_region'] == 0.2
    assert config.options_for('directed_random')['trust_region'] == 0.05


def test_a_parameter_with_no_global_is_rejected():
    with pytest.raises(KeyError, match='not mapped to any runmanager global'):
        config_module.loads(
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.orphan]
min = 0.0
max = 1.0
"""
        )


def test_a_global_taking_an_unknown_parameter_is_rejected():
    with pytest.raises(KeyError, match='not an enabled parameter'):
        config_module.loads(
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
[RUNMANAGER_GLOBALS.G.other]
args = ["missing"]
"""
        )


def test_a_configuration_with_nothing_enabled_says_so():
    with pytest.raises(ValueError, match='no parameters are enabled'):
        config_module.loads(
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = []
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""
        )


def test_a_missing_cost_key_says_so():
    with pytest.raises(KeyError, match='cost_key'):
        config_module.loads(
            """
[ANALYSIS]
groups = ["G"]
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""
        )
