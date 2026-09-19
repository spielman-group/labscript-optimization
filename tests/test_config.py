"""The configuration schema, and the keys it refuses."""

import dataclasses

import pytest

from labscript_optimization import config as config_module

FULL = """
[COMPILATION]
mock = false

[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
groups = ["CMOT", "SHIMS"]

[MLOOP]
session = "run-a"
num_buffered_runs = 3
num_training_runs = 20
max_num_runs = 400
trust_region = 0.05
cost_has_noise = true
controller_type = "gaussian_process"

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

#: The least a file can say and still load: one enabled parameter, its global,
#: and the cost. Every optional setting is left out.
MINIMAL = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
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


def test_the_older_name_for_the_learner_key_is_still_read():
    older = config_module.loads(
        MINIMAL + '[MLOOP]\ncontroller_type = "differential_evolution"\n'
    )
    assert older.learner == 'differential_evolution'


def test_the_learner_key_wins_when_a_file_carries_both_names():
    both = config_module.loads(
        MINIMAL
        + '[MLOOP]\nlearner = "neural_net"\ncontroller_type = "differential_evolution"\n'
    )
    assert both.learner == 'neural_net'


def test_a_file_that_sets_no_options_gets_exactly_the_dataclass_defaults():
    """The two ways of building a Config must agree on every default.

    Production goes through load/loads and the tests construct a Config
    directly; a second copy of any default would let those paths drift apart.
    """
    loaded = config_module.loads(MINIMAL)
    # Only what every file must give. The learner tables come from the file
    # too, so they are handed over rather than defaulted; everything else is
    # left for the dataclass to supply.
    direct = config_module.Config(
        space=loaded.space,
        globals=loaded.globals,
        cost_key=loaded.cost_key,
        learner_options=loaded.learner_options,
    )
    for f in dataclasses.fields(config_module.Config):
        assert getattr(loaded, f.name) == getattr(direct, f.name), f.name


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


def test_a_key_this_package_does_not_act_on_is_rejected():
    """A setting nothing reads is worse than one nobody wrote.

    The file says the option is in force, the session behaves as though it
    never was, and neither side says a word. ``no_delay`` is one M-LOOP took
    and this package has no use for.
    """
    with pytest.raises(ValueError, match='no_delay'):
        config_module.loads(MINIMAL + '[MLOOP]\nno_delay = true\n')


def test_the_rejection_names_the_key_its_table_and_what_that_table_accepts():
    """The message is the whole of what somebody with a stale file gets."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[MLOOP]\nvisualisations = false\n')
    message = str(raised.value)
    assert 'visualisations' in message
    assert 'MLOOP' in message
    assert 'num_buffered_runs' in message


def test_the_retired_ignore_bad_setting_is_rejected():
    """It once suppressed the reporting of a NaN-cost shot. Nothing reads it."""
    with pytest.raises(ValueError, match='ignore_bad'):
        config_module.loads(
            MINIMAL.replace('groups = ["G"]', 'groups = ["G"]\nignore_bad = true')
        )


def test_a_typo_in_a_parameter_table_is_rejected():
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + 'strat = 0.5\n')
    message = str(raised.value)
    assert 'strat' in message
    assert 'MLOOP_PARAMS.G.x' in message
    assert 'start' in message


def test_a_typo_in_a_runmanager_global_table_is_rejected():
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + '[RUNMANAGER_GLOBALS.G.doubled]\nexpr = "lambda v: v"\narg = ["x"]\n'
        )
    message = str(raised.value)
    assert 'arg' in message
    assert 'RUNMANAGER_GLOBALS.G.doubled' in message
    assert 'args' in message


def test_a_misspelt_top_level_table_is_rejected():
    """A table nothing looks for takes every setting in it down with it."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[ANALISYS]\nmaximize = true\n')
    message = str(raised.value)
    assert 'ANALISYS' in message
    assert 'ANALYSIS' in message


def test_a_typo_in_a_group_nobody_switched_on_is_still_rejected():
    """The shape of a parameter table does not depend on its group being active.

    Left to load, the typo waits for the day somebody adds the group to
    ANALYSIS.groups and wonders why the parameter behaves oddly.
    """
    with pytest.raises(ValueError, match='maxx'):
        config_module.loads(MINIMAL + '[MLOOP_PARAMS.OFF.y]\nmin = 0.0\nmaxx = 1.0\n')


def test_a_per_learner_table_may_carry_a_knob_that_learner_does_not_take():
    """Those knobs are the learners', not this schema's.

    The factory passes each learner the ones its constructor takes and drops
    the rest, so that one lab's table serves whichever learner is selected.
    """
    config = config_module.loads(MINIMAL + '[LEARNER.random]\ncost_has_noise = true\n')
    assert config.options_for('random') == {'cost_has_noise': True}
