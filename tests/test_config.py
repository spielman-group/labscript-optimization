"""The configuration schema: what a file is read into, and what it is refused for."""

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization import learners

FULL = """
[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
groups = ["CMOT", "SHIMS", "DERIVED"]

[GENERAL]
num_buffered_runs = 3
max_num_runs = 400
learner = "gaussian_process"

[LEARNER.directed_random]
trust_region = 0.2

[LEARNER.gaussian_process]
trust_region = 0.05
cost_has_noise = true

[PARAMETERS.CMOT.width]
global_name = "CMOTCaptureWidth"
min = 0.01
max = 0.5
start = 0.05

[PARAMETERS.CMOT.disabled_one]
global_name = "NotUsed"
enable = false
min = 0.0
max = 1.0

[PARAMETERS.INACTIVE.elsewhere]
global_name = "AlsoNotUsed"
min = 0.0
max = 1.0

[PARAMETERS.SHIMS.bx]
min = -1.0
max = 1.0
start = 0.0

[PARAMETERS.SHIMS.by]
min = -1.0
max = 1.0
start = 0.0

[RUNMANAGER_GLOBALS.SHIMS.ShimVector]
expr = "lambda a, b: (a, b)"
args = ["bx", "by"]

[RUNMANAGER_GLOBALS.SHIMS.SwitchedOff]
args = ["bx"]
enable = false

[RUNMANAGER_GLOBALS.INACTIVE.NotListed]
args = ["width"]

[RUNMANAGER_GLOBALS.DERIVED.Doubled]
expr = "lambda v: 2 * v"
args = ["bx"]
"""

#: The least a file can say and still load: one enabled parameter, its global,
#: and the cost. Every optional setting is left out. What a test appends to it
#: opens a table of its own, so no test depends on where this ends.
MINIMAL = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""

#: A second parameter, to append to MINIMAL.
Y = '[PARAMETERS.G.y]\nglobal_name = "gy"\nmin = 0.0\nmax = 2.0\n'

DE = MINIMAL + '[GENERAL]\nlearner = "differential_evolution"\n'

TWO_GROUPS = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = {groups}
[PARAMETERS.GA.x]
global_name = "ga"
min = 0.0
max = 1.0
[PARAMETERS.GB.x]
global_name = "gb"
min = 5.0
max = 6.0
"""


@pytest.fixture
def config():
    return config_module.loads(FULL)


def test_listed_groups_decide_what_is_searched_and_what_is_set(config):
    """A parameter or global that is switched off, or in a group nobody listed,
    is not set, so its runmanager global keeps whatever value it holds.
    """
    assert [p.name for p in config.space.parameters] == ['width', 'bx', 'by']
    assert config.globals_for([0.25, 0.1, 0.2]) == {
        'CMOTCaptureWidth': 0.25,
        'ShimVector': (0.1, 0.2),
        'Doubled': 0.2,
    }


def test_an_explorer_and_its_gaussian_process_each_take_their_own_value_of_a_knob(
    config,
):
    """A wide region to explore with and a tight one to refine with. The file
    names no explorer, so this is also the default one reading its own table.
    """
    built = learners.build(config)
    wide = config.space.absolute_trust_region(0.2)
    tight = config.space.absolute_trust_region(0.05)
    assert not np.allclose(wide, tight)
    np.testing.assert_allclose(built.explorer.trust_region, wide)
    np.testing.assert_allclose(built.model.trust_region, tight)


def test_the_explorer_is_named_in_the_gaussian_process_table():
    """Queue depth and budget are held to the selected learner alone: a
    differential evolution explorer beside a budget of ten still loads.
    """
    config = config_module.loads(
        MINIMAL
        + '[GENERAL]\nnum_buffered_runs = 2\nmax_num_runs = 10\n'
        + '[LEARNER.gaussian_process]\nexplorer = "differential_evolution"\n'
    )
    explorer = learners.build(config).explorer
    assert type(explorer) is learners.DifferentialEvolutionLearner


def test_a_setting_left_out_takes_the_documented_default():
    config = config_module.loads(MINIMAL)
    assert config.num_buffered_runs == 2
    assert config.learner == 'gaussian_process'
    assert config.maximize is False


def test_a_name_repeated_in_a_group_nobody_switched_on_still_loads():
    """Switching between groups is what group names are for; only both being
    switched on at once is a collision."""
    config = config_module.loads(TWO_GROUPS.format(groups='["GB"]'))
    assert [p.name for p in config.space.parameters] == ['x']
    assert config.globals_for([5.5]) == {'gb': 5.5}


def test_an_expression_sees_the_names_runmanager_globals_do():
    """And one that fails only at the middle of the ranges, where the load calls
    it once, still loads: a ratio over a range centered on zero.
    """
    config = config_module.loads(
        MINIMAL
        + '[PARAMETERS.G.y]\nmin = -1.0\nmax = 1.0\n'
        + '[RUNMANAGER_GLOBALS.G.decay]\nexpr = "lambda v: exp(-v)"\nargs = ["x"]\n'
        + '[RUNMANAGER_GLOBALS.G.ratio]\nexpr = "lambda a, b: a / b"\nargs = ["x", "y"]\n'
    )
    values = config.globals_for([0.5, 0.25])
    assert values['decay'] == pytest.approx(np.exp(-0.5))
    assert values['ratio'] == 2.0


def test_a_generational_learner_will_not_take_a_queue_depth_as_well():
    with pytest.raises(ValueError, match='num_buffered_runs'):
        config_module.loads(DE + 'num_buffered_runs = 3\n')
    # Any other learner takes one.
    other = config_module.loads(MINIMAL + '[GENERAL]\nnum_buffered_runs = 3\n')
    assert other.num_buffered_runs == 3


@pytest.mark.parametrize(
    'sized, refused, accepted',
    [
        ('', 15, 16),
        ('[LEARNER.differential_evolution]\npopulation_size = 5\n', 9, 10),
    ],
    ids=['the default population', 'a population the file sizes'],
)
def test_a_budget_below_two_whole_generations_is_refused(sized, refused, accepted):
    with pytest.raises(ValueError, match='max_num_runs'):
        config_module.loads(DE + f'max_num_runs = {refused}\n' + sized)
    loaded = config_module.loads(DE + f'max_num_runs = {accepted}\n' + sized)
    assert loaded.max_num_runs == accepted


@pytest.mark.parametrize(
    'setting, refused, accepted',
    [
        ('num_buffered_runs', 0, 1),
        ('seed', -1, 0),
        ('max_num_runs', 0, 1),
        ('max_num_runs_without_better_params', 0, 1),
    ],
)
def test_a_setting_below_its_floor_is_refused_and_the_floor_itself_loads(
    setting, refused, accepted
):
    with pytest.raises(ValueError, match=f'{setting} must be at least'):
        config_module.loads(MINIMAL + f'[GENERAL]\n{setting} = {refused}\n')
    loaded = config_module.loads(MINIMAL + f'[GENERAL]\n{setting} = {accepted}\n')
    assert getattr(loaded, setting) == accepted


# --- the refusals ----------------------------------------------------------
# A refusal names the setting, and its table where the same spelling is a
# setting in one table and nothing in another. The wording is not held.


@pytest.mark.parametrize(
    'knob, tables',
    [
        ('cost_has_noise = true', ['[LEARNER.gaussian_process]']),
        (
            'trust_region = 0.05',
            [
                '[LEARNER.directed_random]',
                '[LEARNER.differential_evolution]',
                '[LEARNER.gaussian_process]',
            ],
        ),
    ],
    ids=['a knob one learner takes', 'the knob three learners take'],
)
def test_a_learner_knob_in_the_general_table_is_refused_naming_where_it_goes(
    knob, tables
):
    """[GENERAL] is read once for a session that may run two learners, so a knob
    there cannot say which one it meant."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[GENERAL]\n{knob}\n')
    message = str(raised.value)
    assert knob.split(' =')[0] in message
    for table in tables:
        assert table in message


UNKNOWN = {
    'ANALYSIS': (
        MINIMAL.replace('groups', 'ignore_bad = true\ngroups'),
        'ignore_bad',
        '[ANALYSIS]',
    ),
    'GENERAL': (MINIMAL + '[GENERAL]\nsession = "run-a"\n', 'session', '[GENERAL]'),
    'PARAMETERS': (
        MINIMAL + '[PARAMETERS.G.y]\nminimum = 2.0\n',
        'minimum',
        '[PARAMETERS.G.y]',
    ),
    'RUNMANAGER_GLOBALS': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: v"\narg = ["x"]\n',
        'arg',
        '[RUNMANAGER_GLOBALS.G.d]',
    ),
    'a group nobody switched on': (
        MINIMAL + '[PARAMETERS.OFF.y]\nmin = 0.0\nmaxx = 1.0\n',
        'maxx',
        '[PARAMETERS.OFF.y]',
    ),
    'a knob its learner does not take': (
        MINIMAL + '[LEARNER.differential_evolution]\nrestart_tolerance = 0.01\n',
        'restart_tolerance',
        '[LEARNER.differential_evolution]',
    ),
    'a learner that takes no knobs': (
        MINIMAL + '[LEARNER.random]\ncost_has_noise = true\n',
        'cost_has_noise',
        '[LEARNER.random]',
    ),
    'the top level': (
        MINIMAL + '[COMPILATION]\nmock = false\n',
        'COMPILATION',
        'the top level',
    ),
}


@pytest.mark.parametrize('text, key, where', UNKNOWN.values(), ids=list(UNKNOWN))
def test_a_key_nothing_reads_is_refused_naming_it_and_its_table(text, key, where):
    """Including the settings UPGRADING.md retires, which a stale file carries."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    message = str(raised.value)
    assert key in message
    assert where in message


def test_the_mloop_tables_are_refused_naming_their_replacements():
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL.replace('[PARAMETERS.', '[MLOOP_PARAMS.')
            + '[MLOOP]\nlearner = "random"\n'
        )
    message = str(raised.value)
    for name in ('[MLOOP]', '[GENERAL]', '[MLOOP_PARAMS]', '[PARAMETERS]'):
        assert name in message


@pytest.mark.parametrize(
    'where, old, new',
    [
        ('[GENERAL]', 'trainer', 'explorer'),
        ('[GENERAL]', 'num_training_runs', 'warmup_observations'),
        ('[GENERAL]', 'num_runs_between_trainer_runs', 'explore_runs'),
        ('[GENERAL]', 'refit_interval', 'batch_size'),
        ('[GENERAL]', 'generation_size', 'batch_size'),
        ('[GENERAL]', 'minimum_observations', 'warmup_observations'),
        ('[LEARNER.gaussian_process]', 'generation_size', 'batch_size'),
    ],
)
def test_a_replaced_setting_is_refused_naming_what_replaces_it(where, old, new):
    """Not aliased: an old key read as its replacement would mean something
    else, a period between explorer shots read as a count of them."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'{where}\n{old} = 4\n')
    message = str(raised.value)
    for part in (where, old, new):
        assert part in message


REFUSED = {
    # Left out.
    'no cost_key': (MINIMAL.replace('cost_key = ["r", "c"]\n', ''), 'ANALYSIS.cost_key'),
    'no groups': (MINIMAL.replace('groups = ["G"]\n', ''), 'ANALYSIS.groups'),
    'no min': (MINIMAL + '[PARAMETERS.G.y]\nmax = 1.0\n', "'min'"),
    'no max': (MINIMAL + '[PARAMETERS.G.y]\nmin = 0.0\n', "'max'"),
    'no args': (MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: v"\n', "'args'"),
    # Names that match nothing, and parameters and globals that do not meet.
    'no group enabled': (
        MINIMAL.replace('groups = ["G"]', 'groups = []'),
        'ANALYSIS.groups',
    ),
    'a listed group no table defines': (
        MINIMAL.replace('groups = ["G"]', 'groups = ["G", "GG"]'),
        "'GG'",
    ),
    'a parameter no global takes': (
        MINIMAL + '[PARAMETERS.G.orphan]\nmin = 0.0\nmax = 1.0\n',
        'orphan',
    ),
    'a global taking a parameter nobody searches': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nargs = ["missing"]\n',
        "'missing'",
    ),
    'one name for two searched parameters': (
        TWO_GROUPS.format(groups='["GA", "GB"]'),
        "'x'",
    ),
    'one global set from two places': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.gx]\nexpr = "lambda v: 2 * v"\nargs = ["x"]\n',
        "'gx'",
    ),
    # An expression that cannot do its job.
    'no expr and no parameter': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nargs = []\n',
        "'d'",
    ),
    'no expr and two parameters': (
        MINIMAL + Y + '[RUNMANAGER_GLOBALS.G.d]\nargs = ["x", "y"]\n',
        "'d'",
    ),
    'an expr that will not evaluate': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: v +"\nargs = ["x"]\n',
        "'d'",
    ),
    'an expr that is not a function': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "2 * 3"\nargs = ["x"]\n',
        "'d'",
    ),
    'an expr that cannot take its parameters': (
        MINIMAL + Y + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda a: a"\nargs = ["x", "y"]\n',
        "'d'",
    ),
    'an expr naming what runmanager globals cannot see': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: math.exp(-v)"\nargs = ["x"]\n',
        "'math'",
    ),
    # A table written one level short.
    'a parameter without its group': (
        MINIMAL + '[PARAMETERS.H]\nglobal_name = "gh"\nmin = 0.0\nmax = 1.0\n',
        '[PARAMETERS.<group>.<name>]',
    ),
    'a global without its group': (
        MINIMAL + '[RUNMANAGER_GLOBALS.H]\nargs = ["x"]\n',
        '[RUNMANAGER_GLOBALS.<group>.<name>]',
    ),
    'a learner knob without its learner': (
        MINIMAL + '[LEARNER]\nbatch_size = 4\n',
        '[LEARNER.<name>]',
    ),
    # A value of the wrong kind, which is refused and never converted.
    'groups as a bare string': (
        MINIMAL.replace('groups = ["G"]', 'groups = "G"'),
        'ANALYSIS.groups',
    ),
    'cost_key as a bare string': (
        MINIMAL.replace('cost_key = ["r", "c"]', 'cost_key = "rc"'),
        'ANALYSIS.cost_key',
    ),
    'cost_key of numbers': (
        MINIMAL.replace('cost_key = ["r", "c"]', 'cost_key = [1, 2]'),
        'cost_key',
    ),
    'cost_key of one name': (
        MINIMAL.replace('cost_key = ["r", "c"]', 'cost_key = ["r"]'),
        'cost_key',
    ),
    'args as a bare string': (
        MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: v"\nargs = "x"\n',
        'RUNMANAGER_GLOBALS.G.d',
    ),
    'maximize quoted': (MINIMAL.replace('groups', 'maximize = "false"\ngroups'), 'maximize'),
    'enable quoted': (
        MINIMAL.replace('min = 0.0', 'enable = "false"\nmin = 0.0'),
        'PARAMETERS.G.x',
    ),
    'seed a fraction': (MINIMAL + '[GENERAL]\nseed = 1.9\n', 'seed'),
    'num_buffered_runs a boolean': (
        MINIMAL + '[GENERAL]\nnum_buffered_runs = true\n',
        'num_buffered_runs',
    ),
    'min a boolean': (
        MINIMAL + Y.replace('min = 0.0', 'min = true'),
        '[PARAMETERS.G.y] min',
    ),
    'start quoted': (MINIMAL + Y + 'start = "1"\n', '[PARAMETERS.G.y] start'),
    'global_name a number': (
        MINIMAL + Y.replace('"gy"', '5'),
        '[PARAMETERS.G.y] global_name',
    ),
    # A learner's table is held to its constructor when the load builds it.
    'a quoted boolean knob': (
        MINIMAL + '[LEARNER.gaussian_process]\ncost_has_noise = "false"\n',
        'cost_has_noise',
    ),
    # A learner that does not exist, refused at load rather than when the
    # apparatus is already running.
    'an unknown learner table': (
        MINIMAL + '[LEARNER.typo]\ntrust_region = 0.2\n',
        "'typo'",
    ),
    'an unknown learner': (
        MINIMAL + '[GENERAL]\nlearner = "gaussain_process"\n',
        "'gaussain_process'",
    ),
}


@pytest.mark.parametrize('text, named', REFUSED.values(), ids=list(REFUSED))
def test_a_file_that_cannot_run_as_written_is_refused_naming_the_setting(text, named):
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    assert named in str(raised.value)
