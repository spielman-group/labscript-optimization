"""The configuration schema, and the keys it refuses."""

import dataclasses
from pathlib import Path

import pytest

from labscript_optimization import config as config_module
from labscript_optimization import learners

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'config_example.toml'

FULL = """
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
learner = "gaussian_process"

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
#: and the cost. Every optional setting is left out. What a test appends to it
#: opens a table of its own, so no test depends on where this ends.
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


def test_a_disabled_parameter_keeps_its_global_out_of_every_shot(config):
    """Switching a parameter off must leave its global at whatever it holds."""
    assert 'disabled_one' not in [p.name for p in config.space.parameters]
    assert 'NotUsed' not in config.globals_for([0.25, 0.1, 0.2])


def test_global_name_is_shorthand_for_a_direct_mapping(config):
    direct = next(g for g in config.globals if g.name == 'CMOTCaptureWidth')
    assert direct.expr is None and direct.args == ('width',)
    assert direct.evaluate({'width': 0.25}) == 0.25


def test_several_parameters_can_feed_one_global(config):
    combined = next(g for g in config.globals if g.name == 'ShimVector')
    assert combined.args == ('bx', 'by')
    assert combined.evaluate({'bx': 0.1, 'by': 0.2}) == (0.1, 0.2)


def test_a_switched_off_global_is_not_set():
    """A parameter switched off is still carried; a global switched off is not.

    The parameter keeps its bounds, because a reader of the file and of
    ``space.disabled`` wants to see what is being held out. A global has
    nothing to hold out: not setting it is the whole of the behaviour.
    """
    config = config_module.loads(
        MINIMAL
        + '[RUNMANAGER_GLOBALS.G.doubled]\n'
        + 'expr = "lambda v: 2 * v"\nargs = ["x"]\nenable = false\n'
    )
    assert [g.name for g in config.globals] == ['gx']


def test_a_global_in_a_group_nobody_switched_on_is_not_set():
    config = config_module.loads(
        MINIMAL
        + '[RUNMANAGER_GLOBALS.OFF.doubled]\nexpr = "lambda v: 2 * v"\nargs = ["x"]\n'
    )
    assert [g.name for g in config.globals] == ['gx']


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


def test_the_learner_is_named_in_the_mloop_table():
    config = config_module.loads(
        MINIMAL + '[MLOOP]\nlearner = "differential_evolution"\n'
    )
    assert config.learner == 'differential_evolution'


def test_a_setting_left_out_takes_the_value_the_documents_promise():
    """What a lab may leave out on the strength of what it was told.

    UPGRADING §4 promises three buffered runs, the Gaussian process is the
    learner a file naming none gets, and a cost is minimised unless the file
    says otherwise -- the flip nothing downstream would show.
    """
    config = config_module.loads(MINIMAL)
    assert config.num_buffered_runs == 3
    assert config.learner == 'gaussian_process'
    assert config.maximize is False


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
    with pytest.raises(ValueError, match='not mapped to any runmanager global'):
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
    with pytest.raises(ValueError, match='not an enabled parameter'):
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
    """And says it plainly.

    Every complaint about the file is a ValueError, including this one. A
    KeyError reprs its argument, so a written-out sentence raised as one
    reaches the reader in quotes with its own quotes backslashed.
    """
    with pytest.raises(ValueError) as raised:
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
    assert str(raised.value).startswith('ANALYSIS.cost_key is required')


@pytest.mark.parametrize('missing', ['min', 'max'])
def test_a_parameter_table_without_its_bounds_says_so(missing):
    """A setting left out is a ValueError like any other complaint."""
    kept = 'min = 0.0' if missing == 'max' else 'max = 1.0'
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + f'[MLOOP_PARAMS.G.y]\nglobal_name = "gy"\n{kept}\n'
        )
    message = str(raised.value)
    assert 'MLOOP_PARAMS.G.y' in message
    assert f"'{missing}'" in message
    assert 'requires' in message


def test_a_global_with_nothing_to_compute_from_says_so():
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + '[RUNMANAGER_GLOBALS.G.doubled]\nexpr = "lambda v: 2 * v"\n'
        )
    message = str(raised.value)
    assert 'RUNMANAGER_GLOBALS.G.doubled' in message
    assert 'args' in message


@pytest.mark.parametrize(
    'text, named',
    [
        (MINIMAL.replace('groups = ["G"]', 'groups = "G"'), 'ANALYSIS.groups'),
        (
            MINIMAL.replace('cost_key = ["r", "c"]', 'cost_key = "rc"'),
            'ANALYSIS.cost_key',
        ),
        (
            MINIMAL + '[RUNMANAGER_GLOBALS.G.d]\nexpr = "lambda v: v"\nargs = "x"\n',
            'RUNMANAGER_GLOBALS.G.d',
        ),
    ],
)
def test_a_list_written_as_a_bare_string_is_rejected(text, named):
    """A string is a sequence of its own characters, so it passes for a list.

    ``groups = "CMOT"`` then selects by substring and works by coincidence,
    until a group name is a substring of another.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    assert named in str(raised.value)


@pytest.mark.parametrize(
    'text, named',
    [
        (MINIMAL.replace('groups', 'maximize = "false"\ngroups'), 'ANALYSIS.maximize'),
        (
            MINIMAL.replace('min = 0.0', 'enable = "false"\nmin = 0.0'),
            'MLOOP_PARAMS.G.x',
        ),
    ],
)
def test_a_quoted_boolean_is_rejected(text, named):
    """Every non-empty string is true, so the quotes invert what was written."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    assert named in str(raised.value)


def test_a_run_count_that_is_not_a_number_is_rejected():
    """Carried through as written it would reach the session as a string.

    ``int()`` raises on its own; what is under test is the wrapping that names
    the setting, because its own message names only the value.
    """
    with pytest.raises(ValueError, match='max_num_runs must be readable as int'):
        config_module.loads(MINIMAL + '[MLOOP]\nmax_num_runs = "many"\n')


@pytest.mark.parametrize(
    'text, spelling, named',
    [
        (
            MINIMAL + '[MLOOP]\ncontroller_type = "differential_evolution"\n',
            'controller_type',
            '[MLOOP]',
        ),
        (MINIMAL + '[MLOOP]\nno_delay = true\n', 'no_delay', '[MLOOP]'),
        (MINIMAL + '[MLOOP]\nvisualisations = false\n', 'visualisations', '[MLOOP]'),
        (
            MINIMAL.replace('groups', 'ignore_bad = true\ngroups'),
            'ignore_bad',
            '[ANALYSIS]',
        ),
        (
            MINIMAL + '[MLOOP_PARAMS.G.y]\nminimum = 2.0\n',
            'minimum',
            '[MLOOP_PARAMS.G.y]',
        ),
        (
            MINIMAL + '[MLOOP_PARAMS.G.y]\nmaximum = 2.0\n',
            'maximum',
            '[MLOOP_PARAMS.G.y]',
        ),
        (MINIMAL + '[COMPILATION]\nmock = false\n', 'COMPILATION', 'the top level'),
    ],
    ids=[
        'controller_type',
        'no_delay',
        'visualisations',
        'ignore_bad',
        'minimum',
        'maximum',
        'COMPILATION',
    ],
)
def test_a_spelling_this_package_retired_is_rejected(text, spelling, named):
    """M-LOOP's and analysislib-mloop's names for things this package renamed,
    dropped, or never had. Kept as aliases they would be spellings to carry for
    ever, in a schema whose argument is that a file says what is in force.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    message = str(raised.value)
    assert spelling in message
    assert named in message


def test_a_typo_is_rejected_naming_the_key_its_table_and_what_that_table_takes():
    """The message is the whole of what somebody with a stale file gets."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL
            + '[RUNMANAGER_GLOBALS.G.doubled]\nexpr = "lambda v: v"\narg = ["x"]\n'
        )
    message = str(raised.value)
    assert "'arg'" in message
    assert 'RUNMANAGER_GLOBALS.G.doubled' in message
    assert 'args' in message


def test_an_expression_that_will_not_evaluate_stops_the_load():
    """Rather than the first proposal, hours later, out through the worker.

    The expression becomes its callable while the file is being read, so a
    mistyped lambda is a load failure naming the global and what was written
    for it, not a session that starts and then falls over on a submission.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL
            + '[RUNMANAGER_GLOBALS.G.doubled]\nexpr = "lambda v: v +"\nargs = ["x"]\n'
        )
    message = str(raised.value)
    assert 'doubled' in message
    assert 'lambda v: v +' in message


def test_an_expression_that_is_not_a_function_stops_the_load():
    """``expr`` is a lambda taking the args in order; a value is not one."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + '[RUNMANAGER_GLOBALS.G.doubled]\nexpr = "2 * 3"\nargs = ["x"]\n'
        )
    message = str(raised.value)
    assert 'doubled' in message
    assert '2 * 3' in message


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


def test_the_example_configuration_loads_and_builds_its_learner():
    """The file every new lab starts from, held to the schema like any other."""
    config = config_module.load(EXAMPLE)
    learner = learners.build(config)
    assert isinstance(learner, learners.TwoPhaseLearner)
    assert learner.num_training == 20
