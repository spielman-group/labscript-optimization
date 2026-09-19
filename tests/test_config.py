"""The configuration schema, and the keys it refuses."""

import dataclasses
import subprocess
import sys
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
        shared_learner_options=loaded.shared_learner_options,
        learner_options=loaded.learner_options,
    )
    for f in dataclasses.fields(config_module.Config):
        assert getattr(loaded, f.name) == getattr(direct, f.name), f.name


def test_loading_configuration_does_not_import_scientific_learners():
    """A lyse routine that never starts a session pays no learner imports."""
    script = (
        "import sys\n"
        "from labscript_optimization import config\n"
        f"config.loads({MINIMAL!r})\n"
        "assert not {'scipy', 'sklearn'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, '-c', script], check=True)


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
        (MINIMAL.replace('groups', 'maximize = "false"\ngroups'), 'maximize'),
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
    """Carried through as written it would reach the session as a string."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[MLOOP]\nmax_num_runs = "many"\n')
    assert str(raised.value) == (
        "max_num_runs must be written as a whole number, got 'many'."
    )


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


def test_a_per_learner_table_rejects_a_knob_that_learner_does_not_take():
    """A named table has one constructor that can define its valid keys."""
    with pytest.raises(ValueError, match=r'\[LEARNER\.random\].*cost_has_noise'):
        config_module.loads(
            MINIMAL + '[LEARNER.random]\ncost_has_noise = true\n'
        )


def test_an_unknown_per_learner_table_is_rejected():
    with pytest.raises(ValueError, match="unknown learner 'typo'"):
        config_module.loads(MINIMAL + '[LEARNER.typo]\ntrust_region = 0.2\n')


def test_a_learner_named_shared_is_not_confused_with_shared_defaults():
    with pytest.raises(ValueError, match="unknown learner 'shared'"):
        config_module.loads(MINIMAL + '[LEARNER.shared]\ntrust_region = 0.2\n')


def test_a_per_learner_table_accepts_that_learners_knob():
    config = config_module.loads(
        MINIMAL + '[LEARNER.directed_random]\ntrust_region = 0.2\n'
    )
    assert config.options_for('directed_random') == {'trust_region': 0.2}


def test_the_example_configuration_loads_and_builds_its_learner():
    """The file every new lab starts from, held to the schema like any other."""
    config = config_module.load(EXAMPLE)
    learner = learners.build(config)
    assert isinstance(learner, learners.TwoPhaseLearner)
    assert learner.num_training == 20


# --- the Gaussian process's batch ------------------------------------------


def test_batch_size_is_the_gaussian_process_knob():
    config = config_module.loads(MINIMAL + '[MLOOP]\nbatch_size = 6\n')
    assert learners.build(config).main.batch_size == 6


@pytest.mark.parametrize(
    'text',
    [
        MINIMAL + '[MLOOP]\ngeneration_size = 4\n',
        MINIMAL + '[LEARNER.gaussian_process]\ngeneration_size = 4\n',
    ],
    ids=['[MLOOP]', '[LEARNER.gaussian_process]'],
)
def test_the_gaussian_process_knob_is_not_spelt_generation_size(text):
    """A generation is what differential evolution's population takes.

    Accepted in either table it would be a setting the file states and
    nothing reads, which is how a lab comes to believe a schedule is in force
    when it is not.
    """
    with pytest.raises(ValueError, match='generation_size'):
        config_module.loads(text)


# --- the budget and the population -----------------------------------------


DE = MINIMAL + '[MLOOP]\nlearner = "differential_evolution"\n'


@pytest.mark.parametrize(
    'text',
    [
        MINIMAL + '[MLOOP]\nrestart_tolerance = 0.01\n',
        MINIMAL + '[LEARNER.differential_evolution]\nrestart_tolerance = 0.01\n',
    ],
    ids=['[MLOOP]', '[LEARNER.differential_evolution]'],
)
def test_a_population_is_never_re_seeded_on_its_own_spread(text):
    """The restart is gone, so the key that sized it is refused.

    Taken at a generation boundary from the costs resolved by then, the
    decision could be changed by a cost arriving afterwards, which would turn
    a block generated as trials into founders of a new epoch;
    ``max_num_runs_without_better_params`` is the stop it was standing in for.
    """
    with pytest.raises(ValueError, match='restart_tolerance'):
        config_module.loads(text)


@pytest.mark.parametrize(
    'written, refused, accepted',
    [('', 15, 16), ('population_size = 5\n', 9, 10)],
    ids=['the default population', 'a population the file sizes'],
)
def test_a_budget_below_two_whole_generations_is_refused(written, refused, accepted):
    """One generation is the population itself; the second is the first to
    evolve it, and a configuration that cannot reach it cannot do what it
    says."""
    with pytest.raises(ValueError, match='max_num_runs') as raised:
        config_module.loads(DE + written + f'max_num_runs = {refused}\n')
    # Not "nothing evolves below this": between one population and two, a
    # generation cut short does evolve some of its slots.
    assert 'cut short' in str(raised.value)
    assert config_module.loads(
        DE + written + f'max_num_runs = {accepted}\n'
    ).max_num_runs == accepted


# --- what a setting may be -------------------------------------------------
#
# Config.__post_init__ is the single authority for the dataclass's own fields,
# so these hold for a Config written out in a script as much as for a file.


@pytest.mark.parametrize(
    'setting, written',
    [
        ('num_buffered_runs', '2.9'),
        ('num_training_runs', '19.5'),
        ('max_num_runs', '400.5'),
        ('max_num_runs_without_better_params', '80.5'),
        ('seed', '1.9'),
    ],
)
def test_a_whole_number_setting_written_as_a_fraction_is_refused(setting, written):
    """Coerced, it would reach the session as the truncated value.

    ``seed = 1.9`` is the one that shows what that costs: the run it makes
    reproducible is not the run the file asks for, and nothing says so.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[MLOOP]\n{setting} = {written}\n')
    assert str(raised.value) == (
        f'{setting} must be written as a whole number, got {float(written)!r}.'
    )


@pytest.mark.parametrize(
    'setting',
    ['num_buffered_runs', 'num_training_runs', 'max_num_runs', 'seed'],
)
def test_a_whole_number_setting_written_as_a_boolean_is_refused(setting):
    """``isinstance(True, int)`` is True, so a bare integer check takes it as 1.

    Which is a plausible number for every one of these, and so a session that
    runs on a setting nobody wrote.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[MLOOP]\n{setting} = true\n')
    assert str(raised.value) == (
        f'{setting} must be written as a whole number, got True.'
    )


@pytest.mark.parametrize('setting', ['learner', 'session'])
def test_a_string_setting_written_as_a_number_is_refused(setting):
    """``str()`` coercion has the flaw ``int()`` coercion has.

    A session labelled 2026 would come back through lyse as the string
    "2026", which is not what the file says and not what a lab filtering on
    the label would write.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[MLOOP]\n{setting} = 2026\n')
    assert str(raised.value) == f'{setting} must be written as a string, got 2026.'


@pytest.mark.parametrize(
    'setting, refused, accepted',
    [
        ('num_buffered_runs', 0, 1),
        ('num_training_runs', -1, 0),
        ('seed', -1, 0),
        ('max_num_runs', 0, 1),
        ('max_num_runs_without_better_params', 0, 1),
    ],
)
def test_a_setting_below_its_floor_is_refused_and_the_floor_itself_loads(
    setting, refused, accepted
):
    """The floors stand between a file and a session that says nothing.

    ``max_num_runs = 0`` is not a clean stop: check_stop is reached only from
    record, so a session that submits nothing never reaches it and never
    explains itself. tests/test_session.py shows what that looks like.
    """
    with pytest.raises(ValueError, match=f'{setting} must be at least'):
        config_module.loads(MINIMAL + f'[MLOOP]\n{setting} = {refused}\n')
    loaded = config_module.loads(MINIMAL + f'[MLOOP]\n{setting} = {accepted}\n')
    assert getattr(loaded, setting) == accepted


@pytest.mark.parametrize(
    'written', ['[1, 2]', '["r", "c", "extra"]', '["r"]'],
)
def test_a_cost_key_that_is_not_two_names_is_refused(written):
    """Two strings, or the uncertainty column is built out of whatever it is.

    ``cost_key = [1, 2]`` gives an uncertainty_key of ``(1, 'u_2')``, and a
    column nothing in the dataframe answers to.
    """
    with pytest.raises(ValueError, match='cost_key must be two strings'):
        config_module.loads(
            MINIMAL.replace('cost_key = ["r", "c"]', f'cost_key = {written}')
        )


def test_a_misspelled_learner_is_refused_while_the_file_is_being_read():
    """Not at build(), which is worker configure with the session starting.

    The per-table check looks at the [LEARNER.<name>] tables, and a file that
    has none of them names its learner only in [MLOOP].
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[MLOOP]\nlearner = "gaussain_process"\n')
    message = str(raised.value)
    assert "unknown learner 'gaussain_process'" in message
    assert 'gaussian_process' in message


# --- one name for one thing ------------------------------------------------


TWO_GROUPS = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = {groups}
[MLOOP_PARAMS.GA.x]
global_name = "ga"
min = 0.0
max = 1.0
[MLOOP_PARAMS.GB.x]
global_name = "gb"
min = 5.0
max = 6.0
"""


def test_one_name_for_two_searched_parameters_is_refused():
    """Both globals would be set from whichever parameter is looked up last.

    Here that drives ``ga``, bounded [0, 1], to a value drawn for ``gb``'s
    range: a shot outside the bounds the file declares.
    """
    with pytest.raises(ValueError, match="'x' names more than one enabled parameter"):
        config_module.loads(TWO_GROUPS.format(groups='["GA", "GB"]'))


def test_a_name_repeated_in_a_group_nobody_switched_on_still_loads():
    """Switching between groups is what the group names are for.

    Two groups holding a parameter of the same name are two settings for one
    knob, one of which is in force; only having both switched on at once is a
    collision.
    """
    config = config_module.loads(TWO_GROUPS.format(groups='["GB"]'))
    assert [p.name for p in config.space.parameters] == ['x']
    assert config.globals_for([5.5]) == {'gb': 5.5}


@pytest.mark.parametrize(
    'text',
    [
        MINIMAL + '[RUNMANAGER_GLOBALS.G.gx]\nexpr = "lambda v: 2 * v"\nargs = ["x"]\n',
        """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
[MLOOP_PARAMS.G.y]
global_name = "gx"
min = 0.0
max = 1.0
""",
    ],
    ids=['an expression over a parameter already mapped directly', 'two parameters'],
)
def test_one_runmanager_global_set_from_two_places_is_refused(text):
    """A global is set once per shot, so the second mapping is the only one
    that happens, and one of the searched dimensions never reaches the
    apparatus at all."""
    with pytest.raises(ValueError, match="'gx' names more than one runmanager global"):
        config_module.loads(text)


# --- an expression and the arguments it is given ---------------------------


@pytest.mark.parametrize(
    'text, named',
    [
        (MINIMAL + '[RUNMANAGER_GLOBALS.G.empty]\nargs = []\n', 'names 0'),
        (
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
min = 0.0
max = 1.0
[MLOOP_PARAMS.G.y]
min = 0.0
max = 1.0
[RUNMANAGER_GLOBALS.G.pair]
args = ["x", "y"]
""",
            'names 2',
        ),
    ],
    ids=['none', 'two'],
)
def test_a_global_with_no_expression_takes_exactly_one_parameter(text, named):
    """With no expr the global is its one argument, passed through.

    Given none it raises an IndexError inside globals_for, mid-session; given
    two it drops the second, so a searched parameter goes nowhere.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    assert named in str(raised.value)


def test_an_expression_that_cannot_take_its_parameters_stops_the_load():
    """Which is the whole reason the expression is compiled while reading.

    Left to the first proposal, this is a TypeError out of globals_for hours
    later, through the worker's error path.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
min = 0.0
max = 1.0
[MLOOP_PARAMS.G.y]
min = 0.0
max = 1.0
[RUNMANAGER_GLOBALS.G.pair]
expr = "lambda a: a"
args = ["x", "y"]
"""
        )
    message = str(raised.value)
    assert 'pair' in message
    assert 'lambda a: a' in message


def test_an_expression_whose_signature_cannot_be_read_is_taken_as_written():
    """``inspect.signature`` has no answer for some callables.

    Refusing on the absence of an answer would refuse a mapping that works,
    so the arity check stands aside where it cannot see.
    """
    config = config_module.loads(
        """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[MLOOP_PARAMS.G.x]
min = 0.0
max = 1.0
[MLOOP_PARAMS.G.y]
min = 0.0
max = 1.0
[RUNMANAGER_GLOBALS.G.larger]
expr = "max"
args = ["x", "y"]
"""
    )
    assert config.globals_for([0.1, 0.9]) == {'larger': 0.9}
