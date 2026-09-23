"""The configuration schema, and the keys it refuses."""

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from labscript_optimization import config as config_module
from labscript_optimization import learners

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'config_example.toml'

FULL = """
[ANALYSIS]
cost_key = ["zTOF", "Nb"]
maximize = true
groups = ["CMOT", "SHIMS"]

[GENERAL]
num_buffered_runs = 3
num_training_runs = 20
max_num_runs = 400
learner = "gaussian_process"
trainer = "directed_random"

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


def test_a_trainer_and_a_main_learner_each_take_their_own_value_of_a_knob(config):
    """The one knob three learners take, given two values in one session.

    A wide region to train with and a tight one to refine with is the first
    thing anyone running two live learners wants, and it is exactly what a
    single table of knobs cannot express.
    """
    built = learners.build(config)
    wide = config.space.absolute_trust_region(0.2)
    tight = config.space.absolute_trust_region(0.05)
    assert not np.allclose(wide, tight)
    np.testing.assert_allclose(built.trainer.trust_region, wide)
    np.testing.assert_allclose(built.main.trust_region, tight)


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
    """Not accepted and dropped for the learners that do not take it.

    [GENERAL] is read once for a session that runs two learners, so a knob here
    has no way of saying which one it meant.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[GENERAL]\n{knob}\n')
    message = str(raised.value)
    assert knob.split(' =')[0] in message
    for table in tables:
        assert table in message


#: The fifteen knobs ``UPGRADING.md`` §3 tells a lab to move out of ``[GENERAL]``,
#: written as that document's table has them. Hard-coded rather than derived
#: from the learners: derived, this would agree with the loader by
#: construction and say nothing about whether the document is true.
MOVED_KNOBS = {
    'trust_region': '0.05',
    'trust_range': '[0.1, 0.25]',
    'trust_gaussian': 'true',
    'explore_fraction': '0.1',
    'cost_has_noise': 'true',
    'population_size': '8',
    'evolution_strategy': '"best1"',
    'mutation_scale': '0.7',
    'cross_over_probability': '0.9',
    'cost_bias': '1.0',
    'uncer_bias': '1.0',
    'refit_interval': '4',
    'length_scale_bounds': '[1e-2, 1e2]',
    'noise_level_bounds': '[1e-5, 1e1]',
    'minimum_observations': '6',
}


@pytest.mark.parametrize('knob, value', sorted(MOVED_KNOBS.items()))
def test_every_knob_the_upgrade_document_moves_is_refused_in_general(knob, value):
    """The lab reads that document, not this file.

    A row of it the loader still accepts is a setting the document says has
    moved and the loader takes where it was, which is the belief the whole
    schema exists to prevent.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[GENERAL]\n{knob} = {value}\n')
    message = str(raised.value)
    assert knob in message
    assert '[LEARNER.' in message


def test_the_learner_is_named_in_the_general_table():
    config = config_module.loads(
        MINIMAL + '[GENERAL]\nlearner = "differential_evolution"\n'
    )
    assert config.learner == 'differential_evolution'


# --- the trainer -----------------------------------------------------------


def test_the_trainer_is_named_in_the_general_table():
    """Which learner runs the training shots, and the periodic ones after them.

    It decides where the first shots of every run land, so it is the lab's to
    choose rather than this package's to fix.
    """
    named = config_module.loads(MINIMAL + '[GENERAL]\ntrainer = "random"\n')
    assert type(learners.build(named).trainer) is learners.RandomLearner
    # And what a file naming none gets: the band around middling costs.
    unnamed = learners.build(config_module.loads(MINIMAL))
    assert type(unnamed.trainer) is learners.DirectedRandomLearner


def test_an_unknown_trainer_is_refused_at_load():
    """Rather than at worker configure, with the apparatus already running."""
    with pytest.raises(ValueError, match="unknown trainer 'directed_randon'"):
        config_module.loads(MINIMAL + '[GENERAL]\ntrainer = "directed_randon"\n')


def test_a_trainer_named_for_a_learner_that_needs_none_is_refused():
    """No trainer is built for such a learner, so the name would name nothing.

    A file that carried it would read as though the first shots came from
    somewhere they do not.
    """
    written = MINIMAL + '[GENERAL]\nlearner = "differential_evolution"\n'
    with pytest.raises(ValueError, match='trainer is not accepted') as raised:
        config_module.loads(written + 'trainer = "random"\n')
    # And the learner it could have been named for.
    assert 'gaussian_process' in str(raised.value)
    assert config_module.loads(written).learner == 'differential_evolution'


def test_the_learner_m_loop_trained_with_cannot_train_here():
    """M-LOOP's machine-learning controllers defaulted to differential
    evolution for the training shots and its periodic runs both. Here that learner
    proposes a whole population at a time and only when none of its proposals
    is outstanding, which a two-phase learner cannot hold a barrier for, so the
    file is refused rather than silently cutting a generation at the handover.
    """
    with pytest.raises(ValueError, match='whole generations of 8') as raised:
        config_module.loads(MINIMAL + '[GENERAL]\ntrainer = "differential_evolution"\n')
    assert 'Name a trainer' in str(raised.value)


def test_how_often_the_trainer_comes_back_is_named_in_the_general_table():
    """It configures the arrangement of the two learners, not either of them.

    Neither learner's constructor could take it: the trainer does not know it
    is behind a main learner, and the main learner does not know how often it
    is stood down. That is what ``[GENERAL]`` carries, and why this is not a
    knob in a ``[LEARNER.<name>]`` table.
    """
    named = config_module.loads(
        MINIMAL + '[GENERAL]\nnum_runs_between_trainer_runs = 4\n'
    )
    assert learners.build(named).num_runs_between_trainer_runs == 4
    # And what a file naming none gets: a run that never goes back.
    unnamed = learners.build(config_module.loads(MINIMAL))
    assert unnamed.num_runs_between_trainer_runs is None


def test_how_often_the_trainer_comes_back_is_refused_where_there_is_no_trainer():
    """No trainer is built for such a learner, so there is nothing to come back.

    Refused beside the trainer's own name and in the same sentence, because a
    file switching learners usually carries both and would otherwise be
    refused twice over.
    """
    written = MINIMAL + '[GENERAL]\nlearner = "differential_evolution"\n'
    with pytest.raises(
        ValueError, match='num_runs_between_trainer_runs is not accepted'
    ):
        config_module.loads(written + 'num_runs_between_trainer_runs = 4\n')
    with pytest.raises(
        ValueError, match='num_runs_between_trainer_runs and trainer are not accepted'
    ):
        config_module.loads(
            written + 'num_runs_between_trainer_runs = 4\ntrainer = "random"\n'
        )


def test_a_warmup_shorter_than_the_learner_needs_stops_the_load():
    """Rather than at worker configure, with the apparatus already running.

    MINIMAL searches one parameter, so the Gaussian process it builds will not
    fit below two usable observations. A file asking to hand over after one is
    asking for a handover that would happen after two.
    """
    with pytest.raises(ValueError, match='num_training_runs is 1') as raised:
        config_module.loads(MINIMAL + '[GENERAL]\nnum_training_runs = 1\n')
    assert 'holds 2 usable observations' in str(raised.value)
    loaded = config_module.loads(MINIMAL + '[GENERAL]\nnum_training_runs = 2\n')
    assert loaded.num_training_runs == 2


def test_a_trainer_that_will_not_propose_from_an_empty_history_is_refused():
    """The trainer proposes the first shot of the run, from an empty history,
    so there is nothing behind it to propose instead.
    """
    with pytest.raises(ValueError, match='will not propose until') as raised:
        config_module.loads(MINIMAL + '[GENERAL]\ntrainer = "gaussian_process"\n')
    assert 'begins with none' in str(raised.value)


def test_a_setting_left_out_takes_the_value_the_documents_promise():
    """What a lab may leave out on the strength of what it was told.

    UPGRADING §5 promises three buffered runs, the Gaussian process is the
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


@pytest.mark.parametrize(
    'load',
    [
        f'config.loads({MINIMAL!r})',
        f'config.loads({MINIMAL + "[LEARNER.gaussian_process]\nrefit_interval = 4\n"!r})',
        f'config.load({str(EXAMPLE)!r})',
    ],
    ids=['a minimal file', 'a table naming the gaussian process', 'the example'],
)
def test_loading_configuration_does_not_import_scientific_learners(load):
    """A lyse routine that never asks for a proposal pays no learner imports.

    Every file here resolves a learner class -- the selected one always, and a
    named table's own as well -- because a constructor is the schema for its
    table; and every file builds the selected learner, because how many
    proposals it makes at a time is a fact about an instance. So the property
    holds of what a learner module imports when it is imported and of what its
    constructor reaches, and not merely of the learners a file leaves alone.
    """
    script = (
        "import sys\n"
        "from labscript_optimization import config\n"
        f"{load}\n"
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
[PARAMETERS.G.orphan]
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
[PARAMETERS.G.x]
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
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
"""
        )


def test_a_listed_group_that_no_table_defines_is_refused_naming_it():
    """A misspelt group matches nothing, and the group it was meant to name is
    then one nobody listed: switched off, its globals never set, while the lab
    believes its parameters are being searched.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL.replace('groups = ["G"]', 'groups = ["G", "SHIMSS", "CMTO"]')
            + '[PARAMETERS.SHIMS.b]\nglobal_name = "gb"\nmin = 0.0\nmax = 1.0\n'
        )
    assert str(raised.value) == (
        "ANALYSIS.groups lists 'SHIMSS', 'CMTO', which no [PARAMETERS.<group>] "
        "or [RUNMANAGER_GLOBALS.<group>] table defines. A group takes part "
        "only through the tables written under it, so a name with none is a "
        "misspelling. The groups this file defines are ['G', 'SHIMS']."
    )


def test_a_group_with_only_globals_in_it_is_defined():
    """A group need carry no parameter of its own: one that only computes a
    global from parameters elsewhere is written under ``RUNMANAGER_GLOBALS``
    alone, and listing it switches that global on.
    """
    config = config_module.loads(
        MINIMAL.replace('groups = ["G"]', 'groups = ["G", "DERIVED"]')
        + '[RUNMANAGER_GLOBALS.DERIVED.doubled]\n'
        + 'expr = "lambda v: 2 * v"\nargs = ["x"]\n'
    )
    assert [g.name for g in config.globals] == ['gx', 'doubled']


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
[PARAMETERS.G.x]
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
            MINIMAL + f'[PARAMETERS.G.y]\nglobal_name = "gy"\n{kept}\n'
        )
    message = str(raised.value)
    assert 'PARAMETERS.G.y' in message
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
            'PARAMETERS.G.x',
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
        config_module.loads(MINIMAL + '[GENERAL]\nmax_num_runs = "many"\n')
    assert str(raised.value) == (
        "max_num_runs must be written as a whole number, got 'many'."
    )


@pytest.mark.parametrize(
    'text, spelling, named',
    [
        (
            MINIMAL.replace('groups', 'ignore_bad = true\ngroups'),
            'ignore_bad',
            '[ANALYSIS]',
        ),
        (
            MINIMAL.replace('groups', 'analysislib_console_log_level = "DEBUG"\ngroups'),
            'analysislib_console_log_level',
            '[ANALYSIS]',
        ),
        (
            MINIMAL.replace('groups', 'analysislib_file_log_level = "DEBUG"\ngroups'),
            'analysislib_file_log_level',
            '[ANALYSIS]',
        ),
        (
            MINIMAL + '[GENERAL]\ncontroller_type = "differential_evolution"\n',
            'controller_type',
            '[GENERAL]',
        ),
        (MINIMAL + '[GENERAL]\nsession = "run-a"\n', 'session', '[GENERAL]'),
        (MINIMAL + '[GENERAL]\nno_delay = true\n', 'no_delay', '[GENERAL]'),
        (
            MINIMAL + '[GENERAL]\nvisualisations = false\n',
            'visualisations',
            '[GENERAL]',
        ),
        (
            MINIMAL + '[GENERAL]\nconsole_log_level = "DEBUG"\n',
            'console_log_level',
            '[GENERAL]',
        ),
        (
            MINIMAL + '[GENERAL]\nconsole_log_string = "%(message)s"\n',
            'console_log_string',
            '[GENERAL]',
        ),
        (MINIMAL + '[GENERAL]\narchive_type = "txt"\n', 'archive_type', '[GENERAL]'),
        (
            MINIMAL + '[GENERAL]\nrestart_tolerance = 0.01\n',
            'restart_tolerance',
            '[GENERAL]',
        ),
        (
            MINIMAL + '[LEARNER.differential_evolution]\nrestart_tolerance = 0.01\n',
            'restart_tolerance',
            '[LEARNER.differential_evolution]',
        ),
        (MINIMAL + '[GENERAL]\ngeneration_size = 4\n', 'generation_size', '[GENERAL]'),
        (
            MINIMAL + '[LEARNER.gaussian_process]\ngeneration_size = 4\n',
            'generation_size',
            '[LEARNER.gaussian_process]',
        ),
        (
            MINIMAL + '[PARAMETERS.G.y]\nminimum = 2.0\n',
            'minimum',
            '[PARAMETERS.G.y]',
        ),
        (
            MINIMAL + '[PARAMETERS.G.y]\nmaximum = 2.0\n',
            'maximum',
            '[PARAMETERS.G.y]',
        ),
        (MINIMAL + '[COMPILATION]\nmock = false\n', 'COMPILATION', 'the top level'),
    ],
    ids=[
        'ANALYSIS.ignore_bad',
        'ANALYSIS.analysislib_console_log_level',
        'ANALYSIS.analysislib_file_log_level',
        'GENERAL.controller_type',
        'GENERAL.session',
        'GENERAL.no_delay',
        'GENERAL.visualisations',
        'GENERAL.console_log_level',
        'GENERAL.console_log_string',
        'GENERAL.archive_type',
        'GENERAL.restart_tolerance',
        'LEARNER.differential_evolution.restart_tolerance',
        'GENERAL.generation_size',
        'LEARNER.gaussian_process.generation_size',
        'PARAMETERS.minimum',
        'PARAMETERS.maximum',
        'COMPILATION',
    ],
)
def test_every_setting_the_upgrade_document_retires_is_refused(text, spelling, named):
    """``UPGRADING.md`` §2 is the list a lab upgrades against: cut these keys,
    and the file loads. One of them still accepted is a setting the document
    says is gone and the loader takes, which is the belief the whole schema
    exists to prevent -- and the lab reads the document, not this file, so a
    row of it that nothing checks is a promise nobody has tested.

    Each is written into the table the document names it in: the same spelling
    can be a setting in one table and meaningless in another, and
    ``[LEARNER.<name>]`` is held to its learner's constructor rather than to
    this module's lists.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    message = str(raised.value)
    assert spelling in message
    assert named in message


@pytest.mark.parametrize(
    'text, old, new, other',
    [
        (
            MINIMAL + '[MLOOP]\nlearner = "random"\n',
            '[MLOOP]',
            '[GENERAL]',
            '[PARAMETERS]',
        ),
        (
            MINIMAL.replace('[PARAMETERS.', '[MLOOP_PARAMS.'),
            '[MLOOP_PARAMS]',
            '[PARAMETERS]',
            '[GENERAL]',
        ),
    ],
    ids=['MLOOP', 'MLOOP_PARAMS'],
)
def test_a_table_named_after_mloop_is_refused_naming_what_replaces_it(
    text, old, new, other
):
    """This package replaces M-LOOP and carries none of its code, so the two
    tables named after it are gone. Nothing is preserved: read under the new
    name, a file written for the old one would load and mean something, and a
    lab would go on typing the name of a tool it is not running.

    The message has to pair the table with its own replacement. Left to the
    top-level spelling check, an old name is a typo and the reader is handed
    every table this package accepts to choose from.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    message = str(raised.value)
    assert old in message
    assert new in message
    assert other not in message


def test_a_file_using_both_old_table_names_is_told_about_both_at_once():
    """An M-LOOP file has both, so reporting one is two loads and two edits."""
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL.replace('[PARAMETERS.', '[MLOOP_PARAMS.')
            + '[MLOOP]\nlearner = "random"\n'
        )
    message = str(raised.value)
    assert '[MLOOP] is now [GENERAL]' in message
    assert '[MLOOP_PARAMS] is now [PARAMETERS]' in message


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
        config_module.loads(MINIMAL + '[PARAMETERS.OFF.y]\nmin = 0.0\nmaxx = 1.0\n')


def test_a_per_learner_table_rejects_a_knob_that_learner_does_not_take():
    """A named table has one constructor that can define its valid keys.

    A learner whose constructor takes nothing but the space and the rng is
    the case the message has to be written for: listing what its table
    accepts would list nothing, and a sentence that trails off reads as one
    that failed rather than as the answer.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + '[LEARNER.random]\ncost_has_noise = true\n'
        )
    assert str(raised.value) == (
        "[LEARNER.random] does not accept 'cost_has_noise'. That learner "
        "takes no knobs at all, so its table holds nothing and is as well "
        "left out."
    )

    with pytest.raises(ValueError) as raised:
        config_module.loads(
            MINIMAL + '[LEARNER.differential_evolution]\ncost_has_noise = true\n'
        )
    assert str(raised.value).endswith(
        "It accepts: cross_over_probability, evolution_strategy, "
        "mutation_scale, population_size, trust_region."
    )


def test_an_unknown_per_learner_table_is_rejected():
    with pytest.raises(ValueError, match="unknown learner 'typo'"):
        config_module.loads(MINIMAL + '[LEARNER.typo]\ntrust_region = 0.2\n')


def test_a_per_learner_table_accepts_that_learners_knob():
    config = config_module.loads(
        MINIMAL + '[LEARNER.directed_random]\ntrust_region = 0.2\n'
    )
    assert config.learner_options['directed_random'] == {'trust_region': 0.2}


@pytest.mark.parametrize(
    'text, message',
    [
        (
            MINIMAL + '[LEARNER.gaussian_process]\ncost_has_noise = "false"\n',
            "cost_has_noise must be written as true or false, unquoted, not "
            "'false'.",
        ),
        (
            MINIMAL + '[LEARNER.directed_random]\ntrust_gaussian = "false"\n',
            "trust_gaussian must be written as true or false, unquoted, not "
            "'false'.",
        ),
        (
            MINIMAL + '[LEARNER.gaussian_process]\nrefit_interval = true\n',
            "refit_interval must be written as a whole number, got True.",
        ),
        (
            MINIMAL + '[LEARNER.gaussian_process]\nlength_scale_bounds = 5\n',
            "length_scale_bounds must be written as a pair of numbers, [low, "
            "high], not 5.",
        ),
        (
            MINIMAL + '[LEARNER.gaussian_process]\ntrust_region = "0.1"\n',
            "trust_region must be written as a number in (0, 1), a fraction of "
            "each parameter's range, or as a list of numbers, one distance per "
            "parameter, not '0.1'.",
        ),
        (
            MINIMAL + '[LEARNER.directed_random]\ntrust_range = 0.5\n',
            "trust_range must be written as a pair of numbers, [low, high], "
            "not 0.5.",
        ),
        (
            MINIMAL
            + '[GENERAL]\nlearner = "differential_evolution"\n'
            + '[LEARNER.differential_evolution]\npopulation_size = 8.9\n',
            "population_size must be written as a whole number, got 8.9.",
        ),
        (
            MINIMAL
            + '[GENERAL]\nlearner = "differential_evolution"\n'
            + '[LEARNER.differential_evolution]\nmutation_scale = 0.8\n',
            "mutation_scale is the range the differential weight is drawn "
            "from, [low, high], not one weight: for a fixed weight of 0.8, "
            "write [0.8, 0.8].",
        ),
    ],
    ids=[
        'cost_has_noise',
        'trust_gaussian',
        'refit_interval',
        'length_scale_bounds',
        'trust_region',
        'trust_range',
        'population_size',
        'mutation_scale',
    ],
)
def test_a_learner_knob_of_the_wrong_kind_stops_the_load(text, message):
    """``[GENERAL]`` is held to its kinds by :class:`Config`, and a learner's
    table by that learner's constructor, which the load builds. Either way a
    quoted boolean, a truncated fraction or a lone number where a pair belongs
    is refused before anything runs, rather than acting as a setting nobody
    wrote.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(text)
    assert str(raised.value) == message


def test_the_example_configuration_loads_and_builds_its_learner():
    """The file every new lab starts from, held to the schema like any other."""
    config = config_module.load(EXAMPLE)
    learner = learners.build(config)
    assert isinstance(learner, learners.TwoPhaseLearner)
    assert learner.num_training == 20


# --- the Gaussian process's refit interval ---------------------------------


def test_refit_interval_is_the_gaussian_process_knob():
    config = config_module.loads(
        MINIMAL + '[LEARNER.gaussian_process]\nrefit_interval = 6\n'
    )
    assert learners.build(config).main.refit_interval == 6


def test_the_old_spelling_of_the_refit_interval_is_refused():
    """It named a batch, and there is no batch left for it to name.

    The key set the period of the exploration schedule as well, which
    uncer_bias now holds, so a file still writing batch_size is asking for a
    schedule it would not get. Taken quietly it would set the refit interval
    and say nothing about the half of its meaning that had gone.
    """
    with pytest.raises(ValueError, match='batch_size'):
        config_module.loads(
            MINIMAL + '[LEARNER.gaussian_process]\nbatch_size = 6\n'
        )


# --- the budget and the population -----------------------------------------


DE = MINIMAL + '[GENERAL]\nlearner = "differential_evolution"\n'


def test_a_generational_learner_will_not_take_a_queue_depth_as_well():
    """Its queue depth is its population, so a second setting for the same
    number is one that can disagree with it.
    """
    with pytest.raises(ValueError, match='num_buffered_runs') as raised:
        config_module.loads(DE + 'num_buffered_runs = 3\n')
    # The depth it would have set instead, so the reader can see the two
    # numbers that would have disagreed.
    assert 'generation of 8' in str(raised.value)

    # A learner that proposes any number at a time still takes one.
    other = config_module.loads(MINIMAL + '[GENERAL]\nnum_buffered_runs = 3\n')
    assert other.num_buffered_runs == 3


def test_a_generation_a_constructor_assigns_is_refused_a_queue_depth_too(monkeypatch):
    """Assigning in ``__init__`` is the ordinary way to declare one.

    Off the class it is the base class's ``None``, and the depth goes
    through: a file then carries a queue depth beside a learner that proposes
    whole generations whatever the depth, a setting read back off the file as
    one thing and acted on as nothing.
    """

    class Ordinary(learners.ParameterSpaceLearner):
        def __init__(self, space, rng, population_size=8):
            super().__init__(space, rng)
            self.generation = int(population_size)

        def propose(self, history, hint):
            return [(p, 'main') for p in self.space.uniform(self.rng, hint)]

    assert Ordinary.generation is None
    monkeypatch.setitem(learners.LEARNERS, 'ordinary', Ordinary)

    written = MINIMAL + '[GENERAL]\nlearner = "ordinary"\n'
    with pytest.raises(ValueError, match='num_buffered_runs') as raised:
        config_module.loads(written + 'num_buffered_runs = 3\n')
    assert 'generation of 8' in str(raised.value)
    # The same learner without the setting is the file that loads.
    assert config_module.loads(written).learner == 'ordinary'


def test_a_generational_learner_behind_a_trainer_is_refused_at_load(monkeypatch):
    """Rather than at worker configure, with the apparatus already running.

    A two-phase learner can neither answer for a generation nor pass one on,
    so it refuses to wrap one. Building the learner at load is what brings
    that refusal forward to the file that asks for the combination.
    """
    monkeypatch.setattr(
        learners, 'NEEDS_TRAINING', frozenset({'differential_evolution'})
    )
    with pytest.raises(ValueError, match='whole generations of 8'):
        config_module.loads(DE)


@pytest.mark.parametrize(
    'sized, refused, accepted',
    [
        ('', 15, 16),
        ('[LEARNER.differential_evolution]\npopulation_size = 5\n', 9, 10),
    ],
    ids=['the default population', 'a population the file sizes'],
)
def test_a_budget_below_two_whole_generations_is_refused(sized, refused, accepted):
    """One generation is the population itself; the second is the first to
    evolve it, and a configuration that cannot reach it cannot do what it
    says."""
    with pytest.raises(ValueError, match='max_num_runs') as raised:
        config_module.loads(DE + f'max_num_runs = {refused}\n' + sized)
    # Not "nothing evolves below this": between one population and two, a
    # generation cut short does evolve some of its slots.
    assert 'cut short' in str(raised.value)
    assert config_module.loads(
        DE + f'max_num_runs = {accepted}\n' + sized
    ).max_num_runs == accepted


def test_the_budget_is_measured_against_the_generation_a_learner_declares(monkeypatch):
    """A learner is free to derive what it proposes at a time.

    What it derived is how many it proposes at a time, and so what two
    generations of it will cost. Predicted from the constructor's default and
    the file's option, the budget would be measured against a number nobody
    runs, and the file that cannot reach its second generation would load.
    """

    class Doubling(learners.ParameterSpaceLearner):
        def __init__(self, space, rng, population_size=4):
            super().__init__(space, rng)
            # Whatever it is handed, it evolves two members per slot.
            self.generation = 2 * int(population_size)

        def propose(self, history, hint):
            return [(p, 'main') for p in self.space.uniform(self.rng, hint)]

    monkeypatch.setitem(learners.LEARNERS, 'doubling', Doubling)

    written = MINIMAL + '[GENERAL]\nlearner = "doubling"\n'
    sized = '[LEARNER.doubling]\npopulation_size = 4\n'
    with pytest.raises(ValueError, match='max_num_runs') as raised:
        config_module.loads(written + 'max_num_runs = 15\n' + sized)
    assert 'cut short' in str(raised.value)
    assert config_module.loads(
        written + 'max_num_runs = 16\n' + sized
    ).max_num_runs == 16


# --- what a setting may be -------------------------------------------------
#
# Config.__post_init__ is the single authority for the dataclass's own fields,
# so these hold for a Config written out in a script as much as for a file.


@pytest.mark.parametrize(
    'setting, written',
    [
        ('num_buffered_runs', '2.9'),
        ('num_training_runs', '19.5'),
        ('num_runs_between_trainer_runs', '4.5'),
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
        config_module.loads(MINIMAL + f'[GENERAL]\n{setting} = {written}\n')
    assert str(raised.value) == (
        f'{setting} must be written as a whole number, got {float(written)!r}.'
    )


@pytest.mark.parametrize(
    'setting',
    [
        'num_buffered_runs',
        'num_training_runs',
        'num_runs_between_trainer_runs',
        'max_num_runs',
        'seed',
    ],
)
def test_a_whole_number_setting_written_as_a_boolean_is_refused(setting):
    """``isinstance(True, int)`` is True, so a bare integer check takes it as 1.

    Which is a plausible number for every one of these, and so a session that
    runs on a setting nobody wrote.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[GENERAL]\n{setting} = true\n')
    assert str(raised.value) == (
        f'{setting} must be written as a whole number, got True.'
    )


@pytest.mark.parametrize('setting', ['learner', 'trainer'])
def test_a_string_setting_written_as_a_number_is_refused(setting):
    """``str()`` coercion has the flaw ``int()`` coercion has.

    A number here is a file that forgot the quotes. Coerced, it would reach
    the lookup that turns this name into a learner as "2026" and be refused
    there for naming no learner -- a complaint about the name, when the fault
    is the quotes.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + f'[GENERAL]\n{setting} = 2026\n')
    assert str(raised.value) == f'{setting} must be written as a string, got 2026.'


@pytest.mark.parametrize(
    'setting, refused, accepted, alongside',
    [
        ('num_buffered_runs', 0, 1, ''),
        # No training shots at all is a floor the dataclass holds, and a file
        # reaches it only beside a learner that asks for no warmup: the
        # Gaussian process MINIMAL would otherwise build refuses a warmup
        # shorter than the two usable observations it needs for one parameter.
        ('num_training_runs', -1, 0, 'learner = "random"\n'),
        ('num_runs_between_trainer_runs', 0, 1, ''),
        ('seed', -1, 0, ''),
        ('max_num_runs', 0, 1, ''),
        ('max_num_runs_without_better_params', 0, 1, ''),
    ],
)
def test_a_setting_below_its_floor_is_refused_and_the_floor_itself_loads(
    setting, refused, accepted, alongside
):
    """The floors stand between a file and a session that says nothing.

    ``max_num_runs = 0`` is not a clean stop: check_stop is reached only from
    record, so a session that submits nothing never reaches it and never
    explains itself. tests/test_session.py shows what that looks like.
    """
    written = MINIMAL + f'[GENERAL]\n{alongside}'
    with pytest.raises(ValueError, match=f'{setting} must be at least'):
        config_module.loads(written + f'{setting} = {refused}\n')
    loaded = config_module.loads(written + f'{setting} = {accepted}\n')
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
    has none of them names its learner only in [GENERAL].
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[GENERAL]\nlearner = "gaussain_process"\n')
    message = str(raised.value)
    assert "unknown learner 'gaussain_process'" in message
    assert 'gaussian_process' in message


# --- one name for one thing ------------------------------------------------


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


def test_a_start_written_for_some_parameters_stops_the_load():
    """The file is where the mistake is made, so the load is where it stops.

    A start on one parameter of two used to be taken, carried through, and
    then dropped at the point of use, because the opening point is one vector
    over all of them: the run opened on a uniform draw with nothing said. A
    setting accepted and not acted on is what this file refuses everywhere
    else.
    """
    with pytest.raises(ValueError, match="'x' has one; 'y' does not"):
        config_module.loads(
            """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
start = 0.25
[PARAMETERS.G.y]
global_name = "gy"
min = 0.0
max = 1.0
"""
        )


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
[PARAMETERS.G.x]
global_name = "gx"
min = 0.0
max = 1.0
[PARAMETERS.G.y]
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
[PARAMETERS.G.x]
min = 0.0
max = 1.0
[PARAMETERS.G.y]
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
[PARAMETERS.G.x]
min = 0.0
max = 1.0
[PARAMETERS.G.y]
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
[PARAMETERS.G.x]
min = 0.0
max = 1.0
[PARAMETERS.G.y]
min = 0.0
max = 1.0
[RUNMANAGER_GLOBALS.G.larger]
expr = "max"
args = ["x", "y"]
"""
    )
    assert config.globals_for([0.1, 0.9]) == {'larger': 0.9}


def test_a_parameter_written_without_its_group_names_the_depth():
    """The ordinary hand-edit slip: [PARAMETERS.<name>] instead of
    [PARAMETERS.<group>.<name>]. Read at face value its settings are entries
    and its values are not tables, so every check after it reads the wrong
    thing and fails on a type.
    """
    written = """
[ANALYSIS]
cost_key = ["r", "c"]
groups = ["G"]
[PARAMETERS.G]
global_name = "gx"
min = 0.0
max = 1.0
"""
    with pytest.raises(ValueError) as raised:
        config_module.loads(written)
    message = str(raised.value)
    assert '[PARAMETERS.G]' in message
    assert '[PARAMETERS.<group>.<name>]' in message
    assert "'global_name', 'max', 'min'" in message


def test_a_global_written_without_its_group_names_the_depth():
    """``args`` is a list, which is iterable, so the spelling check goes
    through it and blames its first element for being a key.
    """
    with pytest.raises(ValueError) as raised:
        config_module.loads(MINIMAL + '[RUNMANAGER_GLOBALS.G]\nargs = ["x"]\n')
    message = str(raised.value)
    assert '[RUNMANAGER_GLOBALS.G]' in message
    assert '[RUNMANAGER_GLOBALS.<group>.<name>]' in message
    assert "'args'" in message


def test_a_parameter_table_written_as_a_value_names_the_depth():
    """``PARAMETERS = 5`` is a known key at the top level, so nothing above
    this refuses it.
    """
    with pytest.raises(ValueError, match=r'\[PARAMETERS\] is written as 5'):
        config_module.loads(
            'PARAMETERS = 5\n[ANALYSIS]\ncost_key = ["r", "c"]\ngroups = ["G"]\n'
        )


def test_a_learner_table_written_one_level_too_deep_names_the_key():
    """The other direction, and already answered by the learner's own
    constructor: the extra level is read as a knob that learner does not take.
    """
    with pytest.raises(ValueError, match="does not accept 'extra'"):
        config_module.loads(
            MINIMAL + '[LEARNER.gaussian_process.extra]\nanything = 1\n'
        )


def test_a_correctly_nested_file_of_each_kind_still_loads():
    config = config_module.loads(
        MINIMAL
        + '[PARAMETERS.G.y]\nmin = 0.0\nmax = 1.0\n'
        + '[RUNMANAGER_GLOBALS.G.gy]\nexpr = "lambda a: a"\nargs = ["y"]\n'
    )
    assert [p.name for p in config.space.parameters] == ['x', 'y']
    assert sorted(g.name for g in config.globals) == ['gx', 'gy']
