import numpy as np
import pytest

from labscript_optimization.observations import (
    DROPPED,
    PENDING,
    Observation,
    best,
    costs_array,
    params_array,
    uncers_array,
    usable,
)

from conftest import observe


def test_unusable_observations_are_excluded_from_fits():
    history = [
        observe('good', [1.0], 5.0),
        observe('bad_flag', [2.0], 1.0, bad=True),
        observe('infinite', [3.0], float('inf')),
        observe('not_a_number', [4.0], float('nan')),
        observe('better', [5.0], 2.0),
    ]
    assert [o.shot_id for o in usable(history)] == ['good', 'better']


def test_a_proposal_that_has_produced_no_cost_cannot_inform_a_fit():
    """A shot still running and one that will never report are both positions
    the session spent, and neither is something to fit to.
    """
    history = [
        observe('waiting', [1.0], None, state=PENDING),
        observe('gone', [2.0], None, state=DROPPED),
        observe('measured', [3.0], 4.0),
    ]
    assert [o.shot_id for o in usable(history)] == ['measured']
    assert best(history).shot_id == 'measured'


def test_best_is_the_lowest_usable_cost():
    history = [
        observe('a', [1.0], 5.0),
        observe('b', [2.0], 0.5, bad=True),
        observe('c', [3.0], 2.0),
    ]
    assert best(history).shot_id == 'c'


def test_best_of_an_empty_history_is_none():
    assert best([]) is None
    assert best([observe('a', [1.0], float('nan'))]) is None


def test_uncertainties_are_absent_unless_some_observation_has_one():
    assert uncers_array([observe('a', [1.0], 1.0)]) is None


def test_observations_without_an_uncertainty_are_treated_as_exact():
    history = [observe('a', [1.0], 1.0, uncer=0.25), observe('b', [2.0], 2.0)]
    np.testing.assert_array_equal(uncers_array(history), [0.25, 0.0])


def test_arrays_keep_the_order_of_the_history():
    history = [observe('a', [1.0, 2.0], 9.0), observe('b', [3.0, 4.0], 8.0)]
    np.testing.assert_array_equal(params_array(history), [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(costs_array(history), [9.0, 8.0])


def test_a_completed_shot_with_no_cost_is_refused():
    """Both defaults taken at their word describe nothing: a shot that ran and
    has no cost is bad, and carries NaN.
    """
    with pytest.raises(ValueError, match='complete and carries no cost'):
        Observation('s', np.zeros(2), None)


def test_a_proposal_with_no_cost_yet_is_not_usable():
    assert not Observation('s', np.zeros(2), None, state=PENDING).usable


def test_a_proposal_that_will_never_report_is_not_usable():
    assert not Observation('s', np.zeros(2), None, state=DROPPED).usable


def test_a_completed_shot_with_a_finite_cost_is_usable():
    assert Observation('s', np.zeros(2), 1.5).usable


def test_a_completed_shot_with_a_nan_cost_is_not_usable():
    assert not Observation('s', np.zeros(2), float('nan'), bad=True).usable
