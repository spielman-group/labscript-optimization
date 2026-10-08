from labscript_optimization.observations import DROPPED, PENDING, best, usable

from conftest import observe


def test_only_a_finite_cost_from_a_completed_unflagged_shot_can_inform_a_fit():
    history = [
        observe('good', [1.0], 5.0),
        observe('bad_flag', [2.0], 1.0, bad=True),
        observe('infinite', [3.0], float('inf')),
        observe('not_a_number', [4.0], float('nan')),
        observe('waiting', [5.0], None, state=PENDING),
        observe('gone', [6.0], None, state=DROPPED),
        observe('better', [7.0], 2.0),
    ]
    assert [o.shot_id for o in usable(history)] == ['good', 'better']


def test_best_is_the_lowest_usable_cost_and_none_without_one():
    history = [
        observe('a', [1.0], 5.0),
        observe('b', [2.0], 0.5, bad=True),
        observe('c', [3.0], 2.0),
        observe('d', [4.0], None, state=PENDING),
    ]
    assert best(history).shot_id == 'c'
    assert best([]) is None
    assert best([observe('a', [1.0], float('nan'))]) is None
