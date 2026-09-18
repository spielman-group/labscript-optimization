import numpy as np
import pytest

from labscript_optimization.space import Parameter, ParameterSpace


def test_disabled_parameters_are_not_searched():
    space = ParameterSpace(
        [
            Parameter('a', 'g_a', 0.0, 1.0),
            Parameter('b', 'g_b', 0.0, 1.0, enable=False),
        ]
    )
    assert space.num_params == 1
    assert space.global_names == ('g_a',)
    assert [p.name for p in space.disabled] == ['b']


def test_a_space_with_nothing_enabled_is_rejected():
    with pytest.raises(ValueError, match='no enabled parameters'):
        ParameterSpace([Parameter('a', 'g_a', 0.0, 1.0, enable=False)])


@pytest.mark.parametrize(
    'kwargs, message',
    [
        (dict(minimum=1.0, maximum=1.0), 'min >= max'),
        (dict(minimum=2.0, maximum=1.0), 'min >= max'),
        (dict(minimum=0.0, maximum=float('inf')), 'finite bounds'),
        (dict(minimum=0.0, maximum=1.0, start=2.0), 'outside'),
    ],
)
def test_impossible_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Parameter('p', 'g_p', **kwargs)


def test_scaling_maps_the_bounds_onto_the_unit_cube(space):
    np.testing.assert_allclose(space.scale(space.minimum), [0.0, 0.0])
    np.testing.assert_allclose(space.scale(space.maximum), [1.0, 1.0])
    point = np.array([1.25, -3.0])
    np.testing.assert_allclose(space.unscale(space.scale(point)), point)


def test_uniform_draws_stay_inside_the_bounds(space, rng):
    assert space.contains(space.uniform(rng, 500)).all()


def test_a_start_is_only_offered_when_every_parameter_has_one():
    both = ParameterSpace(
        [
            Parameter('a', 'g_a', 0.0, 1.0, start=0.5),
            Parameter('b', 'g_b', 0.0, 1.0, start=0.25),
        ]
    )
    np.testing.assert_allclose(both.start, [0.5, 0.25])

    partial = ParameterSpace(
        [Parameter('a', 'g_a', 0.0, 1.0, start=0.5), Parameter('b', 'g_b', 0.0, 1.0)]
    )
    assert partial.start is None


def test_a_fractional_trust_region_scales_with_each_parameter():
    space = ParameterSpace(
        [Parameter('a', 'g_a', 0.0, 100.0), Parameter('b', 'g_b', 0.0, 1.0)]
    )
    np.testing.assert_allclose(space.absolute_trust_region(0.05), [5.0, 0.05])


def test_no_trust_region_means_the_whole_space(space):
    assert space.absolute_trust_region(None) is None


@pytest.mark.parametrize(
    'value, message',
    [
        (1.5, r'\(0, 1\)'),
        (0.0, r'\(0, 1\)'),
        ([1.0], 'shape'),
        ([-1.0, 1.0], 'positive'),
        ([50.0, 50.0], 'wider than the bounds'),
    ],
)
def test_impossible_trust_regions_are_rejected(space, value, message):
    with pytest.raises(ValueError, match=message):
        space.absolute_trust_region(value)
