"""Checks that hold a setting to the kind it was written as.

A learner's constructor is the schema for its ``[LEARNER.<name>]`` table, so
each knob is checked there, where a learner built in Python meets the same check
as one built from a file. A wrong kind is refused, not converted: ``bool("false")``
is true and ``int(8.9)`` is 8. Each check returns the value, as the kind it
names, or raises ``ValueError`` naming the setting.

The configuration's own settings use the same checks. They live here because
:mod:`~labscript_optimization.space` resolves ``trust_region`` and cannot import
:mod:`~labscript_optimization.config`, which imports it.
"""

import numbers
from collections.abc import Collection

import numpy as np


def is_number(value) -> bool:
    """Whether ``value`` is a real number; numpy scalars count, booleans do not."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_numbers(value) -> bool:
    """Whether ``value`` is a list, tuple or one-dimensional array of numbers."""
    # ndim is checked before iterating, which a zero-dimensional array cannot do.
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and getattr(value, "ndim", 1) == 1
        and all(is_number(v) for v in value)
    )


def reject_unknown(table: dict, allowed: frozenset[str], where: str) -> None:
    """Raise for any key of ``table`` not in ``allowed``, naming ``where``."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(
            f"{where} does not accept {', '.join(repr(k) for k in unknown)}. "
            f"It accepts: {', '.join(sorted(allowed)) or 'nothing'}."
        )


def boolean(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be written as true or false, unquoted, not {value!r}."
        )
    return value


def integer(name: str, value) -> int:
    # bool is an Integral, so it is excluded by name.
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be written as a whole number, got {value!r}.")
    return int(value)


def string(name: str, value) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be written as a string in quotes, not {value!r}."
        )
    return value


def array(name: str, value) -> list:
    # A bare string is a sequence of characters, so only a list passes.
    if not isinstance(value, list):
        raise ValueError(
            f"{name} must be written as a list in [brackets], not {value!r}."
        )
    return value


def number(name: str, value) -> float:
    """``value`` as a ``float``."""
    if not is_number(value):
        raise ValueError(
            f"{name} must be written as a number, unquoted, not {value!r}."
        )
    return float(value)


def choice(name: str, value, choices: Collection[str]) -> str:
    # A string first, so an unhashable value is refused here, not in a lookup.
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}")
    return value


def pair(name: str, value) -> tuple[float, float]:
    """``value`` as two floats. Only the shape is checked, not the order."""
    if not is_numbers(value) or len(value) != 2:
        raise ValueError(
            f"{name} must be written as a pair of numbers, [low, high], not "
            f"{value!r}."
        )
    return float(value[0]), float(value[1])
