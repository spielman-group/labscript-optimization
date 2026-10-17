"""What a setting may be written as, and which keys a table may carry.

A learner's constructor is the schema for its ``[LEARNER.<name>]`` table, and
its keyword arguments arrive exactly as the file wrote them. So each knob is
held to its kind in the constructor, where a learner built directly in Python
meets the same check as one built from a file, and refused rather than
converted: ``bool("false")`` is true, ``int(8.9)`` is 8 and ``float("0.1")``
is 0.1, so a conversion hands the learner a setting nobody wrote, and in TOML
each of those is one pair of quotes or one decimal point away.

The configuration's own settings are held to the same checks, so a file hears
one voice whichever table it got wrong. They live here rather than in
:mod:`~labscript_optimization.config` because
:mod:`~labscript_optimization.space` resolves ``trust_region`` and cannot
import that module, which imports it, and because the learners use them while
that module imports the learners only when it builds one.
"""

import numbers

import numpy as np


def is_number(value) -> bool:
    """Whether ``value`` is a real number, numpy's scalars included.

    A boolean is not one, although ``isinstance(True, numbers.Real)`` says it
    is: taken as a number, ``true`` is a weight of 1.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_numbers(value) -> bool:
    """Whether ``value`` is a list, tuple or one-dimensional array of numbers.

    TOML gives a list, and code building a learner directly passes tuples and
    numpy arrays. A string is none of those, although it is a sequence. An
    array is held to one dimension before it is iterated, because a
    zero-dimensional one cannot be; a list is not asked for its ``ndim`` at
    all, because numpy answers a ragged one with an error of its own.
    """
    return (
        isinstance(value, (list, tuple, np.ndarray))
        and getattr(value, "ndim", 1) == 1
        and all(is_number(v) for v in value)
    )


def reject_unknown(table: dict, allowed: frozenset[str], where: str) -> None:
    """Fail on any key of ``table`` that nothing in this package reads.

    The message names the table as well as the key, because the same spelling
    can be a setting in one table and meaningless in another.
    """
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(
            f"{where} does not accept {', '.join(repr(k) for k in unknown)}. "
            f"It accepts: {', '.join(sorted(allowed)) or 'nothing'}."
        )


def boolean(name: str, value) -> bool:
    """``value`` if it is a boolean, else a refusal naming ``name``."""
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be written as true or false, unquoted, not {value!r}."
        )
    return value


def integer(name: str, value) -> int:
    """``value`` as an ``int`` if it is a whole number, else a refusal.

    ``bool`` is excluded by name: ``isinstance(True, numbers.Integral)`` is
    true, and a bare integer check reads ``true`` as 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be written as a whole number, got {value!r}.")
    return int(value)


def string(name: str, value) -> str:
    """``value`` if it is a string, else a refusal naming ``name``."""
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be written as a string in quotes, not {value!r}."
        )
    return value


def array(name: str, value) -> list:
    """``value`` if it is a list, else a refusal naming ``name``.

    A bare string is a sequence of its own characters, so a check that only
    iterates would take ``"G"`` for a list of one group.
    """
    if not isinstance(value, list):
        raise ValueError(
            f"{name} must be written as a list in [brackets], not {value!r}."
        )
    return value


def number(name: str, value) -> float:
    """``value`` as a ``float`` if it is a number, else a refusal."""
    if not is_number(value):
        raise ValueError(
            f"{name} must be written as a number, unquoted, not {value!r}."
        )
    return float(value)


def choice(name: str, value, choices) -> str:
    """``value`` if it is a string naming one of ``choices``, else a refusal.

    A string first, because the lookup hashes what it is given and a list
    written where a name belongs would fail on that in Python's words.
    """
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {tuple(choices)}, got {value!r}")
    return value


def pair(name: str, value) -> tuple[float, float]:
    """``value`` as a tuple of two floats if it is a pair of numbers, else a refusal.

    Only the shape is checked here. Which order the two may come in, and what
    range each may take, is the knob's own to say.
    """
    if not is_numbers(value) or len(value) != 2:
        raise ValueError(
            f"{name} must be written as a pair of numbers, [low, high], not "
            f"{value!r}."
        )
    return float(value[0]), float(value[1])
