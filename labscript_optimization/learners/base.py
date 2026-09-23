"""What a learner is.

A learner is a function from the proposal history and a hint -- how many of
the run's shots to keep in flight -- to whatever its method allows it to
propose now, each proposal with its source. It is given the whole history
every time, in the order the proposals were made, and holding every one of
them: a shot still running and a shot that will never report are both in
there as a position spent without a usable cost. Anything a learner computes
from the history and keeps between calls is a cache: it must hold what an
instance handed the same history for the first time would compute. The one
thing a learner carries that the history does not fix is the position of its
own rng stream, which decides where a draw lands and nothing about what the
learner believes. That rule is what keeps the out-of-order arrival of costs out
of the learners.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ..observations import Observation
from ..space import ParameterSpace


class Learner(ABC):
    """Proposes the next parameters to try.

    This is the whole of what a session drives. A learner that wraps other
    learners is one of these too, so anything holding a learner can hold a
    wrapped one without knowing it.

    Inheriting is not what makes an object usable -- a session proposes from
    anything carrying these two members -- but everything this package calls
    a learner is one.

    A wrapper owes an answer for every attribute something outside a learner
    reads off it: ``generation`` here, read by the session and by the
    configuration, and ``minimum_observations``, read by a wrapper deciding
    how long to train and so carried only by the learners that refuse to
    propose without one. Three answers are honest -- answer for itself, where
    the wrapper's own value is the true one; pass the wrapped learner's on,
    where the wrapper can hold what that value promises; or refuse to be
    built, where it cannot. Leaving one unanswered is none of the three: the
    reader's own default becomes the answer, and the wrapped learner's
    declaration is dropped with nothing said. That is how a generation gets
    lost behind a wrapper while everything still runs.

    These declarations are facts about an instance. Nothing outside a learner
    reads one from a class, a signature, a registry entry or a count: it
    builds a learner and asks that. Every one of those stands in for the
    learner and agrees with it only in the cases that exist on the day it is
    written -- a class attribute misses the learner that assigns in
    ``__init__``, which is the ordinary way; a signature default misses the
    learner that derives or clamps what it was given; a count of proposals is
    not the position one was made at. A learner is therefore free to declare
    however it likes, including from a value its constructor was handed, and
    the declaration is read where it is true.

    Where a proposal came from is not a declaration. It is returned beside the
    proposal, by the call that made it, so it cannot describe some other call,
    and nothing supplies it on a learner's behalf: a proposal returned without
    one is refused by the session rather than recorded as the main learner's.
    """

    #: How many proposals this learner makes at a time, or ``None`` for any
    #: number. A learner that declares one proposes only whole groups of that
    #: many, and only when nothing of the last group is outstanding, which it
    #: reads off the history like anything else. A declaration, not a
    #: delivery policy: the learner holds its own barrier, and what reads this
    #: is the budget refused below two of them, the ``num_buffered_runs``
    #: refused beside one, and the session's count of starved refills, which
    #: skips a learner whose queue empties once a generation by design.
    generation: int | None = None

    @abstractmethod
    def propose(
        self, history: Sequence[Observation], hint: int
    ) -> list[tuple[np.ndarray, str]]:
        """Return what this learner's method allows it to propose now.

        Args:
            history: Every proposal made so far, in the order they were made,
                each with what became of it. The ones still pending are the
                run's shots in flight.
            hint: How many of the run's shots to keep in flight -- the
                session's ``num_buffered_runs`` -- honoured as far as the
                learner's method allows.

        Returns:
            Zero or more ``(params, source)`` pairs, in the order they are to
            be run, so that what the run budget has no room for is cut from
            the end: every ``params`` inside the bounds, and every ``source`` a
            string naming which of the learner's ways of proposing made it,
            which is what that shot reads in the ``phase`` column.
        """


class ParameterSpaceLearner(Learner):
    """A learner that searches a :class:`ParameterSpace` of its own.

    Fixes how one is built: ``space`` first and ``rng`` second, positionally,
    then the learner's own knobs as keyword arguments with defaults. Every
    learner named in a configuration is built as ``cls(space, rng, **options)``
    with the options matched by name against the signature -- so the two
    spelled the other way round searches the wrong thing, and knobs collected
    in ``**kwargs`` are matched by nothing.

    Nothing else is shared. A learner keeps its own state. Where a run
    begins is not part of it: the session proposes the space's configured
    start itself, at the first position of the run, and a learner meets it
    in the history like any other record -- from the very first call, where
    it is the pending record at position 0.
    """

    def __init__(self, space: ParameterSpace, rng: np.random.Generator):
        if not isinstance(space, ParameterSpace) or not isinstance(
            rng, np.random.Generator
        ):
            raise TypeError(
                f"a learner takes a ParameterSpace and a Generator, in that "
                f"order; got {type(space).__name__} and {type(rng).__name__}"
            )
        self.space = space
        self.rng = rng


class InsufficientData(RuntimeError):
    """A learner cannot propose from the history it has been given.

    Raised rather than quietly proposing on some other basis, so that a caller
    given a proposal knows which learner made it. Nothing here catches it: a
    learner is only ever asked to propose from a history long enough for it,
    and the two-phase wrapper is refused at construction unless its training
    phase covers what its main learner needs. Reaching this state is therefore
    a configuration error.
    """
