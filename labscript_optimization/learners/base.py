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
of the learners. The Gaussian process's kernel is an exception, for speed: each
refit starts from where the last one ended.
"""

from typing import Sequence

import numpy as np

from ..observations import PENDING, Observation
from ..space import ParameterSpace


class Learner:
    """Proposes the next parameters to try.

    This is the whole of what a session drives. A learner that wraps other
    learners is one of these too, so anything holding a learner can hold a
    wrapped one without knowing it.

    Pacing is written once, in :meth:`propose`. A learner supplies
    ``ask(history, k)``, its method without the pacing, which returns ``k``
    points as a ``(k, num_params)`` array; one with more than one way of
    proposing overrides :meth:`acquire` instead.

    Inheriting is not what makes an object usable -- a session proposes from
    anything carrying these two members -- but everything this package calls
    a learner is one.

    A wrapper owes an answer for every attribute something outside a learner
    reads off it: ``generation`` here, read by the session and by the
    configuration. Three answers are honest -- answer for itself, where
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
    #: many, and only when nothing of the last group is outstanding: declaring
    #: it is what makes :meth:`propose` hold that barrier. What else reads it
    #: is the budget refused below two of them, the ``num_buffered_runs``
    #: refused beside one, and the session's count of starved refills, which
    #: skips a learner whose queue empties once a generation by design.
    generation: int | None = None

    #: Which records of the history this learner reads when another runs it:
    #: ``"all"``, ``"mine"`` -- the shots it proposed -- or ``"none"``. Read
    #: only by a learner that runs others, as the Gaussian process runs its
    #: explorer.
    history_scope: str = "all"

    def propose(
        self, history: Sequence[Observation], hint: int
    ) -> list[tuple[np.ndarray, str]]:
        """Return what this learner's method allows it to propose now.

        A learner declaring no ``generation`` tops the run's shots in flight up
        to ``hint``, counting every pending record whoever proposed it. One
        declaring a generation proposes the rest of the block of
        ``generation`` positions its next proposal falls in when nothing of the
        last is pending, and nothing otherwise. The number settled on is
        handed to :meth:`acquire`.

        Parameters
        ----------
        history : sequence of Observation
            Every proposal made so far, in the order they were made, each with
            what became of it. The ones still pending are the run's shots in
            flight.
        hint : int
            How many of the run's shots to keep in flight: the session's
            ``num_buffered_runs``. A learner declaring a generation does not
            read it.

        Returns
        -------
        list of (numpy.ndarray, str)
            Zero or more ``(params, source)`` pairs, in the order they are to
            be run, so that what the run budget has no room for is cut from
            the end: every ``params`` inside the bounds, and every ``source`` a
            string naming which of the learner's ways of proposing made it,
            which is what that shot reads in the ``phase`` column.
        """
        if self.generation is None:
            k = hint - sum(o.state == PENDING for o in history)
        else:
            # The configured start, on the call that places it, is pending with
            # no shot id yet: it founds slot 0 and goes out with the rest of its
            # generation rather than being waited for alone.
            if any(o.state == PENDING and o.shot_id is not None for o in history):
                return []
            k = self.generation - len(history) % self.generation
        if k <= 0:
            return []
        return self.acquire(history, k)

    def acquire(
        self, history: Sequence[Observation], k: int
    ) -> list[tuple[np.ndarray, str]]:
        """Return ``k`` proposals, each beside its source.

        The step a learner with more than one way of proposing overrides. By
        default it is ``ask(history, k)``'s points, each sourced ``"main"``.

        Parameters
        ----------
        history : sequence of Observation
            As :meth:`propose` was handed it.
        k : int
            How many proposals :meth:`propose` settled on, at least one.

        Returns
        -------
        list of (numpy.ndarray, str)
            As :meth:`propose` returns them.
        """
        return [(params, "main") for params in self.ask(history, k)]


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
    """A learner's search cannot propose from the history it has been given.

    Raised rather than quietly proposing on some other basis, so that a caller
    given a proposal knows which learner made it. Nothing here catches it: the
    Gaussian process's model raises it from ``ask`` short of its warmup, and
    the learner running it hands warmup to its explorer and never asks the
    model until warmup is over, so a session does not reach it.
    """
