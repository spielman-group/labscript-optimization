"""What a learner is: a function from the proposal history to proposals.

A learner is given the whole history every time, in the order the proposals were
made, and every record is in it: a shot still running and one that will never
report are both a position spent without a usable cost. Anything a learner keeps
between calls is a cache and must hold what a new instance handed the same
history would compute. Only the position of its rng stream is state the history
does not fix, and it decides where a draw lands, not what the learner believes.
That rule keeps the out-of-order arrival of costs out of the learners.

The Gaussian process breaks it twice, for speed: each kernel refit starts from
the last, and whether a batch is ready at a refill depends on how long its fit
took, so a run under it is not reproducible from a seed.
"""

from collections.abc import Sequence

import numpy as np

from ..observations import PENDING, Observation
from ..space import ParameterSpace


class Learner:
    """Proposes the next parameters to try; the whole of what a session drives.

    A learner supplies ``ask(history, k)``, its method without the pacing, which
    returns ``k`` points as a ``(k, num_params)`` array. One with more than one
    way of proposing overrides :meth:`acquire` instead. :meth:`propose` does the
    pacing. A session proposes from anything carrying ``propose`` and
    ``generation``, but every learner this package names inherits from this
    class.

    A learner that wraps others must answer for every attribute read from
    outside a learner, such as ``generation`` for the session and the
    configuration. It may answer for itself, pass the wrapped learner's on, or
    refuse to be built. An attribute left unanswered takes the reader's default
    and drops the wrapped learner's declaration without a word.

    These declarations are facts about an instance, read by building a learner
    and asking it, never from a class, a signature or a registry entry, which
    agree with the learner only in the cases that exist when they are written.
    A learner may declare from a value its constructor was handed.

    Where a proposal came from is not a declaration. The call that made it
    returns the source beside it, and the session refuses a proposal returned
    without one.
    """

    #: How many proposals this learner makes at a time, or ``None`` for any
    #: number. A learner that declares one proposes only whole groups of that
    #: many, and only when nothing of the last group is pending;
    #: :meth:`propose` holds that barrier. The configuration refuses a budget
    #: below two groups and a ``num_buffered_runs`` beside one, and the
    #: session's count of starved refills skips such a learner, whose queue
    #: empties once a generation by design.
    generation: int | None = None

    #: Which records of the history this learner reads when another runs it:
    #: ``"all"``, ``"mine"`` (the shots it proposed) or ``"none"``. Read only by
    #: a learner that runs others, as the Gaussian process runs its explorer.
    history_scope: str = "all"

    def propose(
        self, history: Sequence[Observation], hint: int
    ) -> list[tuple[np.ndarray, str]]:
        """Return what this learner's method allows it to propose now.

        With no ``generation``, tops the run's shots in flight up to ``hint``,
        counting every pending record whoever proposed it. With one, proposes
        the rest of the block of ``generation`` positions the next proposal
        falls in when nothing of the last is pending, and nothing otherwise.
        The count settled on goes to :meth:`acquire`.

        Parameters
        ----------
        history : sequence of Observation
            Every proposal made so far, in the order made, each with what became
            of it. The pending ones are the run's shots in flight.
        hint : int
            How many of the run's shots to keep in flight, the session's
            ``num_buffered_runs``. A learner declaring a generation ignores it.

        Returns
        -------
        list of (numpy.ndarray, str)
            ``(params, source)`` pairs in the order they are to run, so that
            what the run budget has no room for is cut from the end. Each
            ``params`` is inside the bounds, and ``source`` names which of the
            learner's ways of proposing made it: the shot's ``phase``.
        """
        if self.generation is None:
            k = hint - sum(o.state == PENDING for o in history)
        else:
            # The configured start is pending with no shot id on the call that
            # places it. It founds slot 0 and goes out with its generation, not
            # waited for alone.
            if any(o.state == PENDING and o.shot_id is not None for o in history):
                return []
            k = self.generation - len(history) % self.generation
        if k <= 0:
            return []
        return self.acquire(history, k)

    def acquire(
        self, history: Sequence[Observation], k: int
    ) -> list[tuple[np.ndarray, str]]:
        """Return ``k`` proposals as ``(params, source)`` pairs.

        The step a learner with more than one way of proposing overrides. By
        default these are the points of ``ask(history, k)``, each sourced
        ``"main"``. ``history`` is as :meth:`propose` was handed it, and ``k``,
        at least one, is the count it settled on.
        """
        return [(params, "main") for params in self.ask(history, k)]


class ParameterSpaceLearner(Learner):
    """A learner that searches a :class:`~labscript_optimization.space.ParameterSpace`.

    Built as ``cls(space, rng, **options)``: ``space`` and ``rng`` positionally,
    then the learner's own knobs as keyword arguments with defaults, matched by
    name against the signature. The session proposes the space's configured
    start itself, so a learner meets it in the history like any other record,
    from the first call, as the pending record at position 0.
    """

    def __init__(self, space: ParameterSpace, rng: np.random.Generator):
        self.space = space
        self.rng = rng
