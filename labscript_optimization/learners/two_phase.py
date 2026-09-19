"""Run a trainer first, then hand over to the main learner.

A Gaussian process needs a spread of observations before its posterior means
anything, so the first ``num_training`` shots come from a cheap learner that
explores. After that the main learner takes over, and the trainer stays on as
the fallback for any proposal the main learner cannot make.

A proposer wrapping two proposers, not a controller: it proposes the same
way as what it wraps, so it can be wrapped in turn. What it wraps is held to
what it can answer for: a learner declaring a generation is refused, because
two phases cannot hold one barrier between them.
"""

import warnings
from typing import Sequence

import numpy as np

from ..observations import Observation, usable
from .base import InsufficientData, Learner


class TwoPhaseLearner(Learner):
    """Delegate to ``trainer`` while training, then to ``main``.

    A :class:`~labscript_optimization.learners.base.Learner` like the two it
    wraps, so a session cannot tell it apart from either. Not a
    :class:`~labscript_optimization.learners.base.ParameterSpaceLearner`,
    though: it searches no space of its own and so cannot be named in a
    configuration. :func:`~labscript_optimization.learners.build` wraps one
    around the two learners it builds instead.

    Of the three members a session reads, this answers ``last_phase`` for
    itself -- the phase is this learner's own, and naming the trainer, the
    fallback and the main learner is the whole point of it -- and declares
    ``minimum_observations`` of zero, which is true because the trainer is the
    fallback for anything the main learner cannot yet make. ``generation`` it
    can neither answer for nor pass on, so a learner declaring one is refused
    here rather than wrapped.

    The alternative -- forwarding the main learner's generation -- is wrong on
    three counts. During training the trainer proposes and declares no
    barrier, so a forwarded generation would describe a phase that is not
    running: the session would drain the queue every ``generation`` shots
    throughout training, and ``refill`` does not count starvation for a
    learner declaring a generation, so the default shots runmanager hands
    BLACS at each of those drains would be missing from the one number a lab
    is told to watch. The barrier is also not the whole of what a generational
    learner needs: it reads a proposal's role off its position in the history
    it is handed, and behind a trainer that history opens with positions it
    never proposed and results that are not trials of its population -- so
    forwarding would make the queueing honest and leave the algorithm still
    not the one it is named after. And the declaration itself would be false
    of this object, which spends its training phase proposing through a
    learner that makes no such promise.
    Wrapping a generational learner needs the history it is handed to begin
    where its own proposals begin; until something does that, the combination
    is refused rather than approximated.

    Args:
        trainer: The learner used for the training phase, and as the fallback.
        main: The learner used once training is done.
        num_training: How many usable observations to gather before handing
            over.
    """

    #: Zero, and not inherited from either wrapped learner: this learner
    #: always proposes, falling back to the trainer for anything ``main``
    #: cannot make, so it absorbs its main learner's requirement rather than
    #: passing it on.
    minimum_observations = 0

    def __init__(self, trainer, main, num_training: int):
        for role, wrapped in (("trainer", trainer), ("main", main)):
            generation = getattr(wrapped, "generation", None)
            if generation is not None:
                raise ValueError(
                    f"the {role} {type(wrapped).__name__} proposes whole "
                    f"generations of {generation}, and only when none of its "
                    f"proposals is outstanding. A two-phase learner declares "
                    f"no generation of its own, so a session would top its "
                    f"queue up whenever there was room and the generation "
                    f"would go out in pieces. Run "
                    f"{type(wrapped).__name__} without a training phase."
                )

        self.trainer = trainer
        self.main = main
        self.num_training = int(num_training)
        if self.num_training < 0:
            raise ValueError(f"num_training cannot be negative, got {num_training}")

        needed = getattr(main, "minimum_observations", 0)
        if self.num_training < needed:
            # Not an error: the fallback covers it, and the shots are not
            # wasted. But the handover a user configured will not happen when
            # they expect, and neither number is stated in any document, so
            # say it once rather than leave them reading "training (fallback)"
            # and wondering.
            warnings.warn(
                f"{type(main).__name__} needs {needed} usable observations "
                f"before it can propose, and num_training_runs is "
                f"{self.num_training}, so the first proposals after training "
                f"come from {type(trainer).__name__} instead. Raise "
                f"num_training_runs to {needed} or more to hand over cleanly.",
                stacklevel=2,
            )
        # An instance attribute because this is the learner whose phase
        # changes, and answered before anything has been proposed because a
        # session may report its status first.
        self.last_phase = "training"

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        if len(usable(history)) < self.num_training:
            self.last_phase = "training"
            return self.trainer.propose(history, k)
        try:
            proposals = self.main.propose(history, k)
        except InsufficientData:
            # The main learner is past its training runs but still cannot fit,
            # so keep exploring rather than stalling the experiment.
            self.last_phase = "training (fallback)"
            return self.trainer.propose(history, k)
        self.last_phase = "main"
        return proposals
