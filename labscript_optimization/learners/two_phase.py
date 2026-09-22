"""Run a trainer first, then hand over to the main learner.

A Gaussian process needs a spread of observations before its posterior means
anything, so the first ``num_training`` shots come from a second learner, the
trainer, which a configuration names. After that the main learner takes over.
The warmup is held to what the main learner needs: one shorter than that is
refused, because the handover would then not happen at the shot the number
names.

The trainer can also be brought back after the handover, by
``num_runs_between_trainer_runs``: one proposal in every cycle of that many
plus one comes from the trainer instead. A main learner refining around the
best point it has found narrows onto it, and a proposal from the trainer every
so often is what goes on widening the history underneath it. Off unless a
configuration asks for it, because turning it on changes what every existing
file proposes.

A proposer wrapping two proposers, not a controller: it proposes the same
way as what it wraps, so it can be wrapped in turn. What it wraps is held to
what it can answer for: a learner declaring a generation is refused, because
two phases cannot hold one barrier between them.
"""

from typing import Sequence

import numpy as np

from ..observations import Observation, usable
from .base import Learner


class TwoPhaseLearner(Learner):
    """Delegate to ``trainer`` while training, then to ``main``.

    A :class:`~labscript_optimization.learners.base.Learner` like the two it
    wraps, so a session cannot tell it apart from either. Not a
    :class:`~labscript_optimization.learners.base.ParameterSpaceLearner`,
    though: it searches no space of its own and so cannot be named in a
    configuration. :func:`~labscript_optimization.learners.build` wraps one
    around the two learners it builds instead.

    Which learner proposes is a function of the history, like everything else a
    learner computes: the phase is read off the number of usable observations
    in hand, and so is whether this is a position the trainer takes back. No
    count is kept. A counter would part company with the history the first time
    a proposal was dropped or came back with a cost nothing can fit -- the
    position would be spent and the counter would not know it -- and a learner
    handed the same history twice would answer differently the second time.

    Of the three members a session reads, this answers ``last_phase`` for
    itself -- the phase is this learner's own, and naming the trainer and the
    main learner is the whole point of it -- and declares
    ``minimum_observations`` of zero, which is true at both ends: below
    ``num_training`` the trainer proposes, and a trainer that withholds
    proposals of its own is refused, while at ``num_training`` and above the
    main learner can propose, because a warmup shorter than the main learner's
    own requirement is refused. ``generation`` it
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
        trainer: The learner used for the training phase, and for the periodic
            runs after it.
        main: The learner used once training is done.
        num_training: How many usable observations to gather before handing
            over. At least the main learner's own
            ``minimum_observations``, or the handover would not happen where
            this says it does.
        num_runs_between_trainer_runs: How many consecutive proposals come from
            the main learner between one proposal from the trainer and the
            next, once training is over. ``None``, the default, never goes back
            to the trainer.
    """

    #: Zero, and not inherited from either wrapped learner: this learner always
    #: proposes, because each of its two phases is held to a learner that can
    #: propose throughout it, so it absorbs its main learner's requirement
    #: rather than passing it on.
    minimum_observations = 0

    def __init__(
        self,
        trainer,
        main,
        num_training: int,
        num_runs_between_trainer_runs: int | None = None,
    ):
        for role, wrapped, instead in (
            ("trainer", trainer, "Name a trainer that proposes any number at a time"),
            ("main", main, f"Run {type(main).__name__} without a training phase"),
        ):
            # Read straight off the object, with no default standing in for it:
            # a learner declares its own generation, and one that declares
            # nothing is not something this can hold a barrier for.
            generation = wrapped.generation
            if generation is not None:
                raise ValueError(
                    f"the {role} {type(wrapped).__name__} proposes whole "
                    f"generations of {generation}, and only when none of its "
                    f"proposals is outstanding. A two-phase learner declares "
                    f"no generation of its own, so a session would top its "
                    f"queue up whenever there was room and the generation "
                    f"would go out in pieces. {instead}."
                )

        # The trainer proposes the very first shot, from an empty history, and
        # every periodic run after training. A learner that refuses to propose
        # without observations can be neither: at the first shot of the run
        # there is nothing behind it, so it would raise out through the
        # session.
        withheld = getattr(trainer, "minimum_observations", 0)
        if withheld:
            raise ValueError(
                f"the trainer {type(trainer).__name__} will not propose until "
                f"the history holds {withheld} usable observations, and the "
                f"training phase begins with none. Name a trainer that "
                f"proposes from an empty history."
            )

        self.trainer = trainer
        self.main = main
        self.num_training = int(num_training)
        if self.num_training < 0:
            raise ValueError(f"num_training cannot be negative, got {num_training}")

        needed = getattr(main, "minimum_observations", 0)
        if self.num_training < needed:
            # The number would not mean what it says. The handover would happen
            # at ``needed`` and not at ``num_training``, and the shots in
            # between would come from the trainer under a phase called "main"
            # -- a setting read back off the file as one thing and acted on as
            # another.
            raise ValueError(
                f"num_training_runs is {self.num_training}, and "
                f"{type(main).__name__} will not propose until the history "
                f"holds {needed} usable observations, so the handover this "
                f"asks for at {self.num_training} could not happen before "
                f"{needed}. Set num_training_runs to {needed} or more, or "
                f"lower {type(main).__name__}'s minimum_observations to "
                f"{self.num_training}."
            )

        self.num_runs_between_trainer_runs = (
            None
            if num_runs_between_trainer_runs is None
            else int(num_runs_between_trainer_runs)
        )
        if (
            self.num_runs_between_trainer_runs is not None
            and self.num_runs_between_trainer_runs < 1
        ):
            # Zero would be no main-learner runs between one trainer run and
            # the next, which is the main learner never proposing at all --
            # a way of writing "do not run the learner this file names", and
            # not the way anybody would mean to write it. Unset is how the
            # periodic run is turned off.
            raise ValueError(
                f"num_runs_between_trainer_runs is how many proposals come "
                f"from {type(main).__name__} between one from "
                f"{type(trainer).__name__} and the next, so it must be at "
                f"least 1, got {num_runs_between_trainer_runs}. Leave it "
                f"unset for a run that never goes back to the trainer."
            )

        # An instance attribute because this is the learner whose phase
        # changes, and answered before anything has been proposed because a
        # session may report its status first.
        self.last_phase = "training"

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        seen = len(usable(history))
        if seen < self.num_training:
            self.last_phase = "training"
            return self.trainer.propose(history, k)
        period = self.num_runs_between_trainer_runs
        # The turn is taken at the batch boundary: whichever learner the first
        # position of the batch belongs to proposes the whole of it. A batch
        # split between the two would have to say which position it had
        # reached part way through itself, and the only count it could say it
        # from is one proposal per position -- the assumption a counter makes,
        # and wrong for every proposal that is dropped or comes back unusable.
        # It would also report one phase for a batch made by two learners, and
        # cut the main learner's batch, which a Gaussian process conditions on
        # its own earlier points, into pieces.
        if period is not None and (seen - self.num_training) % (period + 1) == period:
            self.last_phase = "periodic trainer"
            return self.trainer.propose(history, k)
        self.last_phase = "main"
        return self.main.propose(history, k)
