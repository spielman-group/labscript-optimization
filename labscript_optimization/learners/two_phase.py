"""Run a trainer first, then hand over to the main learner.

A Gaussian process needs a spread of observations before its posterior means
anything, so the first ``num_training`` shots come from a cheap learner that
explores. After that the main learner takes over, and the trainer stays on as
the fallback for any proposal the main learner cannot make.

A proposer wrapping two proposers, not a controller: it implements the same
:class:`~labscript_optimization.learners.base.Learner` protocol as what it
wraps.
"""

from typing import Sequence

import numpy as np

from ..observations import Observation, usable
from .base import InsufficientData


class TwoPhaseLearner:
    """Delegate to ``trainer`` while training, then to ``main``.

    Args:
        trainer: The learner used for the training phase, and as the fallback.
        main: The learner used once training is done.
        num_training: How many usable observations to gather before handing
            over.
    """

    def __init__(self, trainer, main, num_training: int):
        self.trainer = trainer
        self.main = main
        self.num_training = int(num_training)
        if self.num_training < 0:
            raise ValueError(f"num_training cannot be negative, got {num_training}")
        # An instance attribute because this is the learner whose phase
        # changes, and answered before anything has been proposed because a
        # session may report its status first.
        self.last_phase = "training"

    def training(self, history: Sequence[Observation]) -> bool:
        """Whether the training phase is still running."""
        return len(usable(history)) < self.num_training

    def propose(self, history: Sequence[Observation], k: int) -> np.ndarray:
        if self.training(history):
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
