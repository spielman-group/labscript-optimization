"""One optimisation session: the history, the learner, and the stop conditions.

This is the part that would be a controller in M-LOOP. It is a plain object
with no threads, queues or events in it, so it can be driven by a test one call
at a time and by the worker process in the lab.

History is kept in proposal order. A cost arriving out of order fills in the
slot its tag names, and a shot whose cost never arrives simply stays empty.
That is the whole of the bookkeeping for shots in flight, and it is why the
learners never have to think about them.
"""

from typing import Sequence

import numpy as np

from . import learners
from .observations import Observation, usable
from .runmanager_interface import tag_for


class Session:
    """Drives one optimisation.

    Args:
        config: The session configuration.
        interface: Something with ``check_ready()`` and ``submit(tag, params)``.
        learner: The learner to use, or ``None`` to build the configured one.
    """

    def __init__(self, config, interface, learner=None):
        self.config = config
        self.interface = interface
        self.learner = learners.build(config) if learner is None else learner
        self.proposals: dict[str, np.ndarray] = {}
        self.results: dict[str, tuple[float, float | None, bool]] = {}
        self.next_iteration = 0
        self.stopped: str | None = None

    @property
    def history(self) -> list[Observation]:
        """Completed observations, in the order they were proposed."""
        return [
            Observation(tag, params, *self.results[tag])
            for tag, params in self.proposals.items()
            if tag in self.results
        ]

    @property
    def outstanding(self) -> int:
        """Proposals submitted whose cost has not come back."""
        return len(self.proposals) - len(self.results)

    @property
    def best(self) -> Observation | None:
        seen = usable(self.history)
        return min(seen, key=lambda o: o.cost) if seen else None

    def _runs_since_best(self) -> int:
        seen = usable(self.history)
        if not seen:
            return 0
        costs = [o.cost for o in seen]
        return len(costs) - 1 - int(np.argmin(costs))

    def _check_stop(self) -> None:
        limit = self.config.max_num_runs
        if limit is not None and len(self.results) >= limit:
            self.stopped = f"reached max_num_runs ({limit})"
            return
        patience = self.config.max_num_runs_without_better_params
        if patience is not None and self._runs_since_best() >= patience:
            self.stopped = (
                f"no better parameters in {patience} runs "
                f"(max_num_runs_without_better_params)"
            )

    def record(self, tag: str, cost: float, uncer: float | None, bad: bool) -> bool:
        """Take the cost for one shot.

        A tag this session did not propose is ignored, which is how user shots
        and shots left over from an earlier session pass through harmlessly.
        Returns whether the cost was taken.
        """
        if tag not in self.proposals or tag in self.results:
            return False
        self.results[tag] = (float(cost), uncer, bool(bad))
        self._check_stop()
        return True

    def refill(self) -> list[str]:
        """Submit enough proposals to keep the queue topped up.

        Returns the tags submitted, which is empty once the session has
        stopped.
        """
        if self.stopped:
            return []
        wanted = self.config.num_buffered_runs - self.outstanding
        if self.config.max_num_runs is not None:
            # Do not queue shots beyond the budget; they would be run and
            # charged against a session that has already finished.
            room = self.config.max_num_runs - len(self.proposals)
            wanted = min(wanted, room)
        if wanted <= 0:
            return []

        proposals = np.atleast_2d(self.learner.propose(self.history, wanted))
        submitted = []
        for params in proposals:
            tag = tag_for(self.config.session, self.next_iteration)
            self.next_iteration += 1
            self.proposals[tag] = np.asarray(params, dtype=float)
            self.interface.submit(tag, params)
            submitted.append(tag)
        return submitted

    def status(self) -> dict:
        """A snapshot for the routine to write back as lyse results."""
        best = self.best
        phase = getattr(self.learner, "last_phase", "main")
        return {
            "session": self.config.session,
            "phase": phase,
            "proposed": len(self.proposals),
            "completed": len(self.results),
            "outstanding": self.outstanding,
            "best_cost": None if best is None else best.cost,
            "best_params": None if best is None else best.params.tolist(),
            "best_tag": None if best is None else best.tag,
            "stopped": self.stopped,
        }
