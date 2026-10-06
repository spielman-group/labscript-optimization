"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager has sent
it to lyse and lyse has analyzed it, which is the routine being handed its
row. Every shot carries the identifier runmanager minted for its queue row,
written into the shot file: that is what a cost is matched to a proposal by.
"""

from numbers import Real
from typing import Iterable, Sequence

import numpy as np


class RunmanagerInterface:
    """Submits proposals and reports what became of them.

    Args:
        config: The session configuration.
        client: A ``runmanager.client.RunmanagerClient``, or ``None`` to make
            the default one. Injected so the session can be tested without
            runmanager.
    """

    def __init__(self, config, client=None):
        if client is None:
            from runmanager.client import RunmanagerClient

            client = RunmanagerClient()
        self.config = config
        self.client = client
        self.labscript_file = None
        # The raw Default expressions of every global a session has set, as
        # runmanager held them before the first Start that set it. The worker
        # hands them to the interface of each later session.
        self.original = None
        # The runmanager sequence this session's shots go into, once the first
        # submission has started it. Its index tells it apart from another
        # sequence started in the same second, which shares its id.
        self.sequence = None
        self.sequence_index = None

    def check_ready(self) -> None:
        """Raise if runmanager cannot take a session's shots; pin its labscript file.

        Called at each Start. A global that does not evaluate is a shot that
        will not compile, and every shot this session submits would be one. The
        labscript file is pinned by the first call, and is what
        :meth:`check_unchanged` compares against for the rest of the session, so
        a Start that resumes the run does not move it. The first call also
        records the original values, which :meth:`set_values` restores, of the
        globals it was not handed them for.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimization"
            )

        if self.labscript_file is None:
            # Read first: a failure here leaves both unset for the next Start.
            raw = self.client.get_values(raw=True)
            missing = [g.name for g in self.config.globals if g.name not in raw]
            if missing:
                raise RuntimeError(
                    f"Global {', '.join(missing)} not found in any active group "
                    f"in runmanager"
                )
            self.labscript_file = self.client.get_labscript_file()
            # What an earlier session recorded is what runmanager held before any
            # run, so it wins over what runmanager holds now.
            held = {g.name: raw[g.name] for g in self.config.globals}
            self.original = held | (self.original or {})

    def check_unchanged(self) -> None:
        """Raise if the labscript file has changed since the session started."""
        current = self.client.get_labscript_file()
        if current != self.labscript_file:
            raise RuntimeError(
                f"the labscript file changed from {self.labscript_file!r} to "
                f"{current!r} while this session was running; its shots would "
                f"no longer be the experiment it has been optimizing"
            )

    def submit(self, proposals: Sequence[Sequence[float]]) -> list[str]:
        """Queue one shot per proposal. Returns their shot ids, in order.

        A run is one runmanager sequence: the first submission starts it, and
        every later one names it and joins it.

        A refusal means nothing was queued: submit_shots checks every entry --
        that the globals evaluate, and that each produces exactly one shot --
        and the sequence named before submitting any of them, so a raise here
        leaves nothing behind to account for.
        """
        entries = [self.config.globals_for(p) for p in proposals]
        # A ticked global runs its scan value, or under JIT? the window's value
        # at compile time, rather than the value submitted.
        scan, jit = self.client.get_scan_enabled(), self.client.get_jit_enabled()
        ticked = [
            g.name for g in self.config.globals if scan.get(g.name) or jit.get(g.name)
        ]
        if ticked:
            raise RuntimeError(
                f"Untick Scan? and JIT? in runmanager for {', '.join(ticked)}: "
                f"their shots would not run the values this session submits."
            )
        descriptors = self.client.submit_shots(
            entries, sequence=self.sequence, sequence_index=self.sequence_index
        )
        self.sequence = descriptors[0]["sequence_id"]
        self.sequence_index = descriptors[0]["sequence_index"]
        return [d["shot_id"] for d in descriptors]

    def get_start(self) -> np.ndarray:
        """Read runmanager's current values of the parameters, as a parameter vector.

        A parameter is read from the global that takes it directly, which is
        what ``global_name`` makes. A global an ``expr`` computes cannot be
        turned back into its parameters.

        Raises
        ------
        RuntimeError
            If a parameter has no such global, or its global does not hold a
            real number, or the number lies outside the parameter's bounds.
        """
        values = self.client.get_values()
        start = []
        for parameter in self.config.space.parameters:
            direct = [
                g.name
                for g in self.config.globals
                if g.expr is None and g.args == (parameter.name,)
            ]
            if not direct:
                raise RuntimeError(
                    f"Cannot start from runmanager's values: parameter "
                    f"{parameter.name} reaches runmanager only through an "
                    f"expr, so its value cannot be read back"
                )
            value = values[direct[0]]
            if not isinstance(value, Real) or isinstance(value, bool):
                raise RuntimeError(
                    f"Cannot start from runmanager's values: {direct[0]} is "
                    f"{value!r}, not a number"
                )
            if not parameter.minimum <= value <= parameter.maximum:
                raise RuntimeError(
                    f"Cannot start from runmanager's values: {direct[0]} is "
                    f"{value:g}, outside the range {parameter.minimum:g} to "
                    f"{parameter.maximum:g} of parameter {parameter.name}"
                )
            start.append(float(value))
        return np.array(start)

    def set_values(self, params: Sequence[float] | None = None) -> None:
        """Set runmanager's Default values without submitting a shot.

        Parameters
        ----------
        params : sequence of float, optional
            The parameter vector whose globals to set: exactly what submitting
            it would leave in runmanager's window. Without it, the original
            values recorded at the first Start are written back as written.
        """
        if params is None:
            self.client.set_values(self.original, raw=True)
        else:
            self.client.set_values(self.config.globals_for(params))

    def shot_status(self, shot_ids: Iterable[str]) -> dict[str, dict]:
        """What runmanager says about each of these shots, as it says it.

        ``{shot_id: {'pending': bool, 'state': str}}``, one entry per id asked
        about: runmanager loops over the ids it was handed and answers for each
        of them, so an id it has no row for comes back ``'unknown'`` rather
        than absent. ``pending`` is whether that shot could still produce a
        cost. ``state`` is the queue row's own state, ``'blocked'`` for a row
        sitting behind one an operator has to clear, and ``'unknown'`` for an
        id runmanager does not know.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return {}
        return self.client.shot_status(shot_ids)
