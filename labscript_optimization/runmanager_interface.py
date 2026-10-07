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
        # The runmanager sequence this session's shots go into, once the first
        # submission has started it. Its index tells it apart from another
        # sequence started in the same second, which shares its id.
        self.sequence = None
        self.sequence_index = None

    def check_ready(self) -> dict[str, str] | None:
        """Raise if runmanager cannot take a session's shots.

        Called at each Start. A global that does not evaluate is a shot that
        will not compile, and every shot this session submits would be one.
        Nothing is pinned or kept here, so a Start refused after this call
        leaves the interface as it was.

        Returns
        -------
        dict of str to str or None
            Until :meth:`pin_labscript_file` has run, the raw Default expression
            of each of the configuration's globals, as runmanager holds it now,
            which is what the worker records and :meth:`set_values` restores.
            ``None`` after that.

        Raises
        ------
        RuntimeError
            If runmanager's globals do not evaluate or, until a Start has gone,
            a global the configuration sets is in no active group.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimization"
            )

        if self.labscript_file is None:
            raw = self.client.get_values(raw=True)
            missing = [g.name for g in self.config.globals if g.name not in raw]
            if missing:
                raise RuntimeError(
                    f"Global {', '.join(missing)} not found in any active group "
                    f"in runmanager"
                )
            return {g.name: raw[g.name] for g in self.config.globals}
        return None

    def pin_labscript_file(self) -> None:
        """Note the labscript file that :meth:`check_unchanged` compares against.

        Called by a Start once it has passed every check. The first call pins
        the file, and a Start that resumes the run does not move it.
        """
        if self.labscript_file is None:
            self.labscript_file = self.client.get_labscript_file()

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
        that the globals evaluate, that none has Scan? or JIT? ticked, and that
        each produces exactly one shot -- and the sequence named before
        submitting any of them, so a raise here leaves nothing behind to
        account for.
        """
        entries = [self.config.globals_for(p) for p in proposals]
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

    def set_values(
        self,
        values: Sequence[float] | dict[str, str],
        raw: bool = False,
        skip_missing: bool = False,
    ) -> list[str]:
        """Set runmanager's Default values without submitting a shot.

        Parameters
        ----------
        values : sequence of float or dict
            Without ``raw``, the parameter vector whose globals to set: exactly
            what submitting it would leave in runmanager's window. With
            ``raw``, a dict of global name to the Default expression to write,
            as written.
        raw : bool, optional
            Whether ``values`` are expressions to write as they are.
        skip_missing : bool, optional
            Whether to write the globals that are in an active group and skip
            the rest, rather than refuse them all.

        Returns
        -------
        list of str
            The names skipped for being in no active group, empty unless
            ``skip_missing``.
        """
        return self.client.set_values(
            values if raw else self.config.globals_for(values),
            raw=raw,
            skip_missing=skip_missing,
        )

    def shot_status(self, shot_ids: Iterable[str]) -> dict[str, dict | None]:
        """What runmanager says about each of these shots, as it says it.

        Parameters
        ----------
        shot_ids : iterable of str
            The shots to ask about.

        Returns
        -------
        dict
            One entry per id asked about: runmanager's record of the shot, or
            ``None`` for an id it has not held since it started. The record
            holds ``shot_id``, ``sequence_id``, ``sequence_index``,
            ``run_number``, ``path``, the states ``compile``, ``queue``,
            ``blacs`` and ``lyse``, ``pending``, ``since`` and ``message``.
            ``compile``, ``blacs`` and ``lyse`` are ``None`` for a stage the
            shot left before reaching. ``pending`` is whether the shot may
            still complete in BLACS, and says nothing of lyse.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return {}
        return self.client.shot_status(shot_ids)
