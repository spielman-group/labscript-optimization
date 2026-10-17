"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops: submitting appends to the running queue, with no queue
to start, drain or wait on. A shot is complete when runmanager has sent it to
lyse and lyse has analyzed it, which is the routine being handed its row. Each
shot file runmanager compiles from a queue row carries the identifier it minted
for that row, by which a cost is matched to a proposal; its default shots, which
reach BLACS already compiled, carry none.
"""

from collections.abc import Sequence
from numbers import Real

import numpy as np


class RunmanagerInterface:
    """Submits proposals and reports what became of them."""

    def __init__(self, config, client=None):
        # A client can be injected, so the session can be tested without runmanager.
        if client is None:
            from runmanager.client import RunmanagerClient

            client = RunmanagerClient()
        self.config = config
        self.client = client
        # The session asks the client directly; see
        # ``RunmanagerClient.shot_status`` for the record.
        self.shot_status = client.shot_status
        self.labscript_file = None
        # The runmanager sequence this session's shots join once the first
        # submission starts it. The index tells it from another sequence started
        # in the same second, which shares its id.
        self.sequence = None
        self.sequence_index = None

    def check_ready(self) -> dict[str, str] | None:
        """Raise if runmanager cannot take a session's shots; called at each Start."""
        # Nothing is pinned or kept here, so a Start refused after this call leaves
        # the interface as it was.
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimization"
            )

        # Until the first Start has pinned the file, also check it and the globals,
        # and return the globals' raw Defaults as runmanager holds them now: what
        # the worker records and set_values restores.
        if self.labscript_file is None:
            if not self.client.get_labscript_file():
                raise RuntimeError(
                    "No labscript file is selected; select a labscript file in "
                    "runmanager"
                )
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
        """Pin the labscript file that :meth:`check_unchanged` compares against."""
        # Only the first call pins it, so a Start that resumes the run does not
        # move it.
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
        """Queue one shot per proposal and return their shot ids, in order."""
        # A run is one runmanager sequence, started by the first submission.
        # submit_shots checks every entry and the sequence before queuing any, so
        # a refusal leaves nothing queued.
        entries = [self.config.globals_for(p) for p in proposals]
        descriptors = self.client.submit_shots(
            entries, sequence=self.sequence, sequence_index=self.sequence_index
        )
        self.sequence = descriptors[0]["sequence_id"]
        self.sequence_index = descriptors[0]["sequence_index"]
        return [d["shot_id"] for d in descriptors]

    def get_start(self) -> np.ndarray:
        """Read runmanager's values of the parameters as a parameter vector."""
        values = self.client.get_values()
        start = []
        for parameter in self.config.space.parameters:
            # A global an expr computes cannot be turned back into its parameters.
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
        """Set runmanager's Defaults, submitting no shot; return the names skipped."""
        # Without raw, values is a parameter vector and its globals are set; with
        # raw, a dict of global name to the Default expression to write as written.
        # With skip_missing, a global in no active group is skipped, not refused.
        return self.client.set_values(
            values if raw else self.config.globals_for(values),
            raw=raw,
            skip_missing=skip_missing,
        )
