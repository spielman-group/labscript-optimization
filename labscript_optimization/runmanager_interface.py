"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager has sent
it to lyse and lyse has analyzed it, which is the routine being handed its
row. Every shot carries the identifier runmanager minted for its queue row,
written into the shot file: that is what a cost is matched to a proposal by.

A monitor of its own keeps asking whether runmanager answers, for the window's
light.
"""

import threading
from typing import Iterable, Sequence

#: Seconds between asking runmanager whether it answers. A status light, not a
#: data feed.
POLL_INTERVAL = 2
#: Seconds runmanager is given to answer. runmanager answers a hello off its GUI
#: thread, so one that takes longer than this is one the optimizer cannot reach.
POLL_TIMEOUT = 1
#: The light beside the runmanager label says one thing: whether runmanager
#: answered.
LINK_ICONS = {
    "checking": ":/qtutils/fugue/hourglass",
    "online": ":/qtutils/fugue/tick",
    "offline": ":/qtutils/fugue/exclamation",
}


class RunmanagerStatusMonitor:
    """Keep asking runmanager whether it answers, and report every answer.

    The asking runs on a thread of its own, so a runmanager that has stopped
    answering cannot hold up the window or the session.

    Parameters
    ----------
    on_status : Callable
        Called with each answer: ``{'reachable': True}``, or ``{'reachable':
        False, 'reason': str}`` with why runmanager did not answer.
    client : optional
        What asks, with ``say_hello()``. The default is a
        ``runmanager.client.RunmanagerClient`` that waits :data:`POLL_TIMEOUT`.
    interval : float
        Seconds between asks.
    """

    def __init__(self, on_status, client=None, interval=POLL_INTERVAL):
        if client is None:
            from runmanager.client import RunmanagerClient

            client = RunmanagerClient(timeout=POLL_TIMEOUT)
        self.on_status = on_status
        self.client = client
        self.interval = interval
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.mainloop, daemon=True)

    def start(self):
        self.thread.start()

    def shutdown(self):
        """Stop asking, without waiting for a poll under way to finish."""
        self.stopped.set()

    def poll(self):
        """Ask runmanager once whether it answers, and report the answer."""
        try:
            self.client.say_hello()
        except Exception as exc:
            status = {"reachable": False, "reason": str(exc)}
        else:
            status = {"reachable": True}
        self.on_status(status)

    def mainloop(self):
        while not self.stopped.is_set():
            self.poll()
            self.stopped.wait(self.interval)


def runmanager_link_display(status):
    """Return the ``(state, tooltip)`` for the light beside the runmanager label.

    ``status`` is what :class:`RunmanagerStatusMonitor` reported, or ``None``
    before runmanager has been asked: the state is then ``'checking'``, and
    otherwise ``'online'`` or ``'offline'``.
    """
    if status is None:
        return "checking", "Checking runmanager..."
    if status["reachable"]:
        return "online", "runmanager is responding"
    tooltip = "runmanager is not responding"
    if status.get("reason"):
        tooltip += f"\n{status['reason']}"
    return "offline", tooltip


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

    def check_ready(self) -> None:
        """Raise if runmanager cannot take a session's shots; pin its labscript file.

        Called at each Start. A global that does not evaluate is a shot that
        will not compile, and every shot this session submits would be one. The
        labscript file is pinned by the first call, and is what
        :meth:`check_unchanged` compares against for the rest of the session, so
        a Start that resumes the run does not move it.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimization"
            )

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
