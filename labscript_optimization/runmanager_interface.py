"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager has sent
it to lyse and lyse has analysed it, which is the routine being handed its
row. Every shot carries the identifier runmanager minted for its queue row,
written into the shot file: that is what a cost is matched to a proposal by.
"""

from typing import Iterable, Sequence

#: The state runmanager reports for a shot id it has no row for.
UNKNOWN_SHOT_STATE = "unknown"

#: A shot behind a row runmanager will not hand over -- a rejected head, or
#: one whose compile failed. Nothing further happens to it until an operator
#: moves what is in front of it.
BLOCKED_SHOT_STATE = "blocked"

#: How runmanager's reason begins when it will not add shots to the sequence
#: a submission names: it has no record of it, as after a restart. Nothing is
#: queued.
REFUSED_SEQUENCE = "Cannot add shots to sequence "

#: Seconds runmanager is given to answer the greeting that opens a session.
#: Short, so that a runmanager which is not running is named as the cause in a
#: few seconds rather than a minute later by whichever question happened to be
#: asked first. The session's later requests keep the client's own timeout,
#: labconfig's ``communication_timeout``, which a submission that compiles
#: shots needs; this number is a term of the routine's allowance for the whole
#: of configuration, so the greeting is inside that allowance by construction.
#:
#: A constant, and not labconfig's ``timeouts/liveness_timeout``, which BLACS
#: reads before every exchange. BLACS probes runmanager once per shot, so that
#: number is a trade it has to make: too long and an unreachable runmanager
#: adds dead time to every cycle, too short and a slow link is judged absent.
#: A session greets once, over a round trip that is sub-second on any lab
#: link, so there is no trade here to make -- while a number raised for the
#: sake of BLACS's cycle time would buy the lab nothing here but a longer wait
#: before an absent runmanager is named, and a longer allowance with it.
GREETING_TIMEOUT = 5.0

#: Requests :meth:`RunmanagerInterface.check_ready` makes after the greeting,
#: each of which waits the client's own ``communication_timeout``:
#: ``error_in_globals`` and ``get_labscript_file``. The routine's allowance for
#: configuring the worker is a sum with one of those deadlines per request, so
#: a question added here is visibly a reason to raise this number, and raising
#: it widens that allowance to cover the question.
CHECK_READY_REQUESTS = 2


class RunmanagerInterface:
    """Submits proposals and reports what became of them.

    Args:
        config: The session configuration.
        client: A ``runmanager.remote`` client, or ``None`` to make the default
            one. Injected so the session can be tested without runmanager. Its
            ``timeout`` is how long each request waits, and is held down to
            :data:`GREETING_TIMEOUT` for the greeting.
    """

    def __init__(self, config, client=None):
        if client is None:
            from runmanager import remote

            client = remote.Client()
        self.config = config
        self.client = client
        self.labscript_file = None
        # The runmanager sequence this session's shots go into, once the first
        # submission has started it. Its index tells it apart from another
        # sequence started in the same second, which shares its id.
        self.sequence = None
        self.sequence_index = None

    def check_ready(self) -> None:
        """Raise if runmanager cannot start a session, and pin its labscript file.

        The greeting comes first, and is the only request held to
        :data:`GREETING_TIMEOUT`, so that a runmanager which is not there is
        reported as a runmanager which is not there, within seconds. Asking it
        a question instead leaves the answer to the client's own timeout -- a
        minute where labconfig says nothing -- and names the question that
        failed rather than the runmanager behind it.

        The :data:`CHECK_READY_REQUESTS` questions after it wait that full
        timeout, and the routine's allowance for configuring is summed to
        cover them: a runmanager that greets and then stops answering, its GUI
        thread inside a compile or behind a dialog somebody left open, is
        reported as the question it left unanswered rather than as a worker
        killed mid-wait.

        A global that does not evaluate is a shot that will not compile, and
        every shot this session submits would be one. The file pinned here is
        what :meth:`check_unchanged` compares against for the rest of the
        session.
        """
        # The deadline goes on the client, not on the question, because
        # ``runmanager.remote.Client`` takes its timeout at construction and
        # ``request`` reads ``self.timeout`` on every call: there is no
        # per-request deadline to ask for. So the caller's client is
        # reconfigured for the length of one greeting and put back. This is an
        # interim waiting on one specific change: ``Client.with_timeout`` in
        # runmanager, returning a sibling client for the same host and port
        # with a different deadline. The greeting then becomes
        # ``self.client.with_timeout(GREETING_TIMEOUT).say_hello()`` and
        # nothing here touches an object it does not own.
        patient, self.client.timeout = self.client.timeout, GREETING_TIMEOUT
        try:
            self.client.say_hello()
        except Exception as exc:
            raise RuntimeError(
                f"runmanager did not answer within {GREETING_TIMEOUT:g} "
                f"seconds ({exc!r}); an optimisation session cannot start "
                f"without it"
            ) from exc
        finally:
            self.client.timeout = patient

        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimisation"
            )

        self.labscript_file = self.client.get_labscript_file()

    def check_unchanged(self) -> None:
        """Raise if the labscript file has changed since the session started."""
        current = self.client.get_labscript_file()
        if current != self.labscript_file:
            raise RuntimeError(
                f"the labscript file changed from {self.labscript_file!r} to "
                f"{current!r} while this session was running; its shots would "
                f"no longer be the experiment it has been optimising"
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
