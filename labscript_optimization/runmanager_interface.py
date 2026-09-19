"""Submitting proposals to runmanager, and asking what became of them.

runmanager never stops. Submitting appends to the running queue; there is no
queue to start, drain or wait on. A shot is complete when runmanager has sent
it to lyse and lyse has analysed it, which is the routine being handed its
row. Every shot carries the identifier runmanager minted for its queue row,
written into the shot file: that is what a cost is matched to a proposal by.
"""

from typing import Iterable, Sequence

import numpy as np

#: The state runmanager reports for a shot id it has no row for.
UNKNOWN_SHOT_STATE = "unknown"

#: A shot behind a row runmanager will not hand over -- a rejected head, or
#: one whose compile failed. Nothing further happens to it until an operator
#: moves what is in front of it.
BLOCKED_SHOT_STATE = "blocked"

#: Seconds runmanager is given to answer the greeting that opens a session,
#: when labconfig says nothing. Short against the routine's allowance for the
#: whole of configuration, so that a runmanager which is not running is named
#: as the cause rather than the worker being killed mid-wait and the lab told
#: that it was slow -- a lab that sets the key below longer than that
#: allowance gets the latter back.
#:
#: The key is ``timeouts/liveness_timeout``, which BLACS reads for its own
#: liveness probe, so a lab on a slow link sets one number and both
#: applications honour it. Deliberately not ``communication_timeout``: that
#: one allows runmanager to do work -- evaluating globals, compiling shots --
#: while this one measures a network round trip. The session's later requests
#: keep the client's own timeout, which is ``communication_timeout``, because
#: a submission that compiles shots needs it.
GREETING_TIMEOUT = 5.0


class RunmanagerInterface:
    """Submits proposals and reports what became of them.

    Args:
        config: The session configuration.
        client: A ``runmanager.remote`` client, or ``None`` to make the default
            one. Injected so the session can be tested without runmanager. Its
            ``timeout`` is how long each request waits.
        greeting_timeout: Seconds the greeting is allowed, or ``None`` to take
            labconfig's ``timeouts/liveness_timeout``, falling back to
            :data:`GREETING_TIMEOUT`. Injected for the same reason the client
            is: read here rather than at import, so that the module has no
            side effect on being imported and the deadline is a value a caller
            can supply rather than a file a test has to arrange.
    """

    def __init__(self, config, client=None, greeting_timeout=None):
        if client is None:
            from runmanager import remote

            client = remote.Client()
        if greeting_timeout is None:
            from labscript_utils.labconfig import LabConfig

            greeting_timeout = LabConfig().getfloat(
                "timeouts", "liveness_timeout", fallback=GREETING_TIMEOUT
            )
        self.config = config
        self.client = client
        self.greeting_timeout = greeting_timeout
        self.labscript_file = None

    def check_ready(self) -> None:
        """Raise if runmanager cannot start a session, and pin its labscript file.

        The greeting comes first, and is the only request held to
        ``greeting_timeout``, so that a runmanager which is not there is
        reported as a runmanager which is not there. Asking it a question
        instead leaves the answer to the client's own timeout, which outlasts
        the routine's allowance for the whole of configuration: the worker is
        killed mid-wait and the lab reads that the worker was slow.

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
        # ``self.client.with_timeout(self.greeting_timeout).say_hello()`` and
        # nothing here touches an object it does not own.
        patient, self.client.timeout = self.client.timeout, self.greeting_timeout
        try:
            self.client.say_hello()
        except Exception as exc:
            raise RuntimeError(
                f"runmanager did not answer within {self.greeting_timeout:g} "
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

        A refusal means nothing was queued: submit_shots checks every entry --
        that the globals evaluate, and that each produces exactly one shot --
        before submitting any of them, so a raise here leaves nothing behind
        to account for.
        """
        entries = [
            self.config.globals_for(np.asarray(p, dtype=float)) for p in proposals
        ]
        return [d["shot_id"] for d in self.client.submit_shots(entries)]

    def shot_status(self, shot_ids: Iterable[str]) -> dict[str, dict]:
        """What runmanager says about each of these shots, as it says it.

        ``{shot_id: {'pending': bool, 'state': str}}``, one entry per id asked
        about: runmanager loops over the ids it was handed and answers for each
        of them, so an id it has no row for comes back ``'unknown'`` rather
        than absent. ``pending`` is whether that shot could still produce a
        cost. ``state`` is the queue row's own state, or ``'submitted'`` for a
        shot runmanager has taken on but has no row for yet, ``'blocked'`` for
        a row sitting behind one an operator has to clear, and ``'unknown'``
        for an id runmanager does not know.
        """
        shot_ids = list(shot_ids)
        if not shot_ids:
            return {}
        return self.client.shot_status(shot_ids)
