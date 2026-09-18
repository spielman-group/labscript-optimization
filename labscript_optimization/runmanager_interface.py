"""Submitting proposals to runmanager.

runmanager never stops. ``engage()`` compiles whatever the globals currently
expand to and appends the resulting shots to the running queue, so submitting
is just setting values and engaging; there is no queue to start, drain or wait
on. A shot is complete when runmanager sends it to lyse, which is the routine
being called on it.

Each shot is stamped with a tag global. That is what a cost is matched to a
proposal by, which means shots can come back in any order, user shots can be
mixed into the queue, and runmanager can mint default shots when the queue runs
dry, without any of it needing to be accounted for here.
"""

from typing import Sequence

import numpy as np

#: Global holding the session name, so a restart cannot be confused with the
#: run before it.
SESSION_GLOBAL = "mloop_session"

#: Global holding the index of the shot within the session.
ITERATION_GLOBAL = "mloop_iteration"


def tag_for(session: str, iteration: int) -> str:
    """The tag identifying one proposal."""
    return f"{session}:{iteration}"


def parse_tag(tag: str) -> tuple[str, int]:
    """Split a tag back into its session and iteration."""
    session, _, iteration = tag.rpartition(":")
    return session, int(iteration)


class RunmanagerInterface:
    """Sets globals and engages runmanager for each proposal.

    Args:
        config: The session configuration.
        client: A ``runmanager.remote`` client, or ``None`` to make the default
            one. Injected so the worker can be tested without runmanager.
    """

    def __init__(self, config, client=None):
        self.config = config
        if client is None:
            from runmanager import remote

            client = remote.Client()
        self.client = client

    def check_ready(self) -> None:
        """Raise if runmanager cannot accept the globals this session needs.

        Failing here is the point: a missing global or a broken expression
        should stop the session at the first shot, with a message naming what
        is wrong, rather than quietly optimising the wrong thing.
        """
        if self.client.error_in_globals():
            raise RuntimeError(
                "runmanager reports an error in its globals; fix it before "
                "starting an optimisation"
            )
        present = set(self.client.get_globals())
        required = {g.name for g in self.config.globals} | {
            SESSION_GLOBAL,
            ITERATION_GLOBAL,
        }
        missing = sorted(required - present)
        if missing:
            raise RuntimeError(
                f"runmanager has no globals named {missing}. Create them in "
                f"an active group; {SESSION_GLOBAL} and {ITERATION_GLOBAL} "
                f"carry the tag that costs are matched by."
            )

    def submit(self, tag: str, params: Sequence[float]) -> None:
        """Set the globals for one proposal and engage.

        One shot per engage. A batch could be submitted as a scan list in a
        single engage, but runmanager expands independent scans as an outer
        product, so a batch of k over n parameters would have to be zipped to
        avoid compiling k**n shots. Engaging once per shot keeps that off the
        table, and the batch sizes here are small.
        """
        session, iteration = parse_tag(tag)
        values = self.config.globals_for(np.asarray(params, dtype=float))
        values[SESSION_GLOBAL] = session
        values[ITERATION_GLOBAL] = iteration
        self.client.set_globals(values)
        self.client.engage()


class MockInterface:
    """Accepts proposals without a runmanager behind it.

    Selected by ``[COMPILATION] mock = true``. Useful for checking that a
    configuration loads, that the globals it computes are the ones intended,
    and that the worker starts, without compiling anything.
    """

    def __init__(self, config):
        self.config = config
        self.submitted: list[tuple[str, dict]] = []

    def check_ready(self) -> None:
        pass

    def submit(self, tag: str, params: Sequence[float]) -> None:
        session, iteration = parse_tag(tag)
        values = self.config.globals_for(np.asarray(params, dtype=float))
        values[SESSION_GLOBAL] = session
        values[ITERATION_GLOBAL] = iteration
        self.submitted.append((tag, values))
        print(f"mock submit {tag}: {values}", flush=True)


def interface_for(config):
    """The interface a configuration asks for."""
    return MockInterface(config) if config.mock else RunmanagerInterface(config)
