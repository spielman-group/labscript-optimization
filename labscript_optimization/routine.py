"""The lyse multishot routine.

A lab analysis routine is two lines::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('mloop_config.toml')

Adding the routine to lyse starts the session; removing it, restarting it, or
reaching the run budget stops it. :data:`SHOT_RESULTS` is written onto each
shot the session proposed, as lyse results under :data:`RESULTS_GROUP`, so
the best cost and where the search has got to are columns of the dataframe.

The routine itself does almost nothing: it reads the cost for the shot it was
called on, hands it to the worker, and waits for the worker to say where the
session has got to.
"""

import atexit
import os
import subprocess
import sys

import numpy as np

#: The lyse results group the session's status is written to, and so the first
#: level of every column it produces: ``df[('labscript_optimization',
#: 'best_cost')]``. lyse names a routine's group after the routine's file, so a
#: lab collides with this only by naming a routine after the package it imports.
RESULTS_GROUP = "labscript_optimization"

#: The status keys written onto a shot, and so the columns the session
#: produces: where the search has got to, and whether it has stopped. The rest
#: of the status is the session's bookkeeping, one answer for the whole run
#: that would be repeated onto every shot of it; :func:`optimise` returns all
#: of it.
SHOT_RESULTS = ("phase", "best_cost", "best_params", "best_shot_id", "stopped")

#: Seconds the routine waits for the worker to answer the message it has just
#: sent. Generous for an answer that is a dictionary and a socket hop, and
#: short against a shot cycle.
REPLY_TIMEOUT = 2.0

#: Seconds allowed for the worker to load its configuration and establish its
#: first runmanager connection. This is startup, not part of a shot cycle.
CONFIGURE_TIMEOUT = 30.0


def latest(dataframe):
    """The most recent shot, as a one-row frame.

    A frame rather than the row ``dataframe.iloc[-1]``: pandas resolves a key
    shallower than the column MultiIndex against a frame's columns, whatever
    the frame's depth, where against a row the same key names a sub-Series.
    """
    return dataframe.iloc[[-1]]


def value(shot, key):
    """One column of a one-row frame, or ``None`` if there is no such column.

    ``key`` may be shallower than the frame's MultiIndex, whose padding levels
    are empty.
    """
    if key not in shot:
        return None
    return shot[key].iloc[-1]


def extract(dataframe, config):
    """Read the shot id and cost of the most recent shot.

    Returns ``(shot_id, cost, uncer, bad)``, or ``None`` when there is no id to
    read: lyse reads the identifier runmanager wrote into the file as a column,
    and it is empty for one of runmanager's default shots, which go to BLACS
    already compiled and so never have an id written into them. An id that is
    there does not make the shot the session's -- runmanager mints one for
    every row it compiles, a user's own shots included -- and which ids belong
    to the session is the session's own answer. The sign flip for ``maximize``
    happens here, once, so everything downstream minimises.
    """
    if not len(dataframe):
        return None

    shot = latest(dataframe)
    shot_id = value(shot, "shot_id")
    if shot_id is None or shot_id == "":
        return None

    cost, uncer = float("nan"), None
    raw = value(shot, config.cost_key)
    if raw is not None:
        cost = float(raw)
        measured = value(shot, config.uncertainty_key)
        if measured is not None and np.isfinite(float(measured)):
            uncer = float(measured)

    bad = not np.isfinite(cost)
    if not bad and config.maximize:
        cost = -cost
    return shot_id, cost, uncer, bad


def save_status(filepath, status) -> None:
    """Write :data:`SHOT_RESULTS` onto one shot, as lyse results.

    lyse reads the attributes of ``/results/<group>`` back as dataframe
    columns, so each key written becomes ``df[(RESULTS_GROUP, key)]`` against
    the shot the routine ran on. Only attributes are read that way, which is
    why ``best_params`` is saved with ``save_result`` although it is a list --
    ``save_result_array`` would write it as a dataset, into a part of the file
    the dataframe never looks at. A value the session does not have yet is
    written as NaN, because an h5 attribute cannot be ``None``.

    A write that fails is reported to lyse's output and otherwise passed over.
    """
    try:
        import lyse

        run = lyse.Run(filepath)
        run.set_group(RESULTS_GROUP)
        # One open for the whole status. Left to itself each save_result opens
        # and locks the file again, and this runs inline in lyse.
        with run.open("r+"):
            for name in SHOT_RESULTS:
                reported = status[name]
                run.save_result(name, float("nan") if reported is None else reported)
    except Exception as exc:
        print(
            f"could not write the optimisation status to {filepath}: {exc!r}",
            file=sys.stderr,
        )


def start_worker(config_path, process_tree=None):
    """Spawn the optimisation worker and configure it.

    Waits for configuration to finish, then returns
    ``(to_worker, from_worker, popen)``.
    """
    # zprocess sends the class itself to the child, so the parent needs it.
    # Imported here rather than above so that a routine which never starts a
    # session does not pay for the learners the worker brings with it.
    from .worker import Worker

    if process_tree is None:
        from labscript_utils.ls_zprocess import ProcessTree

        # Inside a lyse analysis subprocess this is the tree already connected
        # to lyse, so the worker becomes a child of this routine's process and
        # goes away with it.
        process_tree = ProcessTree.instance()

    worker = Worker(process_tree, startup_timeout=30)
    to_worker, from_worker = worker.start()
    handles = to_worker, from_worker, worker.child
    try:
        to_worker.put(("configure", os.path.abspath(config_path)))
        _, status = _drain(from_worker, CONFIGURE_TIMEOUT)
        if status is None:
            raise TimeoutError(
                f"the optimisation worker did not configure within "
                f"{CONFIGURE_TIMEOUT:g} seconds"
            )
    except BaseException:
        _stop_worker(handles)
        raise
    # The Popen, not the Process: stopping the worker escalates from a
    # request to terminate and then to kill, which zprocess does not do.
    return handles


def _drain(from_worker, timeout=None):
    """Wait for the worker's answer to the message just sent, and return it.

    Returns ``(recorded, status)``: whether the session took the observation
    just sent, and where the session has got to. The two travel in one message,
    so a status can never be read against another shot's answer.

    The answer is this shot's own: the worker's reply is still crossing a
    socket while this runs. ``timeout`` bounds the wait for its first message
    and defaults to :data:`REPLY_TIMEOUT`; reaching it returns ``(False,
    None)`` and this shot goes unreported.

    Anything behind the answer is swept up too, but only if it is already
    waiting, which is how an error from the slow work behind an earlier reply
    arrives without being waited for. Raises if the worker reported an error.
    """
    recorded, status, error = False, None, None
    timeout = REPLY_TIMEOUT if timeout is None else timeout
    while True:
        try:
            kind, payload = from_worker.get(timeout=timeout)
        except TimeoutError:
            break
        # Answered; from here on take only what is already waiting.
        timeout = 0
        if kind == "error":
            error = payload
        else:
            recorded, status = payload
    if error is not None:
        raise RuntimeError(f"the optimisation worker failed:\n{error}")
    return recorded, status


def optimise(config_path, storage=None, dataframe=None):
    """Advance the optimisation by one shot. The lyse routine entry point.

    Args:
        config_path: The TOML configuration.
        storage: Where to keep the worker between shots. Defaults to
            ``lyse.routine_storage``.
        dataframe: The shots to read. Defaults to the shot the routine was
            called on, asked of lyse.

    Returns:
        The whole status the worker sends in answer to this invocation, or
        ``None`` if the worker does not answer within :data:`REPLY_TIMEOUT`.
        When the session took this shot's cost, :func:`save_status` has
        written :data:`SHOT_RESULTS` of that status onto the shot. Worker
        configuration is acknowledged before the worker is stored, so the
        first shot receives its own answer like every later shot.
    """
    if storage is None or dataframe is None:
        import lyse

        storage = lyse.routine_storage if storage is None else storage
        if dataframe is None:
            # One sequence because a run is one sequence, and one shot of it
            # because only the most recent is read: a sequence grows by a row
            # every time this is called, and asking for the whole of it would
            # make each invocation cost more than the one before.
            dataframe = lyse.data(n_sequences=1, n_shots=1)

    if getattr(storage, "optimisation_worker", None) is None:
        from . import config as config_module

        # Read once and kept for the life of the session. The worker holds the
        # configuration it was started with, so re-reading the file each shot
        # would let an edit mid-session leave the two disagreeing about what
        # the cost is -- a flipped maximize driving the search the wrong way.
        storage.optimisation_config = config_module.load(config_path)
        storage.optimisation_worker = start_worker(config_path)
        # The ordinary shutdown, where lyse asks the analysis subprocess to
        # quit. A killed subprocess does not run this and the worker is left
        # to zprocess's heartbeat.
        atexit.register(stop_worker, storage)

    config = storage.optimisation_config
    to_worker, from_worker, _ = storage.optimisation_worker

    observation = extract(dataframe, config)
    if observation is not None:
        # Reported whether or not its cost is usable: the shot has run and
        # lyse has analysed it, so withholding it would leave its id awaited
        # until a reconcile quietly dropped it, understating the runs spent.
        to_worker.put(("observe", observation))
    else:
        to_worker.put(("shot", None))

    recorded, status = _drain(from_worker)
    # The session took this shot's cost, so this shot is one it proposed. The
    # id alone does not say so: runmanager mints one for every queue row it
    # compiles, and writing the status onto a shot the session never proposed
    # would put a column of somebody else's numbers against a user's own shot.
    if recorded:
        save_status(value(latest(dataframe), "filepath"), status)
    return status


def exited_within(popen, timeout=5) -> bool:
    """Whether the worker has exited, waiting up to ``timeout`` seconds for it.

    Waiting is also reaping: a child nobody waits for stays a zombie for as
    long as the lyse analysis subprocess lives.
    """
    try:
        popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _stop_worker(handles) -> None:
    """Stop and reap one spawned worker, including a partly started one."""
    to_worker, _, popen = handles
    try:
        to_worker.put(("quit", None))
    except Exception:
        # A pipe that will not carry the request changes nothing about what
        # follows: the worker is signalled and reaped either way.
        pass
    if exited_within(popen):
        return
    popen.terminate()
    if exited_within(popen):
        return
    popen.kill()
    # Nothing stronger is available, and blocking lyse's shutdown on a worker
    # stuck in the kernel would help nobody.
    exited_within(popen)


def stop_worker(storage=None) -> None:
    """Ask the worker to quit, and see that it has. Safe to call when there is none.

    Restarting the routine is the ordinary way to begin a fresh session, so a
    worker left behind here is one left behind every time.
    """
    if storage is None:
        import lyse

        storage = lyse.routine_storage
    handles = getattr(storage, "optimisation_worker", None)
    if handles is None:
        return
    storage.optimisation_worker = None
    _stop_worker(handles)
