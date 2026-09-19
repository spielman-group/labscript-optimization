"""The lyse multishot routine.

A lab analysis routine is two lines::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('mloop_config.toml')

Adding the routine to lyse starts the session; removing it, restarting it, or
reaching the run budget stops it. Progress is written onto each shot the
optimiser can claim, as lyse results under :data:`RESULTS_GROUP`, so the best
cost and the rest of the session's status are columns of the dataframe.

The routine itself does almost nothing: it reads the cost for the shot it was
called on, hands it to the worker, and waits for the worker to say where the
session has got to. Everything slow happens in the worker *after* that answer
goes out, because lyse calls multishot routines inline and a slow one delays
every shot behind it.

Every shot the routine can identify is reported, including one whose cost is
not usable. Such a shot has run and lyse has analysed it, so no cost for it is
ever coming: it is reported as a bad observation, which counts it as a
completed run and keeps it out of the fits.
"""

import atexit
import os
import subprocess
import sys

import numpy as np

from .runmanager_interface import SHOT_ID_ATTR

WORKER_PATH = os.path.join(os.path.dirname(__file__), "worker.py")

#: The lyse results group the session's status is written to, and so the first
#: level of every column it produces: ``df[('labscript_optimization',
#: 'best_cost')]``. lyse names a routine's group after the routine's file, so
#: the package's own name is a group a lab collides with only by naming a
#: routine after the package it imports.
RESULTS_GROUP = "labscript_optimization"

#: Seconds the routine waits for the worker to answer the message it has just
#: sent. Generous for an answer that is a dictionary and a socket hop, and
#: short against a shot cycle, which is what it is traded against: see
#: :func:`_drain`.
REPLY_TIMEOUT = 2.0


def latest(dataframe, key):
    """The most recent value of one column, or ``None`` if there is no such column.

    lyse labels its columns with a MultiIndex and pads every label out to the
    depth of the deepest one, so an analysis result is ``('routine', 'result')``
    and the shot's file is ``('filepath', '')`` -- deeper still in a sequence
    whose shots carry images. Either key reads as the column itself rather than
    as a frame of what sits beneath it, because the levels the padding added
    are empty, so the depth of the frame does not matter here.
    """
    if key not in dataframe:
        return None
    return dataframe[key].iloc[-1]


def shot_id_of(filepath) -> str | None:
    """The identifier runmanager wrote into a shot file, if it wrote one.

    lyse does not carry this into its dataframe, so it is read from the file.
    A shot without it is not one this optimiser submitted -- a user's own, or
    one of runmanager's default shots, which deliberately carry none. Nor is
    one whose file cannot be read for an identifier at all.
    """
    import h5py

    try:
        with h5py.File(filepath, "r") as f:
            shot_id = f.attrs.get(SHOT_ID_ATTR)
    except (OSError, KeyError, RuntimeError, ValueError):
        # These are the ways h5py reports a file that is missing, still being
        # written, or damaged. A shot the optimiser cannot identify is not one
        # it can claim, and passing it over costs nothing, whereas raising
        # would take the whole session down through lyse's error path. The
        # list is named rather than bare on purpose: an interrupt or an
        # out-of-memory error says nothing about the shot and must propagate.
        return None
    if shot_id is None:
        return None
    if isinstance(shot_id, bytes):
        shot_id = shot_id.decode()
    return str(shot_id)


def extract(dataframe, config):
    """Read the shot id and cost of the most recent shot.

    Returns ``(shot_id, cost, uncer, bad)``, or ``None`` when the shot carries
    no identifier and so belongs to somebody else.

    The sign flip for ``maximize`` happens here, once, so that everything
    downstream minimises.
    """
    if not len(dataframe):
        return None

    filepath = latest(dataframe, "filepath")
    if filepath is None:
        return None
    shot_id = shot_id_of(filepath)
    if shot_id is None:
        return None

    cost, uncer = float("nan"), None
    raw = latest(dataframe, config.cost_key)
    if raw is not None:
        cost = float(raw)
        measured = latest(dataframe, config.uncertainty_key)
        if measured is not None and np.isfinite(float(measured)):
            uncer = float(measured)

    bad = not np.isfinite(cost)
    if not bad and config.maximize:
        cost = -cost
    return shot_id, cost, uncer, bad


def save_status(filepath, status) -> None:
    """Write the session's status onto one shot, as lyse results.

    lyse reads the attributes of ``/results/<group>`` back as dataframe
    columns, so this is what makes progress visible: each key of the status
    becomes ``df[(RESULTS_GROUP, key)]`` against the shot the routine ran on.
    Only attributes are read that way, which is why ``best_params`` is saved
    with ``save_result`` although it is a list -- ``save_result_array`` would
    write it as a dataset, into a part of the file the dataframe never looks
    at.

    A value the session does not have yet is written as NaN, because an h5
    attribute cannot be ``None`` and NaN is what pandas means by a missing
    value in a column of any type; it is also what lyse itself puts in a
    column a shot carries nothing for.

    A write that fails is reported to lyse's output and otherwise passed over.
    The status is a progress report: losing one is worth less than the
    optimisation that stopping here would end.
    """
    try:
        import lyse

        run = lyse.Run(filepath)
        run.set_group(RESULTS_GROUP)
        # One open for the whole status. Left to itself each save_result opens
        # and locks the file again, and this runs inline in lyse.
        with run.open("r+"):
            for name, value in status.items():
                run.save_result(name, float("nan") if value is None else value)
    except Exception as exc:
        print(
            f"could not write the optimisation status to {filepath}: {exc!r}",
            file=sys.stderr,
        )


def start_worker(config_path, process_tree=None):
    """Spawn the optimisation worker and configure it.

    Returns ``(to_worker, from_worker, popen)``.
    """
    if process_tree is None:
        from labscript_utils.ls_zprocess import ProcessTree

        # Inside a lyse analysis subprocess this is the tree already connected
        # to lyse, so the worker becomes a child of this routine's process and
        # goes away with it.
        process_tree = ProcessTree.instance()

    to_worker, from_worker, popen = process_tree.subprocess(
        WORKER_PATH, startup_timeout=30
    )
    to_worker.put(("configure", os.path.abspath(config_path)))
    return to_worker, from_worker, popen


def _drain(from_worker):
    """Wait for the worker's answer to the message just sent, and return it.

    Waiting, rather than taking whatever happens to have arrived, is what makes
    the answer this shot's own: the worker's reply is on its way across a
    socket while this runs, so a poll issued microseconds after the request
    finds nothing on the first shot and the shot before's status thereafter.

    The wait is bounded because lyse calls multishot routines inline and
    serially, so whatever is spent here delays every shot behind this one. It
    is not a budget for the worker's work -- the reply is built from memory and
    sent before the worker fits anything or asks runmanager anything -- but the
    limit on how long a worker that has stopped answering can hold lyse up, and
    reaching it means this shot goes unreported rather than that anything is
    lost.

    Once something has arrived, anything further is taken only if it is already
    waiting. The slow work that follows a reply can still fail, and its error
    is worth having as soon as it can be had, but the routine does not wait on
    work it deliberately did not wait for.

    Raises if the worker reported an error, so it surfaces through lyse's
    normal error path.
    """
    status, error, timeout = None, None, REPLY_TIMEOUT
    while True:
        try:
            kind, payload = from_worker.get(timeout=timeout)
        except TimeoutError:
            break
        # Answered. What follows is swept up if it is here and left if it is
        # not, which is how an error from the work behind an earlier reply is
        # picked up without waiting for one.
        timeout = 0
        if kind == "error":
            error = payload
        else:
            status = payload
    if error is not None:
        raise RuntimeError(f"the optimisation worker failed:\n{error}")
    return status


def optimise(config_path, storage=None, dataframe=None):
    """Advance the optimisation by one shot. The lyse routine entry point.

    Args:
        config_path: The TOML configuration.
        storage: Where to keep the worker between shots. Defaults to
            ``lyse.routine_storage``.
        dataframe: The shots to read. Defaults to ``lyse.data(n_sequences=1)``.

    Returns:
        The status the worker sends in answer to this invocation, already
        written onto the shot by :func:`save_status`, or ``None`` if the
        worker does not answer within :data:`REPLY_TIMEOUT`. On the shot that
        starts the session the answer is to the configuration this invocation
        also sent, so it counts a session that has proposed nothing yet.
    """
    if storage is None or dataframe is None:
        import lyse

        storage = lyse.routine_storage if storage is None else storage
        dataframe = lyse.data(n_sequences=1) if dataframe is None else dataframe

    if getattr(storage, "optimisation_worker", None) is None:
        from . import config as config_module

        # Read once, at the moment the worker reads it for itself, and kept
        # for the life of the session. The worker holds the configuration it
        # was started with, so a routine that re-read the file each shot would
        # extract costs under a cost_key or a maximize the learner knows
        # nothing about the moment somebody edited it -- a flipped maximize
        # driving the search the wrong way without a word. Re-reading would
        # also parse the file and rebuild the parameter space every shot, in a
        # routine whose whole job is to return before it delays lyse.
        storage.optimisation_config = config_module.load(config_path)
        storage.optimisation_worker = start_worker(config_path)
        # Covers the ordinary shutdown, where lyse asks the analysis
        # subprocess to quit and it exits cleanly. A killed subprocess does
        # not run this, and the worker is left to zprocess's heartbeat.
        atexit.register(stop_worker, storage)

    config = storage.optimisation_config
    to_worker, from_worker, _ = storage.optimisation_worker

    observation = extract(dataframe, config)
    if observation is not None:
        # A shot whose cost is unusable is reported too, as a bad observation.
        # The shot has run and lyse has analysed it, so no cost for it will
        # ever arrive: withholding it would leave its id awaited until a
        # reconcile silently recorded it as dropped, understating the runs
        # spent against max_num_runs.
        to_worker.put(("observe", observation))
    else:
        to_worker.put(("status", None))

    status = _drain(from_worker)
    # Only a shot the optimiser can claim carries the status. A shot with no
    # id is somebody else's -- the user's own, or one of runmanager's defaults
    # -- and a session that has proposed nothing has no shot of its own yet at
    # all, which is what the answer to the configuration message describes.
    if observation is not None and status and status.get("submitted"):
        save_status(latest(dataframe, "filepath"), status)
    return status


def _exited_within(popen, timeout=5) -> bool:
    """Whether the worker has exited, waiting up to ``timeout`` seconds for it.

    Waiting is also reaping: a child nobody waits for stays a zombie for as
    long as the lyse analysis subprocess lives.
    """
    try:
        popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def stop_worker(storage=None) -> None:
    """Ask the worker to quit, and see that it has. Safe to call when there is none.

    Restarting the routine is the ordinary way to begin a fresh session, so a
    worker left behind here is one left behind every time, and they accumulate.
    """
    if storage is None:
        import lyse

        storage = lyse.routine_storage
    handles = getattr(storage, "optimisation_worker", None)
    if handles is None:
        return
    to_worker, _, popen = handles
    storage.optimisation_worker = None
    try:
        to_worker.put(("quit", None))
    except Exception:
        # A pipe that will not carry the request changes nothing about what
        # follows: the worker is signalled and reaped either way.
        pass
    if _exited_within(popen):
        return
    popen.terminate()
    if _exited_within(popen):
        return
    popen.kill()
    # Nothing stronger is available. A killed worker that is still not reaped
    # is stuck in the kernel, and blocking lyse's shutdown on it would help
    # nobody.
    _exited_within(popen)
