"""The lyse multishot routine.

A lab analysis routine is two lines::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('mloop_config.toml')

Adding the routine to lyse starts the session; removing it, restarting it, or
reaching the run budget stops it. Progress appears as results on the routine.

The routine itself does almost nothing: it reads the cost for the shot it was
called on, hands it to the worker, and returns. Everything slow happens in the
worker, because lyse calls multishot routines inline and a slow one delays
every shot behind it.

Every shot the routine can identify is reported, including one whose cost is
not usable. Such a shot has run and lyse has analysed it, so no cost for it is
ever coming: it is reported as a bad observation, which counts it as a
completed run and keeps it out of the fits.
"""

import atexit
import os
import subprocess

import numpy as np

from .runmanager_interface import SHOT_ID_ATTR

WORKER_PATH = os.path.join(os.path.dirname(__file__), "worker.py")


def latest(dataframe, key):
    """The most recent value of one column, or ``None`` if there is no such column.

    lyse pads its column labels into a MultiIndex, so an analysis result is the
    column ``('routine', 'result')``. Indexing by a bare name then gives a
    frame rather than a value. Accepting both shapes keeps this usable against
    a plain dataframe in a test.
    """
    if key not in dataframe:
        return None
    column = dataframe[key]
    if getattr(column, "ndim", 1) > 1:
        if column.shape[1] != 1:
            return None
        column = column.iloc[:, 0]
    return column.iloc[-1]


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
    """Take every reply waiting, and return the last status.

    Raises if the worker reported an error, so it surfaces through lyse's
    normal error path.
    """
    status, error = None, None
    while True:
        try:
            kind, payload = from_worker.get(timeout=0)
        except TimeoutError:
            break
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
        The worker's status, which the caller may save as lyse results.
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

    return _drain(from_worker)


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
