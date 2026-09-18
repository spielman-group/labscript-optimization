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
"""

import atexit
import os

import numpy as np

from .runmanager_interface import ITERATION_GLOBAL, SESSION_GLOBAL, tag_for

WORKER_PATH = os.path.join(os.path.dirname(__file__), "worker.py")


def latest(dataframe, key):
    """The most recent value of one column, or ``None`` if there is no such column.

    lyse pads its column labels into a MultiIndex, so a runmanager global named
    ``x`` is really the column ``('x', '')`` while an analysis result is
    ``('routine', 'result')``. Indexing by the bare name then gives a frame
    rather than a value. Accepting both shapes keeps this usable against a
    plain dataframe in a test.
    """
    if key not in dataframe:
        return None
    column = dataframe[key]
    if getattr(column, "ndim", 1) > 1:
        if column.shape[1] != 1:
            return None
        column = column.iloc[:, 0]
    return column.iloc[-1]


def extract(dataframe, config):
    """Read the tag and cost of the most recent shot.

    Returns ``(tag, cost, uncer, bad)``, or ``None`` when the shot carries no
    tag and so belongs to somebody else.

    The sign flip for ``maximize`` happens here, once, so that everything
    downstream minimises.
    """
    if not len(dataframe):
        return None

    session = latest(dataframe, SESSION_GLOBAL)
    iteration = latest(dataframe, ITERATION_GLOBAL)
    if session is None or iteration is None:
        return None
    try:
        if np.isnan(iteration):
            return None
    except (TypeError, ValueError):
        pass
    try:
        tag = tag_for(str(session), int(iteration))
    except (TypeError, ValueError):
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
    return tag, cost, uncer, bad


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

    from . import config as config_module

    config = config_module.load(config_path)

    if getattr(storage, "optimisation_worker", None) is None:
        storage.optimisation_worker = start_worker(config_path)
        # Covers the ordinary shutdown, where lyse asks the analysis
        # subprocess to quit and it exits cleanly. A killed subprocess does
        # not run this, and the worker is left to zprocess's heartbeat.
        atexit.register(stop_worker, storage)

    to_worker, from_worker, _ = storage.optimisation_worker

    observation = extract(dataframe, config)
    if observation is not None:
        tag, cost, uncer, bad = observation
        # A shot with no cost yet is not an observation. Leaving it
        # unreported is what lets a multishot routine average several
        # repeats and only then produce a number.
        if not (bad and config.ignore_bad):
            to_worker.put(("observe", (tag, cost, uncer, bad)))
        else:
            to_worker.put(("status", None))
    else:
        to_worker.put(("status", None))

    return _drain(from_worker)


def stop_worker(storage=None) -> None:
    """Ask the worker to quit. Safe to call when there is none."""
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
        popen.wait(timeout=5)
    except Exception:
        popen.terminate()
