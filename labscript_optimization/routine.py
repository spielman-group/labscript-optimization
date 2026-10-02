"""The lyse multishot routine.

A lab's routine file is a subclass of :class:`OptimizationRoutine`::

    LYSE_MODE = "gui"
    from labscript_optimization.routine import OptimizationRoutine

    class Optimization(OptimizationRoutine):
        config_path = "optimization_config.toml"

Adding it to lyse's multishot routines opens the optimizer window with a
paused session; Start begins submitting shots. Removing or restarting the
routine, or reaching the run budget, stops it. :data:`SHOT_RESULTS` is saved
into lyse's dataframe, as lyse results under :data:`RESULTS_GROUP` in the row
of each shot the session proposed, so the best cost, where the search has got
to, and what proposed each shot are columns of it.

lyse runs a multishot routine once per drained batch of singleshot analyses
rather than once per shot, and names that batch's files in ``paths``. Every
one of them is a run the session spent, so every one of them is handed over.
"""

import queue
import sys
from pathlib import Path

import lyse
import numpy as np

from . import config as config_module
from .runmanager_interface import RunmanagerInterface
from .window import WindowController
from .worker import Worker

#: The lyse results group the session's status is written to, and so the first
#: level of every column it produces: ``df[('labscript_optimization',
#: 'best_cost')]``. lyse names a routine's group after the routine's file, so a
#: lab collides with this only by naming a routine after the package it imports.
RESULTS_GROUP = "labscript_optimization"

#: The keys written onto a shot, and so the columns the session produces: what
#: proposed the shot, where the search has got to, and whether it has stopped.
#:
#: ``phase`` is the shot's own: the source the session recorded when it
#: proposed that shot, which the worker's verdict on the shot carries. It is
#: not the phase of whatever was proposed most recently, which is another shot
#: whenever more than one is in flight. The other four come from the session's
#: status, whose remaining keys are its bookkeeping, one answer for the whole
#: run that would be repeated onto every shot of it; the window shows all of
#: it.
#:
#: ``stopped`` is here rather than with the bookkeeping because it is a marker
#: and not a tally. A counter carries a running total onto every shot and says
#: nothing about the one it lands on; ``stopped`` is empty until the session
#: ends, so the first shot carrying a reason is the shot the run ended on, and
#: where it sits in the column is the answer to the question a lab asks of a
#: finished run.
SHOT_RESULTS = ("phase", "best_cost", "best_params", "best_shot_id", "stopped")

#: What each of :data:`SHOT_RESULTS` is saved as while the session has
#: nothing to report for it.
#:
#: lyse gives a dataframe column one dtype, and the shots already saved fix
#: it: a stand in of a different type than the value it holds a place for
#: types the column against that value, and the shot that finally has one
#: cannot be written into it. So each empty here carries the type of the value
#: that replaces it -- ``""`` for the string-valued keys, an empty list for the
#: parameter vector, NaN only for the float that NaN is the empty of. lyse
#: reads a shot with no identifier back as ``""`` for the same reason.
#:
#: None of these collide with a value the session reports: ``stopped`` is a
#: sentence, ``best_shot_id`` is an id runmanager minted, and a configuration
#: with no enabled parameters is refused, so ``best_params`` is never empty.
NO_VALUE_YET = {
    "phase": "",
    "best_cost": float("nan"),
    "best_params": [],
    "best_shot_id": "",
    "stopped": "",
}


def analysed(paths):
    """The rows of lyse's dataframe for the files in ``paths``.

    The rows come in one request, in the dataframe's order. A file named twice,
    after a failed pass, is one row, and a BLACS rerun is a file of its own
    carrying the same shot id, which passes through harmlessly because the
    session takes a cost for an id once.
    """
    if not paths:
        return []
    # Columns sorted, because lyse keeps them in the order they were added and
    # pandas warns about lexsort depth when an unsorted MultiIndex is read by
    # a key shallower than it, as :func:`extract` reads it.
    return lyse.data(where={"filepath": paths}).sort_index(axis=1)


def extract(shots, config):
    """Read the shot ids and costs of ``shots``, rows of lyse's dataframe.

    Returns two lists in step: the file of each shot there is an id to read,
    and its ``(shot_id, cost, uncer, bad)``. lyse reads the identifier
    runmanager wrote into the file as a column, and it is empty for one of
    runmanager's default shots, which go to BLACS already compiled and so
    never have an id written into them. An id that is there does not make the
    shot the session's -- runmanager mints one for every row it compiles, a
    user's own shots included -- and which ids belong to the session is the
    session's own answer. A shot with an id is read whether or not its cost is
    usable: it has run and lyse has analysed it, so withholding it would leave
    its id awaited until a reconcile quietly dropped it, understating the runs
    spent. The sign flip for ``maximize`` happens here, on the way in, so
    everything downstream minimises; the session puts it back in the best
    cost it reports.
    """
    # A column at a time off the frame rather than a row at a time: pandas
    # resolves a key shallower than the column MultiIndex, whose padding levels
    # are empty, against a frame's columns, where against a row the same key
    # names a sub-Series. A column the frame does not have is None throughout.
    keys = "filepath", "shot_id", config.cost_key, config.uncertainty_key
    columns = [shots[k] if k in shots else [None] * len(shots) for k in keys]
    filepaths, observations = [], []
    for filepath, shot_id, raw, measured in zip(*columns):
        if shot_id is None or shot_id == "":
            continue
        cost, uncer = float("nan"), None
        if raw is not None:
            cost = float(raw)
            if measured is not None and np.isfinite(float(measured)):
                uncer = float(measured)
        bad = not np.isfinite(cost)
        if not bad and config.maximize:
            cost = -cost
        filepaths.append(filepath)
        observations.append((shot_id, cost, uncer, bad))
    return filepaths, observations


def save_status(filepath, status) -> None:
    """Save :data:`SHOT_RESULTS` of ``status`` against one shot, as lyse results.

    ``status`` is what this shot is to carry: the session's status, with the
    shot's own ``phase`` beside it.

    Each key becomes ``df[(RESULTS_GROUP, key)]`` in that shot's row of lyse's
    dataframe, and is saved there alone, with ``save_to_h5=False``: lyse sets
    it into the row, and the shot file is not opened. The dataframe is where
    the status is read, and writing it into the file as well would take the
    file's h5 lock once per shot, inline in lyse. ``best_params`` is saved
    with ``save_result`` although it is a list, because ``save_result`` is
    what reaches the dataframe; ``save_result_array`` writes a dataset into
    the file and nothing more. A value the session does not have yet is saved
    as its :data:`NO_VALUE_YET` stand in, which has the type of the value it
    holds a place for, so that the column is one dtype from the first shot
    onwards.

    A save that fails is reported to lyse's output and otherwise passed over.
    """
    try:
        import lyse

        run = lyse.Run(filepath)
        run.set_group(RESULTS_GROUP)
        for name in SHOT_RESULTS:
            reported = status[name]
            if reported is None:
                reported = NO_VALUE_YET[name]
            run.save_result(name, reported, save_to_h5=False)
    except Exception as exc:
        print(
            f"could not write the optimization status to {filepath}: {exc!r}",
            file=sys.stderr,
        )


class OptimizationRoutine(lyse.Routine):
    """One optimization session, as a lyse GUI routine.

    A lab's routine file subclasses this and sets :attr:`config_path`. The
    configuration is read once, when lyse starts the routine; restart the
    routine after editing it.

    Attributes
    ----------
    config_path : str
        The TOML configuration, relative to the routine file's folder.
    interface_factory : callable
        What the configuration is turned into a runmanager interface by.
    """

    config_path = None
    icon = str(Path(__file__).with_name("optimizer.svg"))
    interface_factory = RunmanagerInterface

    def __init__(self):
        text = Path(self.config_path).read_text(encoding="utf-8")
        self.config = config_module.loads(text)
        ui = self.load_ui(Path(__file__).with_name("window.ui"))
        commands = queue.Queue()
        window = WindowController(ui, commands)
        self.worker = Worker(
            self.config, text, window, commands, self.interface_factory
        )

    def run(self, path, paths):
        if paths is None:
            raise ValueError(
                "labscript_optimization's routine runs on the shots of a "
                "multishot pass; add it to lyse's multishot routines."
            )
        filepaths, observations = extract(analysed(paths), self.config)
        self.worker.hand_over(filepaths, observations, save_status)

    def close(self):
        self.worker.quit()
