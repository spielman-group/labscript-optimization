"""The lyse multishot routine.

A lab's routine is a folder, ``<name>.lyse``, whose ``lyse_routine.py`` holds a
subclass of :class:`OptimizationRoutine`::

    from labscript_optimization.routine import OptimizationRoutine

    class Optimization(OptimizationRoutine):
        config_path = "optimization_config.toml"

Adding it to lyse's multishot routines opens the optimizer window with a
paused session; Start begins submitting shots. Removing or restarting the
routine, or reaching the run budget, stops it. :data:`SHOT_RESULTS` is saved
into lyse's dataframe, as lyse results under the routine's
:attr:`~OptimizationRoutine.group` in the row of each shot the session
proposed, so the best cost, where the search has got to, and what proposed
each shot are columns of it.

lyse runs a multishot routine once per drained batch of singleshot analyses
rather than once per shot, and names that batch's files in ``paths``. Every
one of them is a run the session spent, so every one of them is handed over.
"""

import inspect
import queue
import sys
from pathlib import Path

import lyse
import numpy as np

from . import config as config_module
from .runmanager_interface import RunmanagerInterface
from .window import WindowController
from .worker import Worker

#: The keys written onto a shot, and so the columns the session produces.
#: They say what proposed the shot and where the search has got to.
#:
#: ``phase`` is the shot's own: the source the session recorded when it
#: proposed that shot, which the worker's verdict on the shot carries. It is
#: not the phase of whatever was proposed most recently, which is another shot
#: whenever more than one is in flight. The other two come from the session's
#: status, whose remaining keys are its bookkeeping, one answer for the whole
#: run that would be repeated onto every shot of it; the window shows all of
#: it.
SHOT_RESULTS = ("phase", "best_cost", "best_params")

#: What each of :data:`SHOT_RESULTS` is saved as while the session has
#: nothing to report for it.
#:
#: lyse gives a dataframe column one dtype, and the shots already saved fix
#: it: a stand in of a different type than the value it holds a place for
#: types the column against that value, and the shot that finally has one
#: cannot be written into it. So each empty here carries the type of the value
#: that replaces it -- ``""`` for the string, an empty list for the parameter
#: vector, NaN only for the float that NaN is the empty of. lyse reads a shot
#: with no identifier back as ``""`` for the same reason.
#:
#: None of these collide with a value the session reports: a configuration
#: with no enabled parameters is refused, so ``best_params`` is never empty.
NO_VALUE_YET = {
    "phase": "",
    "best_cost": float("nan"),
    "best_params": [],
}


def extract(shots, config):
    """Read the shot ids and costs of ``shots``, rows of lyse's dataframe.

    lyse reads the identifier runmanager wrote into the file as a column, and
    it is empty for one of runmanager's default shots, which go to BLACS
    already compiled and so never have an id written into them. An id that is
    there does not make the shot the session's -- runmanager mints one for
    every row it compiles, a user's own shots included -- and which ids belong
    to the session is the session's own answer. A shot with an id is read
    whether or not its cost is usable: it has run and lyse has analyzed it, so
    withholding it would leave its id awaited until a reconcile quietly dropped
    it, understating the runs spent. The sign flip for ``maximize`` happens
    here, on the way in, so everything downstream minimizes; the session puts
    it back in the best cost it reports.

    Parameters
    ----------
    shots : pandas.DataFrame
        Rows of lyse's dataframe, one per shot.
    config : labscript_optimization.config.Config
        Names the cost and uncertainty columns, and whether to maximize.

    Returns
    -------
    filepaths : list of str
        The file of each shot there is an id to read.
    observations : list of tuple
        ``(shot_id, cost, uncer, bad)`` for each of those files, in step with
        ``filepaths``. ``bad`` is true when the cost is missing or not finite,
        and ``cost`` is NaN when it is missing. ``uncer`` is ``None`` when the
        uncertainty is missing or not finite.
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


class OptimizationRoutine(lyse.Routine):
    """One optimization session, as a lyse GUI routine.

    A lab's ``lyse_routine.py`` subclasses this and sets :attr:`config_path`. The
    configuration is read when lyse starts the routine, which fails if the file
    does not load, and again at each Reset.
    """

    #: The TOML configuration, relative to the routine folder. A subclass must
    #: set it.
    config_path = None

    #: The lyse results group the session's status is saved in, and so the first
    #: level of every column it produces: ``df[('labscript_optimization',
    #: 'best_cost')]``.
    group = "labscript_optimization"

    #: The routine's icon in lyse, the package's ``optimizer.svg``.
    #:
    #: :meta hide-value:
    icon = str(Path(__file__).with_name("optimizer.svg"))

    #: Called with the session's :class:`~labscript_optimization.config.Config`
    #: to build the runmanager interface the session submits shots through.
    interface_factory = RunmanagerInterface

    def _load_config(self):
        text = self._config_file.read_text(encoding="utf-8")
        return config_module.loads(text), text

    def __init__(self):
        if self.config_path is None:
            raise ValueError(
                f"{type(self).__name__} must set config_path to its TOML "
                f"configuration file."
            )
        # The folder of the lab's subclass, whatever the working directory is.
        folder = Path(inspect.getfile(type(self))).parent
        self._config_file = folder / self.config_path

        from runmanager.client import RunmanagerClient

        # Only for runmanager's address: the light asks through a client of
        # its own.
        client = RunmanagerClient()
        ui = self.load_ui(Path(__file__).with_name("window.ui"))
        self.saved_widgets(ui.start_from_runmanager)
        commands = queue.Queue()
        window = WindowController(
            ui, commands, client.host, client.port, self._config_file
        )
        # The worker reads the file at once, so one that does not load raises
        # here, and lyse shows no window.
        self.worker = Worker(
            self._load_config, window, commands, self.interface_factory
        )
        self.link = window.link
        self.link.start()

    def run(self):
        """Hand the shots of lyse's current multishot pass to the session."""
        if self.paths is None:
            raise ValueError(
                "labscript_optimization's routine runs on the shots of a "
                "multishot pass; add it to lyse's multishot routines."
            )
        # An empty pass, from Run multishot with nothing analyzed, reads nothing.
        shots = []
        if self.paths:
            # Columns sorted, because lyse keeps them in the order they were
            # added and pandas warns about lexsort depth when an unsorted
            # MultiIndex is read by a key shallower than it, as extract reads it.
            shots = self.data(where={"filepath": self.paths}).sort_index(axis=1)
        filepaths, observations = extract(shots, self.worker.config)
        self.worker.hand_over(filepaths, observations, self.save_status)

    def save_status(self, filepath, status) -> None:
        """Save :data:`SHOT_RESULTS` of ``status`` into one shot's row of lyse's dataframe.

        Each key becomes ``df[(group, key)]``, saved with ``save_to_h5=False``,
        so the shot file is not opened: writing it as well would take its h5
        lock once per shot, inline in lyse. A value of ``None`` is saved as its
        :data:`NO_VALUE_YET`. A shot whose file is gone is reported to lyse's
        output and passed over.

        Parameters
        ----------
        filepath : str
            The shot file whose row is set.
        status : dict
            The session's status, with the shot's own ``phase`` beside it.
        """
        # The only save that can fail: lyse.Run refuses a file that is not
        # there, though the save itself never opens it.
        try:
            run = self.get_run(filepath)
        except FileNotFoundError as exc:
            print(f"Not saving the optimization status: {exc}", file=sys.stderr)
            return
        for name in SHOT_RESULTS:
            reported = status[name]
            if reported is None:
                reported = NO_VALUE_YET[name]
            # save_result although best_params is a list, because it is what
            # reaches the dataframe; save_result_array writes the file alone.
            run.save_result(name, reported, save_to_h5=False)

    def close(self):
        """Ask the session to stop and stop watching runmanager.

        Called when lyse removes or restarts the routine.
        """
        self.link.shutdown()
        self.worker.quit()
