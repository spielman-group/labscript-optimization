"""The lyse plugin.

lyse loads it when its labconfig's ``[lyse/plugins]`` section has
``labscript_optimization = True``. Its window, opened from lyse's Optimizer
menu, runs one session at a time from the TOML configuration file chosen there,
and Start begins submitting shots. After all analysis of a shot, the plugin
hands the shot's cost to the session and saves :data:`SHOT_RESULTS` into lyse's
dataframe, as ``df[(GROUP, name)]``, in the row of each shot the session
proposed, so the best cost, where the search has got to, and what proposed each
shot are columns of it.
"""

import queue
from functools import partial
from pathlib import Path

import numpy as np
from lyse.plugins import BasePlugin
from qtutils import UiLoader, inmain, inmain_later
from qtutils.qt import QtGui
from runmanager.client import RunmanagerClient

from . import config as config_module
from .runmanager_interface import RunmanagerInterface
from .window import WindowController
from .worker import Worker

#: The lyse results group the plugin's columns are saved under, and so the first
#: level of each: ``df[('labscript_optimization', 'best_cost')]``.
GROUP = "labscript_optimization"

#: The keys written onto a shot, and so the columns the session produces. They
#: say what proposed the shot, and where the search has got to.
#:
#: ``phase`` is the shot's own, the source recorded when it was proposed, not
#: that of the latest proposal. The other two come from the session's status,
#: whose remaining keys are one answer for the whole run, shown in the window
#: and not repeated onto every shot.
SHOT_RESULTS = ("phase", "best_cost", "best_params")

#: What the best columns are saved as before there is a best; a shot's
#: ``phase`` always has a value.
#:
#: lyse fixes a column's dtype from the first value saved, so each stand-in has
#: the type of the value that replaces it: an empty list for the parameter
#: vector, NaN for the cost. A real ``best_params`` is never empty, since a
#: configuration with no enabled parameters is refused.
NO_VALUE_YET = {
    "best_cost": float("nan"),
    "best_params": [],
}


def extract(shots, config):
    """Read the shot ids and costs of ``shots``, rows of lyse's dataframe.

    A shot with no ``shot_id`` is skipped, as runmanager's default shots, which
    reach BLACS already compiled, have none. An id does not make a shot the
    session's, since runmanager mints one for every row it compiles, a user's
    own included; which ids belong to the session is the session's answer. A
    shot with an id is read whether or not its cost is usable: it has run and
    been analyzed, and withholding it would leave its id awaited until a
    reconcile dropped it, understating the runs spent. The sign flip for
    ``maximize`` happens here, on the way in, so everything downstream
    minimizes; the session puts it back in the best cost it reports.

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
    # Read a column at a time: against a frame's columns pandas resolves a key
    # shallower than the MultiIndex, but against a row it names a sub-Series. A
    # column the frame lacks is None throughout.
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


class Plugin(BasePlugin):
    """The optimizer in lyse: its window, its session and its remote commands."""

    #: Called with the session's :class:`~labscript_optimization.config.Config`
    #: to build the runmanager interface the session submits shots through.
    interface_factory = RunmanagerInterface

    def __init__(self):
        self.commands = queue.Queue()
        ui = UiLoader().load(str(Path(__file__).with_name("window.ui")))
        ui.setWindowIcon(QtGui.QIcon(str(Path(__file__).with_name("optimizer.svg"))))
        # Only for runmanager's address: the light asks through a client of its own.
        client = RunmanagerClient()
        self.window = WindowController(ui, self.commands, client.host, client.port)
        self.shots = None
        self.worker = None

    def plugin_setup_complete(self, data):
        self.shots = data["shots_model"]
        self.worker = Worker(
            self._load_config,
            self.window,
            self.commands,
            data["output_box"].output,
            self.interface_factory,
        )
        self.window.link.start()

    def _load_config(self):
        # The worker's thread reads the path the window holds, a plain string.
        if not self.window.config_file:
            raise ValueError("No configuration file is chosen. Choose one above.")
        path = Path(self.window.config_file)
        return config_module.load(path), path.read_text(encoding="utf-8")

    def get_event_handlers(self):
        return {"analysis_done": self.on_analysis_done}

    def on_analysis_done(self, filepath, outcomes):
        """Hand the shot's cost to the session, and save its status when it answers."""
        config = self.worker.config
        # None until a configuration file has loaded.
        if config is None:
            return
        filepaths, observations = extract(inmain(self._row, filepath), config)
        reply = self.worker.hand_over(observations)
        reply.add_done_callback(partial(self._save_status, filepaths))

    def _row(self, filepath):
        # lyse keeps columns in the order added, and pandas warns about lexsort
        # depth when an unsorted MultiIndex is read by a key shallower than it,
        # as extract does.
        row = self.shots.row_number_by_filepath[filepath]
        return self.shots.dataframe.iloc[[row]].sort_index(axis=1)

    def _save_status(self, filepaths, reply):
        # On the worker's thread, which has reported any error.
        if reply.exception() is not None:
            return
        recorded, status = reply.result()
        updated = {}
        for filepath, source in zip(filepaths, recorded):
            # A source comes back only for a shot the session proposed and had
            # not yet recorded; runmanager mints an id for every row it compiles.
            if source is not None:
                values = status | {"phase": source}
                updated[filepath] = {
                    (GROUP, name): NO_VALUE_YET[name]
                    if values[name] is None
                    else values[name]
                    for name in SHOT_RESULTS
                }
        if updated:
            inmain_later(self.shots.update_rows, updated)

    def get_menu_contributions(self):
        return [
            {
                "location": "menubar",
                "path": "Optimizer",
                "name": "Show window",
                "action": self._show_window,
            }
        ]

    def _show_window(self, checked=False):
        self.window.ui.show()
        self.window.ui.raise_()
        self.window.ui.activateWindow()

    def get_save_data(self):
        ui = self.window.ui
        geometry = ui.geometry()
        return {
            "config_file": self.window.config_file,
            "start_from_runmanager": ui.start_from_runmanager.isChecked(),
            "visible": ui.isVisible(),
            "geometry": [
                geometry.x(), geometry.y(), geometry.width(), geometry.height()
            ],
        }

    def restore_save_data(self, data):
        ui = self.window.ui
        if "geometry" in data:
            ui.setGeometry(*data["geometry"])
        if "start_from_runmanager" in data:
            ui.start_from_runmanager.setChecked(data["start_from_runmanager"])
        if "visible" in data:
            ui.setVisible(data["visible"])
        if "config_file" in data:
            self.window.set_config_file(data["config_file"])

    def close(self):
        self.window.link.shutdown()
        if self.worker is not None:
            self.worker.quit()

    def get_remote_handler(self):
        return self

    def handle_start(self, from_runmanager=False):
        """Start or resume the optimization, as the window's Start does. With
        from_runmanager, a run's first Start opens at runmanager's current values."""
        self.commands.put(("start", None, from_runmanager))

    def handle_pause(self):
        """Pause the optimization, as the window's Pause does; start resumes it."""
        self.commands.put(("pause", None, None))

    def handle_reset(self):
        """End the session and open a new one, reading the configuration file again."""
        self.commands.put(("reset", None, None))

    def handle_set_config_file(self, path):
        """Choose the TOML configuration file; the session is reset to read it."""
        inmain(self.window.set_config_file, str(path))

    def handle_get_status(self):
        """Return the window's phase, the configuration file, the shot counts and
        the best cost and parameters so far."""
        return inmain(
            lambda: {
                "phase": self.window.ui.phase_value.text(),
                "config_file": self.window.config_file,
            }
            | self.window.status
        )
