#####################################################################
#                                                                   #
# /labscript_optimization/window.py                                 #
#                                                                   #
# Copyright 2026, JQI                                               #
# Author: Ian Spielman                                              #
#                                                                   #
# This file is part of labscript-optimization, in the labscript     #
# suite (see http://labscriptsuite.org), and is licensed under the  #
# MIT License. See the LICENSE file in the root of the project.     #
#                                                                   #
#####################################################################

"""The optimizer's live controls and status window."""

from pathlib import Path

from qtutils import UiLoader, inmain_decorator
from qtutils.qt import QtWidgets


class OptimizerWindow(QtWidgets.QMainWindow):
    def closeEvent(self, event):
        event.ignore()
        self.hide()


class WindowController:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        path = Path(__file__).with_suffix(".ui")
        self.ui = UiLoader().load(str(path), OptimizerWindow())
        for name in ("start", "pause", "reset"):
            button = getattr(self.ui, f"{name}_button")
            button.clicked.connect(
                lambda checked=False, command=name: self.command_queue.put(
                    (command, None, None)
                )
            )

    @inmain_decorator(wait_for_return=False)
    def update(self, status, computing, sources, gaussian_process):
        stopped = status.get("stopped")
        paused = status.get("paused", True)
        if stopped:
            phase = f"Ended: {stopped}"
        elif paused:
            phase = "Paused"
        elif computing:
            phase = "Batch computing"
        elif gaussian_process and "main" in sources:
            phase = "Batch out"
        elif gaussian_process:
            phase = "Warmup"
        else:
            phase = "Running"
        self.ui.phase_value.setText(phase)

        for name in (
            "submitted", "completed", "awaiting", "dropped", "blocked", "starved"
        ):
            getattr(self.ui, f"{name}_value").setText(str(status.get(name, 0)))
        cost = status.get("best_cost")
        self.ui.best_cost_value.setText("—" if cost is None else f"{cost:g}")
        params = status.get("best_params")
        self.ui.best_params_value.setText("—" if params is None else str(params))

        self.ui.start_button.setEnabled(not stopped and paused)
        self.ui.pause_button.setEnabled(not stopped and not paused)
        self.ui.reset_button.setEnabled("paused" in status)
