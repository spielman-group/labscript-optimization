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

import pyqtgraph as pg
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
        for button, name in (
            (self.ui.start_button, "start"),
            (self.ui.pause_button, "pause"),
            (self.ui.reset_button, "reset"),
        ):
            button.clicked.connect(
                lambda checked=False, command=name: self.command_queue.put(
                    (command, None, None)
                )
            )
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(220)
        self.plot.setLabel("bottom", "Shot order")
        self.plot.setLabel("left", "Cost")
        self.plot.addLegend()
        self.ui.plot_layout.addWidget(self.plot)
        colors = {
            "start": "#0072b2",
            "warmup": "#e69f00",
            "main": "#009e73",
            "explore": "#cc79a7",
        }
        self.points = {
            source: self.plot.plot(
                [], [], pen=None, symbol="o", symbolBrush=color, name=source.title()
            )
            for source, color in colors.items()
        }
        self.best_line = self.plot.plot(
            [], [], pen=pg.mkPen("#eeeeee", width=2), name="Best so far"
        )

    @inmain_decorator(wait_for_return=False)
    def update(self, status, computing, observations, gaussian_process, maximize):
        stopped = status.get("stopped")
        paused = status.get("paused", True)
        if stopped:
            phase = f"Ended: {stopped}"
        elif paused:
            phase = "Paused"
        elif computing:
            phase = "Batch computing"
        elif gaussian_process and any(source == "main" for source, _ in observations):
            phase = "Batch out"
        elif gaussian_process:
            phase = "Warmup"
        else:
            phase = "Running"
        self.ui.phase_value.setText(phase)

        for label, name in (
            (self.ui.submitted_value, "submitted"),
            (self.ui.completed_value, "completed"),
            (self.ui.awaiting_value, "awaiting"),
            (self.ui.dropped_value, "dropped"),
            (self.ui.blocked_value, "blocked"),
            (self.ui.starved_value, "starved"),
        ):
            label.setText(str(status.get(name, 0)))
        cost = status.get("best_cost")
        self.ui.best_cost_value.setText("—" if cost is None else f"{cost:g}")
        params = status.get("best_params")
        self.ui.best_params_value.setText("—" if params is None else str(params))

        self.ui.start_button.setEnabled(not stopped and paused)
        self.ui.pause_button.setEnabled(not stopped and not paused)
        self.ui.reset_button.setEnabled("paused" in status)

        points = {source: ([], []) for source in self.points}
        best_x, best_y = [], []
        best = None
        for shot, (source, cost) in enumerate(observations, start=1):
            if cost is None:
                continue
            x, y = points[source]
            x.append(shot)
            y.append(cost)
            best = cost if best is None else (max if maximize else min)(best, cost)
            best_x.append(shot)
            best_y.append(best)
        for source, (x, y) in points.items():
            self.points[source].setData(x, y)
        self.best_line.setData(best_x, best_y)
