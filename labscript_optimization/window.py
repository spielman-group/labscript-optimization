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

import pyqtgraph as pg
from qtutils import inmain_decorator
from qtutils.qt import QtGui, QtWidgets


class WindowController:
    def __init__(self, ui, command_queue):
        self.command_queue = command_queue
        self.ui = ui
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
        # The title row sits above the axes, so a legend there covers no data.
        plot_item = self.plot.getPlotItem()
        plot_item.layout.removeItem(plot_item.titleLabel)
        self.legend = pg.LegendItem(colCount=5)
        plot_item.layout.addItem(self.legend, 0, 1)
        self.listed = []
        self.ui.plot_layout.addWidget(self.plot)
        self.ui.parameters_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.ui.config_text.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        )
        colors = {
            "start": "#0072b2",
            "warmup": "#e69f00",
            "main": "#009e73",
            "explore": "#cc79a7",
        }
        self.points = {}
        for source, color in colors.items():
            self.add_points(source, color)
        self.best_line = self.plot.plot([], [], pen=pg.mkPen("#eeeeee", width=2))

    def add_points(self, source, color):
        self.points[source] = self.plot.plot(
            [], [], pen=None, symbol="o", symbolBrush=color
        )

    @inmain_decorator(wait_for_return=False)
    def show_config(self, config, text):
        self.ui.method_value.setText(config.learner.replace("_", " ").capitalize())
        self.ui.config_text.setPlainText(text)
        table = self.ui.parameters_table
        table.setRowCount(len(config.space.parameters))
        for row, parameter in enumerate(config.space.parameters):
            start = "—" if parameter.start is None else f"{parameter.start:g}"
            cells = (
                parameter.name,
                f"{parameter.minimum:g}",
                f"{parameter.maximum:g}",
                start,
            )
            for column, cell in enumerate(cells):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(cell))

    @inmain_decorator(wait_for_return=False)
    def update(self, status, computing, observations, gaussian_process, maximize):
        stopped = status.get("stopped")
        # An empty status is a session still opening.
        opening = not status
        paused = status.get("paused", True)
        if stopped:
            phase = f"Ended: {stopped}"
        elif opening:
            phase = "Opening"
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
        table = self.ui.parameters_table
        for row in range(table.rowCount()):
            cell = "—" if params is None else f"{params[row]:g}"
            table.setItem(row, 4, QtWidgets.QTableWidgetItem(cell))

        self.ui.start_button.setEnabled(not stopped and paused and not opening)
        self.ui.pause_button.setEnabled(not stopped and not paused)
        # Reset also retries an opening that failed.
        self.ui.reset_button.setEnabled("paused" in status or bool(stopped))

        points = {source: ([], []) for source in self.points}
        best_x, best_y = [], []
        best = None
        for shot, (source, cost) in enumerate(observations, start=1):
            if cost is None:
                continue
            if source not in self.points:
                self.add_points(source, pg.intColor(len(self.points)))
            x, y = points.setdefault(source, ([], []))
            x.append(shot)
            y.append(cost)
            best = cost if best is None else (max if maximize else min)(best, cost)
            best_x.append(shot)
            best_y.append(best)
        for source, (x, y) in points.items():
            self.points[source].setData(x, y)
        self.best_line.setData(best_x, best_y)

        # Listed only once it has points: most learners never propose some.
        # Rebuilt only when that changes, since new entries are drawn once
        # before the legend's layout places them.
        listed = [
            (self.points[source], source.title())
            for source, (x, _) in points.items()
            if x
        ]
        if best_x:
            listed.append((self.best_line, "Best so far"))
        if listed != self.listed:
            self.legend.clear()
            for item, name in listed:
                self.legend.addItem(item, name)
            self.listed = listed
