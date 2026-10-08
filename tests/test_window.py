#####################################################################
#                                                                   #
# /tests/test_window.py                                             #
#                                                                   #
# Copyright 2026, JQI                                               #
# Author: Ian Spielman                                              #
#                                                                   #
# This file is part of labscript-optimization, in the labscript     #
# suite (see http://labscriptsuite.org), and is licensed under the  #
# MIT License. See the LICENSE file in the root of the project.     #
#                                                                   #
#####################################################################

from pathlib import Path
from queue import Queue

from qtutils import UiLoader

from conftest import SESSION_CONFIG
from labscript_optimization import config as config_module
from labscript_optimization import window as window_module
from labscript_optimization.window import WindowController


def test_window_shows_progress_and_controls(qt_application, monkeypatch, tmp_path):
    commands = Queue()
    ui = UiLoader().load(str(Path(window_module.__file__).with_suffix('.ui')))
    config_file = tmp_path / "config.toml"
    # The indicator is never started, so nothing is asked.
    window = WindowController(ui, commands, "localhost", 42523, config_file)
    window.ui.show()
    qt_application.processEvents()

    window.update(
        {"paused": True, "submitted": 2, "completed": 1},
        False,
        (),
        True,
        True,
    )
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Paused"
    assert window.ui.submitted_value.text() == "2"
    assert window.ui.completed_value.text() == "1"
    # Start needs runmanager answering as well as a session that can start, and
    # needs no Reset once it does.
    assert not window.ui.start_button.isEnabled()
    # The worker is told of the first answer, whichever it is, and then of
    # each change.
    window.link.on_answer(False, "gone")
    assert commands.get_nowait() == ("link", None, False)
    window.link.on_answer(True, None)
    assert commands.get_nowait() == ("link", None, True)
    qt_application.processEvents()
    window.ui.start_button.click()
    assert commands.get_nowait() == ("start", None, False)

    # A Start can open the run at runmanager's values, and the table shows the
    # start the session will use.
    window.show_config(config_module.loads(SESSION_CONFIG), SESSION_CONFIG)
    window.update({"paused": True, "start": [0.25]}, False, (), True, False)
    qt_application.processEvents()
    assert window.ui.parameters_table.item(0, 3).text() == "0.25"
    window.ui.start_from_runmanager.setChecked(True)
    window.ui.start_button.click()
    assert commands.get_nowait() == ("start", None, True)

    # A Reset session has submitted nothing and can still restore the original
    # values. It has no cost yet, so there is no best to set.
    window.update({"paused": True, "restorable": True}, False, (), True, False)
    qt_application.processEvents()
    assert not window.ui.set_best_button.isEnabled()
    window.ui.restore_button.click()
    assert commands.get_nowait()[0] == "restore"
    best = {"paused": True, "submitted": 2, "best_cost": 1.0, "restorable": True}
    window.update(best, False, (), True, False)
    qt_application.processEvents()
    window.ui.set_best_button.click()
    assert commands.get_nowait()[0] == "set_best"

    # While the run is going, runmanager's values are the run's.
    window.update(best | {"paused": False}, True, (), True, False)
    qt_application.processEvents()
    assert not window.ui.set_best_button.isEnabled()
    assert not window.ui.restore_button.isEnabled()
    window.update({"paused": False}, True, (), True, False)
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Batch computing"
    window.ui.pause_button.click()
    assert commands.get_nowait()[0] == "pause"
    window.ui.reset_button.click()
    assert commands.get_nowait()[0] == "reset"

    # A pause that something other than the user caused says why, and Start
    # waits for runmanager to be heard again.
    reason = best | {"pause_reason": "runmanager is not answering"}
    window.update(reason, False, (), True, False)
    window.link.on_answer(False, "gone")
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Paused: runmanager is not answering"
    assert not window.ui.start_button.isEnabled()
    assert not window.ui.set_best_button.isEnabled()
    assert not window.ui.restore_button.isEnabled()

    # Answering again does not start a session that has ended, but does let its
    # values be set.
    window.link.on_answer(True, None)
    ended = best | {"paused": False, "stopped": "reached max_num_runs (2)"}
    window.update(ended, False, (), True, False)
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Ended: reached max_num_runs (2)"
    assert not window.ui.start_button.isEnabled()
    assert window.ui.set_best_button.isEnabled()
    assert window.ui.restore_button.isEnabled()

    # An empty status is a session opening: no control works until it has.
    window.update({}, False, (), True, False)
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Opening"
    buttons = (
        window.ui.start_button,
        window.ui.pause_button,
        window.ui.reset_button,
        window.ui.set_best_button,
        window.ui.restore_button,
    )
    assert not any(button.isEnabled() for button in buttons)

    # Each usable cost is plotted at the position of its shot, with a gap where
    # a shot has none, in a series for what proposed it, and a line follows the
    # best so far.
    shots = (("main", 2.0), ("explore", 3.0), ("main", None), ("main", 1.0))
    window.update({"paused": False}, False, shots, False, False)
    qt_application.processEvents()
    drawn = [
        (list(x), list(y))
        for x, y in (item.getData() for item in window.plot.listDataItems())
        if x is not None
    ]
    assert ([1, 4], [2.0, 1.0]) in drawn
    assert ([2], [3.0]) in drawn
    assert ([1, 2, 4], [2.0, 2.0, 1.0]) in drawn

    # The button and the right-click item open the configuration file, over
    # this window.
    opened = []
    monkeypatch.setattr(
        window_module,
        "open_in_editor",
        lambda path, parent=None: opened.append((path, parent)),
    )
    window.ui.edit_config_button.click()
    window.ui.edit_config_action.trigger()
    assert opened == [(config_file, window.ui)] * 2
