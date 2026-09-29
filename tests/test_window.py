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

from queue import Queue

from labscript_optimization.window import WindowController


def test_window_shows_progress_controls_and_can_reopen(qt_application):
    commands = Queue()
    window = WindowController(commands)
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
    window.ui.start_button.click()
    assert commands.get_nowait()[0] == "start"

    window.update({"paused": False}, True, (), True, False)
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Batch computing"
    window.ui.pause_button.click()
    assert commands.get_nowait()[0] == "pause"
    window.ui.reset_button.click()
    assert commands.get_nowait()[0] == "reset"

    window.update({"stopped": "reached max_num_runs (2)"}, False, (), True, False)
    qt_application.processEvents()
    assert window.ui.phase_value.text() == "Ended: reached max_num_runs (2)"
    assert not window.ui.start_button.isEnabled()
    window.ui.close()
    assert not window.ui.isVisible()
    window.ui.show()
    assert window.ui.isVisible()
