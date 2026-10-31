Troubleshooting
===============

Where errors appear
-------------------

A session that has stopped shows ``Ended:`` and the reason in the window's phase. lyse's output box holds tracebacks, in red, and one line, ``The optimization has stopped: <reason>``, printed once for each session. A configuration that does not load ends the session in the window, with the reason and no traceback.

How an error is reported depends on what the session was doing:

* During the opening, Start and Reset, and when a Gaussian process batch finishes, the message becomes the reason in the window. The traceback goes to lyse's output box, except for a configuration file that does not load at a Reset, which is yours to fix and prints none. lyse goes on analyzing.
* During the work after a shot, the window reads ``Ended: stopped by an error``, and the traceback goes to lyse's output box as soon as it happens. lyse goes on analyzing.
* When Start finds an error in runmanager's globals or cannot use its values for the start, or runmanager stops answering during a run, the session pauses and the window reads ``Paused:`` and the reason. Nothing is printed, and the session does not end.

The messages quoted below are the reason in the first and third cases and the error in the traceback in the second.

An ended session cannot be resumed. After fixing the cause, press Reset and then Start.

runmanager is not answering
---------------------------

The light beside "runmanager" in the window shows an exclamation mark, its tooltip gives the reason, and Start is disabled. Nothing is printed. Start runmanager. Once the light shows a tick, Start enables itself, with no Reset.

If runmanager stops answering during a run, for instance with its GUI inside a compile or behind an open dialog, the request it left unanswered pauses the session after the client's own timeout. The phase reads ``Paused: runmanager is not answering``. The shots in flight are kept, and no traceback is printed. While the light shows an exclamation mark, the session does not ask runmanager what became of its shots, so Reset and Pause do not wait for that timeout. When the light shows a tick again, press Start to resume the same run. A runmanager whose GUI is busy still answers the light, so Start can be enabled while its requests time out. The session then stays paused with the request's timeout as the reason.

If runmanager was restarted, and not just busy, it no longer knows the run's sequence. The next submission after Start is refused and the session ends; see :ref:`troubleshooting:A refused sequence join`.

No labscript file is selected
-----------------------------

Start leaves the session paused, and the phase reads ``Paused: No labscript file is selected; select a labscript file in runmanager``. Select the labscript file in runmanager, then press Start again.

The globals do not evaluate
---------------------------

Start leaves the session paused, and the phase reads ``Paused: runmanager reports an error in its globals; fix it before starting an optimization``. Fix the globals in runmanager, then press Start again.

A global that evaluates when you press Start can still stop a later submission. runmanager refuses a batch whose globals cannot be evaluated, and its message becomes the reason. A global the configuration sets that is in no active group in runmanager is refused at Start, which leaves the session paused with ``Global <name> not found in any active group in runmanager``. Only the Starts before the first one that goes check this. A global that leaves the active groups after that makes runmanager refuse the next submission, and the session ends with runmanager's reason.

Start from runmanager values is refused
---------------------------------------

With the box ticked, the run's first Start reads runmanager's current values of the parameters. If it cannot use them, the session stays paused and the phase reads ``Paused:`` and one of these reasons. Nothing is printed. Fix the value in runmanager, or untick the box, then press Start again.

* ``Cannot start from runmanager's values: <global> is <value>, outside the range <min> to <max> of parameter <name>``. The global's value is not between the parameter's ``min`` and ``max``.
* ``Cannot start from runmanager's values: <global> is <value>, not a number``. The global evaluates to something other than a real number, such as a string, a list, an array or a boolean.
* ``Cannot start from runmanager's values: parameter <name> reaches runmanager only through an expr, so its value cannot be read back``. Only a parameter with a global of its own, from ``global_name`` or from a global with no ``expr``, can be read from runmanager. Untick the box for this configuration.

Scan? or JIT? ticked
--------------------

runmanager refuses a submission that sets a global whose Scan? or JIT? box is ticked, and queues nothing, because a ticked global runs its scan values, or under JIT? the value in runmanager's window, instead of the value submitted. The session ends with runmanager's message, ``Cannot submit a value for a global with its Scan? or JIT? box ticked, since the shot would not run it: <globals>.``, which names every global that has a box ticked. It is the reason in the window when Start or a finished Gaussian process batch made the submission, and the error in the traceback when the work after a shot did; see :ref:`troubleshooting:Where errors appear`. Untick the boxes in runmanager, press Reset and then Start.

A scan on a global the configuration does not set makes runmanager refuse the submission, because the entry would not produce exactly one shot. Its message names the globals that expand it.

The labscript file changed during a session
-------------------------------------------

The session notes runmanager's labscript file at its first Start that is not refused and compares it before every submission. If it has changed, the session stops with ``the labscript file changed from '<old>' to '<new>' while this session was running; its shots would no longer be the experiment it has been optimizing``. Press Reset. The new session notes whichever file runmanager has when you press Start, and starts with an empty history.

A refused sequence join
-----------------------

A run is one runmanager sequence, and runmanager remembers a sequence only until it restarts. If runmanager restarts during a run, it refuses the next submission, and the session stops with runmanager's reason, such as ``Cannot add shots to sequence <id>: runmanager has no record of it, or has two and was not told which``. runmanager also refuses to join a sequence of a different labscript file. The plugin does not raise: it prints the reason once and lyse goes on analyzing. Press Reset. The new session starts a new sequence, with an empty history.

A missing cost column
---------------------

The plugin reads the cost from the column that ``cost_key`` names. If the column is not in the shot's row of lyse's dataframe when all analysis of the shot has ended, or the cost is NaN or infinite, the shot is recorded as a bad observation and nothing says why. Completed counts the shot, Best cost stays "—", the plot stays empty, and a Gaussian process stays in Warmup, which counts only usable observations.

Check that:

* ``cost_key`` names the group and the result as they are saved: the file name of a classic script without its ``.py``, or the class of a GUI routine, then the result.
* The routine that saves the cost is active in lyse, in either routine box.
* That routine does not fail on the shots.

A run with no usable cost still ends at the limits the configuration sets, ``no better parameters in <N> runs (max_num_runs_without_better_params)`` or ``reached max_num_runs (<N>)``, because both count every completed shot.

Starved counts
--------------

Starved counts each time the session ran and found none of its shots queued. runmanager then gave BLACS a default shot, so the apparatus kept running, but the optimizer did not get that shot. BLACS asks for its next shot as soon as it finishes one, which is before the optimizer has seen the cost and proposed a replacement. Raise ``num_buffered_runs`` in ``[GENERAL]``, which is 2 by default, and press Reset. The learners that top up a queue keep exactly that many of the session's shots in flight.

``differential_evolution`` proposes a whole generation and empties the queue once per generation, so it does not count starved shots. It refuses ``num_buffered_runs``, because its queue depth is its ``population_size``.

A stopped session thread
------------------------

If the session thread has stopped, the plugin raises ``The optimization session's thread has stopped, so nothing will answer. Restart lyse.`` at every shot. lyse logs it and goes on analyzing. Restart lyse: the history is lost with the thread.

The configuration does not load
-------------------------------

The session reads its configuration when a file is chosen, at each Reset, and when lyse starts with a file saved in its configuration. If no file is chosen, the window's phase reads ``Ended: No configuration file is chosen. Choose one above.`` If the file is unreadable or a setting is refused, the phase reads ``Ended:`` followed by the message, which names the setting. Nothing is printed, and there is no session until you fix the file and press Reset, or choose another file. :doc:`configuration` describes the settings.
