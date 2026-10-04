Troubleshooting
===============

Where errors appear
-------------------

A session that has stopped shows ``Ended:`` and the reason in the window's phase. The routine's Output dock holds tracebacks, and one line, ``The optimization has stopped: <reason>``, printed once for each session. A configuration that does not load is reported in lyse's own output box.

How an error is reported depends on what the session was doing:

* During the opening, Start and Reset, and when a Gaussian process batch finishes, the message becomes the reason in the window. The traceback goes to the Output dock. The routine does not raise, and lyse goes on analyzing.
* During the work after a shot, the window reads ``Ended: stopped by an error``. The routine raises the error in lyse, in that pass or the next, and lyse then pauses analysis. An error raised in the next pass has its traceback in the Output dock as soon as it happens. When you resume, the shots of the failed pass are part of the next pass.
* When Start finds an error in runmanager's globals, or runmanager stops answering during a run, the session pauses and the window reads ``Paused:`` and the reason. Nothing is printed, and the session does not end.

The messages quoted below are the reason in the first and third cases and the error the routine raises in the second.

An ended session cannot be resumed. After fixing the cause, press Reset and then Start.

runmanager is not answering
---------------------------

The light beside "runmanager" in the window shows an exclamation mark, its tooltip gives the reason, and Start is disabled. Nothing is printed. Start runmanager. Once the light shows a tick, Start enables itself, with no Reset.

If runmanager stops answering during a run, for instance with its GUI inside a compile or behind an open dialog, the request it left unanswered pauses the session after the client's own timeout. The phase reads ``Paused: runmanager is not answering``. The shots in flight are kept, and no traceback is printed. When the light shows a tick again, press Start to resume the same run. A runmanager whose GUI is busy still answers the light, so Start can be enabled while its requests time out. The session then stays paused with the request's timeout as the reason.

If runmanager was restarted, and not just busy, it no longer knows the run's sequence. The next submission after Start is refused and the session ends; see :ref:`troubleshooting:A refused sequence join`.

The globals do not evaluate
---------------------------

Start leaves the session paused, and the phase reads ``Paused: runmanager reports an error in its globals; fix it before starting an optimization``. Fix the globals in runmanager, then press Start again.

A global that evaluates when you press Start can still stop a later submission. runmanager refuses a batch whose globals cannot be evaluated, and its message becomes the reason. A ``global_name`` in the configuration that is in no active group in runmanager is refused with ``Global <name> not found in any active group``.

Scan? or JIT? ticked
--------------------

Before each submission the session checks the globals the configuration sets. If any has Scan? or JIT? ticked, the session stops with ``Untick Scan? and JIT? in runmanager for <globals>: their shots would not run the values this session submits.`` A ticked global runs its scan values, or under JIT? the value in runmanager's window, instead of the value submitted. Untick the boxes in runmanager, press Reset and then Start.

A scan on a global the configuration does not set makes runmanager refuse the submission, because the entry would not produce exactly one shot. Its message names the globals that expand it.

The labscript file changed during a session
-------------------------------------------

The session notes runmanager's labscript file at its first Start and compares it before every submission. If it has changed, the session stops with ``the labscript file changed from '<old>' to '<new>' while this session was running; its shots would no longer be the experiment it has been optimizing``. Press Reset. The new session notes whichever file runmanager has when you press Start, and starts with an empty history.

A refused sequence join
-----------------------

A run is one runmanager sequence, and runmanager remembers a sequence only until it restarts. If runmanager restarts during a run, it refuses the next submission, and the session stops with runmanager's reason, such as ``Cannot add shots to sequence <id>: runmanager has no record of it, or has two and was not told which``. runmanager also refuses to join a sequence of a different labscript file. The session does not raise: the routine prints the reason once and lyse goes on analyzing. Press Reset. The new session starts a new sequence, with an empty history.

A missing cost column
---------------------

The routine reads the cost from the column that ``cost_key`` names. If the column is not in lyse's dataframe when the routine runs, or the cost is NaN or infinite, the shot is recorded as a bad observation and nothing says why. Completed counts the shot, Best cost stays "—", the plot stays empty, and a Gaussian process stays in Warmup, which counts only usable observations.

Check that:

* ``cost_key`` names the group and the result as they are saved: the file name of a classic script without its ``.py``, or the class of a GUI routine, then the result.
* The routine that saves the cost is in the Singleshot routines box, or above the optimization routine in the Multishot routines box.
* That routine does not fail on the shots.

A run with no usable cost still ends at the limits the configuration sets, ``no better parameters in <N> runs (max_num_runs_without_better_params)`` or ``reached max_num_runs (<N>)``, because both count every completed shot.

Starved counts
--------------

Starved counts each time the session ran and found none of its shots queued. runmanager then gave BLACS a default shot, so the apparatus kept running, but the optimizer did not get that shot. BLACS asks for its next shot as soon as it finishes one, which is before the optimizer has seen the cost and proposed a replacement. Raise ``num_buffered_runs`` in ``[GENERAL]``, which is 2 by default, and restart the routine. The learners that top up a queue keep exactly that many of the session's shots in flight.

``differential_evolution`` proposes a whole generation and empties the queue once per generation, so it does not count starved shots. It refuses ``num_buffered_runs``, because its queue depth is its ``population_size``.

A stopped session thread
------------------------

If the session thread has stopped, the routine raises ``The optimization session's thread has stopped, so nothing will answer; its traceback is above. Restart the routine.`` at every shot, and lyse pauses analysis. The traceback is in the routine's Output dock. In lyse, right-click the routine and choose "restart worker process for selected routines", then resume analysis. The history is lost with the thread.

The folder is added as a singleshot routine
-------------------------------------------

A singleshot routine gets one shot file at a time, so the optimization routine raises ``labscript_optimization's routine runs on the shots of a multishot pass; add it to lyse's multishot routines.`` at every shot, and lyse pauses analysis. The window still opens. Remove the routine from the Singleshot routines box and add the folder to the Multishot routines box.

The configuration does not load
-------------------------------

The routine reads its configuration when lyse starts it. If the file is unreadable or a setting is refused, lyse's output box shows the message, which names the setting, and no window opens. lyse reports the error again at each analysis until you fix the file and restart the routine. :doc:`configuration` describes the settings.
