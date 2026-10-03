Introduction
============

labscript-optimization optimizes an experiment online. You choose the runmanager globals to vary, and write a lyse analysis that computes a cost from each shot. The package proposes values for the globals, runmanager runs them as shots, and lyse returns the costs, until a limit is reached or you stop the run. Your labscript file does not change: the optimization only sets runmanager globals.

It is one lyse GUI routine, :class:`~labscript_optimization.routine.OptimizationRoutine`, in lyse's Multishot routines box. There is no program of its own to start.

How it works
------------

Behind the routine is one *session*: the history of every proposal made, one learner that makes them, and the rules that stop the run. The session runs on a thread of its own in the routine's lyse worker process, and the routine's window controls it. A cycle goes round three programs:

#. The session submits the learner's proposals to runmanager, one shot per proposal. runmanager compiles each shot and adds it to its running queue, which is never stopped or waited on. The globals in runmanager's window are left at the last proposal submitted.
#. BLACS runs the shots. lyse analyses each one, and your singleshot routine computes its cost.
#. lyse runs the optimization routine on the batch of shots it has just analysed. The routine reads each shot's id and cost from lyse's dataframe, hands them to the session and returns. The session takes the costs, saves its progress into the dataframe (see :doc:`results`), and submits what the learner proposes next.

A run is one runmanager sequence: the first submission starts it and every later one joins it.

Shot ids
~~~~~~~~

runmanager mints an id for every queue row it compiles and writes it into the shot file, and lyse reads it as the ``shot_id`` column. The session records the id of each shot it submits and matches costs to proposals by it. Shots can therefore come back in any order, and shots that are not the session's pass through unharmed: your own shots in the same queue, and the default shots runmanager makes to keep the apparatus busy, which carry no id. The session takes a cost for an id once, so a shot that BLACS reruns is not counted twice.

The session keeps no count of shots in flight. Each time the routine runs it asks runmanager which of the awaited shots can still produce a cost, and gives up on a shot that was cancelled or deleted instead of waiting for it for ever. The window counts such shots as dropped.

The session thread
~~~~~~~~~~~~~~~~~~

lyse runs a multishot routine inline, so a slow one delays every shot behind it. The routine therefore does no fitting and no runmanager traffic. It hands the observations to the session thread and waits at most two seconds for the reply, which the thread sends before it talks to runmanager again. A reply that arrives later is saved onto its shots at the next pass.

lyse runs the routine once per batch of analysed shots, not once per shot, and names every shot of the batch in the routine's ``paths``. The routine hands over all of them.

Learners
--------

The ``learner`` setting of the configuration chooses what proposes shots. :doc:`learners` describes each in full.

``random``
    Draws uniformly from the whole parameter space.

``directed_random``
    Draws near previously seen points, by default points with poor costs, so it explores rather than refines.

``differential_evolution``
    Evolves a population. It proposes a whole generation at once and waits for all of it.

``gaussian_process``
    Fits a Gaussian process to the usable observations and proposes the points it favours, in batches computed on a background thread. An explorer learner keeps runmanager's queue filled meanwhile.

Relation to M-LOOP
------------------

labscript-optimization replaces M-LOOP and its lyse plugin analysislib-mloop. It carries over the random, differential evolution and Gaussian process learners but not Nelder-Mead or the neural network, and keeps the shape of the configuration with some tables and settings renamed. It imports nothing from either, and needs only numpy, scipy and scikit-learn beside the suite. An analysislib-mloop configuration does not load unchanged: see :doc:`migrating`.
