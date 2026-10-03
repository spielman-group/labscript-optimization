Configuration
=============

A TOML file configures one optimization session: the cost to optimize, the parameters to search and the runmanager globals they set, the learner, and what stops the run.

Where the file is read
----------------------

The routine's ``lyse_routine.py`` sets ``config_path`` on its :class:`~labscript_optimization.routine.OptimizationRoutine` subclass:

.. code-block:: python

    class Optimization(OptimizationRoutine):
        config_path = "optimization_config.toml"

``config_path`` is relative to the routine folder. The file is read once, when lyse starts the routine, so restart the routine after editing it. A file that is refused fails the routine's load, and the reason appears in lyse's output.

The example file
----------------

``examples/optimization.lyse/optimization_config.toml`` in the repository presents every option. A commented line shows a key at its default, or an example value where the key has no default. The sections below describe each table.

.. literalinclude:: ../../examples/optimization.lyse/optimization_config.toml
    :language: toml
    :caption: optimization_config.toml

``[ANALYSIS]``
--------------

What the session optimizes, and which parameter groups take part.

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``cost_key``
      - list of two strings
      - required
      - ``[routine_name, result_name]``, the lyse dataframe column that holds the cost. The column ``[routine_name, "u_<result_name>"]`` holds the uncertainty on the cost, and is read when it exists; only ``gaussian_process`` uses it. The cost must come from a singleshot routine, or from a multishot routine above this one in lyse's list. A shot whose cost column is missing, or whose cost is not finite, is recorded as a bad observation, which no learner uses to propose.
    * - ``maximize``
      - boolean
      - ``false``
      - ``true`` makes larger costs better. The cost is negated on the way in and the session reports ``best_cost`` in the sign of your cost column.
    * - ``groups``
      - list of strings
      - required
      - The groups whose ``[PARAMETERS...]`` and ``[RUNMANAGER_GLOBALS...]`` tables take part; see `Groups`_. A missing or empty list leaves no parameters, and the file is refused.

``[GENERAL]``
-------------

The session's own settings. A learner's knobs do not go here; they go in ``[LEARNER.<name>]``.

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``learner``
      - string
      - ``"gaussian_process"``
      - ``random``, ``directed_random``, ``differential_evolution`` or ``gaussian_process``. :doc:`learners` describes each.
    * - ``max_num_runs``
      - whole number, at least 1
      - no limit
      - The run budget. The session stops when this many shots have completed, and submits no more than the budget has room for: completed and still-awaited shots count against it, dropped shots do not. A shot with an unusable cost is completed. ``differential_evolution`` needs a budget of at least two generations, ``2 * population_size``.
    * - ``max_num_runs_without_better_params``
      - whole number, at least 1
      - no limit
      - The session stops when this many completed shots, in proposal order, follow the one holding the best cost. Every completed shot counts, including one with an unusable cost, so a detector that has died still ends the run.
    * - ``num_buffered_runs``
      - whole number, at least 1
      - 2
      - How many of the session's shots to keep in runmanager's queue. One shot in flight is the one BLACS is running, so at 2 one is waiting when BLACS asks for the next. At 1, the queue holds none of the session's shots at that moment and runmanager hands BLACS a default shot, which the window counts as ``starved``; raise it if ``starved`` keeps climbing. ``differential_evolution`` refuses it: its queue depth is its generation, ``population_size``.
    * - ``seed``
      - whole number, at least 0
      - unset
      - Seeds the random number generator the learners draw from. Unset, it is seeded from system entropy. A run under ``gaussian_process`` is not reproducible from its seed, because which slots a batch's points take depends on how long each fit takes.

When a limit is reached the session stops proposing and records the reason in the ``stopped`` result. Shots already in runmanager's queue still run, and their costs are still recorded.

.. _configuration-parameters:

``[PARAMETERS.<group>.<name>]``
-------------------------------

One searched parameter. ``<group>`` is a name of your choosing, which ``[ANALYSIS] groups`` lists to switch the group on. ``<name>`` is the parameter's name, unique among the searched parameters, and is what a global's ``args`` refers to.

Parameters are ordered by group, in the order each group first appears under ``[PARAMETERS]``, then by table within the group. A per-parameter ``trust_region`` list follows this order.

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``min``, ``max``
      - numbers
      - required
      - The bounds of the search, finite, with ``min`` below ``max``. They are in the units of the value the parameter hands to its global or to the global's ``expr``.
    * - ``start``
      - number, from ``min`` to ``max``
      - none
      - Where the run begins. The session proposes this point first and once, whichever learner runs, and its shot reads ``start`` in the ``phase`` column. It is one point over every searched parameter, so write ``start`` on all of them or on none.
    * - ``enable``
      - boolean
      - ``true``
      - ``false`` leaves the parameter out of the search. It gets no mapping, so its runmanager global keeps the value it already holds, and no global's ``args`` may name it.
    * - ``global_name``
      - string
      - none
      - The runmanager global that takes the parameter's value directly. Leave it out when the parameter reaches runmanager only through the ``args`` of a ``[RUNMANAGER_GLOBALS...]`` table. A searched parameter that no global takes is refused.

``[RUNMANAGER_GLOBALS.<group>.<name>]``
---------------------------------------

A runmanager global computed from one or more parameters. ``<name>`` is the global's name in runmanager, not a parameter's.

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``args``
      - list of strings
      - required
      - The names of the searched parameters that feed the global, from any listed group. They reach ``expr`` in this order, as Python floats.
    * - ``expr``
      - string
      - none
      - The source of a Python lambda taking ``args`` in order, such as ``"lambda t0, dt: t0 + dt"``. Without it, ``args`` must name one parameter and the global takes its value. The expression is evaluated with ``eval`` when the file loads, so the file is as trusted as your analysis routines. A lambda that cannot be built, is not callable, or cannot take that many arguments is refused at load. It sees Python's builtins and its arguments only: a free name such as ``math`` loads without complaint and fails when the first proposal is turned into globals.
    * - ``enable``
      - boolean
      - ``true``
      - ``false`` leaves the global unset.

Global names are unique across the listed groups, whether they come from a ``global_name`` or from a ``[RUNMANAGER_GLOBALS...]`` table, because each global is set once per shot.

Every global the session sets must exist in an active runmanager group, with Scan? and JIT? unticked. The load does not check this. runmanager refuses a name no active group has, and the session refuses to submit while a global it sets has Scan? or JIT? ticked; either stops the session with an error. After each submission runmanager's globals hold the last values submitted.

Groups
------

A group is the name shared by the tables written under it, in ``[PARAMETERS]`` and ``[RUNMANAGER_GLOBALS]`` alike. ``[ANALYSIS] groups`` lists the groups that take part.

* A listed group's parameters are searched, and its globals are set.
* A group that is defined and not listed is switched off, and nothing under it is part of the session. Its keys, required keys and value types are still checked; bounds, ``args`` and ``expr`` are checked only once the group is listed.
* A name in ``groups`` that no table defines is refused, because it is a misspelling that would otherwise leave its group switched off without a word.
* ``enable = false`` on a parameter or global inside a listed group switches off that one entry.

A global's ``args`` may name a parameter in another listed group.

``[LEARNER.<name>]``
--------------------

The knobs of the learner ``<name>``. A key the learner does not take is refused, and every value is held to the kind its knob takes, so a quoted ``"false"``, a fraction where a whole number belongs, or one number where a pair belongs is refused rather than converted. A table can be left out, and a knob left out takes its default.

The tables for learners the session does not build are checked for their keys but not read, so one file can carry the settings of several learners and switch between them with ``[GENERAL] learner``. ``gaussian_process`` also builds its explorer, the learner its ``explorer`` knob names, from that learner's own table. A knob both take, such as ``trust_region``, is written twice, once in each table, and each gets its own value.

A ``trust_region`` is written as a number in (0, 1), which is a fraction of each parameter's range, or as a list of numbers, which are absolute distances, one per searched parameter in the :ref:`order the parameters are given <configuration-parameters>`. Each distance is positive and no wider than its parameter's range.

.. _configuration-learner-random:

``random``
~~~~~~~~~~

Takes no knobs. A table written for it must be empty.

.. _configuration-learner-directed-random:

``directed_random``
~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``trust_region``
      - number or list
      - 0.05
      - The largest distance of a draw from its center, in the form described above.
    * - ``trust_range``
      - ordered pair in [0, 1]
      - ``[0.1, 0.25]``
      - Which observed costs may center a draw: 0 is the worst cost seen and 1 the best. ``[1, 1]`` centers on the best point.
    * - ``trust_gaussian``
      - boolean
      - ``false``
      - Draw from a Gaussian of width ``trust_region`` about the center, clipped to the bounds, instead of uniformly within ``trust_region``.
    * - ``explore_fraction``
      - number in [0, 1]
      - 0.0
      - The share of draws made uniformly over the whole space.

.. _configuration-learner-differential-evolution:

``differential_evolution``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``population_size``
      - whole number
      - 8
      - The number of members, which is also the number of shots in a generation and the queue depth. It must exceed the number of other members the strategy draws on: at least 3 for ``best1``, 4 for ``rand1``, 5 for ``best2`` and 6 for ``rand2``.
    * - ``evolution_strategy``
      - string
      - ``"best1"``
      - The mutation: ``best1``, ``best2``, ``rand1`` or ``rand2``.
    * - ``mutation_scale``
      - pair of numbers, ``[low, high]``
      - ``[0.5, 1.0]``
      - The range the differential weight is drawn from, uniformly, once per generation. ``low`` is not negative and not above ``high``. For a fixed weight write the same number twice.
    * - ``cross_over_probability``
      - number in [0, 1]
      - 0.7
      - The chance that a coordinate of a trial comes from the mutant rather than from the member it replaces.
    * - ``trust_region``
      - number or list
      - unset
      - A point for a slot with no member to evolve, once any member has a cost, and the replacement for a coordinate outside the bounds, are drawn within this distance of the best member. Unset, they are drawn over the whole space.

.. _configuration-learner-gaussian-process:

``gaussian_process``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 18 16 14 52

    * - Key
      - Type
      - Default
      - Meaning
    * - ``explorer``
      - string
      - ``"directed_random"``
      - The learner that proposes the warmup shots and fills the queue while a batch is computed: ``random``, ``directed_random`` or ``differential_evolution``, with its knobs in its own table.
    * - ``warmup_observations``
      - whole number, at least 1
      - ``max(5, 2 * n)``, for ``n`` searched parameters
      - The number of usable observations the explorer gathers before the Gaussian process proposes. A shot with an unusable cost, or one given up on, does not count.
    * - ``batch_size``
      - whole number, at least 1
      - 4
      - The number of points computed together, each conditioned on the ones before it.
    * - ``explore_runs``
      - whole number, at least 0
      - 1
      - The least number of explorer shots in each batch cycle. 0 guarantees none and leaves the explorer to fill the queue.
    * - ``uncer_bias``
      - number or list of numbers
      - ``[0.0, 1.0, 2.0, 3.0]``
      - The weights on predicted uncertainty, one per point of a batch, starting again at the first for every batch. A number is that weight on every point, and an empty list is refused. A weight of 0 is purely greedy and a larger one looks further from measured points.
    * - ``cost_bias``
      - number
      - 1.0
      - The weight on the predicted cost.
    * - ``trust_region``
      - number or list
      - unset
      - Restricts the search to this distance around the best point seen. Unset, it searches the whole space.
    * - ``cost_has_noise``
      - boolean
      - ``true``
      - Adds a white-noise term to the kernel. ``false`` says every cost is measured exactly, and then a history in which only some shots carry an uncertainty is refused when a batch is computed.
    * - ``length_scale_bounds``
      - pair of numbers, ``[low, high]``
      - ``[1e-2, 1e2]``
      - The bounds on each parameter's length scale, in units of the unit cube the parameters are scaled onto. The load checks only that there are two numbers, and the fit expects them positive and in order. When a fit leaves length scales at an end of the bounds, a warning names the parameters, and is repeated only when that set changes.
    * - ``noise_level_bounds``
      - pair of numbers, ``[low, high]``
      - ``[1e-5, 1e1]``
      - The bounds on the white-noise level, in units of the standardized cost. Read only when ``cost_has_noise`` is on, and checked as ``length_scale_bounds`` is.

How a file is checked
---------------------

The whole file is checked when it loads, and the first problem stops the load with a message that names the setting.

* A key that its table does not know is refused, with the keys the table accepts. This covers the top-level tables, ``[ANALYSIS]``, ``[GENERAL]``, each parameter and global table, and each ``[LEARNER.<name>]`` table, which is held to its learner's constructor. A ``[LEARNER.<name>]`` table for a learner that does not exist is refused, as is a ``learner`` of that name.
* A learner's knob written in ``[GENERAL]`` is refused, naming the ``[LEARNER.<name>]`` tables it can go in. ``[GENERAL]`` is read once for the session, and a session can run two learners, so a knob there could not tell them apart.
* A required key that a table leaves out is refused. ``min`` and ``max`` are required in a parameter table, and ``args`` in a globals table.
* A table written one level short, such as ``[PARAMETERS.<name>]`` with the settings under it, is refused, because only the innermost table carries settings.
* The names this package replaced are refused with their replacements named: the tables ``[MLOOP]`` and ``[MLOOP_PARAMS]``, and six old Gaussian process settings. :doc:`migrating` has the mapping. Any other setting the package no longer has is refused as an unknown key.
* Values are held to their kind and never converted. A boolean is written ``true`` or ``false`` unquoted, and a list in brackets.
* ``num_buffered_runs`` beside ``differential_evolution``, and a ``max_num_runs`` below two generations of it, are refused.
* A parameter that is searched and reaches no global, a global whose ``args`` name a parameter that is not searched, and a repeated parameter or global name are refused.

What the load cannot know is found later: that the globals exist in runmanager, a free name in an ``expr``, whether the Gaussian process's bounds are positive and in order, and the values in the table of a learner the session does not build.
