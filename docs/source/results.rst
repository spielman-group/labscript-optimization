Results
=======

The plugin saves the session's progress into lyse's dataframe, in the group ``labscript_optimization``, in the row of each shot the session proposed. Each key below is a column. For example, ``df[("labscript_optimization", "best_cost")]`` is the best cost so far:

.. code-block:: python

    import lyse

    df = lyse.data()
    best_cost = df[("labscript_optimization", "best_cost")]

The columns are written to the dataframe alone and not into the shot files, so shots that lyse loads afresh come without them.

.. list-table::
    :header-rows: 1
    :widths: 18 12 70

    * - Column
      - Type
      - Value
    * - ``phase``
      - string
      - What proposed this shot. ``start`` is the configured start. Otherwise it is the source the learner gave the proposal: ``main`` for the learner itself, and for the Gaussian process's own batch points, and ``warmup`` or ``explore`` for the shots of the Gaussian process's explorer during and after warmup.
    * - ``best_cost``
      - float
      - The best usable cost so far, in the units and sign of your cost column. A maximized cost is saved as measured.
    * - ``best_params``
      - list of floats
      - The parameter values of the shot with the best cost, in real units, in the order of the window's parameter table.

``phase`` is the shot's own. It is the source recorded when the session proposed that shot, however far the run has moved on since, and it is not the phase of whatever was proposed most recently. The other two columns are the session's state when it took the shot's cost.

The status counters and the reason the session stopped are one answer for the whole run rather than anything about a shot, so they are not saved. The window shows them, and the reason is printed once to lyse's output box.

Empty values
------------

When the session has nothing to report for a key, the shot is saved with an empty value of that key's own type, so that every column has one type from the first shot on.

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Column
      - Empty value
    * - ``best_cost``
      - ``NaN``
    * - ``best_params``
      - an empty list

The best columns are empty until a shot has a usable cost.

Which shots are written
-----------------------

The session writes onto a shot id it proposed and onto no other. runmanager mints a shot id for every queue row it compiles, so carrying one does not make a shot the session's. Your own shots, runmanager's default shots, and a shot the session has already counted, rerun or analyzed again, are left alone, and their rows have no value in these columns.

A shot is saved when the session thread answers the plugin's hand-over of its cost, by which time lyse may have gone on to later shots. If lyse's dataframe has lost the shot's row, lyse warns in its output box and the save is dropped.
