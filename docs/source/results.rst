Results
=======

The session saves its progress into lyse's dataframe as lyse results in the group ``labscript_optimization``, in the row of each shot it proposed. Each key below is a column. For example, ``df[("labscript_optimization", "best_cost")]`` is the best cost so far:

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
    * - ``best_shot_id``
      - string
      - The shot id of the shot with the best cost.
    * - ``stopped``
      - string
      - Why the session stopped, as the window shows it after ``Ended:``. It is empty until the session stops.

``phase`` is the shot's own. It is the source recorded when the session proposed that shot, however far the run has moved on since, and it is not the phase of whatever was proposed most recently. The other four columns are the session's state when it took the costs of the pass that handed over the shot, so the shots of one pass share them.

The status counters are one answer for the whole run rather than anything about a shot, so they are not saved. The window shows them.

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
    * - ``phase``, ``best_shot_id``, ``stopped``
      - an empty string

The best columns are empty until a shot has a usable cost, and ``stopped`` is empty until the session stops. Shots that are saved after that carry the reason.

Which shots are written
-----------------------

The session writes onto a shot id it proposed and onto no other. runmanager mints a shot id for every queue row it compiles, so carrying one does not make a shot the session's. Your own shots, runmanager's default shots, and a rerun of a shot the session has already counted are left alone, and their rows have no value in these columns.

A shot is saved when the session takes its cost, in the pass that hands it over. When the session's reply is slower than the routine will wait, the shot is saved at the next pass. A save that fails is printed in the routine's output and passed over.
