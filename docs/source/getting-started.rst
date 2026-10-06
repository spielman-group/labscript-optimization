Getting started
===============

This page goes from installation to a running optimization. It assumes that runmanager, BLACS and lyse already run your experiment.

Install
-------

Install the package into the Python environment that lyse runs in, from a clone of the repository and with the ``lyse`` extra:

.. code-block:: bash

    git clone https://github.com/spielman-group/labscript-optimization
    pip install -e "./labscript-optimization[lyse]"

The routine runs in a lyse worker process, so an installation anywhere else is one lyse cannot import. The package needs Python 3.11 or newer. The extra adds labscript-utils, lyse, runmanager, qtutils and pyqtgraph. The routine needs the Development branches of lyse and runmanager from the Spielman group's labscript suite.

Write the cost routine
----------------------

The optimization needs one number per shot, the cost, which your own analysis computes. Write the analysis as a singleshot routine, a classic script or a GUI routine in lyse's Singleshot routines box, that saves the cost as a lyse result:

.. code-block:: python

    import lyse

    run = lyse.Run(lyse.path)
    run.save_result("atom_number", atom_number)
    run.save_result("u_atom_number", atom_number_uncertainty)

Saved from a script named ``my_routine.py``, the result is the column ``("my_routine", "atom_number")`` of lyse's dataframe, because a classic script saves in a results group named for its file. A GUI routine that saves through ``get_run()`` uses a group named for its class. The configuration names the column with ``cost_key = ["my_routine", "atom_number"]``. The uncertainty ``u_atom_number`` beside it is optional. The Gaussian process uses it as the uncertainty on each cost. A shot whose cost is missing or NaN is recorded as a bad observation: it counts as a completed run and is left out of the fits.

lyse runs the singleshot routines on a shot before the multishot routines, so the cost is in the dataframe when the optimization routine runs. A cost computed by a multishot routine works only if that routine is above the optimization routine in the Multishot routines box, because lyse runs routines in list order.

Copy the example routine folder
-------------------------------

``examples/optimization.lyse`` in the repository is a routine folder to copy:

.. code-block:: text

    optimization.lyse/
        lyse_routine.py
        optimization_config.toml

The folder's name must end in ``.lyse``. It is the routine's name in lyse and the title of its window. ``lyse_routine.py`` is all the code the routine needs:

.. literalinclude:: ../../examples/optimization.lyse/lyse_routine.py
    :language: python
    :lines: 14-

``config_path`` is relative to the folder. Edit ``optimization_config.toml`` for your experiment: ``cost_key`` and ``maximize`` in ``[ANALYSIS]``, the ``groups`` that take part, and one ``[PARAMETERS.<group>.<name>]`` table for each parameter, giving the runmanager global it sets (``global_name``), its ``min`` and ``max``, and optionally a ``start``. Every global must be in an active group in runmanager. :doc:`configuration` describes every key. The routine reads the file when lyse starts it and again at each Reset, so press Reset after editing the file.

Add the folder to lyse
----------------------

In lyse's Multishot routines box, open the menu of the plus button, choose "Add a GUI routine folder (.lyse)…" and select the folder. The routine appears in the box and its window opens with a paused session, and the light beside "runmanager" shows whether runmanager is answering. Start is enabled once it does. A phase that starts with "Ended" means the session could not be built; see :doc:`troubleshooting`.

Start the optimization
----------------------

Check that runmanager is running with your labscript file and output folder set, that its globals evaluate, and that Scan? and JIT? are unticked for the globals the configuration sets. BLACS must be running to execute the shots. Then press Start in the window.

The session submits its first proposals, which begin with the configured start when every parameter has one. The counts and the plot in the window follow the shots as lyse analyses them, and the status columns appear in lyse's dataframe. The run ends when it reaches ``max_num_runs`` or ``max_num_runs_without_better_params``, or when you remove or restart the routine. When it ends, the best parameters are set as runmanager's values. Pause stops new submissions. :doc:`window` describes the window and :doc:`results` the columns.
