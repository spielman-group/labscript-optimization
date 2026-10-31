Getting started
===============

This page goes from installation to a running optimization. It assumes that runmanager, BLACS and lyse already run your experiment.

Install
-------

Install the package into the Python environment that lyse runs in, from a clone of the repository and with the ``lyse`` extra:

.. code-block:: bash

    git clone https://github.com/spielman-group/labscript-optimization
    pip install -e "./labscript-optimization[lyse]"

The plugin runs in lyse's own process, so an installation anywhere else is one lyse cannot import. The package needs Python 3.13 or newer. The extra adds lyse, runmanager, qtutils and pyqtgraph; labscript-utils is installed without it. The plugin needs the Development branches of lyse and runmanager from the Spielman group's labscript suite.

Enable the plugin
-----------------

The installation registers the plugin with lyse, under the ``lyse.plugins`` entry point, as ``labscript_optimization``. lyse loads a plugin only when its flag in the ``[lyse/plugins]`` section of the labconfig is true. Add the flag and restart lyse:

.. code-block:: ini

    [lyse/plugins]
    labscript_optimization = True

lyse's menu bar then has an Optimizer menu.

Write the cost routine
----------------------

The optimization needs one number per shot, the cost, which your own analysis computes. Write the analysis as a lyse routine, single-shot or multishot, a classic script or a GUI routine, that saves the cost as a lyse result:

.. code-block:: python

    import lyse

    run = lyse.Run(lyse.path)
    run.save_result("atom_number", atom_number)
    run.save_result("u_atom_number", atom_number_uncertainty)

Saved from a script named ``my_routine.py``, the result is the column ``("my_routine", "atom_number")`` of lyse's dataframe, because a classic script saves in a results group named for its file. A GUI routine that saves through ``get_run()`` uses a group named for its class. The configuration names the column with ``cost_key = ["my_routine", "atom_number"]``. The uncertainty ``u_atom_number`` beside it is optional. The Gaussian process uses it as the uncertainty on each cost. A shot whose cost is missing or NaN is recorded as a bad observation: it counts as a completed run and is left out of the fits.

The plugin reads the shot's row once all analysis of the shot has ended: its single-shot routines and, if any multishot routine is active, the multishot pass that includes it. The routine that saves the cost may be either kind, wherever it is in its box. The optimizer is not a routine and is added to no routine box.

Choose the configuration file
-----------------------------

``examples/optimization_config.toml`` in the repository is a configuration file to copy. Edit it for your experiment: ``cost_key`` and ``maximize`` in ``[ANALYSIS]``, the ``groups`` that take part, and one ``[PARAMETERS.<group>.<name>]`` table for each parameter, giving the runmanager global it sets (``global_name``), its ``min`` and ``max``, and optionally a ``start``. Every global must be in an active group in runmanager. :doc:`configuration` describes every key.

Show the optimizer window from lyse's menu bar, with Optimizer > Show window, and press Choose… beside "Configuration file" to select the file. The path is absolute, and lyse saves it with its configuration. Choosing a file resets the session to read it, and so does Reset, so press Reset after editing the file. The session opens paused, and the light beside "runmanager" shows whether runmanager is answering. Start is enabled once it does. A phase that starts with "Ended" means the session could not be built; with no file chosen it reads "Ended: No configuration file is chosen. Choose one above." See :doc:`troubleshooting`.

Start the optimization
----------------------

Check that runmanager is running with your labscript file and output folder set, that its globals evaluate, and that Scan? and JIT? are unticked for the globals the configuration sets. BLACS must be running to execute the shots. Then press Start in the window.

The session submits its first proposals, which begin with the configured start when every parameter has one. The counts and the plot in the window follow the shots as lyse analyzes them, and the status columns appear in lyse's dataframe. The run ends when it reaches ``max_num_runs`` or ``max_num_runs_without_better_params``, or when you quit lyse. When it ends, the best parameters are set as runmanager's values. Pause stops new submissions. :doc:`window` describes the window and :doc:`results` the columns.
