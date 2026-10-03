API Reference
=============

The lyse routine
----------------

.. The class docstring's Attributes section already documents the two excluded.

.. automodule:: labscript_optimization.routine
    :members: OptimizationRoutine, extract, save_status, RESULTS_GROUP
    :exclude-members: config_path, interface_factory
    :undoc-members:
    :show-inheritance:

Configuration
-------------

.. automodule:: labscript_optimization.config
    :members: load, loads, Config
    :undoc-members:
    :show-inheritance:

The parameter space
-------------------

.. automodule:: labscript_optimization.space
    :members: Parameter, ParameterSpace
    :undoc-members:
    :show-inheritance:

Learners
--------

.. automodule:: labscript_optimization.learners
    :members: LEARNERS, build, knobs_by_learner, validate_options
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.base
    :members: Learner, ParameterSpaceLearner, InsufficientData
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.random
    :members: RandomLearner, DirectedRandomLearner
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.differential_evolution
    :members: DifferentialEvolutionLearner
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.gaussian_process
    :members: GaussianProcessLearner
    :undoc-members:
    :show-inheritance:
