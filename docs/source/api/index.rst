API Reference
=============

.. automodule:: labscript_optimization

The lyse plugin
---------------

.. automodule:: labscript_optimization.plugin
    :members: Plugin, extract, GROUP, SHOT_RESULTS, NO_VALUE_YET
    :undoc-members:
    :show-inheritance:

Configuration
-------------

.. automodule:: labscript_optimization.config
    :members: load, loads, from_dict, Config, GlobalMapping
    :undoc-members:
    :show-inheritance:

The parameter space
-------------------

.. automodule:: labscript_optimization.space
    :members: Parameter, ParameterSpace
    :undoc-members:
    :show-inheritance:

Observations
------------

.. automodule:: labscript_optimization.observations
    :members: Observation, PENDING, DROPPED, COMPLETE
    :undoc-members:
    :show-inheritance:

Learners
--------

.. automodule:: labscript_optimization.learners
    :members: LEARNERS, build, validate_options
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.base
    :members: Learner, ParameterSpaceLearner
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.random
    :members: RandomLearner, DirectedRandomLearner
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.differential_evolution
    :members: DifferentialEvolutionLearner, STRATEGIES
    :undoc-members:
    :show-inheritance:

.. automodule:: labscript_optimization.learners.gaussian_process
    :members: GaussianProcessLearner, GaussianProcess, WARMUP_SOURCE, BATCH_SOURCE, EXPLORE_SOURCE
    :undoc-members:
    :show-inheritance:
