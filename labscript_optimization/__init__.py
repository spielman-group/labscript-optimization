"""Machine-learning online optimization of labscript suite experiments.

A lyse routine proposes shots, runmanager runs them, and the costs come back
through lyse. The learners are ordinary objects, driven through one method,
so they can be used on their own::

    from labscript_optimization.learners import GaussianProcessLearner
    from labscript_optimization.space import Parameter, ParameterSpace

    space = ParameterSpace([Parameter('x', 0.0, 1.0)])
    learner = GaussianProcessLearner(space, numpy.random.default_rng())
    proposals = learner.propose(history, hint=4)  # (params, source) pairs

In a lab, the entry point is a lyse routine folder whose ``lyse_routine.py``
holds a subclass of
:class:`~labscript_optimization.routine.OptimizationRoutine`::

    from labscript_optimization.routine import OptimizationRoutine

    class Optimization(OptimizationRoutine):
        config_path = 'optimization_config.toml'
"""
