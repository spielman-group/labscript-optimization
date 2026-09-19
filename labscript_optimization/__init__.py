"""Machine-learning online optimisation of labscript suite experiments.

A lyse routine proposes shots, runmanager runs them, and the costs come back
through lyse. The learners are ordinary objects with one method and no
threads, so they can be used on their own::

    from labscript_optimization.learners import GaussianProcessLearner
    from labscript_optimization.space import Parameter, ParameterSpace

    space = ParameterSpace([Parameter('x', 0.0, 1.0)])
    learner = GaussianProcessLearner(space, numpy.random.default_rng())
    proposals = learner.propose(history, k=4)

In a lab, the entry point is the routine::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('mloop_config.toml')
"""
