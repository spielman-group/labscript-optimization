"""Machine-learning online optimization of labscript suite experiments.

A lyse plugin proposes shots, runmanager runs them, and the costs come back
through lyse. The learners are ordinary objects, driven through one method,
so they can be used on their own::

    import numpy as np

    from labscript_optimization.learners import GaussianProcessLearner
    from labscript_optimization.space import Parameter, ParameterSpace

    space = ParameterSpace([Parameter('x', 0.0, 1.0)])
    learner = GaussianProcessLearner(space, np.random.default_rng())
    history = []  # one Observation per proposal made so far
    proposals = learner.propose(history, hint=4)  # (params, source) pairs

In a lab, it is a lyse plugin, :class:`~labscript_optimization.plugin.Plugin`,
which lyse loads when its labconfig's ``[lyse/plugins]`` section has
``labscript_optimization = True``.
"""
