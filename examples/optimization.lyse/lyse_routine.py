#####################################################################
#                                                                   #
# /examples/optimization.lyse/lyse_routine.py                       #
#                                                                   #
# Copyright 2026, JQI                                               #
# Author: Ian Spielman                                              #
#                                                                   #
# This file is part of labscript-optimization, in the labscript     #
# suite (see http://labscriptsuite.org), and is licensed under the  #
# MIT License. See the LICENSE file in the root of the project.     #
#                                                                   #
#####################################################################

"""A lyse routine folder that runs labscript-optimization.

Copy the folder and edit optimization_config.toml for your lab. In lyse's
Multishot routines box, choose "Add a GUI routine folder (.lyse)…" and select
the folder. The cost column the configuration names must come from a singleshot
routine, or from a multishot routine above this one.
"""

from labscript_optimization.routine import OptimizationRoutine


class Optimization(OptimizationRoutine):
    config_path = "optimization_config.toml"
