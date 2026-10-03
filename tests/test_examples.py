#####################################################################
#                                                                   #
# /tests/test_examples.py                                           #
#                                                                   #
# Copyright 2026, JQI                                               #
# Author: Ian Spielman                                              #
#                                                                   #
# This file is part of labscript-optimization, in the labscript     #
# suite (see http://labscriptsuite.org), and is licensed under the  #
# MIT License. See the LICENSE file in the root of the project.     #
#                                                                   #
#####################################################################

from pathlib import Path

from labscript_optimization import config

EXAMPLE = Path(__file__).resolve().parent.parent / 'examples' / 'optimization.lyse'


def test_the_example_configuration_loads():
    """The file every lab starts from is held to the schema like any other."""
    config.load(EXAMPLE / 'optimization_config.toml')
