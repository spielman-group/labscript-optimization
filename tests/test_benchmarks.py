"""The benchmark harness still drives the learner that ships.

``benchmarks/de_pipeline.py`` is not part of the package and its sweeps take
minutes, so the suite runs one seed of it at a budget of twelve shots. What
that proves is worth the second it costs: a harness nobody runs rots into
something that cannot be run at all, and the numbers in
``benchmarks/README.md`` point at it.

It also holds the harness to the rule the numbers depend on -- every arm runs
to the same number of *completed* shots, and nothing is left in the queue
unscored. Counting proposals submitted instead scores the arms on different
numbers of shots, and that is invisible in a sweep's output: the rows look
ordinary and only the aggregate moves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import de_pipeline  # noqa: E402


def test_every_benchmark_arm_completes_the_budget_it_was_given():
    rows = [de_pipeline.run(*job) for job in de_pipeline.jobs("smoke")]

    assert {(row["variant"], row["function"]) for row in rows} == {
        (variant, function)
        for variant in de_pipeline.VARIANTS
        for function in de_pipeline.FUNCTIONS
    }
    for row in rows:
        # The budget is a count of completed shots, so an arm that stops with
        # shots still in flight has been scored on fewer of them than another
        # -- which is what charged the generational arm for a whole generation
        # it was never allowed to finish.
        assert row["completed"] == row["budget"], row
        # And nothing is submitted once the budget is claimed, so the two arms
        # cost the apparatus the same as well.
        assert row["submitted"] == row["budget"], row
        assert row["dropped"] == 0, row
        assert row["best"] is not None, row
