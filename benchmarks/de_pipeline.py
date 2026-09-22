"""Measure the differential evolution learner that ships, end to end.

Each run is a real :class:`~labscript_optimization.session.Session`, built by
the shipped configuration loader from a real configuration file, with a fake
runmanager standing in for the queue. The generation barrier in ``refill``, the
position walk in ``replay`` and the history's pending entries are therefore
exercised as they are in a lab, rather than by a reimplementation written to
match them.

Two variants, differing from each other in one thing only:

``generational``
    The shipped learner. Its ``generation`` declaration makes the session
    submit one whole population and ask for nothing until all of it has been
    answered for.

``asynchronous``
    The same shipped learner with the barrier lifted and nothing else changed
    -- a subclass whose only content is ``generation = None`` -- driven by the
    same session at ``num_buffered_runs`` depth. Same replay, same mutation,
    same crossover, same bounds handling. It is a reference point for feedback
    latency, not an alternative on offer, since it also runs the apparatus at a
    shallower queue.

**Every arm runs to the same number of completed shots.** The session's budget
is ``max_num_runs`` against shots *completed*, and it submits nothing once that
budget is claimed, so the loop here simply runs until the session stops and
lands every shot it submitted. Counting submissions instead is the defect that
invalidated every benchmark taken before 2026-09-19: it stops with a whole
generation still in the queue, and charges the generational arm for shots it
was never allowed to use. See ``README.md`` beside this file.

This is a benchmark, not a test: it measures quality, where the suite in
``tests/`` proves behaviour. Its standing value is regression checking of what
ships. The one part of it the suite does run is the ``smoke`` sweep, which
proves the harness still drives the current learner.

Usage::

    python benchmarks/de_pipeline.py run barrier --out benchmarks/results/barrier_4d.csv
    python benchmarks/de_pipeline.py report barrier benchmarks/results/barrier_4d.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from labscript_optimization import config as config_module
from labscript_optimization.learners import DifferentialEvolutionLearner
from labscript_optimization.session import Session
from labscript_optimization.space import Parameter, ParameterSpace

#: Every function is searched on the same box, as the earlier sweeps did.
BOX = (-5.12, 5.12)

#: The four analytic test functions, each taking its dimension from the point
#: it is handed. All four have their minimum at the origin except Rosenbrock,
#: whose minimum is at ``(1, ..., 1)`` -- both inside the box.
FUNCTIONS = {
    "rastrigin": lambda p: float(
        10 * len(p) + np.sum(p**2 - 10 * np.cos(2 * np.pi * p))
    ),
    "sphere": lambda p: float(np.sum(p**2)),
    "ackley": lambda p: float(
        -20 * np.exp(-0.2 * np.sqrt(np.mean(p**2)))
        - np.exp(np.mean(np.cos(2 * np.pi * p)))
        + 20
        + np.e
    ),
    "rosenbrock": lambda p: float(
        np.sum(100 * (p[1:] - p[:-1] ** 2) ** 2 + (1 - p[:-1]) ** 2)
    ),
}

#: The variants this harness can run. Ordered as the tables print them.
VARIANTS = ("generational", "asynchronous")

#: Variants no longer runnable from this checkout, kept so that archived rows
#: naming them still load and still print. ``former`` is the pre-slice-4
#: learner, which was deleted rather than copied here; see ``README.md``.
ARCHIVED_VARIANTS = ("former", "former_norestart")

FIELDS = (
    "variant",
    "function",
    "dimension",
    "population_size",
    "budget",
    "seed",
    "best",
    "completed",
    "submitted",
    "dropped",
    "starved",
    "refills",
)


class AsynchronousDifferentialEvolution(DifferentialEvolutionLearner):
    """The shipped learner with the generation barrier lifted, and nothing else.

    A session asks a learner declaring no generation for enough proposals to
    keep ``num_buffered_runs`` in the queue, whenever there is room. Every
    other line of the algorithm is the shipped one.
    """

    @property
    def generation(self):
        return None


class FakeRunmanager:
    """Stands in for runmanager. Every shot runs; none is lost.

    The same answers ``tests/conftest.py``'s fake gives, with sets rather than
    a list so that a six-hundred-shot session does not spend its time in
    membership tests.
    """

    def __init__(self):
        self.count = 0
        self.finished = set()

    def check_ready(self):
        pass

    def check_unchanged(self):
        pass

    def submit(self, proposals):
        ids = [f"shot-{self.count + i}" for i in range(len(proposals))]
        self.count += len(ids)
        return ids

    def shot_status(self, shot_ids):
        return {
            shot_id: (
                {"pending": False, "state": "unknown"}
                if shot_id in self.finished
                else {"pending": True, "state": "running"}
            )
            for shot_id in shot_ids
        }


def configuration(dimension, budget, learner, population_size, seed, buffered=None):
    """A configuration file, as text, for one run of a sweep.

    Written as TOML and parsed by the shipped loader, so that the session under
    measurement is built the way a lab's is, refusals and all.
    """
    lines = [
        "[ANALYSIS]",
        'cost_key = ["bench", "cost"]',
        'groups = ["G"]',
        "[MLOOP]",
        f'learner = "{learner}"',
        f"max_num_runs = {budget}",
        f"seed = {seed}",
    ]
    if buffered is not None:
        lines.append(f"num_buffered_runs = {buffered}")
    # The population belongs to the learner that evolves one. The random arm
    # writes the table too and never reads it: it is the same file with one
    # learner name changed, which is what makes the two arms comparable.
    lines += [
        "[LEARNER.differential_evolution]",
        f"population_size = {population_size}",
    ]
    for i in range(dimension):
        lines += [
            f"[MLOOP_PARAMS.G.p{i}]",
            f'global_name = "g{i}"',
            f"min = {BOX[0]}",
            f"max = {BOX[1]}",
        ]
    return "\n".join(lines) + "\n"


def run(variant, function, dimension, population_size, budget, seed, depth=3):
    """One optimisation. Returns a row of what it spent and what it found.

    ``generational`` goes all the way through the shipped loader, which builds
    the learner and holds the budget to two whole generations. The asynchronous
    reference is not a learner a configuration can name, so the session is
    handed one directly; everything else about the session is the same.

    One shot completes per turn of the loop, drawn uniformly from those in
    flight, so costs arrive out of order as they do in a lab. No drops and no
    NaN costs: this measures sample efficiency, and the role invariants under
    loss are the test suite's business.
    """
    cost = FUNCTIONS[function]
    interface = FakeRunmanager()

    if variant == "generational":
        config = config_module.loads(
            configuration(
                dimension, budget, "differential_evolution", population_size, seed
            )
        )
        session = Session(config, interface)
    elif variant == "asynchronous":
        config = config_module.loads(
            configuration(
                dimension, budget, "random", population_size, seed, buffered=depth
            )
        )
        space = ParameterSpace([Parameter(f"p{i}", *BOX) for i in range(dimension)])
        session = Session(
            config,
            interface,
            AsynchronousDifferentialEvolution(
                space, np.random.default_rng(seed), population_size=population_size
            ),
        )
    else:
        raise ValueError(
            f"unknown variant {variant!r}; this harness runs {VARIANTS}. "
            f"{ARCHIVED_VARIANTS} were measured against code that has since "
            f"been deleted -- see benchmarks/README.md"
        )

    order = np.random.default_rng(seed + 991)
    in_flight = []
    refills = 0
    while not session.stopped:
        session.reconcile()
        submitted = session.refill()
        if submitted:
            refills += 1
            in_flight.extend(submitted)
        if not in_flight:
            break
        shot_id = in_flight.pop(int(order.integers(len(in_flight))))
        interface.finished.add(shot_id)
        session.record(shot_id, cost(session.proposals[shot_id]), None, False)

    status = session.status()
    return {
        "variant": variant,
        "function": function,
        "dimension": dimension,
        "population_size": population_size,
        "budget": budget,
        "seed": seed,
        "best": status["best_cost"],
        "completed": status["completed"],
        "submitted": status["submitted"],
        "dropped": status["dropped"],
        "starved": status["starved"],
        "refills": refills,
    }


def jobs(sweep, seeds=None):
    """The argument tuples one named sweep is made of, in table order.

    ``barrier``
        What the generation barrier costs. Four functions at four parameters,
        N over 8 / 16 / 60, budgets of 120 / 240 / 600 shots, 16 seeds. The
        twelve (function, N) cells at each budget are the cells the ratios and
        the win counts are taken over.
    ``dimension``
        How many members, against how many parameters. Four functions at two,
        four and eight parameters, N over the multiples of the parameter count
        between 4 and 32, budgets of 240 and 600 with 1200 added at eight
        parameters, 12 seeds.
    ``smoke``
        One seed at a budget of twelve shots over every variant and every
        function: enough to prove the harness still drives the current learner,
        and small enough for the test suite to run it.
    """
    if sweep == "barrier":
        seeds = 16 if seeds is None else seeds
        for function in FUNCTIONS:
            for population_size in (8, 16, 60):
                for budget in (120, 240, 600):
                    for variant in VARIANTS:
                        for seed in range(seeds):
                            yield (
                                variant, function, 4, population_size, budget, seed
                            )
    elif sweep == "dimension":
        seeds = 12 if seeds is None else seeds
        for dimension in (2, 4, 8):
            populations = [
                k * dimension
                for k in (1, 2, 4, 8, 16)
                if 4 <= k * dimension <= 32
            ]
            budgets = (240, 600, 1200) if dimension >= 8 else (240, 600)
            for function in FUNCTIONS:
                for population_size in populations:
                    for budget in budgets:
                        for variant in VARIANTS:
                            for seed in range(seeds):
                                yield (
                                    variant,
                                    function,
                                    dimension,
                                    population_size,
                                    budget,
                                    seed,
                                )
    elif sweep == "smoke":
        seeds = 1 if seeds is None else seeds
        for variant in VARIANTS:
            for function in FUNCTIONS:
                for seed in range(seeds):
                    yield (variant, function, 2, 4, 12, seed)
    else:
        raise ValueError(f"unknown sweep {sweep!r}")


def load(paths):
    """Raw rows from one or more results files, gathered into cells.

    A cell is one (variant, function, dimension, population_size, budget); its
    rows are one per seed. Lines beginning with ``#`` are the provenance header
    every results file carries.
    """
    cells = defaultdict(list)
    for path in paths:
        with open(path) as f:
            reader = csv.DictReader(line for line in f if not line.startswith("#"))
            for row in reader:
                cells[
                    (
                        row["variant"],
                        row["function"],
                        int(row["dimension"]),
                        int(row["population_size"]),
                        int(row["budget"]),
                    )
                ].append(row)
    return cells


def median(cells, key):
    return float(np.median([float(row["best"]) for row in cells[key]]))


def cell(cells, key):
    best = [float(row["best"]) for row in cells[key]]
    return f"{np.median(best):.3f} / {np.mean(best):.3f}"


def report_barrier(cells):
    """The tables Issue 2 cites for the barrier and the walk it replaced."""
    functions = list(FUNCTIONS)
    populations = (8, 16, 60)
    budgets = (120, 240, 600)
    present = [
        v
        for v in VARIANTS + ARCHIVED_VARIANTS
        if any(key[0] == v for key in cells)
    ]
    titles = {
        "generational": "generational",
        "asynchronous": "asynchronous d3",
        "former": "former d3",
        "former_norestart": "former d3, no restart",
    }

    print("#### The whole 4-D table, median / mean of the best cost found\n")
    print("| function | N | shots | " + " | ".join(titles[v] for v in present) + " |")
    print("|---|---|---|" + "---|" * len(present))
    for function in functions:
        for population_size in populations:
            for budget in budgets:
                row = [function, str(population_size), str(budget)]
                row += [
                    cell(cells, (v, function, 4, population_size, budget))
                    for v in present
                ]
                print("| " + " | ".join(row) + " |")

    print("\n#### The barrier's price\n")
    print("| shots | generational / asynchronous, median over the 12 cells "
          "| cells where generational is worse |")
    print("|---|---|---|")
    for budget in budgets:
        ratios = [
            median(cells, ("generational", function, 4, population_size, budget))
            / median(cells, ("asynchronous", function, 4, population_size, budget))
            for function in functions
            for population_size in populations
        ]
        worse = sum(1 for r in ratios if r > 1)
        print(f"| {budget} | {np.median(ratios):.2f} | {worse} of {len(ratios)} |")

    if "former" not in present:
        return

    print("\n#### The correct algorithm against the walk it replaced\n")
    print("| function | " + " | ".join(f"{b} shots" for b in budgets) + " | all |")
    print("|---|---|---|---|---|")
    overall = defaultdict(int)
    for function in functions:
        row = [function]
        won = 0
        for budget in budgets:
            wins = sum(
                1
                for population_size in populations
                if median(cells, ("generational", function, 4, population_size, budget))
                < median(cells, ("former", function, 4, population_size, budget))
            )
            overall[budget] += wins
            won += wins
            row.append(f"{wins} of {len(populations)}")
        row.append(f"{won} of {len(populations) * len(budgets)}")
        print("| " + " | ".join(row) + " |")
    total = sum(overall.values())
    cells_per_budget = len(functions) * len(populations)
    print(
        "| **all four** | "
        + " | ".join(f"**{overall[b]} of {cells_per_budget}**" for b in budgets)
        + f" | **{total} of {cells_per_budget * len(budgets)}** |"
    )

    print("\nWhere the former walk's advantage comes from: cells of twelve in "
          "which it beats\nthe generational learner, with its restart on and off.\n")
    for budget in budgets:
        counts = []
        for variant in ("former", "former_norestart"):
            counts.append(
                sum(
                    1
                    for function in functions
                    for population_size in populations
                    if median(cells, (variant, function, 4, population_size, budget))
                    < median(
                        cells, ("generational", function, 4, population_size, budget)
                    )
                )
            )
        print(
            f"  {budget:4d} shots: restart on {counts[0]} of 12, "
            f"restart off {counts[1]} of 12"
        )

    print("\nAnd the largest population, against the other two: cells of twelve "
          "(function,\nbudget) in which N=60 holds the worst median of the three.\n")
    for variant in ("generational", "asynchronous"):
        worst = sum(
            1
            for function in functions
            for budget in budgets
            if median(cells, (variant, function, 4, 60, budget))
            == max(
                median(cells, (variant, function, 4, n, budget)) for n in populations
            )
        )
        print(f"  {variant}: {worst} of 12")


def report_dimension(cells):
    """The tables Issue 2 cites for the population size against the dimension."""
    functions = list(FUNCTIONS)
    blocks = [(2, 240), (2, 600), (4, 240), (4, 600), (8, 240), (8, 600), (8, 1200)]
    sizes = {
        dimension: [
            k * dimension for k in (1, 2, 4, 8, 16) if 4 <= k * dimension <= 32
        ]
        for dimension in (2, 4, 8)
    }

    for variant in VARIANTS:
        print(f"#### Members against parameters -- {variant}, median / mean\n")
        print("| D | N | N/D | shots | " + " | ".join(functions) + " |")
        print("|---|---|---|---|" + "---|" * len(functions))
        for dimension, budget in blocks:
            for population_size in sizes[dimension]:
                row = [
                    str(dimension),
                    str(population_size),
                    str(population_size // dimension),
                    str(budget),
                ]
                row += [
                    cell(cells, (variant, function, dimension, population_size, budget))
                    for function in functions
                ]
                print("| " + " | ".join(row) + " |")
        print()

    print("#### Which N takes each (parameters, budget) block\n")
    print("A block goes to the N holding the lowest median on the most functions.\n")
    print("| D | shots | " + " | ".join(VARIANTS) + " |")
    print("|---|---|---|---|")
    won = defaultdict(lambda: defaultdict(int))
    for dimension, budget in blocks:
        row = [str(dimension), str(budget)]
        for variant in VARIANTS:
            per_size = defaultdict(int)
            for function in functions:
                per_size[
                    min(
                        sizes[dimension],
                        key=lambda n: median(
                            cells, (variant, function, dimension, n, budget)
                        ),
                    )
                ] += 1
            most = max(per_size.values())
            winners = sorted(n for n in per_size if per_size[n] == most)
            for n in winners:
                won[variant][n] += 1
            counts = ", ".join(f"N={n}: {per_size[n]}" for n in sorted(per_size))
            row.append(
                " and ".join(f"N={n}" for n in winners) + f" ({counts})"
            )
        print("| " + " | ".join(row) + " |")
    for variant in VARIANTS:
        tally = ", ".join(f"N={n}: {won[variant][n]}" for n in sorted(won[variant]))
        print(f"\n  {variant}: blocks taken or shared -- {tally} (of {len(blocks)})")

    print("\n  and counted by single (dimension, budget, function) cells:")
    for variant in VARIANTS:
        per_size = defaultdict(int)
        for dimension, budget in blocks:
            for function in functions:
                per_size[
                    min(
                        sizes[dimension],
                        key=lambda n: median(
                            cells, (variant, function, dimension, n, budget)
                        ),
                    )
                ] += 1
        total = sum(per_size.values())
        tally = ", ".join(f"N={n}: {per_size[n]}" for n in sorted(per_size))
        print(f"    {variant}: {tally} (of {total})")

    print("\n#### The floor: what a population of four does with 2.5x the budget\n")
    print("| variant | D | function | 240 shots | 600 shots | change |")
    print("|---|---|---|---|---|---|")
    for population_size in (4, 8):
        for variant in VARIANTS:
            for dimension in (2, 4):
                for function in functions:
                    short = median(
                        cells, (variant, function, dimension, population_size, 240)
                    )
                    long = median(
                        cells, (variant, function, dimension, population_size, 600)
                    )
                    change = 100.0 if short == 0 else 100 * (short - long) / short
                    print(
                        f"| {variant}, N={population_size} | {dimension} | {function} "
                        f"| {short:.3f} | {long:.3f} | {change:.1f}% |"
                    )


def report_truncation(withdrawn, equal):
    """What the withdrawn stopping rule did, against what replaced it.

    ``withdrawn`` holds rows taken while the loop counted proposals submitted;
    ``equal`` holds the barrier sweep's own rows, where every arm runs to the
    same number of completed shots. Both are the barrier sweep's cells.
    """
    functions = list(FUNCTIONS)
    populations = (8, 16, 60)
    budgets = (120, 240, 600)

    print("#### The withdrawn stopping rule, applied to the shipped learner\n")
    print("| shots | N | shots the generational arm was scored on "
          "| ratio under that rule | ratio at equal completed shots |")
    print("|---|---|---|---|---|")
    for budget in budgets:
        for population_size in populations:
            scored = int(
                np.median(
                    [
                        int(row["completed"])
                        for function in functions
                        for row in withdrawn[
                            ("generational", function, 4, population_size, budget)
                        ]
                    ]
                )
            )
            ratios = [
                np.median(
                    [
                        median(cells, ("generational", f, 4, population_size, budget))
                        / median(cells, ("asynchronous", f, 4, population_size, budget))
                        for f in functions
                    ]
                )
                for cells in (withdrawn, equal)
            ]
            print(
                f"| {budget} | {population_size} | {scored} of {budget} "
                f"| {ratios[0]:.2f} | {ratios[1]:.2f} |"
            )

    print("\n| shots | median ratio over the 12 cells, withdrawn rule "
          "| at equal completed shots |")
    print("|---|---|---|")
    for budget in budgets:
        ratios = [
            np.median(
                [
                    median(cells, ("generational", f, 4, n, budget))
                    / median(cells, ("asynchronous", f, 4, n, budget))
                    for f in functions
                    for n in populations
                ]
            )
            for cells in (withdrawn, equal)
        ]
        print(f"| {budget} | {ratios[0]:.2f} | {ratios[1]:.2f} |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="run a sweep and write its raw rows")
    runner.add_argument("sweep", choices=("barrier", "dimension", "smoke"))
    runner.add_argument("--out", help="where to write the rows (default: stdout)")
    runner.add_argument("--seeds", type=int, help="override the sweep's seed count")

    reporter = sub.add_parser("report", help="aggregate raw rows into the tables")
    reporter.add_argument("sweep", choices=("barrier", "dimension", "truncation"))
    reporter.add_argument(
        "rows",
        nargs="+",
        help="results files to read; `truncation` takes the rows taken under "
             "the withdrawn rule first and the barrier sweep's rows second",
    )

    args = parser.parse_args(argv)

    if args.command == "report":
        if args.sweep == "barrier":
            report_barrier(load(args.rows))
        elif args.sweep == "dimension":
            report_dimension(load(args.rows))
        else:
            withdrawn, equal = args.rows
            report_truncation(load([withdrawn]), load([equal]))
        return

    work = list(jobs(args.sweep, args.seeds))
    out = open(args.out, "w", newline="") if args.out else sys.stdout
    print(f"{len(work)} runs", file=sys.stderr, flush=True)
    try:
        writer = csv.DictWriter(out, FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor() as pool:
            futures = [pool.submit(run, *job) for job in work]
            for n, future in enumerate(futures, 1):
                writer.writerow(future.result())
                if n % 200 == 0:
                    print(f"  {n}/{len(work)}", file=sys.stderr, flush=True)
    finally:
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
