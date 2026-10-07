Learners
========

A learner proposes the parameter values of the next shots from the history of the shots so far. ``[GENERAL] learner`` names the one a session runs, and ``[LEARNER.<name>]`` carries its knobs. :doc:`configuration` lists every knob with its type, range and default.

All four learners search the box set by the parameters' ``min`` and ``max``, and all of them minimize: ``maximize`` flips the sign of the cost before a learner sees it. A learner uses only usable costs, which are finite and not bad. A shot that is still running, that has a bad cost, or that was given up on contributes no cost.

.. list-table::
    :header-rows: 1
    :widths: 24 36 14 26

    * - Learner
      - Proposes
      - Fits a model
      - Shots kept queued
    * - ``random``
      - Uniform draws over the box.
      - no
      - ``num_buffered_runs``
    * - ``directed_random``
      - Draws near a point already seen.
      - no
      - ``num_buffered_runs``
    * - ``differential_evolution``
      - One whole generation at a time.
      - no
      - ``population_size``
    * - ``gaussian_process``
      - Batches from a fitted model, with an explorer filling the queue.
      - yes
      - ``num_buffered_runs``

When the parameters carry a ``start``, the session proposes it first and once, whichever learner runs. The learner meets it in the history as an ordinary observation.

``random``
----------

Draws every point uniformly over the box. It reads no history and fits nothing. Choose it as the reference the other learners have to beat, or to survey the space. It takes no knobs.

``directed_random``
-------------------

Draws each point near one already seen. A draw is centered on a usable observation whose cost lies in the band ``trust_range`` sets, measured from the worst cost seen (0) to the best (1), and lands within ``trust_region`` of that center, drawn uniformly, or with ``trust_gaussian`` from a Gaussian of that width, clipped to the box. A share ``explore_fraction`` of the draws ignores the band and the region and covers the whole box. ``trust_region`` is always a region, so to search the whole box set ``explore_fraction = 1``, or choose ``random``. With no usable observation yet, it draws uniformly. When no observation's cost lies in the band, it centers on the best point.

The default band lies toward the poor end of the costs, so the search spreads over mediocre points and explores. That is why ``gaussian_process`` uses it as its default explorer. A band near the best end, such as ``trust_range = [0.9, 1.0]``, makes it refine around the best points instead. Choose it for a search that needs no fit and stays close to what has been measured.

Its knobs are in :ref:`[LEARNER.directed_random] <configuration-learner-directed-random>`.

``differential_evolution``
--------------------------

Evolves a population, one whole generation at a time. The first generation is ``population_size`` founders drawn uniformly over the box, and the configured start, when there is one, takes the first slot. Each later generation proposes one trial per slot. A trial is a mutant built from other members, crossed over with the member of its slot, with a coordinate that falls outside the bounds redrawn rather than clipped. The strategy sets the mutant: ``best`` strategies start from the best member and ``rand`` strategies from a random one, and the digit is the number of difference pairs added, scaled by a weight drawn from ``mutation_scale`` once per generation. A slot keeps the lowest usable cost its proposals have produced, and a cost that arrives after its generation has moved on still competes for its own slot.

A whole generation goes out together, and nothing is proposed until every shot of the last has completed or been given up on. The queue therefore empties once per generation. ``population_size`` is both the population and the queue depth, so ``num_buffered_runs`` is refused, and ``starved`` is not counted. ``max_num_runs`` must allow at least two generations; its last generation is cut short where the budget runs out.

It builds no model of the cost, and it selects on a single measurement per slot, so a lucky measurement stays in its slot. Choose it for a cost that a Gaussian process fits poorly and whose shot-to-shot noise is small. `benchmarks/README.md <https://github.com/spielman-group/labscript-optimization/blob/Development/benchmarks/README.md>`_ in the repository holds the sweep behind the default of 8 members, run over four analytic test functions at two to eight parameters: 8 suits the budgets a lab usually runs, 16 suits budgets past a thousand shots, and 4 mostly stalls.

Its knobs are in :ref:`[LEARNER.differential_evolution] <configuration-learner-differential-evolution>`.

``gaussian_process``
--------------------

The default learner. It fits a Gaussian process to the history and searches its posterior for the next points. It computes each batch of points on a thread of its own, so the apparatus never waits on a fit, and an explorer keeps the queue filled in the meantime. Choose it when a shot is expensive and the cost varies smoothly enough with the parameters to model: it spends computation to save shots. It needs scikit-learn and scipy.

Its knobs are in :ref:`[LEARNER.gaussian_process] <configuration-learner-gaussian-process>`.

Model and acquisition
~~~~~~~~~~~~~~~~~~~~~

The model is a scikit-learn ``GaussianProcessRegressor`` with an RBF kernel that has one length scale per parameter. It is fit to the parameters scaled onto the unit cube and the cost standardized. With ``cost_has_noise``, which is on by default, a white-noise term lets the fit attribute scatter to the measurement rather than to structure. When the cost has an uncertainty column, each observation's own variance enters the fit too.

The next point minimizes ``cost_bias * predicted_cost - uncer_bias * predicted_standard_deviation`` over the box, or over ``trust_region`` around the best point seen, by L-BFGS-B from several starting points. ``uncer_bias`` is a list of weights walked by position within a batch, starting again at its first weight in every batch. A weight of 0 gives a purely greedy point and a larger weight looks further from measured points, so the default schedule opens greedily and widens across a batch.

A batch has ``batch_size`` points, each conditioned on the ones before it: a point is folded into the fit at its predicted cost, so the next one does not chase the same uncertain corner. The kernel's hyperparameters are refit once per batch, starting from the last refit's values.

The cycle
~~~~~~~~~

**Warmup.** Until the history holds ``warmup_observations`` usable observations, the explorer alone proposes, keeping ``num_buffered_runs`` shots queued. The count is of usable observations, so a shot with a bad cost, or one given up on, does not advance it, and the configured start counts like any other shot. Warmup ends at the count. Explorer shots already queued at that moment still run, and their costs join the fit as they land.

**Batches.** After warmup the model computes a batch in the background, from the history as it stands when the computation begins. While it computes, each refill keeps ``num_buffered_runs`` shots queued with explorer shots. When the batch is ready, its points take the queue's free places ahead of any explorer shot, and the explorer fills the queue again once they have gone out. The next batch is computed once every point of the last has come back or been given up on, so it has every answer it asked for. Explorer shots never hold it up, and the batch is not conditioned on those still in flight. When ``max_num_runs`` leaves room for less than a refill, the explorer's shots are cut first.

A shot is given up on as soon as runmanager says that no cost can come for it, which the session asks each time the routine runs. If every point of a batch is deleted from the queue and nothing else reaches lyse, the next batch waits until the routine has run once more.

**The explorer.** ``explorer`` names the learner whose shots run the warmup and fill the queue after it: ``random``, ``directed_random`` or ``differential_evolution``, each built from its own ``[LEARNER.<name>]`` table. The Gaussian process cannot be its own explorer, because it proposes nothing until it is warmed up. The explorer is handed the part of the history it reads: ``directed_random`` reads all of it, ``differential_evolution`` only the explorer shots, and ``random`` none. The Gaussian process's model reads all of it.

As the explorer, ``differential_evolution`` is asked for a point or two at a time over its own shots with no generation barrier, so it runs asynchronously, with the differential weight drawn afresh for each request, and its generation and budget rules do not apply. Until enough of its slots hold usable costs to breed from, which is three for the default ``best1``, its shots are founder draws over the whole box, or within ``trust_region`` of the best member when that is set and some member has a cost. That makes it a wider explorer than ``directed_random``.

**explore_runs.** A ready batch waits until at least ``explore_runs`` explorer shots have gone out since the previous batch's last point. For the first batch the count runs from the start of the run, so the warmup shots count. The explorer fills the queue as a batch's points come back, so a cycle holds about ``num_buffered_runs`` explorer shots when a fit is instant and more while it is slow. ``explore_runs`` adds only what that leaves short, so at its default it normally adds nothing. At 0 the explorer only fills the queue.

Which slots a batch's points take depends on how long each fit takes, so a run under ``gaussian_process`` is not reproducible from its ``seed``.

A noisy cost
~~~~~~~~~~~~

``best_cost`` is the lowest cost any shot measured. It is one measurement, not an estimate of the cost at ``best_params``, and on a noisy cost it is optimistic, so re-measure at ``best_params`` to estimate that cost. ``differential_evolution`` keeps the lowest measurement of each slot and so keeps a lucky one. ``gaussian_process`` with ``cost_has_noise`` on models the scatter, and proposes against the model's predicted cost, an average over the shots near a point. For a noisy cost, prefer ``gaussian_process``.

The phase column
----------------

Each shot's ``phase`` result says what proposed it; :doc:`results` describes the columns. ``start`` is the configured start. Under ``gaussian_process``, ``warmup`` is an explorer shot during warmup, ``main`` is a point of the Gaussian process's own batch and ``explore`` is an explorer shot after warmup. Under the other learners every shot except the start reads ``main``.
