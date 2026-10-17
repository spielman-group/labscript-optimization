# labscript-optimization deep pass

The planning document for this repo's code-health pass
(`issues/code_health_campaign.md` in the meta repo): the contract, then one
section per step of the deep pass. Nothing is removed until Ian has decided
each entry. It is deleted when the pass is done.

## The contract

The behaviours the tests protect, one line per test, read from the test names
(`pytest --collect-only`, `CodeHealth` at 3d633bc: 100 tests, 216 cases with
parametrization). An italic note marks a test that a step-1 candidate would
remove with its behaviour, or one the contract itself questions.

**tests/test_config.py**

- A budget below two whole generations is refused.
- A file that cannot run as written is refused naming the setting.
- A generational learner will not take a queue depth as well.
- A key nothing reads is refused naming it and its table.
- A learner knob in the general table is refused naming where it goes. *(step 1, #2)*
- A name repeated in a group nobody switched on still loads.
- A replaced setting is refused naming what replaces it. *(step 1, #1)*
- A setting below its floor is refused and the floor itself loads.
- A setting left out takes the documented default.
- An explorer and its gaussian process each take their own value of a knob.
- An expression sees the names runmanager globals do.
- Listed groups decide what is searched and what is set.
- The explorer is named in the gaussian process table.
- The mloop tables are refused naming their replacements. *(step 1, #1)*

**tests/test_examples.py**

- The example configuration loads.

**tests/test_learners.py**

- A batch is computed from the history as its computation began.
- A gaussian process batch does not repeat itself.
- A history with only some uncertainties is refused without noise.
- A knob that cannot mean what it says is refused naming it.
- A length scale at a bound is reported when the set of them changes. *(step 1, #9)*
- A model that raises stops the session at the next refill.
- A proposal is bred from the member of the slot its position names.
- Differential evolution explores over its own shots alone.
- Differential evolution finds the minimum.
- Directed random explores the whole space when told to. *(step 1, #6)*
- Directed random falls back to the best point when the band is empty.
- Directed random prefers mediocre points over the best one.
- Directed random searches near the usable points it has seen.
- Each batch cycle holds at least explore runs explorer shots.
- Each batch walks the exploration schedule from its first weight.
- Every strategy runs at its smallest population and refuses less.
- Gaussian process finds the minimum.
- Gaussian process uses per point uncertainties.
- Learners filling to the hint keep exactly that many in flight.
- Mutation scale sets the differential weight.
- Proposals have the requested shape and stay in bounds.
- The explorer proposes the warmup and the shots that fill the queue.
- The next batch is computed once none of the last is pending.
- The warmup scales with the search above a floor of five.
- Warmup ends at a count of usable observations and not of shots.
- While a batch computes the queue is topped up with explorer shots.
- With explore runs of zero the explorer only fills.

**tests/test_lyse_routine.py**

- The routine runs multishot passes and refuses singleshot ones.

**tests/test_observations.py**

- Best is the lowest usable cost and none without one.
- Only a finite cost from a completed unflagged shot can inform a fit.

**tests/test_routine.py**

- A pass reads its own files whatever order lyse keeps its columns in.
- A shot carrying no identifier is not the sessions to read.
- A shot is read as the proposal it answers and its cost.
- A shot with no usable cost is still read and marked bad.
- The shot that first has a value can be written into its column.
- The status is written onto the shot as lyse results and not into its file.

**tests/test_runmanager_interface.py**

- A labscript file changed mid session is refused.
- A start is refused while runmanager cannot sustain a session.
- Every global is submitted as a python value.
- Every submission after the first joins its sequence.
- Runmanagers values are read back as the start.
- The original values are read until a start goes.
- Values are set from a point or as originally written.

**tests/test_session.py**

- A changed labscript file is refused before anything is submitted.
- A cost arriving after a shot was given up on is still taken.
- A gaussian process session runs end to end with each explorer.
- A generation goes out whole and waits to be answered for.
- A generation opening on the configured start is still whole.
- A lost shot is not charged against the run budget.
- A new session waits for start.
- A refused submission is raised and leaves the session untouched.
- A sequence runmanager cannot add to ends the session.
- A session counts the refills that found its queue empty. *(step 1, #7)*
- A session gives up after runs that bring nothing better.
- A session out of patience keeps that reason as its last shots land.
- A session stops at the run budget and stays stopped.
- A shot that has only just run is not treated as lost.
- A shot that will never arrive stops being waited on.
- A shot whose cost cannot come is dropped at the first reconcile.
- Costs arriving out of order land in proposal order.
- Only the first cost of a shot this session submitted is taken.
- Pause keeps taking costs but submits nothing.
- Status counts every shot by what became of it.
- The best cost is reported in the labs own sign.
- The budget may cut the last generation short.
- The configured start is the first proposal and the only one.
- The history tells a shot still coming from one that never will.
- The queue is kept filled to the buffer depth.

**tests/test_space.py**

- A fractional trust region scales with each parameter.
- A space with nothing enabled is rejected.
- A start is one point over every searched parameter or none.
- A trust region near an edge does not reach past it.
- Containment is answered once per row. *(contract: a ParameterSpace helper, not a user behaviour; candidate to go)*
- Impossible parameters are rejected.
- Impossible trust regions are rejected.
- Scaling maps the bounds onto the unit cube. *(contract: a ParameterSpace helper, not a user behaviour; candidate to go)*

**tests/test_window.py**

- Window shows progress and controls. *(contract: one test for eight behaviours, since only one WindowController may be built in a run; candidate to rename to what it checks: the phase line, the counts, Start / Set best / Restore enabling against the light, the Start payload and column, the plot, the edit button, the light's message)*

**tests/test_worker.py**

- A failure after a reply stops the session and is raised next.
- A reply later than the deadline is saved by the next hand over.
- A session thread that has died is reported. *(step 1, #12)*
- A start runmanager refuses leaves the session paused and a later one goes.
- Only the shots the session proposed get their status.
- Reset reads the file again and a file that does not load ends the session.
- Runmanager not answering mid run pauses the session and start resumes it.
- Runmanagers values are set from the window and when the session ends.

## Step 1: behaviour no one asked for

The whole package is new (no `master`), so "not upstream" holds for every
line; M-LOOP and analysislib-mloop (read-only, `~/Code/Python/M-LOOP`,
`~/Code/Python/analysislib-mloop`) stand in for upstream, since the package
replaces them. Evidence for each entry:

- **Added:** the commit `git log -S` names.
- **Request:** a grep of the meta repo's history (`git log --all -S`), its
  `issues/*.md` and `CODEX.md`, and of this repo's `issues/`; "none found"
  means only that these show no ask. Ian's requests in this package's
  sessions (the window, the light, Set best / Restore, Start from runmanager
  values, Reset reloading the file, the edit button) are not candidates.
- **Users considered:** a lab's configuration file and lyse routine folder,
  its lyse analysis reading the result columns, and code building a learner
  directly (the package docstring shows it). No other repo imports
  labscript-optimization: `git -C <repo> grep labscript_optimization` over
  every repo at `Development` (hyde and desktop-app at `HEAD`) finds only
  labscript-agents' copied docstring (`routine.py:144`) and copied file
  banner (`window.py:3`), which are step 2's; the control, this repo, finds 37
  files.

Lines are non-blank lines, comments and docstrings included, at 3d633bc.

| # | Behaviour | Where | Lines | Added | Request | Who sees it | Proposal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | M-LOOP's old table names (`[MLOOP]`, `[MLOOP_PARAMS]`) and six renamed GP keys are refused by name, naming the replacement | `config.py` `RENAMED_TABLES`, `RENAMED_KEYS`, their blocks in `check_keys` | ~55 | 0ed2a44, e8af71e | none found; Ian kept UPGRADING's renamed-key text (docs findings) | a lab converting an M-LOOP file | Drop the special refusals: the generic unknown-key refusal still stops the file, and UPGRADING.md has the mapping |
| 2 | A learner knob written in `[GENERAL]` gets its own message naming the `[LEARNER.<name>]` tables it belongs in | `config.py` `reject_misplaced_knobs`; `learners/__init__.py` `knobs_by_learner` | ~45 | b093535 | none found | a lab with a misplaced knob | Drop it: the generic refusal ("[GENERAL] does not accept 'x'. It accepts: ...") still stops the file |
| 3 | A parameter or global table written one level short gets its own message | `config.py` `require_tables` and its four calls | ~35 | 8d6c3c0 | none found | a lab with a mis-nested table | Keep or drop: without it the file still fails, but on a type error that names neither the table nor the setting |
| 4 | Guards for someone writing a new learner: knobs collected in `**kwargs`, `space`/`rng` in the wrong order or type, a learner with no `ask`; and `InsufficientData`, which no session can reach | `learners/__init__.py` `_constructor_parameters`; `learners/base.py` `ParameterSpaceLearner.__init__`, `InsufficientData` | ~37 | 72558db, a21d6a1, 472ffd7 | none found | no user path: `LEARNERS` is a fixed table, and the GP never asks its model short of warmup | Drop |
| 5 | An `Observation` that is complete with no cost is refused | `observations.py` `Observation.__post_init__` | ~20 | 472ffd7 | none found | no user path: `Session.record` always passes a float | Drop |
| 6 | Two directed-random knobs M-LOOP does not have: `trust_gaussian` (Gaussian rather than uniform draws in the trust region) and `explore_fraction` (a share of whole-space draws) | `learners/random.py` | ~14, plus docs and example rows | 472ffd7 | none found; not in M-LOOP | a lab's `[LEARNER.directed_random]` table | Ian decides: keep either, both or neither |
| 7 | The session counts "starved" refills (queue found empty), shown in the window | `session.py` `starved`; `window.ui`/`window.py` row; docs | ~15 | 071397b | none found | the window's Status tab | Ian decides |
| 8 | Two lyse result columns beyond the best cost and parameters: `best_shot_id` and `stopped` (the stop reason on every shot of the last pass) | `routine.py` `SHOT_RESULTS`, `NO_VALUE_YET`; `session.py` status; docs/results.rst | ~10 code, plus docs | 9783948 | none found | a lab's analysis reading the columns | Ian decides |
| 9 | scikit-learn's "length scale at its bound" warning is filtered and reported once, when the set of parameters at a bound changes | `learners/gaussian_process.py` `LENGTH_SCALE_AT_BOUND`, the filter, `report_length_scale_bounds` | ~55 | aaba24a | none found | the routine's Output dock | Simplify: let scikit-learn's own warning through (repeated each refit), or filter it out entirely; Ian decides |
| 10 | A single number written for `mutation_scale` gets its own message suggesting `[x, x]` | `learners/differential_evolution.py` | ~9 | 472ffd7..a21d6a1 | none found | a lab's DE table | Drop: the generic pair refusal still stops it |
| 11 | A GP expr whose lambda cannot take its `args` is refused at load (by its signature) | `config.py` `GlobalMapping.__post_init__` | ~16 | 472ffd7 | none found | a lab with a wrong-arity expr | Keep or drop: without it the first submission fails with a TypeError and the run ends |
| 12 | A hand-over finding the session thread dead says so and drops what it waits for | `worker.py` `hand_over` | ~8 | 260d472 | a code-quality review's finding, whose fixes Ian approved | only after a bug kills the thread | Keep (cheap) or drop |
| 13 | `directed_random` with `trust_region` unset searches the whole space, as a random learner, though the docs say a trust region is always a region | `learners/random.py` `ask`; `space.py` `absolute_trust_region` returns None | ~3 | 472ffd7 | none found | a lab's table | Make the docs and code agree: refuse `None` here, or document it |

## Step 2: duplication

### 2.2 Shared with labscript-agents

Both packages are new-style lyse routines (`OptimizationRoutine`,
`AgentRoutine`, on `lyse.Routine` in `lyse/routine.py`). labscript-agents'
`routine.py`, `worker.py` and `window.py` began as copies of this package's;
an AST comparison of the two `CodeHealth` branches finds these names identical
in both: `SHOT_RESULTS`, `NO_VALUE_YET`, `analyzed`, `extract`, `save_status`,
`REPLY_TIMEOUT`, `Worker.hand_over`, `Worker.quit`. Agreed with
labscript-agents' session, which answered for its side: labscript-agents writes
nothing to lyse, uses no costs, and does no slow work in `run()`.

| # | Item | Copies | Should it exist? | One copy, where |
| --- | --- | --- | --- | --- |
| A | `analyzed(paths)`: the rows of a pass's files | both, identical | Yes, but it is `lyse.Routine.data(where={'filepath': paths})` plus a column sort | Both call `self.data(...)`; the sort, if still needed, moves into `lyse.Routine.data` (lyse's owner) |
| B | `save_status`, `SHOT_RESULTS`, `NO_VALUE_YET`, `RESULTS_GROUP`: the optimizer's status columns | both, identical | In labscript-optimization only; agents' are unused copies | labscript-optimization, writing through `lyse.Routine.get_run` and `group` rather than `lyse.Run` and `set_group`; labscript-agents deletes its copies |
| C | `extract(shots, config)`: shot ids and costs, `maximize`'s sign, bad costs | both, identical | In labscript-optimization only | labscript-optimization; labscript-agents deletes its copy |
| D | `Worker` with `hand_over`, `REPLY_TIMEOUT`, `quit`: a thread whose late replies are saved on a later pass | both, identical; agents' `worker.py` is unused | Yes, for labscript-optimization, whose `run()` hands slow work to a thread | labscript-optimization's `worker.py`; labscript-agents deletes its copy. A `lyse.Routine` facility only once a second routine needs it |
| E | The routine folder: `Path(inspect.getfile(type(self))).parent` | both `routine.py` | Yes | `lyse.Routine` holds it as the private `_folder`; make it public there (lyse's owner) and both use it |
| F | The runmanager light: `RunmanagerClient()` for its address, `LinkIndicator('runmanager', host, port)` | both | Yes | labscript-utils' `LinkIndicator` is already the one copy; the few lines of wiring stay in each |
| G | Reading the routine's TOML: `tomllib` here, `labscript_profile.toml_config.load_toml_file` in agents | one each | Yes | `load_toml_file` in both (labscript-utils' step-2 entry 11 moved agents to it) |
| H | `RunmanagerInterface.shot_status` | both | Here a bare pass-through of the client's; agents' skips the request for no ids and backs a tool | Here: dropped, the session calling the client through the interface's `client`; agents keeps its own |
| I | "This shot is ours; take it once": the session's `proposals`/`results` here, agents' `_engaged`/`_reported`/`take_unreported` | one each | Yes | Each keeps its own: a few lines, and each package needs its own records of the shots anyway (Ian decides) |

### 2.1 Within this package, and against the rest of the suite

| # | Item | Copies | Proposal |
| --- | --- | --- | --- |
| J | Two sets of value checks: `knobs.py` (`boolean`, `integer`, `number`, `choice`, `pair`) for learner knobs, and `config.py`'s `require_type` plus `Config.__post_init__`'s own bool-is-not-an-int and integer checks for the session's settings | `knobs.py`, `config.py` | One set, `knobs.py`, used by `config.py` too; the librarian is asked whether the suite already has one |
| K | Two unknown-key refusals: `config.reject_unknown` and `learners.validate_options`' own message, with its "takes no knobs at all" variant | `config.py`, `learners/__init__.py` | `validate_options` calls `reject_unknown` |
| L | Two repeated-name checks written the same way: repeated globals (`Config.__post_init__`) and repeated parameters (`ParameterSpace.__init__`) | `config.py`, `space.py` | Keep: two lines each, on different objects |
| M | `in_background`: a daemon thread returning a `Future`, because `ThreadPoolExecutor` joins its threads at exit | `learners/gaussian_process.py` | Keep unless the librarian finds the suite already has one |

## Step 3: root causes

Every line of this package postdates the fork, so the guards here are those
that work around another module's behaviour, traced to it.

| # | Guard | Cause | Proposal |
| --- | --- | --- | --- |
| 3.1 | `analyzed` sorts the frame's columns, because pandas warns of lexsort depth when an unsorted MultiIndex is read by a shallower key, as `extract` does | `lyse.data` returns columns in the order they were added (`lyse/dataframe_utilities.py` sorts only the rows) | Keep the one-line sort here (it is 2.2 row A): sorting in `lyse.data` would reorder every lab's columns |
| 3.2 | `NO_VALUE_YET`: typed stand-ins (`""`, NaN, `[]`) saved until a status value exists | lyse fixes a column's dtype from the first value saved, and a later value of another type cannot be written into it | Keep: a lyse change to that would touch every lab's dataframe. Ian decides |
| 3.3 | `RunmanagerInterface.check_unchanged` and `pin_labscript_file`: a round trip before every submission, refusing a labscript file changed mid-run | runmanager itself refuses to add shots to a sequence of another labscript file (`compile_and_queue_shots`, `SequenceRefused`), which ends the session with runmanager's reason. It compares the script's base name only, so a file of the same name in another folder passes there and not here | Drop both and the per-submission round trip, and rely on runmanager's refusal; or keep this check for the same-name case. Ian decides. The no-labscript-file refusal in `check_ready` stays either way |
| 3.4 | `save_status` catches any exception and prints it | a save into a row lyse no longer has; lyse already warns of that at the reply (`lyse/filebox.py:435`) | Trace what else can raise; if nothing, drop the catch |
| 3.5 | `Worker.hand_over`'s bounded wait, and late replies saved on a later pass | lyse saves a routine's results only inside `run()` | Keep: it is lyse's design |
| 3.6 | `GlobalMapping.evaluate` converts a numpy scalar with `.item()` | runmanager writes a submitted value as its `repr`, which names numpy | Keep: Ian settled it ("You convert, not runmanager") |

## Step 4: dead code, compatibility leftovers and branches

| # | Item | Evidence | Proposal |
| --- | --- | --- | --- |
| 4.1 | `ParameterSpace.contains` | used only by tests (`git grep '\bcontains\b'` over this repo; no other repo imports the package) | Drop, with `test_containment_is_answered_once_per_row`; the learner tests check bounds with numpy |
| 4.2 | `InsufficientData` | raised by the GP's model short of warmup, which no session reaches (step 1 #4) | Drop with step 1 #4 |
| 4.3 | `benchmarks/` (`de_pipeline.py` 692 lines, README, four result CSVs) | a developer sweep that informed the DE decisions; nothing runs it since its smoke test went | Keep as a tool, or remove it: Ian decides |
| 4.4 | Branch `LinkCheck` (`3d95d7d`, on origin) | every commit in `Development`; Ian said "keep LinkCheck for now"; CODEX.md keeps it with the suite-wide LinkCheck effort | Delete under cb8ef42 once Ian releases it, or rename `spielman_retired/LinkCheck` |
| 4.5 | Compatibility leftovers | none: no upstream, and no old interface kept for a caller (M-LOOP's old names are step 1 #1) | Nothing |
