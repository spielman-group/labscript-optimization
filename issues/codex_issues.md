# labscript-optimization: known-issue resolution

Work items decomposed from [`codex_issues_proposal.md`](../codex_issues_proposal.md),
which is the PRD for this batch and carries the measurements, the rejected
alternatives and the reasoning behind every decision below. Read the relevant
PRD section before starting a slice; these items state *what*, the PRD states
*why*.

House rules that apply to every slice:

- Every test must be **proven to fail** by mutating the code it guards. Run the
  mutation, capture the failure, revert it, report both. A test that cannot be
  made to fail is a finding, not a deliverable.
- A comment or docstring states what the code does now. No "previously", no
  "used to". Prose that describes behaviour is a claim under test: when
  behaviour changes, grep for every sentence that described the old behaviour.
- No settings accepted for compatibility. An unknown or unusable key is refused
  at load, naming the key and what is accepted instead.
- Branch is `Development`. Baseline is 224 tests; recount before relying on it.

**Revised after a fourth review round.** Slices 5, 7, 8 and 9 below replace what
was filed earlier. The cause was a defect found while checking Ian's assertion
that nothing in this package times out on the experiment cycling — true, and
enumerating the five timeouts to confirm it showed that the one remaining bound
misattributes results under ordinary load. That finding reorders the rest: the
runmanager `Client.with_timeout` change is **not** being pursued as a statement
of work, because no defect depends on it; it is a tidy-up for runmanager's own
backlog, which would also let BLACS stop mutating a client's deadline per call.

## Checklist

- [x] Slice 1: Routine hands over every shot lyse analysed
- [x] Slice 2: Configuration is validated at load
- [x] Slice 3: What the lab reads — `best_cost` units and cost ordering
- [x] Slice 4: Textbook generational differential evolution
- [x] Slice 5: The greeting's deadline is a constant again
- [x] Slice 6: A wrapper cannot silently drop a generation barrier
- [x] Slice 7: A reply says which request it answers
- [x] Slice 8: The configure deadline covers the waits inside it
- [ ] Slice 9: A learner's declarations are facts about an instance
- [ ] Slice 10: Benchmark the shipped DE
- [ ] Slice 11: Test cleanup

---

## Slice 1: Routine hands over every shot lyse analysed

### Type

`AFK`

### What to build

**Prerequisite, no review needed:** install the package editable into the
labscript conda environment, and document that step wherever a lab is told how
to set the package up. Every other suite repo is already installed this way;
this one is not, which means nothing in it has ever been exercised through
lyse. Do **not** add a unit test asserting the installation — that tests the
workstation, not the repository.

lyse runs multishot routines once per *drained* analysis batch, not once per
shot. In steady state, where singleshot analysis keeps up, that is one shot per
invocation and the current code is correct. When a shot becomes incomplete
while the previous one is still being analysed — or analysis is paused and
resumed, or lyse starts with shots already in the box — several shots are
analysed and the multishot routine runs once. The routine asks for the last row
only, so the rest are never handed to the worker: their costs never reach the
learner, and `dropped` climbs for shots that ran and were analysed perfectly
well.

The routine must hand over **every shot in the current sequence that it has not
already reported, and that carries a shot id** — not every row in the
dataframe. It remembers the last row it handled and asks for progressively more
rows until that row is in the frame or the frame is shorter than the request.
On the very first invocation there is no remembered row: remember the last one
and hand over nothing, rather than reaching back over the sequence so far.

Two constraints on the worker protocol, both load-bearing:

- **One message per invocation, one reply.** That lockstep is what closed the
  reply-offset defect; handing over *m* observations must not become *m*
  messages. Send one message carrying the observations, and answer with one
  verdict per observation — a list or tuple — because `save_status` writes into
  each shot's own file and needs to know which shots the session actually took.
- **An invocation that finds no new rows still sends a message.** `reconcile`
  and `refill` run only in the trailing work after a reply, so a routine that
  returns early stops reconciling entirely, and a generation that was blocked
  and then cleared by an operator would never be revived.

Also in this slice, because it is the same startup path: a failed configure
must name its cause. `CONFIGURE_TIMEOUT` is 30 s while the runmanager client
takes labconfig's `communication_timeout`, which falls back to 60 s — so with
runmanager down the worker is killed mid-wait and the failure reads "did not
configure within 30 seconds", never saying that runmanager did not answer. Give
the interface's client a short timeout, or greet runmanager before configuring.

### Acceptance criteria

- [ ] Two shots analysed in one pass are both handed over — mutation: ask for
      one row; the second is never sent
- [ ] A row already handed over is not sent again — mutation: drop the
      remembered row, asserting on the messages sent, because the session's
      duplicate-rejection would otherwise mask it
- [ ] A pile-up larger than the first request is fetched whole — mutation:
      remove the escalation; the oldest row is missed
- [ ] An invocation that finds no new rows still sends one message — mutation:
      return early when nothing is new
- [ ] The first invocation of a session hands over nothing and does not fetch
      the whole sequence
- [ ] A shot the session never proposed is handed over harmlessly and is not
      written to
- [ ] One message per invocation regardless of how many observations it carries
- [ ] A configure failure caused by runmanager not answering says so
- [ ] `test_only_the_shot_it_was_called_on_is_asked_of_lyse` is deleted; it pins
      the wrong contract
- [ ] The package imports from outside the checkout in the labscript env

### Blocked by

None - can start immediately.

### User stories covered

- PRD "Issue 0 — the routine discards shots lyse analysed"
- PRD "Issue 1 — worker reply offset", the residue subsection
- PRD "Issue 7 (a) — Not installed"

---

## Slice 2: Configuration is validated at load

### Type

`AFK`

### What to build

Every remaining way a configuration can be accepted and then misbehave, closed
in one pass. `Config.__post_init__` becomes the single authority for the
dataclass's own fields, so a `Config` built directly in a script is checked the
same as one parsed from TOML; `from_dict` stops coercing in its `present` calls
and passes values through as written. Coercion inside `Parameter` stays — the
example file legitimately writes integer bounds.

Refuse, each naming the offending key and what is accepted:

- Values written as the wrong type. Integers must be exact and must exclude
  `bool` (`isinstance(True, int)` is `True` in Python, so every integer check
  needs that guard); `maximize` must be an exact boolean; `learner` and
  `session` must be strings, since string coercion has the same flaw integer
  coercion does.
- Out-of-range values that produce a silently dead session:
  `num_buffered_runs` below 1, `num_training_runs` below 0, `seed` below 0,
  `max_num_runs` and `max_num_runs_without_better_params` at 0 or below when
  set. A session configured this way today submits nothing, counts no
  starvation and never explains itself.
- A learner name that is not a known learner. The per-table option check never
  looks at the *selected* learner, so a misspelling loads and fails later at
  worker configure.
- Collisions in the parameter-to-global mapping: two parameters sharing a name
  (in the parameter space, whose invariant it is — a learner handed such a
  space is malformed with no configuration in sight), two mappings sharing a
  global name, and a mapping with no expression that does not take exactly one
  argument. Today two active groups each defining `x` feed one parameter's
  value to both globals, driving one of them outside its own declared bounds.
- An expression whose callable cannot accept the number of arguments
  configured. Today this loads and raises inside the session, mid-run, which is
  exactly what compiling the expression at construction exists to prevent. Let
  a callable whose signature cannot be read pass rather than refusing it.

**`population_size` becomes the population size.** It keeps its name and means
the number of members, default **8**, and no multiplier survives in any form.
This matches the DE literature, where NP is given directly and the
dimension-scaled figure is a rule of thumb for choosing it. Nothing is
preserved for the sake of existing files: a file carrying M-LOOP's
`population_size = 15` now means 15 members rather than 15 x the parameter
count, which moves towards the measured optimum. State the new meaning in
`UPGRADING.md`; refuse nothing and convert nothing.

**The Gaussian process knob `generation_size` becomes `batch_size`.** The
differential evolution work adds a `generation` attribute to the learner base,
named for what the DE literature calls it, and the Gaussian process already
used the same word for an unrelated thing — the period of its exploration
schedule and its kernel refit interval, which batch Bayesian optimisation calls
a batch. Renaming the Gaussian process knob leaves each learner using its own
field's standard word, rather than leaving two senses of "generation" side by
side in one document. It is a public key, so refuse it under the old name and
list the rename in `UPGRADING.md`. Nineteen occurrences across the learner, the
shared-key list, the example configuration and the tests; the learner's module
prose uses the word in sentences too, not only as an identifier.

With it: the hard floor stays the mutation strategy's own requirement, since
that is a correctness constraint; the roughly eight-member search floor is
prose, because searching badly is the user's business. Refuse
`max_num_runs < 2 x population_size` — below two generations a population
cannot evolve, so the configuration cannot do what it claims — but the message
must not say nothing evolves below that line, because a truncated second
generation does evolve some slots.

### Acceptance criteria

- [ ] Each refusal above has a test proven by deleting its check
- [ ] A float written where an integer belongs is refused, and the message says
      integer — mutation: restore coercion; it loads truncated
- [ ] `true` written where an integer belongs is refused — mutation: use a bare
      integer check; it loads as 1
- [ ] A dead-session setting is refused, and a follow-on assertion shows what it
      would otherwise do: three refills returning nothing, no starvation
      counted, no stop reason
- [ ] A misspelled learner is refused at load, not at worker configure
- [ ] A parameter named twice across two active groups is refused; the same name
      in a group that is switched off still loads
- [ ] `population_size = 15` builds fifteen members — mutation: multiply by the
      parameter count; forty-five
- [ ] `UPGRADING.md` states `population_size`'s new meaning
- [ ] `generation_size` is refused at load and `batch_size` does what it did —
      mutation: leave the old key accepted; a file setting it loads and the
      setting is silently ignored
- [ ] No occurrence of `generation_size` survives in the learner, the shared-key
      list, the example configuration, the tests or the learner's prose
- [ ] Prose claiming every key is "acted on" is corrected: a key must be one the
      package knows, and whether it is acted on depends on the learner selected

### Blocked by

None - can start immediately.

### User stories covered

- PRD "Issue 3 — parameter-to-global collisions"
- PRD "Issue 4 — coercive scalar settings"
- PRD "Issue 2", the population decision

---

## Slice 3: What the lab reads — `best_cost` units and cost ordering

### Type

`AFK`

### What to build

A lab maximising a figure of merit of 7 currently reads `best_cost = -7`: the
routine flips the sign once so everything downstream minimises, and the session
reports that internal value unchanged. Report it in the units and sign of the
lab's own cost column. The column keeps its name; `best_loss` is not adopted,
because the minimisation convention is the package's internal business and
should not be exported into the dataframe a physicist reads. Correct the
`UPGRADING.md` bullet that says `maximize` "still flips the sign", which
describes the internal treatment and misleads about the column.

Separately, document a requirement nothing states: the column named by the cost
key must exist by the time this routine runs. lyse runs multishot routines in
list order, so the cost must come from a singleshot routine or from a multishot
routine above this one. A cost column that is not there yet is not waited for —
the shot is recorded as a bad observation, and so is every shot of the run. This
belongs where a reader is standing when they add the routine to lyse, not in the
configuration file, whose cost-key comment already covers what the key means.

### Acceptance criteria

- [ ] With maximisation on and two observations, the reported best cost is the
      larger measurement in the lab's own sign, and the reported best shot is
      the one that measured it — mutation: remove the sign restoration;
      second mutation: select the maximum internally, and the wrong shot is named
- [ ] The mirror case with maximisation off, so the first cannot be satisfied by
      negating unconditionally
- [ ] `UPGRADING.md` says what `maximize` means for the reported column
- [ ] The cost-ordering requirement is documented once, where a reader adds the
      routine to lyse

### Blocked by

None - can start immediately.

### User stories covered

- PRD "Issue 5 — `best_cost` reports the internal sign"
- PRD "Issue 6 — the cost must already exist when the routine runs"

---

## Slice 4: Textbook generational differential evolution

### Type

`AFK`

### What to build

The largest slice, and the one most likely to come back from review. It may be
split at the seam between the walk and the submission boundary if that proves
better; both halves are independently verifiable.

Today the session gives learners only completed observations, in proposal
order, and the differential evolution learner reconstructs its population and
target slot by *counting* what it is given. At buffer depth d every trial
therefore competes against the slot d-1 further on: at the shipped default,
essentially every trial has always competed against the wrong incumbent. The
decision is textbook generational DE — scipy's deferred updating — because we
cannot distribute something, call it differential evolution, and have it do
something else.

**History carries every proposal, in order, with its state.** A proposal that
has not resolved and one that never will are both visible to the learner as a
position spent without a usable cost. Keep the two states distinct in the
record even though the learner does not need the distinction: the session and
diagnostics do, and it cannot be recovered later without changing the seam
again. Do not represent both as a missing cost.

**Slot is position, not count.** Founding the population from the first N
*usable* records is wrong: a dropped or unusable founder spills founding into
the next block, promoting a trial to founder and re-indexing every role after
it — and when that founder returns late, every role moves again. Instead, slot
is the position within the block for founders and trials alike; a slot's member
is its best usable result; a slot whose founder produced no cost stays vacant
until a later trial for it lands, and a trial for a vacant slot is drawn
founder-style because there is no incumbent to cross over with.

**Whole generations are submitted at once.** One attribute on the learner base —
`generation`, an integer or nothing — declares "proposes only whole groups of
this many, and only when none of its proposals is outstanding". DE sets it to
the population size. The name is the DE literature's, which is why the Gaussian
process knob that used to share the word is renamed in the configuration slice. Refill submits a whole generation when nothing of the
session's is outstanding and nothing otherwise. This is a declaration in the
same family as the minimum-observations and phase attributes, not a delivery
policy: the proposal call keeps its signature, and whether anything is
outstanding remains runmanager's answer rather than a locally kept count.

With the boundary: refuse `num_buffered_runs` for a learner that declares a
generation, because it would duplicate the population size and a setting that
duplicates another can disagree with it. Do not count starvation at a
generation boundary — the queue empties there by design, and a counter that
fires by design is noise in the one number the documentation tells a lab to
watch. Say in the documentation that a generational learner empties the queue
once per generation.

**Remove `restart_tolerance`,** in both the code and the configuration. It is
the one remaining source of order dependence — the restart decision is taken at
a boundary from the costs resolved by then, and a cost arriving afterwards can
change it, turning a block generated as trials into founders of a new epoch. It
is also harmful within lab budgets: it re-seeds a population that has converged
to within a fraction of its initial spread but not to the minimum, and it is
why the shipped code loses on the slowly-converging test functions. The
no-better-parameters stop is what a lab actually wants there. Refuse the key and
list it in `UPGRADING.md`.

**Write down the late-cost semantics.** A shot behind a row that will not
compile is blocked and dropped; the operator deletes the red row, the shots run,
and their costs arrive after the next generation was built without them. That
is correct and is what textbook DE would have produced had the cost arrived in
time. Two generations sitting in the queue in that state is not a fault, and the
starvation and outstanding-shot prose must not call it one.

### Acceptance criteria

- [ ] A dropped founder leaves its slot vacant and shifts no other slot —
      mutation: found by usable count
- [ ] A late-returning founder changes no other proposal's role — mutation:
      found by usable count
- [ ] Roles assigned at proposal equal roles read at replay, under random
      completion order, drops, unusable costs and late returns, **checked after
      every arrival** — mutation: any counting walk. Checking only the final
      replay cannot fail: it is order-independent by construction and reports
      nothing wrong however broken the intermediate behaviour is
- [ ] An unusable trial leaves its slot's member unchanged — mutation: walk the
      usable subset
- [ ] With crossover disabled, a trial for a block position shares all but one
      coordinate with that position's member — mutation: target by completed
      count
- [ ] The configured start point is proposed once — mutation: completed-only
      history
- [ ] Refill proposes nothing while a generation is outstanding and exactly a
      full generation when none is — mutation: the top-up formula
- [ ] Starvation stays at zero across a clean generational run — mutation: count
      at boundaries
- [ ] `num_buffered_runs` in a file whose learner declares a generation is
      refused — mutation: delete the check; it loads and refill tops up
      mid-generation
- [ ] A pending proposal is not usable, is not counted as completed, and does
      not count towards the runs-since-best stop
- [ ] `restart_tolerance` is refused at load and gone from the code
- [ ] Documentation states the late-cost semantics and that a generational
      learner empties the queue once per generation

### Blocked by

- Slice 2: Configuration is validated at load — the population size becomes a
  member count there and this slice's generation equals it, and the Gaussian
  process knob is renamed there so the word `generation` is free

### User stories covered

- PRD "Issue 2 — differential evolution: textbook generational DE", all
  subsections

---

## Slice 5: The greeting's deadline is a constant again

### Type

`AFK`

### What to build

Revert the labconfig read introduced when this slice was first implemented. The
greeting's deadline goes back to a constant.

The reasoning that put it there was mine and does not survive scrutiny. BLACS
reads `timeouts/liveness_timeout` because it probes runmanager **once per
shot**, so the key is a real trade for it: a longer probe costs cycle time on
every shot, a shorter one gives false "runmanager unavailable" on a slow link.
This package greets **once per session**, over a round trip that is sub-second
on any lab link. There is no trade to make, so the knob buys nothing — and a
lab that raises the number for BLACS's sake, which is the only reason it is
configurable, silently reinstates the failure the greeting exists to prevent.

Keep a comment saying the key exists, that BLACS reads it, and why this greeting
does not: one number per session against one per shot.

The interim that holds the injected client to a short deadline stays, with its
comment naming `Client.with_timeout`. That is filed for runmanager's backlog and
is not blocking anything.

### Acceptance criteria

- [x] The greeting's deadline is a constant; nothing reads labconfig for it
- [x] A comment says the key exists, that BLACS reads it, and why this does not
- [x] The tests and fixtures added for the configurable value are removed, not
      left asserting a path that no longer exists
- [x] `tests/` no longer imports `labscript_utils` unless something else needs it

### Blocked by

None - can start immediately.

### User stories covered

- PRD "Issue 1 — worker reply offset", the residue subsection

---

## Slice 6: A wrapper cannot silently drop a generation barrier

### Type

`AFK`

### What to build

`TwoPhaseLearner` wraps a trainer and a main learner and does not forward the
main learner's `generation`. Nothing that needs training declares one today —
only the Gaussian process is wrapped, and it has no population — so this cannot
bite now. It is filed because of what happens when it does: a generational
learner placed behind a trainer would have its barrier silently dropped, the
session would top the queue up mid-generation, and the algorithm would quietly
stop being the algorithm it is named after. That is precisely the failure the
differential evolution slice exists to correct, left reachable by a second
route.

Forwarding is not obviously right — during training the trainer proposes, and
the trainer has no barrier, so a forwarded generation would describe a phase
that is not running. Refusing is the honest alternative: a wrapper that cannot
honour a barrier its main learner declares should say so at construction rather
than discard it. Decide which, and write down why the other was not chosen.

The general form of the rule is worth stating in the learner contract: an
attribute a session acts on must be either honoured or refused by anything that
wraps a learner. `last_phase` and `minimum_observations` should be checked
against that rule at the same time, since the same argument applies to both.

### Acceptance criteria

- [x] A generational learner behind a trainer either keeps its barrier or is
      refused at construction — mutation: whichever is chosen, the opposite
      behaviour passes silently today
- [x] The learner contract states what a wrapper owes an attribute the session
      acts on
- [x] `last_phase` and `minimum_observations` are checked against the same rule,
      and any gap is fixed or recorded

### Blocked by

None - can start immediately. The differential evolution slice is what makes it
reachable, and that has landed.

### User stories covered

- Found while implementing PRD "Issue 2"; not in the PRD, which did not
  anticipate a wrapped generational learner.

---

## Slice 7: A reply says which request it answers

### Type

`AFK`

### What to build

**The largest remaining slice, and the one a lab would feel.** The routine and
the worker exchange messages with nothing saying which reply answers which
request; they rely on order. Order is not reliable, and under generational
differential evolution it is routinely wrong.

Reproduced against the shipped code, driving the real worker with trailing work
of five seconds and shots arriving every 0.4 s:

```
  inv      handed over   verdict
    1           shot-1      None
    2           shot-2      None
    3    SOMEONE-ELSES      True  <-- a shot the session never proposed
    4           shot-4      None
    5           shot-5      True
```

The third invocation hands over a shot the session never proposed and receives
`True`, so the optimiser's results are written onto somebody else's shot. The
session's own shots receive no verdict and are not written to at all. This is
the defect Slice 1 closed, reached by a second route.

**Why it fires in ordinary operation.** The worker replies *before* its trailing
work, so a late reply never means a dead worker — it means the worker is still
inside the previous request's `reconcile` and `refill`. Under generational
submission that trailing work is `submit_shots` for a whole population once per
generation, with runmanager evaluating and writing N files through its GUI
thread: many seconds, by construction. A Gaussian process fit on a long history
is the second such case. And it does not self-correct: while the load lasts,
every invocation times out and every verdict lands one shot late.

**Why an unbounded wait is not enough on its own**, both verified by reading the
worker:

- An error from trailing work is sent *after* the reply it follows, so the
  stream can read `status_k, error_k, status_k+1`. A reader that takes the first
  message as its answer raises on a healthy shot and leaves the real reply in
  the pipe.
- A request whose handling raises before the reply produces an error and **no**
  status. The reader then waits forever on a worker that is alive, so a liveness
  check does not fire either.

So: number the requests. Every message the routine sends carries a request
number — a counter in the routine's storage, not the shot id, which is absent
for a no-observation message and can legitimately repeat when BLACS re-runs a
file. Every message the worker sends carries the number of the request it
belongs to. The reply to a request is the first `status` or `error` carrying its
number; a later error carrying that number is trailing work that failed, raised
when it is seen and naming which request it came from.

The worker owes exactly one `status` per request, unless the request itself
failed, in which case the error is the reply.

**The wait learns to tell dead from busy.** The drain takes the worker's process
handle and polls it between short waits, so a worker that has died is reported
as dead within about a second rather than at whatever deadline is set — and the
deadline is then only ever reached by a live worker doing slow work.

**Late verdicts land on the right shot.** The routine remembers the filepath
against the request number for requests it stopped waiting on — a handful at
most — and when that status arrives during a later drain, writes it onto *that*
file if the session took it.

**The bound stays as it is.** It already does its job: the routine puts its
message and waits a couple of seconds, so it never holds lyse up for longer than
that, whatever the worker is doing. Nothing about its value needs revisiting —
the defect here is that a wait which ends early attributes the next reply to the
wrong shot, and that is wrong at any bound. Numbering fixes it; the number does
not come into it.

### Acceptance criteria

- [x] With trailing work longer than the bound, every status is written onto the
      shot that produced it, and a shot the session never proposed is never
      written to — mutation: attribute by arrival order; the table above
      reproduces
- [x] A request whose handling raises still yields exactly one message carrying
      its number — mutation: drop the guarantee; the wait for that request never
      ends
- [x] An error from trailing work is raised when seen, names the request it came
      from, and is not mistaken for the next request's reply — mutation: treat
      any first message as the answer; a healthy shot raises
- [x] A worker whose child has exited is reported as dead within about a second,
      and the message says the worker died rather than that it was slow —
      mutation: remove the poll; the test waits out the bound

### Blocked by

None - can start immediately.

### User stories covered

- PRD "Issue 1 — worker reply offset". The PRD treats the offset as closed; it
  is closed only for the startup case, and this is the general one.

---

## Slice 8: The configure deadline covers the waits inside it

### Type

`AFK`

### What to build

An outer deadline set below the sum of the bounded waits inside it will always
fire first and name the wrong cause. That is the shape of the original defect,
and it is still present after the greeting fix.

Verified: after the greeting restores the client's own deadline, `check_ready`
calls `error_in_globals()` and then `get_labscript_file()`, each at the client's
full `communication_timeout` — 60 s by labconfig default — underneath a 30 s
configure deadline. A runmanager that says hello and then stops answering, its
GUI thread inside a compile or behind a modal dialog somebody left open, is an
ordinary lab state and produces exactly the original failure: the worker killed
mid-wait, the lab told the worker was slow.

Derive the configure deadline from what it must cover — the greeting's deadline
plus the client's own deadline for each further request `check_ready` makes,
plus a margin — reading the client's number the way `runmanager.remote.Client`
reads it, since that is the number the worker's client will actually use.
Document it as the sum, so that adding a third request to `check_ready` is
visibly a reason to change it.

With defaults that is a long time for a stalled routine to wait, which is
exactly why this slice follows the one that teaches the drain to tell a dead
worker from a slow one: a generous backstop is cheap once a dead worker is
reported in a second.

### Acceptance criteria

- [x] The derived deadline grows when the client's deadline grows — mutation:
      restore a constant; a labconfig giving a larger value leaves it unchanged
- [x] A runmanager that greets and then times out on a later request surfaces
      that request's own failure, naming it — mutation: set the outer deadline
      below the client's; the routine reports its own timeout instead
- [x] The relation "the greeting's deadline lies inside the configure deadline"
      still holds and is still tested

### Blocked by

- Slice 7: A reply says which request it answers — the drain's liveness poll is
  what makes a generous deadline acceptable, and both change the same function.

### User stories covered

- PRD "Issue 1 — worker reply offset", the residue subsection

---

## Slice 9: A learner's declarations are facts about an instance

### Type

`AFK`

### What to build

Four places read a fact about a learner from something standing in for the
learner, and each agrees with the truth only in the single case that exists
today.

- The refusal of `num_buffered_runs` beside a generational learner reads
  `generation` **off the class**. Differential evolution declares it as a
  `property`, so the class attribute is a truthy property object and the check
  passes by accident. A learner setting `self.generation` in `__init__` — the
  ordinary way — shows the base class's `None` and is invisible. Reproduced: the
  file loads, the session runs at a queue depth the learner never asked for, the
  queue is topped up mid-generation, and because starvation is deliberately not
  counted for a generational learner, the resulting default shots are missing
  from the one number the README tells a lab to watch.
- The budget refusal reads the constructor's **signature default** and the
  file's option to predict what an instance's population will be. Today the
  prediction matches. The day a learner clamps or derives its population, the
  check and the instance disagree with nothing said.
- Slice 4's defect was a **count** standing for a position.
- Slice 6's was a **wrapper** standing for the learner it wraps.

**The pattern, and the rule that closes it:** a fact about an object read from a
proxy for it — a count for a position, a wrapper for the wrapped, a class or a
signature for the instance. State in the learner contract, beside the rule Slice
6 added: *a learner's declarations are facts about an instance; nothing outside
a learner reads them from a class, a signature, a registry entry or a count — it
asks an instance.*

**Stop predicting: build the learner at load and ask it.** Configuration loading
already imports the selected learner's module to inspect its constructor, so
building it costs a constructor call and nothing else — no fitting happens at
construction. Then both refusals read the instance, and the budget check becomes
two whole generations of whatever the instance actually declares, which is the
same number for differential evolution and the right generalisation. Both
class-level reads go.

A bonus worth knowing: building at load wraps the Gaussian process in its
two-phase learner, so Slice 6's refusal of a wrapped generational learner starts
firing at load instead of at worker configure, with the apparatus already
running.

**Do not require a class-level declaration instead.** Differential evolution's
generation depends on `population_size`, which comes from the file, so a
class-level flag would need a second instance-level number beside it — two
declarations for one fact, which is the shape just removed from
`population_size`.

Add a registry contract test: construct every registered learner over a
one-parameter space with defaults, and assert its phase is a string and its
generation is either nothing or a positive whole number. That holds a future
learner to the declaration however it chooses to make it.

### Acceptance criteria

- [ ] A learner that declares its generation in `__init__` is refused a queue
      depth exactly as one declaring it on the class — mutation: restore the
      class-level read; the reproduction above is accepted again
- [ ] The budget refusal reads the built learner, not a signature default —
      mutation: restore the signature read; a learner that derives its own
      population is checked against the wrong number
- [ ] The registry contract test covers every registered learner — mutations:
      register a learner whose generation is zero, and one with no phase; each
      must fail this test and no other
- [ ] The learner contract states the rule

### Blocked by

None - can start immediately.

### User stories covered

- Found while validating Slices 4 and 6; not in the PRD, which did not
  anticipate reading a learner's declarations off anything but a learner.

---


## Slice 10: Benchmark the shipped DE

### Type

`AFK`

### What to build

Every benchmark behind the PRD was run against scratch reimplementations, not
against the code that ships. Re-run the comparison against the merged
implementation and record the result beside the decision it supports.

What the earlier sweeps established, and what this must therefore check: that
the generation barrier costs sample efficiency against an asynchronous variant
and that the gap does not close with budget; that the default population size of
eight is the right default across parameter counts; and that the strongly
multimodal test function remains the case where the correct algorithm loses to
the shipped one's accidental mixing. If the shipped implementation disagrees
with any of those, that is a finding about the implementation, not a reason to
revisit the decision.

Carry the caveats into whatever documentation cites the numbers: analytic test
functions only, a bounded range of parameter counts and budgets, and one
mutation strategy.

### Acceptance criteria

- [ ] The comparison runs against the shipped learner, not a reimplementation
- [ ] Population sizes either side of the default are measured at more than one
      parameter count
- [ ] Results are recorded with their caveats, and any documented guidance on
      sizing the population cites them
- [ ] A disagreement with the PRD's expectations is reported rather than
      quietly absorbed

### Blocked by

- Slice 4: Textbook generational differential evolution

### User stories covered

- PRD "Issue 2", the benchmark and dimension-sweep subsections

---

## Slice 11: Test cleanup

### Type

`AFK`

### What to build

Use the test-cleanup skill. The slices above are written test-first with a
mutation proof for every assertion, which is the right way to build them and
the wrong thing to leave behind in full: some of those tests exist only to drive
a development loop and pin implementation details that a later change should be
free to move.

Remove or rewrite the scaffolding; keep the tests that enforce behaviour a lab
depends on, the public interfaces, and the seams that were deliberately placed.
This is not a coverage-reduction pass — the aim is the smallest useful
behavioural safety net, not a smaller number.

Two classes deserve particular attention, because this package has produced both
repeatedly: a test that cannot fail is worse than no test, so anything surviving
this pass should have a mutation that kills it; and a test that asserts a
published name against the constant that defines it proves nothing.

Three specific items carried forward from earlier slices:

- **A test whose name claims more than it covers.** The test asserting that
  loading a configuration imports neither scipy nor scikit-learn passes on a
  minimal file, but the shipped example configuration *does* import both,
  because it carries a table for a named learner and every named table has its
  class resolved. Verified. Either make the property true for named tables or
  rename the test to the narrower thing it actually guards; a name that
  overclaims is the failure mode this project keeps finding.
- **The timeout relation test**, asserting one constant is smaller than another.
  It guards a real invariant that no behavioural test covers, which is why it
  survives, but see whether the design change to the greeting removes both
  constants and the test with them.
- **Nothing tests on the oldest supported interpreter.** `pyproject.toml`
  requires 3.11; the suite only ever runs on 3.14, whose deferred annotations
  hid an unresolvable annotation name that would have raised at import on 3.11.
  That bug is fixed, and the next one of its kind is still invisible.

### Acceptance criteria

- [ ] Every surviving test has a mutation that makes it fail, and that mutation
      is recorded
- [ ] Tests that only pinned an implementation detail of the development loop
      are gone
- [ ] Behaviour a lab depends on is still covered: what is written onto a shot,
      what is refused at load, what the session reports, and the DE role
      invariants
- [ ] The suite's size before and after is reported, with the reasoning for
      anything removed that looked load-bearing

### Blocked by

- Slices 1 through 10

### User stories covered

- Repository practice: "A check that does not run looks exactly like a clean
  one"
