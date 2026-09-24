"""The lyse multishot routine.

A lab analysis routine is two lines::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('optimisation_config.toml')

Adding the routine to lyse starts the session; removing it, restarting it, or
reaching the run budget stops it. :data:`SHOT_RESULTS` is saved into lyse's
dataframe, as lyse results under :data:`RESULTS_GROUP` in the row of each
shot the session proposed, so the best cost, where the search has got to, and
what proposed each shot are columns of it.

The routine itself does almost nothing: it reads the costs of the shots lyse
has analysed since it last ran, hands them to the worker, and waits for the
worker to say where the session has got to.

Each message it sends carries a request number, and each message the worker
sends carries the number of the request it belongs to. A worker still inside
the work behind an earlier reply takes longer to answer than the routine is
willing to wait, which is ordinary under generational submission; the number
is what puts that answer onto the shots that earned it when it comes.

lyse runs a multishot routine once per drained batch of singleshot analyses
rather than once per shot. Where analysis keeps up that is one shot an
invocation, and where it does not -- a shot arriving while the one before it
is still being analysed, analysis paused and resumed, or lyse started with
shots already in the box -- it is several. lyse names them in ``lyse.paths``.
Every one of them is a run the session spent, so every one of them is handed
over.
"""

import atexit
import os
import subprocess
import sys
import time

import numpy as np

from . import runmanager_interface

#: The lyse results group the session's status is written to, and so the first
#: level of every column it produces: ``df[('labscript_optimization',
#: 'best_cost')]``. lyse names a routine's group after the routine's file, so a
#: lab collides with this only by naming a routine after the package it imports.
RESULTS_GROUP = "labscript_optimization"

#: The keys written onto a shot, and so the columns the session produces: what
#: proposed the shot, where the search has got to, and whether it has stopped.
#:
#: ``phase`` is the shot's own: the source the session recorded when it
#: proposed that shot, which the worker's verdict on the shot carries. It is
#: not the phase of whatever was proposed most recently, which is another shot
#: whenever more than one is in flight. The other four come from the session's
#: status, whose remaining keys are its bookkeeping, one answer for the whole
#: run that would be repeated onto every shot of it; :func:`optimise` returns
#: all of it.
#:
#: ``stopped`` is here rather than with the bookkeeping because it is a marker
#: and not a tally. A counter carries a running total onto every shot and says
#: nothing about the one it lands on; ``stopped`` is empty until the session
#: ends, so the first shot carrying a reason is the shot the run ended on, and
#: where it sits in the column is the answer to the question a lab asks of a
#: finished run.
SHOT_RESULTS = ("phase", "best_cost", "best_params", "best_shot_id", "stopped")

#: What each of :data:`SHOT_RESULTS` is saved as while the session has
#: nothing to report for it.
#:
#: lyse gives a dataframe column one dtype, and the shots already saved fix
#: it: a stand in of a different type than the value it holds a place for
#: types the column against that value, and the shot that finally has one
#: cannot be written into it. So each empty here carries the type of the value
#: that replaces it -- ``""`` for the string-valued keys, an empty list for the
#: parameter vector, NaN only for the float that NaN is the empty of. lyse
#: reads a shot with no identifier back as ``""`` for the same reason.
#:
#: None of these collide with a value the session reports: ``stopped`` is a
#: sentence, ``best_shot_id`` is an id runmanager minted, and a configuration
#: with no enabled parameters is refused, so ``best_params`` is never empty.
NO_VALUE_YET = {
    "phase": "",
    "best_cost": float("nan"),
    "best_params": [],
    "best_shot_id": "",
    "stopped": "",
}

#: Seconds the routine waits for the worker to answer the message it has just
#: sent. Generous for an answer that is a dictionary and a socket hop, and
#: short against a shot cycle.
REPLY_TIMEOUT = 2.0

#: Seconds :func:`configure_timeout` allows the worker on top of the runmanager
#: traffic it is derived from: reading the configuration file and building the
#: learner. That is work rather than a bounded wait, so it brings no deadline
#: of its own to the sum.
CONFIGURE_MARGIN = 10.0

#: Seconds between one look at the worker's process and the next while waiting
#: for a reply. A worker that has died is reported as dead within about this
#: long, so a deadline is only ever reached by a live worker doing slow work.
LIVENESS_POLL = 0.5

#: The number carried by the ``configure`` request, and so where the routine's
#: request counter starts. Every later message the routine sends is numbered
#: from here upwards, and every message the worker sends carries the number of
#: the request it belongs to.
CONFIGURE_REQUEST = 0


def analysed():
    """The rows of lyse's dataframe for the shots analysed since the last pass.

    ``lyse.paths`` names them, and is ``None`` outside lyse: with none named
    there is nothing to ask lyse for, and this is ``[]``. The rows come in one
    request, in the dataframe's order. A file named twice, after a failed
    pass, is one row, and a BLACS rerun is a file of its own carrying the same
    shot id, which passes through harmlessly because the session takes a cost
    for an id once.
    """
    import lyse

    paths = lyse.paths
    if not paths:
        return []
    return lyse.data(where={"filepath": paths})


def extract(shots, config):
    """Read the shot ids and costs of ``shots``, rows of lyse's dataframe.

    Returns two lists in step: the file of each shot there is an id to read,
    and its ``(shot_id, cost, uncer, bad)``. lyse reads the identifier
    runmanager wrote into the file as a column, and it is empty for one of
    runmanager's default shots, which go to BLACS already compiled and so
    never have an id written into them. An id that is there does not make the
    shot the session's -- runmanager mints one for every row it compiles, a
    user's own shots included -- and which ids belong to the session is the
    session's own answer. A shot with an id is read whether or not its cost is
    usable: it has run and lyse has analysed it, so withholding it would leave
    its id awaited until a reconcile quietly dropped it, understating the runs
    spent. The sign flip for ``maximize`` happens here, on the way in, so
    everything downstream minimises; the session puts it back in the best
    cost it reports.
    """
    # A column at a time off the frame rather than a row at a time: pandas
    # resolves a key shallower than the column MultiIndex, whose padding levels
    # are empty, against a frame's columns, where against a row the same key
    # names a sub-Series. A column the frame does not have is None throughout.
    keys = "filepath", "shot_id", config.cost_key, config.uncertainty_key
    columns = [shots[k] if k in shots else [None] * len(shots) for k in keys]
    filepaths, observations = [], []
    for filepath, shot_id, raw, measured in zip(*columns):
        if shot_id is None or shot_id == "":
            continue
        cost, uncer = float("nan"), None
        if raw is not None:
            cost = float(raw)
            if measured is not None and np.isfinite(float(measured)):
                uncer = float(measured)
        bad = not np.isfinite(cost)
        if not bad and config.maximize:
            cost = -cost
        filepaths.append(filepath)
        observations.append((shot_id, cost, uncer, bad))
    return filepaths, observations


def save_status(filepath, status) -> None:
    """Save :data:`SHOT_RESULTS` of ``status`` against one shot, as lyse results.

    ``status`` is what this shot is to carry: the session's status, with the
    shot's own ``phase`` beside it.

    Each key becomes ``df[(RESULTS_GROUP, key)]`` in that shot's row of lyse's
    dataframe, and is saved there alone, with ``save_to_h5=False``: lyse sets
    it into the row, and the shot file is not opened. The dataframe is where
    the status is read, and writing it into the file as well would take the
    file's h5 lock once per shot, inline in lyse. ``best_params`` is saved
    with ``save_result`` although it is a list, because ``save_result`` is
    what reaches the dataframe; ``save_result_array`` writes a dataset into
    the file and nothing more. A value the session does not have yet is saved
    as its :data:`NO_VALUE_YET` stand in, which has the type of the value it
    holds a place for, so that the column is one dtype from the first shot
    onwards.

    A save that fails is reported to lyse's output and otherwise passed over.
    """
    try:
        import lyse

        run = lyse.Run(filepath)
        run.set_group(RESULTS_GROUP)
        for name in SHOT_RESULTS:
            reported = status[name]
            if reported is None:
                reported = NO_VALUE_YET[name]
            run.save_result(name, reported, save_to_h5=False)
    except Exception as exc:
        print(
            f"could not write the optimisation status to {filepath}: {exc!r}",
            file=sys.stderr,
        )


def configure_timeout():
    """Seconds the worker is allowed to configure itself in.

    The sum of the waits it contains, because a deadline shorter than that sum
    fires first and names the wrong cause: the worker killed mid-wait, and the
    lab told that the worker was slow when runmanager was the one that stopped
    answering. The terms are

    * :data:`~labscript_optimization.runmanager_interface.GREETING_TIMEOUT`,
      the one request runmanager is held to a short deadline for;
    * the client's own deadline once for each request ``check_ready`` makes
      after the greeting, of which there are
      :data:`~labscript_optimization.runmanager_interface.CHECK_READY_REQUESTS`,
      so that a third question asked there is visibly a reason to change this;
    * :data:`CONFIGURE_MARGIN`, for the worker's own startup work.

    The client's deadline is labconfig's ``timeouts/communication_timeout``,
    read here as ``runmanager.remote.Client.__init__`` reads it, fallback and
    all, because that is the number the worker's client will wait. With the
    fallback the sum is a little over two minutes, which is a long time for a
    stalled lyse routine -- and affordable because :func:`_drain` looks at the
    worker's process while it waits, so a worker that has died is reported
    within :data:`LIVENESS_POLL` and only a live worker ever reaches the
    deadline.
    """
    # Deferred, so that importing this module reads no files: the routine is
    # imported by lyse whether or not a session is ever started.
    from labscript_utils.labconfig import LabConfig

    client_timeout = LabConfig().getfloat(
        "timeouts", "communication_timeout", fallback=60
    )
    return (
        runmanager_interface.GREETING_TIMEOUT
        + runmanager_interface.CHECK_READY_REQUESTS * client_timeout
        + CONFIGURE_MARGIN
    )


def start_worker(config_path, process_tree=None):
    """Spawn the optimisation worker and configure it.

    Configuring is :data:`CONFIGURE_REQUEST`, the session's first request, and
    it is given :func:`configure_timeout`. Waits for its reply, then returns
    ``(to_worker, from_worker, popen)``.
    """
    # zprocess sends the class itself to the child, so the parent needs it.
    # Imported here rather than above so that a routine which never starts a
    # session does not pay for the learners the worker brings with it.
    from .worker import Worker

    if process_tree is None:
        from labscript_utils.ls_zprocess import ProcessTree

        # Inside a lyse analysis subprocess this is the tree already connected
        # to lyse, so the worker becomes a child of this routine's process and
        # goes away with it.
        process_tree = ProcessTree.instance()

    # zprocess's own deadline, and a different wait: how long the child has to
    # start and connect back, which is over before the configure request goes
    # out and the deadline below starts.
    worker = Worker(process_tree, startup_timeout=30)
    to_worker, from_worker = worker.start()
    handles = to_worker, from_worker, worker.child
    try:
        # Inside the try: a worker already spawned is reaped even if its
        # deadline is what could not be worked out.
        allowed = configure_timeout()
        to_worker.put(
            ("configure", CONFIGURE_REQUEST, os.path.abspath(config_path))
        )
        status = _drain(
            from_worker, worker.child, CONFIGURE_REQUEST, {}, allowed
        )
        if status is None:
            raise TimeoutError(
                f"the optimisation worker did not configure within "
                f"{allowed:g} seconds"
            )
    except BaseException:
        _stop_worker(handles)
        raise
    # The Popen, not the Process: stopping the worker escalates from a
    # request to terminate and then to kill, which zprocess does not do.
    return handles


def _drain(from_worker, popen, request, pending, timeout=None):
    """Wait for the worker's reply to ``request``, and return the status in it.

    The reply to a request is the first message carrying its number. Every
    other message is handled by what it means rather than by when it arrived,
    which is what keeps a reply the routine stopped waiting for from being
    read as the answer to the shots it is holding now.

    ``pending`` maps a request number to the shot files that request handed
    over. A status is written onto the shots held against its own number, for
    each shot the session took, whichever drain it arrives in, and its number
    is dropped from ``pending`` once it has been. Each of those shots is
    written with its own ``phase``, the source its verdict carries. A status
    whose number ``pending`` no longer holds -- a request that handed nothing
    over, or one written to already -- has nothing to write.

    An error raises, naming the request it carries. That is the reply to a
    request whose handling failed, and trailing work that failed behind a
    request already answered; the session has stopped either way.

    ``timeout`` bounds the wait and defaults to :data:`REPLY_TIMEOUT`;
    reaching it returns ``None`` and the shots this request handed over are
    written to when its status arrives in a later drain. The worker's process
    is looked at every :data:`LIVENESS_POLL` seconds while waiting, so a
    worker that has died is reported as one rather than waited out.

    Anything behind the reply is swept up too, but only if it is already
    waiting: waiting for more would hand lyse back the delay the worker exists
    to absorb, once per shot.
    """
    deadline = time.monotonic() + (REPLY_TIMEOUT if timeout is None else timeout)
    answer = None
    while True:
        if answer is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            waiting = min(LIVENESS_POLL, remaining)
        else:
            waiting = 0
        try:
            kind, number, payload = from_worker.get(timeout=waiting)
        except TimeoutError:
            if answer is not None:
                return answer
            if popen.poll() is not None:
                raise RuntimeError(
                    f"the optimisation worker died without answering request "
                    f"{request}"
                )
            continue
        if kind == "error":
            raise RuntimeError(
                f"the optimisation worker failed handling request {number}:"
                f"\n{payload}"
            )
        recorded, status = payload
        for filepath, source in zip(pending.pop(number, ()), recorded):
            # The session taking a cost is what says the shot is one it
            # proposed: the id alone does not, because runmanager mints one
            # for every queue row it compiles, and writing the status onto a
            # shot the session never proposed would put a column of somebody
            # else's numbers against a user's own shot.
            if source is not None:
                save_status(filepath, status | {"phase": source})
        if number == request:
            answer = status


def optimise(config_path, storage=None, shots=None):
    """Hand over the shots analysed since last time. The lyse routine entry point.

    Args:
        config_path: The TOML configuration.
        storage: Where to keep the worker between invocations. Defaults to
            ``lyse.routine_storage``.
        shots: The shots to hand over, rows of lyse's dataframe. Defaults to
            :func:`analysed`.

    Returns:
        The whole status the worker sends in answer to this invocation, or
        ``None`` if the worker does not answer within :data:`REPLY_TIMEOUT`.
        For each shot the session took, :func:`save_status` has written
        :data:`SHOT_RESULTS` onto it: that shot's own ``phase``, and the rest
        from that status. The status holds no ``phase`` of its own, because
        one request can hand over shots that different learners proposed. An
        answer that misses the deadline is not lost: the shots this
        invocation handed over are remembered against its request number, and
        a later invocation writes that status onto them when it arrives.
        Worker configuration is acknowledged before the worker is stored, so
        the first invocation receives its own answer like every later one.
        The first invocation whose status carries a stop reason prints why;
        later invocations of the same session get the same status back and do
        not print it again.
    """
    if storage is None:
        import lyse

        storage = lyse.routine_storage

    if getattr(storage, "optimisation_worker", None) is None:
        from . import config as config_module

        # Read once and kept for the life of the session. The worker holds the
        # configuration it was started with, so re-reading the file each shot
        # would let an edit mid-session leave the two disagreeing about what
        # the cost is -- a flipped maximize driving the search the wrong way.
        storage.optimisation_config = config_module.load(config_path)
        storage.optimisation_worker = start_worker(config_path)
        # Configuring was this session's first request; the counter carries
        # on from it.
        storage.optimisation_request = CONFIGURE_REQUEST
        # The shots handed over by each request still awaiting its status, so
        # that a status arriving after the routine gave up waiting for it is
        # written onto the shots that produced it. A handful of entries at
        # most: the worker owes one status per request.
        storage.optimisation_pending = {}
        # Whether this session has already told lyse why it stopped. Every
        # pass of a stopped session gets the same status back, and this is
        # what keeps it from being printed again on each one.
        storage.optimisation_stop_printed = False
        # The ordinary shutdown, where lyse asks the analysis subprocess to
        # quit. A killed subprocess does not run this and the worker is left
        # to zprocess's heartbeat.
        atexit.register(stop_worker, storage)

    config = storage.optimisation_config
    to_worker, from_worker, popen = storage.optimisation_worker
    if shots is None:
        shots = analysed()
    handed, observations = extract(shots, config)

    storage.optimisation_request += 1
    request = storage.optimisation_request
    # Remembered before the message goes out, because the reply is what clears
    # it: whether it arrives inside this invocation's wait or three
    # invocations later, it is written onto these shots and no others.
    storage.optimisation_pending[request] = handed
    if observations:
        # One message however many shots it carries. The routine waits for one
        # reply, so the verdicts a second message earned would go unread until
        # a later invocation, leaving shots of this batch unwritten for as long
        # as that took.
        to_worker.put(("observe", request, tuple(observations)))
    else:
        # An invocation with nothing to report still sends one. Reconciling
        # and refilling happen in the worker's trailing work, after it has
        # replied, so a routine that returned here would stop the session
        # giving up on shots that are not coming and stop it reviving a
        # generation an operator has unblocked.
        to_worker.put(("shot", request, None))

    status = _drain(from_worker, popen, request, storage.optimisation_pending)
    stopped = status is not None and status.get("stopped")
    if stopped and not storage.optimisation_stop_printed:
        # lyse shows what a routine prints. Nothing is raised, so lyse goes on
        # analysing and the shots still in flight are still taken. Printed
        # once per session: every later pass gets the same status back.
        print(f"The optimisation has stopped: {status['stopped']}")
        storage.optimisation_stop_printed = True
    return status


def exited_within(popen, timeout=5) -> bool:
    """Whether the worker has exited, waiting up to ``timeout`` seconds for it.

    Waiting is also reaping: a child nobody waits for stays a zombie for as
    long as the lyse analysis subprocess lives.
    """
    try:
        popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _stop_worker(handles) -> None:
    """Stop and reap one spawned worker, including a partly started one."""
    to_worker, _, popen = handles
    try:
        # Numberless: nothing is owed in reply, and nothing is waiting for one.
        to_worker.put(("quit", None, None))
    except Exception:
        # A pipe that will not carry the request changes nothing about what
        # follows: the worker is signalled and reaped either way.
        pass
    if exited_within(popen):
        return
    popen.terminate()
    if exited_within(popen):
        return
    popen.kill()
    # Nothing stronger is available, and blocking lyse's shutdown on a worker
    # stuck in the kernel would help nobody.
    exited_within(popen)


def stop_worker(storage=None) -> None:
    """Ask the worker to quit, and see that it has. Safe to call when there is none.

    Restarting the routine is the ordinary way to begin a fresh session, so a
    worker left behind here is one left behind every time.
    """
    if storage is None:
        import lyse

        storage = lyse.routine_storage
    handles = getattr(storage, "optimisation_worker", None)
    if handles is None:
        return
    storage.optimisation_worker = None
    _stop_worker(handles)
