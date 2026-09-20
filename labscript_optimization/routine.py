"""The lyse multishot routine.

A lab analysis routine is two lines::

    import labscript_optimization.routine as optimisation
    optimisation.optimise('mloop_config.toml')

Adding the routine to lyse starts the session; removing it, restarting it, or
reaching the run budget stops it. :data:`SHOT_RESULTS` is written onto each
shot the session proposed, as lyse results under :data:`RESULTS_GROUP`, so
the best cost and where the search has got to are columns of the dataframe.

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
shots already in the box -- it is several. Every one of them is a run the
session spent, so every one of them is handed over.
"""

import atexit
import os
import subprocess
import sys
import time

import numpy as np

#: The lyse results group the session's status is written to, and so the first
#: level of every column it produces: ``df[('labscript_optimization',
#: 'best_cost')]``. lyse names a routine's group after the routine's file, so a
#: lab collides with this only by naming a routine after the package it imports.
RESULTS_GROUP = "labscript_optimization"

#: The status keys written onto a shot, and so the columns the session
#: produces: where the search has got to, and whether it has stopped. The rest
#: of the status is the session's bookkeeping, one answer for the whole run
#: that would be repeated onto every shot of it; :func:`optimise` returns all
#: of it.
SHOT_RESULTS = ("phase", "best_cost", "best_params", "best_shot_id", "stopped")

#: Seconds the routine waits for the worker to answer the message it has just
#: sent. Generous for an answer that is a dictionary and a socket hop, and
#: short against a shot cycle.
REPLY_TIMEOUT = 2.0

#: Seconds allowed for the worker to load its configuration and establish its
#: first runmanager connection. This is startup, not part of a shot cycle.
CONFIGURE_TIMEOUT = 30.0

#: Seconds between one look at the worker's process and the next while waiting
#: for a reply. A worker that has died is reported as dead within about this
#: long, so a deadline is only ever reached by a live worker doing slow work.
LIVENESS_POLL = 0.5

#: The number carried by the ``configure`` request, and so where the routine's
#: request counter starts. Every later message the routine sends is numbered
#: from here upwards, and every message the worker sends carries the number of
#: the request it belongs to.
CONFIGURE_REQUEST = 0

#: Rows asked of lyse first: the row this routine handled last, and the shot
#: analysed since. That is the steady state, where analysis keeps up, in one
#: request; a batch larger than this is reached by doubling.
FIRST_REQUEST = 2


def value(shot, key):
    """One column of a one-row frame, or ``None`` if there is no such column.

    A one-row frame rather than the row ``dataframe.iloc[-1]``: pandas
    resolves a key shallower than the column MultiIndex against a frame's
    columns, whatever the frame's depth, where against a row the same key
    names a sub-Series. ``key`` may be shallower than the frame's MultiIndex,
    whose padding levels are empty.
    """
    if key not in shot:
        return None
    return shot[key].iloc[-1]


def catch_up(handled):
    """Ask lyse for every shot of this sequence analysed after ``handled``.

    ``handled`` is the filepath of the row this routine last handled, or
    ``None`` on the first invocation of a session. The frame returned holds
    that row and everything after it, which is what tells the caller which
    rows are new.

    Asks for :data:`FIRST_REQUEST` rows and doubles until the handled row is
    in the frame or the sequence has no more rows to give, so a pile-up of any
    size is fetched whole while the steady state costs one request. The first
    invocation asks for one row: it hands nothing over, and reaching back over
    a sequence already hundreds of rows long would fetch all of it to make
    nothing of it.
    """
    import lyse

    if handled is None:
        return lyse.data(n_sequences=1, n_shots=1)
    wanted = FIRST_REQUEST
    while True:
        # One sequence because a run is one sequence: the rows of whatever ran
        # before this session are not shots it can report on.
        frame = lyse.data(n_sequences=1, n_shots=wanted)
        if len(frame) < wanted or handled in list(frame["filepath"]):
            return frame
        wanted *= 2


def unreported(dataframe, handled):
    """The rows after ``handled``, in order: the shots to hand over.

    None of them on the first invocation of a session, where there is no
    handled row: those shots were analysed before the session existed, and
    the sequence so far is not what it spent its runs on.

    All of them when the handled row is not in the frame, which is a sequence
    that has moved on further than the frame reaches, or a new sequence
    entirely. Handing over a row twice is harmless -- the session takes a cost
    for a shot id once -- and skipping one is a run it never hears about.
    """
    if handled is None:
        return dataframe.iloc[0:0]
    paths = list(dataframe["filepath"])
    if handled not in paths:
        return dataframe
    return dataframe.iloc[paths.index(handled) + 1 :]


def extract(shot, config):
    """Read the shot id and cost of one shot, given as a one-row frame.

    Returns ``(shot_id, cost, uncer, bad)``, or ``None`` when there is no id to
    read: lyse reads the identifier runmanager wrote into the file as a column,
    and it is empty for one of runmanager's default shots, which go to BLACS
    already compiled and so never have an id written into them. An id that is
    there does not make the shot the session's -- runmanager mints one for
    every row it compiles, a user's own shots included -- and which ids belong
    to the session is the session's own answer. The sign flip for ``maximize``
    happens here, on the way in, so everything downstream minimises; the
    session puts it back in the best cost it reports.
    """
    shot_id = value(shot, "shot_id")
    if shot_id is None or shot_id == "":
        return None

    cost, uncer = float("nan"), None
    raw = value(shot, config.cost_key)
    if raw is not None:
        cost = float(raw)
        measured = value(shot, config.uncertainty_key)
        if measured is not None and np.isfinite(float(measured)):
            uncer = float(measured)

    bad = not np.isfinite(cost)
    if not bad and config.maximize:
        cost = -cost
    return shot_id, cost, uncer, bad


def save_status(filepath, status) -> None:
    """Write :data:`SHOT_RESULTS` onto one shot, as lyse results.

    lyse reads the attributes of ``/results/<group>`` back as dataframe
    columns, so each key written becomes ``df[(RESULTS_GROUP, key)]`` against
    that shot. Only attributes are read that way, which is why ``best_params``
    is saved with ``save_result`` although it is a list --
    ``save_result_array`` would write it as a dataset, into a part of the file
    the dataframe never looks at. A value the session does not have yet is
    written as NaN, because an h5 attribute cannot be ``None``.

    A write that fails is reported to lyse's output and otherwise passed over.
    """
    try:
        import lyse

        run = lyse.Run(filepath)
        run.set_group(RESULTS_GROUP)
        # One open for the whole status. Left to itself each save_result opens
        # and locks the file again, and this runs inline in lyse.
        with run.open("r+"):
            for name in SHOT_RESULTS:
                reported = status[name]
                run.save_result(name, float("nan") if reported is None else reported)
    except Exception as exc:
        print(
            f"could not write the optimisation status to {filepath}: {exc!r}",
            file=sys.stderr,
        )


def start_worker(config_path, process_tree=None):
    """Spawn the optimisation worker and configure it.

    Configuring is :data:`CONFIGURE_REQUEST`, the session's first request.
    Waits for its reply, then returns ``(to_worker, from_worker, popen)``.
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

    worker = Worker(process_tree, startup_timeout=30)
    to_worker, from_worker = worker.start()
    handles = to_worker, from_worker, worker.child
    try:
        to_worker.put(
            ("configure", CONFIGURE_REQUEST, os.path.abspath(config_path))
        )
        status = _drain(
            from_worker, worker.child, CONFIGURE_REQUEST, {}, CONFIGURE_TIMEOUT
        )
        if status is None:
            raise TimeoutError(
                f"the optimisation worker did not configure within "
                f"{CONFIGURE_TIMEOUT:g} seconds"
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
    over. A status is written onto the files held against its own number, for
    each shot the session took, whichever drain it arrives in, and its number
    is dropped from ``pending`` once it has been. A status whose number
    ``pending`` no longer holds -- a request that handed nothing over, or one
    written to already -- has nothing to write.

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
        for filepath, taken in zip(pending.pop(number, ()), recorded):
            # The session taking a cost is what says the shot is one it
            # proposed: the id alone does not, because runmanager mints one
            # for every queue row it compiles, and writing the status onto a
            # shot the session never proposed would put a column of somebody
            # else's numbers against a user's own shot.
            if taken:
                save_status(filepath, status)
        if number == request:
            answer = status


def optimise(config_path, storage=None, dataframe=None):
    """Hand over the shots analysed since last time. The lyse routine entry point.

    Args:
        config_path: The TOML configuration.
        storage: Where to keep the worker and the sequence's place between
            invocations. Defaults to ``lyse.routine_storage``.
        dataframe: The shots to read, the row handled last among them.
            Defaults to what :func:`catch_up` asks of lyse.

    Returns:
        The whole status the worker sends in answer to this invocation, or
        ``None`` if the worker does not answer within :data:`REPLY_TIMEOUT`.
        For each shot the session took, :func:`save_status` has written
        :data:`SHOT_RESULTS` of that status onto it. An answer that misses the
        deadline is not lost: the shots this invocation handed over are
        remembered against its request number, and a later invocation writes
        that status onto them when it arrives. Worker configuration is
        acknowledged before the worker is stored, so the first invocation
        receives its own answer like every later one.
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
        # A session opens having handled nothing: the rows already in lyse's
        # box were analysed before it existed.
        storage.optimisation_last_row = None
        # Configuring was this session's first request; the counter carries
        # on from it.
        storage.optimisation_request = CONFIGURE_REQUEST
        # The shots handed over by each request still awaiting its status, so
        # that a status arriving after the routine gave up waiting for it is
        # written onto the shots that produced it. A handful of entries at
        # most: the worker owes one status per request.
        storage.optimisation_pending = {}
        # The ordinary shutdown, where lyse asks the analysis subprocess to
        # quit. A killed subprocess does not run this and the worker is left
        # to zprocess's heartbeat.
        atexit.register(stop_worker, storage)

    config = storage.optimisation_config
    to_worker, from_worker, popen = storage.optimisation_worker
    if dataframe is None:
        dataframe = catch_up(storage.optimisation_last_row)

    shots = unreported(dataframe, storage.optimisation_last_row)
    if len(dataframe):
        storage.optimisation_last_row = value(dataframe.iloc[[-1]], "filepath")

    handed, observations = [], []
    for position in range(len(shots)):
        shot = shots.iloc[[position]]
        observation = extract(shot, config)
        if observation is None:
            # One of runmanager's default shots, carrying no queue-row id.
            continue
        # Handed over whether or not its cost is usable: the shot has run and
        # lyse has analysed it, so withholding it would leave its id awaited
        # until a reconcile quietly dropped it, understating the runs spent.
        handed.append(value(shot, "filepath"))
        observations.append(observation)

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

    return _drain(from_worker, popen, request, storage.optimisation_pending)


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
