"""The scheduler daemon. One instance runs for the whole server (an admin
starts it once, in tmux or as a systemd service). It only ever reads and
writes JSON files in the shared spool -- it decides which GPUs a job gets,
but never launches anything and never touches a user's own files. Each
user's `corral launcher` (see `corral/launcher.py`) does the actual
launching, running entirely under that user's own OS account. Scheduling
policy (FIFO with bounded backfill) is documented on `_plan_schedule` below.
"""
from __future__ import annotations

import fcntl
import os
import signal
import sys
import time

from . import config, gpu, jobstore
from .models import Job


def _acquire_lock():
    config.ensure_dirs()
    lock_file = open(config.LOCK_PATH, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"[corral] another daemon already holds the lock at {config.LOCK_PATH} -- exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file  # keep a reference alive for the life of the process


def _sweep_cancelled() -> None:
    """Removes CANCELLED records from spool/done/ past CANCELLED_RETENTION_SEC --
    a short-lived audit trail, unlike COMPLETED/FAILED records, which are kept
    forever.
    """
    cutoff = time.time() - config.CANCELLED_RETENTION_SEC
    for job in jobstore.list_jobs(config.DONE_DIR):
        if job.state != "CANCELLED" or not job.finished_at:
            continue
        finished_ts = time.mktime(time.strptime(job.finished_at, "%Y-%m-%dT%H:%M:%S"))
        if finished_ts < cutoff:
            (config.DONE_DIR / f"{job.id}.json").unlink(missing_ok=True)


def _plan_schedule(
    pending: list[Job],
    eligible: list[int],
    total_gpus: int,
    skip_counts: dict[str, int],
    max_skips: int,
) -> list[tuple[Job, list[int]]]:
    """Pure decision function -- no tmux/filesystem I/O, only mutates
    `skip_counts` -- so it's cheap to unit-test directly.

    `pending` is processed in the order given -- it's the caller's job to
    order it (FIFO by submission time, with urgent jobs sorted first; see
    `_sort_pending`). Bounded backfill: jobs that fit launch in that order.
    The first job that doesn't fit becomes "the head" for this poll; later
    jobs can still backfill into whatever GPUs are left over, for up to
    `max_skips` consecutive polls of the head being blocked. Past that,
    every free GPU is reserved for the head instead, so it can't be starved
    forever. `max_skips=0` is plain strict FIFO.

    Impossible jobs (more GPUs than the server has) are always skipped and
    never become "the head".
    """
    available = set(eligible)
    assignments: list[tuple[Job, list[int]]] = []
    head_job: Job | None = None
    backfill_paused = False

    for job in pending:
        if job.n_gpus > total_gpus:
            continue  # impossible job -- caller warns; never blocks, never "the head"

        if backfill_paused:
            break  # head's skip budget is exhausted -- every remaining free GPU is its alone

        free_now = sorted(available)
        if len(free_now) >= job.n_gpus:
            chosen = free_now[: job.n_gpus]
            available -= set(chosen)
            assignments.append((job, chosen))
            skip_counts.pop(job.id, None)  # it ran; forget its skip history
            continue

        # This job can't run yet.
        if head_job is None:
            head_job = job
            skip_counts[job.id] = skip_counts.get(job.id, 0) + 1
            if skip_counts[job.id] > max_skips:
                backfill_paused = True
                break  # budget exhausted -- reserve everything free for the head, stop scanning
        # else: a later job that also doesn't fit right now -- just keep scanning past it.

    # Forget skip history for anything no longer pending (it launched, finished, or was cancelled).
    pending_ids = {j.id for j in pending}
    for jid in list(skip_counts):
        if jid not in pending_ids:
            del skip_counts[jid]

    return assignments


def _sort_pending(pending: list[Job]) -> list[Job]:
    """Urgent jobs are considered before normal ones; FIFO order is preserved
    within each priority tier. `_plan_schedule` has no notion of priority --
    it just treats the first job in whatever order it's given as the head, so
    sorting here is the entire mechanism.
    """
    return sorted(pending, key=lambda j: (0 if j.priority == "urgent" else 1, j.id))


def _cap_eligible_gpus(eligible: list[int], total_gpus: int, committed: int, reserved_gpus: int) -> list[int]:
    """Trims `eligible` (free GPU indices, sorted ascending) so corral never
    holds more than `total_gpus - reserved_gpus` GPUs at once, counting
    `committed` GPUs already granted or running. Pure, so an admin raising or
    lowering the reservation live is trivial to unit-test without a daemon.
    """
    capacity = max(0, total_gpus - reserved_gpus)
    allowed_new = max(0, capacity - committed)
    return eligible[:allowed_new]


def _grant_assignments(assignments: list[tuple[Job, list[int]]]) -> None:
    for job, chosen in assignments:
        job.gpu_ids = chosen
        job.state = "GRANTED"
        jobstore.move_job(job, config.PENDING_DIR, config.GRANTED_DIR)
        print(f"[corral] granted job {job.id} ({job.name}) GPUs {chosen} for user={job.user} -- their launcher will start it")


def run(poll_interval: float | None = None) -> None:
    config.ensure_dirs()
    lock_file = _acquire_lock()  # noqa: F841 -- held for the process lifetime
    poll_interval = poll_interval or config.POLL_INTERVAL_SEC

    free_streak: dict[int, int] = {}
    backfill_skip_counts: dict[str, int] = {}
    print(f"[corral] daemon started, pid={os.getpid()}, spool={config.SPOOL_DIR}, poll={poll_interval}s")

    def _shutdown(signum, frame):
        print("[corral] shutdown signal received -- exiting. Granted/running jobs are unaffected, "
              "each user's own launcher keeps tracking them; restart the daemon to resume scheduling.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        _sweep_cancelled()

        try:
            gpus = gpu.detect_gpus()
        except RuntimeError as e:
            print(f"[corral] GPU detection failed: {e}", file=sys.stderr)
            time.sleep(poll_interval)
            continue

        # GPUs already granted or running are never eligible, regardless of what
        # nvidia-smi says -- a freshly-launched job's memory usage ramps up
        # gradually during weight loading and can look "free" for its first poll
        # or two, so relying on nvidia-smi alone can double-book a GPU. Re-reading
        # from disk each poll (rather than keeping in-memory state) keeps this
        # correct even though launchers, not this process, move jobs in and out
        # of these directories.
        reserved = set()
        for j in jobstore.list_jobs(config.GRANTED_DIR):
            reserved.update(j.gpu_ids or [])
        for j in jobstore.list_jobs(config.RUNNING_DIR):
            reserved.update(j.gpu_ids or [])

        seen_this_poll = set()
        for g in gpus:
            seen_this_poll.add(g.index)
            is_free = g.mem_used_mib < config.FREE_MEM_THRESHOLD_MIB and g.index not in reserved
            free_streak[g.index] = free_streak.get(g.index, 0) + 1 if is_free else 0
        for idx in list(free_streak):
            if idx not in seen_this_poll:
                del free_streak[idx]

        eligible = sorted(i for i, streak in free_streak.items() if streak >= config.FREE_STREAK_REQUIRED)
        eligible = _cap_eligible_gpus(eligible, len(gpus), len(reserved), config.reserved_gpus())

        pending = _sort_pending(jobstore.list_jobs(config.PENDING_DIR))
        for job in pending:
            if job.n_gpus > len(gpus):
                print(
                    f"[corral] WARNING: job {job.id} ({job.name}) requests {job.n_gpus} GPU(s) but "
                    f"this server only has {len(gpus)} total -- it can never run. "
                    f"Cancel it with `corral cancel {job.id}`.",
                    file=sys.stderr,
                )

        assignments = _plan_schedule(pending, eligible, len(gpus), backfill_skip_counts, config.BACKFILL_MAX_SKIPS)
        _grant_assignments(assignments)

        time.sleep(poll_interval)
