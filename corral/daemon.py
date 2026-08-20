"""The scheduler daemon. One instance runs for the whole server (an admin
starts it once, in tmux or as a systemd service); every user's `corral submit`
just drops a JSON file in the shared spool directory for this process to pick
up. Scheduling policy (FIFO with bounded backfill) is documented on
`_plan_schedule` below.
"""
from __future__ import annotations

import fcntl
import os
import signal
import sys
import time
from pathlib import Path

from . import config, gpu, jobstore, tmux_exec
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


def _reap_running(running: dict[str, Job]) -> None:
    for job_id in list(running):
        job = running[job_id]
        # The exitcode file always lives alongside the log file, wherever that is.
        log_dir_path = Path(job.log_path)
        ec_path = log_dir_path.with_name(f"{job_id}.exitcode")
        if not ec_path.exists():
            continue
        try:
            exit_code = int(ec_path.read_text().strip())
        except ValueError:
            exit_code = -1
        job.exit_code = exit_code
        # `corral cancel` drops this sentinel before interrupting a running job,
        # so we record CANCELLED here regardless of the process's actual exit code.
        cancel_marker = log_dir_path.with_name(f"{job_id}.cancelled")
        if cancel_marker.exists():
            job.state = "CANCELLED"
            cancel_marker.unlink(missing_ok=True)
        else:
            job.state = "COMPLETED" if exit_code == 0 else "FAILED"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        jobstore.move_job(job, config.RUNNING_DIR, config.DONE_DIR)
        print(f"[corral] job {job_id} ({job.name}) finished: state={job.state}, exit={exit_code}, GPUs {job.gpu_ids} freed")
        del running[job_id]


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

    Bounded backfill: jobs that fit launch in FIFO order. The first job that
    doesn't fit becomes "the head" for this poll; later jobs can still
    backfill into whatever GPUs are left over, for up to `max_skips`
    consecutive polls of the head being blocked. Past that, every free GPU
    is reserved for the head instead, so it can't be starved forever.
    `max_skips=0` is plain strict FIFO.

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


def _launch(job: Job, gpu_ids: list[int]) -> None:
    job.gpu_ids = gpu_ids
    job.state = "RUNNING"
    job.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    log_dir = config.log_dir_for(job.cwd)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.id}.log"
    exitcode_path = log_dir / f"{job.id}.exitcode"
    job.log_path = str(log_path)
    tmux_exec.launch(job.id, gpu_ids, job.cmd, job.cwd, str(log_path), str(exitcode_path))
    jobstore.move_job(job, config.PENDING_DIR, config.RUNNING_DIR)
    print(f"[corral] launched job {job.id} ({job.name}) on GPUs {gpu_ids} for user={job.user}")


def _launch_assignments(assignments: list[tuple[Job, list[int]]], running: dict[str, Job]) -> None:
    for job, chosen in assignments:
        try:
            _launch(job, chosen)
            running[job.id] = job
        except Exception as e:
            # A launch failure must never take down scheduling for every other
            # job on the server (see README's Design notes) -- fail just this one.
            job.state = "FAILED"
            job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            jobstore.move_job(job, config.PENDING_DIR, config.DONE_DIR)
            print(
                f"[corral] ERROR: failed to launch job {job.id} ({job.name}): {e} -- "
                f"marked FAILED, GPUs {chosen} released back to the pool.",
                file=sys.stderr,
            )


def run(poll_interval: float | None = None) -> None:
    config.ensure_dirs()
    lock_file = _acquire_lock()  # noqa: F841 -- held for the process lifetime
    poll_interval = poll_interval or config.POLL_INTERVAL_SEC

    # Resume tracking of jobs already RUNNING on disk from a previous daemon instance
    # (e.g. after a restart). Their GPUs stay reserved until an exitcode file appears --
    # they are never relaunched.
    running: dict[str, Job] = {j.id: j for j in jobstore.list_jobs(config.RUNNING_DIR)}
    if running:
        print(f"[corral] resuming tracking of {len(running)} already-running job(s)")

    free_streak: dict[int, int] = {}
    backfill_skip_counts: dict[str, int] = {}
    print(f"[corral] daemon started, pid={os.getpid()}, spool={config.SPOOL_DIR}, poll={poll_interval}s")

    def _shutdown(signum, frame):
        print("[corral] shutdown signal received -- exiting. Running jobs are unaffected, "
              "they keep running in their own tmux windows; restart the daemon to resume tracking them.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        _reap_running(running)
        _sweep_cancelled()

        try:
            gpus = gpu.detect_gpus()
        except RuntimeError as e:
            print(f"[corral] GPU detection failed: {e}", file=sys.stderr)
            time.sleep(poll_interval)
            continue

        # A GPU already assigned to a job we're tracking as RUNNING is never eligible,
        # regardless of what nvidia-smi says -- a freshly-launched job's memory usage
        # ramps up gradually during weight loading and can look "free" for its first
        # poll or two, so relying on nvidia-smi alone can double-book a GPU.
        reserved = set()
        for j in running.values():
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

        pending = sorted(jobstore.list_jobs(config.PENDING_DIR), key=lambda j: j.id)
        for job in pending:
            if job.n_gpus > len(gpus):
                print(
                    f"[corral] WARNING: job {job.id} ({job.name}) requests {job.n_gpus} GPU(s) but "
                    f"this server only has {len(gpus)} total -- it can never run. "
                    f"Cancel it with `corral cancel {job.id}`.",
                    file=sys.stderr,
                )

        assignments = _plan_schedule(pending, eligible, len(gpus), backfill_skip_counts, config.BACKFILL_MAX_SKIPS)
        _launch_assignments(assignments, running)

        time.sleep(poll_interval)
