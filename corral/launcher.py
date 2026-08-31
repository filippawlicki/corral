"""Per-user job launcher. Runs entirely as the submitting user's own OS
account, auto-started by their first `corral submit` into a tmux session
named `<CORRAL_TMUX_SESSION>-<user>` (see `cli._ensure_launcher`). It never
touches another user's files: the central `corral daemon` only decides which
GPUs a job gets, writing that into the shared spool as state GRANTED; this
process does the actual launching and reaping for jobs belonging to the user
running it, so every file a job creates -- logs, checkpoints, anything -- is
owned by the person who submitted it.
"""
from __future__ import annotations

import fcntl
import getpass
import os
import signal
import sys
import time
from pathlib import Path

from . import config, jobstore, tmux_exec


def _acquire_lock(user: str):
    config.ensure_dirs()
    lock_path = config.SPOOL_DIR / f"launcher-{user}.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[corral] a launcher for {user} is already running -- exiting.", file=sys.stderr)
        sys.exit(1)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file  # keep a reference alive for the life of the process


def _launch_granted(user: str, session: str) -> None:
    for job in jobstore.list_jobs(config.GRANTED_DIR):
        if job.user != user:
            continue
        log_dir = config.log_dir_for(job.cwd)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job.id}.log"
        exitcode_path = log_dir / f"{job.id}.exitcode"
        job.log_path = str(log_path)
        try:
            tmux_exec.launch(job.id, job.gpu_ids or [], job.cmd, job.cwd, str(log_path), str(exitcode_path), session)
        except Exception as e:
            job.state = "FAILED"
            job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            jobstore.move_job(job, config.GRANTED_DIR, config.DONE_DIR)
            print(f"[corral] ERROR: failed to launch job {job.id} ({job.name}): {e} -- marked FAILED, GPUs {job.gpu_ids} released.", file=sys.stderr)
            continue
        job.state = "RUNNING"
        job.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        jobstore.move_job(job, config.GRANTED_DIR, config.RUNNING_DIR)
        print(f"[corral] launched job {job.id} ({job.name}) on GPUs {job.gpu_ids}")


def _reap_running(user: str) -> None:
    for job in jobstore.list_jobs(config.RUNNING_DIR):
        if job.user != user or not job.log_path:
            continue
        log_dir_path = Path(job.log_path)
        ec_path = log_dir_path.with_name(f"{job.id}.exitcode")
        if not ec_path.exists():
            continue
        try:
            exit_code = int(ec_path.read_text().strip())
        except ValueError:
            exit_code = -1
        job.exit_code = exit_code
        # `corral cancel` drops this sentinel before interrupting a running job,
        # so we record CANCELLED here regardless of the process's actual exit code.
        cancel_marker = log_dir_path.with_name(f"{job.id}.cancelled")
        if cancel_marker.exists():
            job.state = "CANCELLED"
            cancel_marker.unlink(missing_ok=True)
        else:
            job.state = "COMPLETED" if exit_code == 0 else "FAILED"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        jobstore.move_job(job, config.RUNNING_DIR, config.DONE_DIR)
        print(f"[corral] job {job.id} ({job.name}) finished: state={job.state}, exit={exit_code}, GPUs {job.gpu_ids} freed")


def run(poll_interval: float | None = None) -> None:
    user = getpass.getuser()
    lock_file = _acquire_lock(user)  # noqa: F841 -- held for the process lifetime
    poll_interval = poll_interval or config.POLL_INTERVAL_SEC
    session = config.launcher_session(user)

    print(f"[corral] launcher started for {user}, pid={os.getpid()}, session={session}")

    def _shutdown(signum, frame):
        print(f"[corral] launcher shutdown -- running jobs are unaffected, they keep running in "
              f"tmux session '{session}'; restart the launcher (`corral launcher`) to resume tracking them.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    while True:
        _reap_running(user)
        _launch_granted(user, session)
        time.sleep(poll_interval)
