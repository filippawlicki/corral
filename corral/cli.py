from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config, gpu, jobstore
from .daemon import run as daemon_run
from .launcher import run as launcher_run
from .models import Job
from .tmux_exec import interrupt as tmux_interrupt


def cmd_gpus(args) -> None:
    try:
        gpus = gpu.detect_gpus()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    reserved: dict[int, Job] = {}
    for j in jobstore.list_jobs(config.RUNNING_DIR) + jobstore.list_jobs(config.GRANTED_DIR):
        for idx in (j.gpu_ids or []):
            reserved[idx] = j

    held_back = config.reserved_gpus()
    if held_back:
        print(f"{held_back} GPU(s) held back by admin (see `corral reserve`) -- corral will never schedule onto them.\n")

    print(f"{len(gpus)} GPU(s) detected:\n")
    print(f"{'IDX':>3}  {'NAME':<22} {'MEM USED / TOTAL':>20}  STATUS")
    for g in gpus:
        if g.index in reserved:
            j = reserved[g.index]
            status = f"busy -- corral job {j.id} ({j.user})"
        elif g.mem_used_mib >= config.FREE_MEM_THRESHOLD_MIB:
            status = "busy -- external process"
        else:
            status = "free"
        mem = f"{g.mem_used_mib} / {g.mem_total_mib} MiB"
        print(f"{g.index:>3}  {g.name:<22} {mem:>20}  {status}")


def _ensure_launcher(user: str) -> None:
    """Starts this user's launcher (their own tmux session, their own OS
    account) if it isn't already running -- so `corral submit` needs no
    separate setup step. See README's Design notes.
    """
    session = config.launcher_session(user)
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        return
    # tmux does NOT forward the calling shell's environment into a new session
    # by default -- without `-e` (tmux >= 3.2), the launcher would silently
    # fall back to every CORRAL_* default instead of this user's actual config.
    env_flags = [f for k, v in os.environ.items() if k.startswith("CORRAL_") for f in ("-e", f"{k}={v}")]
    # Re-exec via this same interpreter/module rather than relying on a
    # `corral` entry point being on PATH -- works regardless of install method.
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, *env_flags, sys.executable, "-m", "corral.cli", "launcher"],
        check=False,
    )
    time.sleep(0.3)  # give tmux a moment to fail fast if the command couldn't start at all
    if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
        print(f"[corral] started your launcher (tmux session '{session}') -- it runs your jobs as you")
    else:
        print(
            f"[corral] warning: couldn't start your launcher automatically. Run `corral launcher` "
            f"yourself (e.g. `tmux new-session -d -s {session} corral launcher`) or this job will "
            f"never move past GRANTED.",
            file=sys.stderr,
        )


def cmd_submit(args) -> None:
    config.ensure_dirs()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("error: no command given. Usage: corral submit --gpus N -- <command...>", file=sys.stderr)
        sys.exit(1)
    if args.gpus < 1:
        print("error: --gpus must be at least 1", file=sys.stderr)
        sys.exit(1)

    user = getpass.getuser()
    job = Job(
        id=jobstore.new_job_id(),
        name=args.name or os.path.basename(cmd[0]),
        user=user,
        n_gpus=args.gpus,
        cmd=cmd,
        cwd=os.path.abspath(args.cwd or os.getcwd()),
        priority="urgent" if args.urgent else "normal",
        submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    _ensure_launcher(user)
    jobstore.write_job(job, config.PENDING_DIR)
    print(f"submitted job {job.id} ({job.name}), requesting {job.n_gpus} GPU(s)"
          + (", priority=urgent" if args.urgent else ""))
    print(f"  check status:  corral queue")
    print(f"  view log:      corral log {job.id} -f")


def cmd_queue(args) -> None:
    running = sorted(jobstore.list_jobs(config.RUNNING_DIR), key=lambda j: j.id)
    granted = sorted(jobstore.list_jobs(config.GRANTED_DIR), key=lambda j: j.id)
    pending = sorted(jobstore.list_jobs(config.PENDING_DIR), key=lambda j: j.id)
    done = sorted(jobstore.list_jobs(config.DONE_DIR), key=lambda j: j.id)[-args.last:]

    def _print_table(title: str, jobs: list[Job]) -> None:
        print(f"\n{title} ({len(jobs)}):")
        if not jobs:
            print("  (none)")
            return
        print(f"  {'ID':<20} {'USER':<12} {'NAME':<22} {'GPUS':<5} {'PRI':<7} {'STATE':<10} GPU_IDS")
        for j in jobs:
            pri = "urgent" if j.priority == "urgent" else "-"
            print(f"  {j.id:<20} {j.user:<12} {j.name[:22]:<22} {j.n_gpus:<5} {pri:<7} {j.state:<10} {j.gpu_ids or '-'}")

    _print_table("RUNNING", running)
    _print_table("GRANTED (waiting for the owner's launcher to start them)", granted)
    _print_table("PENDING (FIFO order)", pending)
    _print_table(f"RECENTLY FINISHED (last {args.last})", done)


def _find_job(job_id: str) -> tuple[Job | None, str | None]:
    for where, d in (
        ("pending", config.PENDING_DIR),
        ("granted", config.GRANTED_DIR),
        ("running", config.RUNNING_DIR),
        ("done", config.DONE_DIR),
    ):
        p = d / f"{job_id}.json"
        if p.exists():
            return jobstore.read_job(p), where
    return None, None


def cmd_status(args) -> None:
    job, _where = _find_job(args.job_id)
    if job is None:
        print(f"error: no job with id {args.job_id}", file=sys.stderr)
        sys.exit(1)
    for k, v in job.to_dict().items():
        print(f"{k}: {v}")
    if job.log_path and os.path.exists(job.log_path):
        print("\n--- last 20 lines of log ---")
        with open(job.log_path) as f:
            print("".join(f.readlines()[-20:]))


def cmd_cancel(args) -> None:
    job, where = _find_job(args.job_id)
    if job is None:
        print(f"error: no job with id {args.job_id}", file=sys.stderr)
        sys.exit(1)
    retention_min = config.CANCELLED_RETENTION_SEC // 60
    if where == "pending":
        job.state = "CANCELLED"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        jobstore.move_job(job, config.PENDING_DIR, config.DONE_DIR)
        print(
            f"cancelled pending job {job.id} (never started, no GPUs were reserved). "
            f"Kept as CANCELLED in `corral queue`/`status` for ~{retention_min} more minute(s)."
        )
    elif where == "granted":
        job.state = "CANCELLED"
        job.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        jobstore.move_job(job, config.GRANTED_DIR, config.DONE_DIR)
        print(
            f"cancelled job {job.id} before its launcher started it (GPUs {job.gpu_ids} released). "
            f"Kept as CANCELLED for ~{retention_min} more minute(s)."
        )
    elif where == "running":
        session = config.launcher_session(job.user)
        tmux_interrupt(job.id, session)
        # Sentinel file tells the owner's launcher this exit was a user-requested
        # cancel, so it records CANCELLED instead of COMPLETED/FAILED once it reaps it.
        if job.log_path:
            Path(job.log_path).with_name(f"{job.id}.cancelled").touch()
        print(
            f"sent Ctrl-C to running job {job.id} (tmux window {job.id[:30]} in session "
            f"'{session}'). It will be reaped automatically once its process exits, "
            f"and recorded as CANCELLED (kept for ~{retention_min} more minute(s)). "
            f"If it doesn't stop, attach with `tmux attach -t {session}` and kill it by hand."
        )
    else:
        print(f"job {job.id} already finished (state={job.state}), nothing to cancel")


def cmd_log(args) -> None:
    job, _where = _find_job(args.job_id)
    if job is None or not job.log_path:
        print(f"error: no log found for job {args.job_id}", file=sys.stderr)
        sys.exit(1)
    if args.follow:
        os.execvp("tail", ["tail", "-f", job.log_path])
    else:
        with open(job.log_path) as f:
            print(f.read())


def cmd_reserve(args) -> None:
    config.ensure_dirs()
    if args.gpus is None:
        print(f"{config.reserved_gpus()} GPU(s) currently held back from corral scheduling")
        return
    if args.gpus < 0:
        print("error: --gpus must be >= 0", file=sys.stderr)
        sys.exit(1)
    config.set_reserved_gpus(args.gpus)
    print(f"holding back {args.gpus} GPU(s) -- takes effect on the daemon's next poll, no restart needed")


def cmd_daemon(args) -> None:
    daemon_run(poll_interval=args.poll_interval)


def cmd_launcher(args) -> None:
    launcher_run(poll_interval=args.poll_interval)


def main() -> None:
    ap = argparse.ArgumentParser(prog="corral", description="A simple FIFO GPU job queue for a shared server.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gpus", help="show detected GPUs and their current status")
    p.set_defaults(func=cmd_gpus)

    p = sub.add_parser("submit", help="submit a job to the queue")
    p.add_argument("--gpus", type=int, required=True, help="number of GPUs this job needs")
    p.add_argument("--name", type=str, default=None, help="optional job name (default: the command)")
    p.add_argument("--cwd", type=str, default=None, help="working directory for the job (default: cwd)")
    p.add_argument("--urgent", action="store_true", help="schedule ahead of normal-priority jobs (e.g. a deadline)")
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run, e.g. -- python train.py")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("queue", help="show pending/running/recently finished jobs")
    p.add_argument("--last", type=int, default=10, help="how many recently finished jobs to show")
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("status", help="show full details for one job")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("cancel", help="cancel a pending job, or interrupt a running one")
    p.add_argument("job_id")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("log", help="show a job's log")
    p.add_argument("job_id")
    p.add_argument("-f", "--follow", action="store_true")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("reserve", help="hold back N GPUs from corral scheduling, live (admin)")
    p.add_argument("--gpus", type=int, default=None, help="GPUs to hold back; omit to show the current value")
    p.set_defaults(func=cmd_reserve)

    p = sub.add_parser("daemon", help="run the scheduler daemon in the foreground (admin runs this once)")
    p.add_argument("--poll-interval", type=float, default=None)
    p.set_defaults(func=cmd_daemon)

    p = sub.add_parser("launcher", help="run your personal job launcher (auto-started by `corral submit`)")
    p.add_argument("--poll-interval", type=float, default=None)
    p.set_defaults(func=cmd_launcher)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
