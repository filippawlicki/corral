"""All configuration is via environment variables so there is nothing to edit
in the source to deploy this on a new server -- see README.md for the admin
setup steps. Every value has a sane single-user default so `corral` also
works out of the box for one person on their own machine.
"""
from __future__ import annotations

import os
from pathlib import Path

CORRAL_HOME = Path(os.environ.get("CORRAL_HOME", os.path.expanduser("~/.corral"))).resolve()
SPOOL_DIR = CORRAL_HOME / "spool"
PENDING_DIR = SPOOL_DIR / "pending"
GRANTED_DIR = SPOOL_DIR / "granted"
RUNNING_DIR = SPOOL_DIR / "running"
DONE_DIR = SPOOL_DIR / "done"
LOCK_PATH = SPOOL_DIR / "daemon.lock"

# Where job logs are written. Job *state* always lives under CORRAL_HOME, so
# `corral queue`/`status`/`log` work regardless -- this only controls where
# the raw output bytes land. If unset, each job's logs default to a
# `.corral/logs/` folder inside the directory it was submitted from (see
# `log_dir_for`). See README's Design notes for the permissions this implies.
_log_dir_env = os.environ.get("CORRAL_LOG_DIR")
LOG_DIR: Path | None = Path(_log_dir_env).resolve() if _log_dir_env else None


def log_dir_for(job_cwd: str) -> Path:
    """Resolves the log directory for a job submitted from `job_cwd`."""
    return LOG_DIR if LOG_DIR is not None else Path(job_cwd) / ".corral" / "logs"

# Base name for per-user launcher tmux sessions -- each user's jobs run in
# their own session, `<TMUX_SESSION>-<user>`, under their own OS account.
TMUX_SESSION = os.environ.get("CORRAL_TMUX_SESSION", "corral")


def launcher_session(user: str) -> str:
    return f"{TMUX_SESSION}-{user}"

# How often the daemon polls nvidia-smi and re-evaluates the queue.
POLL_INTERVAL_SEC = float(os.environ.get("CORRAL_POLL_INTERVAL", "10"))

# A GPU must be seen "free" for this many consecutive polls before it's considered
# eligible for scheduling. Guards against a GPU that looks idle for one poll because
# a job (ours or someone else's) is still ramping up during weight loading.
FREE_STREAK_REQUIRED = int(os.environ.get("CORRAL_FREE_STREAK", "2"))

# A GPU with less than this much memory used is considered "free". Nonzero because
# idle GPUs can show a small amount of driver/context overhead even with nothing
# meaningful running on them.
FREE_MEM_THRESHOLD_MIB = int(os.environ.get("CORRAL_FREE_MEM_THRESHOLD_MIB", "512"))

# Consecutive polls a blocked head-of-queue job may be skipped by smaller
# jobs backfilling into GPUs it doesn't need, before they're reserved for it
# instead. 0 disables backfill (plain strict FIFO).
BACKFILL_MAX_SKIPS = int(os.environ.get("CORRAL_BACKFILL_MAX_SKIPS", "5"))

# How long a CANCELLED record is kept in spool/done/ as a short audit trail
# before the daemon sweeps it away. COMPLETED/FAILED records are never swept.
CANCELLED_RETENTION_SEC = int(os.environ.get("CORRAL_CANCELLED_RETENTION_SEC", "600"))


def ensure_dirs() -> None:
    for d in (PENDING_DIR, GRANTED_DIR, RUNNING_DIR, DONE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if LOG_DIR is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    # else: per-job log dirs are created lazily at launch time, since where
    # they live depends on each job's own cwd.
