# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the full test suite:
```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

Run a single test file:
```bash
PYTHONPATH=. python3 -m unittest tests.test_scheduling -v
```

Run a single test case or method:
```bash
PYTHONPATH=. python3 -m unittest tests.test_scheduling.TestBoundedBackfill.test_small_job_backfills_while_head_is_blocked -v
```

Install locally for manual/live testing:
```bash
pip install --user -e .
```

No linter or formatter is configured in this repo (no ruff/black/flake8 config anywhere) -- don't assume one exists when reviewing or writing changes. There is no build step; `corral` is pure-stdlib Python, installed via `pyproject.toml`'s setuptools entry point (`corral = "corral.cli:main"`).

## Architecture

Corral is a FIFO GPU job queue for a shared server. There is no database and no RPC -- all coordination is JSON files on a shared filesystem, plus tmux for process execution.

### Job state is directory location, not just a field

A job is one JSON file (`corral/jobstore.py`). Its lifecycle is which directory it currently lives in under `$CORRAL_HOME/spool/`:

```
pending/  ->  granted/  ->  running/  ->  done/
```

The `state` field on the `Job` dataclass mirrors the directory for display, but the directory is the source of truth. Moving a job between states is `jobstore.move_job()`: write to the new directory, then unlink from the old one -- not atomic across the two operations, so a crash mid-move can briefly leave a job in both directories. Every reader (`list_jobs`, the CLI, the daemon, the launcher) tolerates that.

### Two processes, split by privilege boundary

This is the load-bearing design decision in the codebase -- get it wrong and you reintroduce a real file-ownership bug that motivated this split in the first place:

- **`corral/daemon.py`** (`corral daemon`, one instance, admin-run) is the scheduler. It reads `pending/`, decides which GPUs each job gets (`_plan_schedule`, a pure function with no I/O -- unit-test it directly), and writes that decision by moving the job into `granted/`. It must never launch a process or write outside its own spool directory. If you're tempted to have the daemon touch tmux or a user's files, stop -- that's exactly what this split exists to prevent, since the daemon's own OS account would otherwise end up owning every file a job creates.
- **`corral/launcher.py`** (`corral launcher`, one per user, auto-started by that user's first `corral submit`, runs entirely under their own account) does the actual work. It polls `granted/` for jobs where `job.user == getpass.getuser()`, launches them via `tmux_exec.launch()` into that user's own `<CORRAL_TMUX_SESSION>-<user>` tmux session, and moves them to `running/`. It also reaps `running/` jobs for that same user once their `.exitcode` file appears, moving them to `done/`.

Every file a job's command creates -- logs, checkpoints, anything -- ends up owned by the submitting user because the launcher genuinely runs as them, not because of any filesystem permission grant.

### GPU double-booking prevention

`corral/gpu.py` shells out to `nvidia-smi` on every daemon poll; there's no persistent GPU state to get out of sync. A GPU counts as free only if it's under `CORRAL_FREE_MEM_THRESHOLD_MIB`, *and* not already reserved by a job the daemon currently sees in `granted/` or `running/`, *and* it's looked free for `CORRAL_FREE_STREAK` consecutive polls in a row (`daemon.run`'s `free_streak` dict) -- that last rule guards against a job that's still ramping up its own GPU memory usage looking falsely idle for a poll or two. The daemon re-reads `granted/`/`running/` from disk every poll rather than caching in memory, because a separate process (each user's launcher) mutates those directories concurrently.

### Scheduling policy: bounded backfill

`daemon._plan_schedule` is the function to read closely before changing scheduling behavior -- it's pure (no I/O, mutates only the `skip_counts` dict passed into it) and has the heaviest test coverage in the repo (`tests/test_scheduling.py`). Plain FIFO wastes idle GPUs behind a blocked head-of-queue job; unbounded backfill can starve that head-of-queue job forever. The compromise: smaller jobs may backfill into GPUs the head doesn't need, for up to `CORRAL_BACKFILL_MAX_SKIPS` consecutive polls, after which every free GPU is reserved for the head until it launches. `CORRAL_BACKFILL_MAX_SKIPS=0` degenerates to plain strict FIFO -- keep that regression test green.

### Cancellation and the tee/SIGINT interaction

`corral cancel` on a running job does two things (`cli.cmd_cancel`): touches a `{job_id}.cancelled` sentinel file next to the log, then sends Ctrl-C into that job's tmux window. The launcher's reap logic (`launcher._reap_running`) checks for that sentinel and, if present, records `CANCELLED` regardless of the process's actual exit code -- otherwise a job killed by SIGINT would misleadingly show as `FAILED`.

Ctrl-C in a tmux window hits every process in the pipeline at once, including `tee`. `tmux_exec.launch()`'s inner shell script wraps `tee` as `(trap '' INT; exec tee ...)` so it survives long enough for a job's own `trap ... INT` cleanup handler to finish writing to it before the pipeline actually dies. `${PIPESTATUS[0]}` (not `tee`'s own exit code) is what gets written to the `.exitcode` file. If you touch this script, re-verify against a job that traps SIGINT and writes output from inside the trap -- that's the exact scenario this exists for.

`CANCELLED` records are swept from `done/` after `CORRAL_CANCELLED_RETENTION_SEC` (`daemon._sweep_cancelled`); `COMPLETED`/`FAILED` records are kept forever. That asymmetry is deliberate, not an oversight.

### Configuration is per-process, read once at import time

Everything in `corral/config.py` is read from environment variables at module import time -- there is no config file and no reload. Two consequences that matter when developing:

- **Tests must patch `config` module attributes** (`patch.object(config, "PENDING_DIR", ...)`), not re-import the module or mutate env vars after the fact -- follow the pattern in any existing test file.
- **Each long-running process reads its own environment once at startup.** The daemon and every user's launcher are separate OS processes; a config change (e.g. `CORRAL_LOG_DIR`) only takes effect for a launcher after that launcher process is restarted, not just re-exported in a shell.

### tmux does not forward the calling shell's environment

Non-obvious and easy to reintroduce as a bug: `tmux new-session` does **not** give the new session's process the environment of the shell that ran the `tmux` command -- only an explicit `-e VAR=value` (tmux >= 3.2) does. `cli._ensure_launcher()` (which auto-spawns a user's launcher on their first `corral submit`) builds `-e` flags for every `CORRAL_*` variable found in the current process's environment before calling `tmux new-session`, so a new `CORRAL_*` config var is picked up automatically. If you ever add another place in this codebase that spawns a tmux session for a long-running corral process, apply the same treatment.

## Testing conventions

- Stdlib `unittest` only -- no pytest, no test-time dependencies, matching the project's zero-dependency philosophy. Write new tests as a `TestCase`, not with pytest fixtures/parametrize.
- Tests that touch the filesystem use `tempfile.TemporaryDirectory()` plus `patch.object(config, ...)` on the specific spool directories under test (see `tests/test_daemon.py`, `tests/test_launcher.py`).
- Tests that touch tmux/subprocess mock `subprocess.run` directly (`tests/test_tmux_exec.py`) -- there is no real-tmux test mode.
- `_plan_schedule` (daemon.py) and the launch/reap functions (launcher.py) are pure or near-pure by design, specifically so they're unit-testable without real GPUs or a real tmux server. Preserve that when modifying them -- push I/O to the caller rather than folding it into these functions.
