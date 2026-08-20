# Corral

A simple FIFO GPU job queue for a shared research server. Think Slurm, but
small enough to read start to finish in an afternoon: no database, no
cluster daemon on every node, no config DSL. One Python process, one shared
directory of JSON files, and tmux.

```
corral gpus                            # what GPUs does this server have, and who's using them?
corral submit --gpus 2 -- python train.py
corral queue                           # what's running, what's waiting, what just finished
corral log <job-id> -f                 # tail a job's output
corral cancel <job-id>                 # cancel a pending job, or interrupt a running one
```

## Why

On a shared GPU box with a handful of researchers, everyone ends up hand-rolling
the same thing: check `nvidia-smi`, guess whether a GPU is really free, launch
in tmux so it survives a disconnect, hope nobody else launches onto the same
GPU thirty seconds later. Corral is that pattern, written once, shared by
everyone on the machine.

It is intentionally not trying to be Slurm: no accounting, no fair-share
scheduling, no cgroups-based isolation, no multi-node support. It solves one
problem -- **don't double-book a GPU, and run jobs in the order people asked
for them** -- for a single shared machine.

## How it works

- **GPU detection** is automatic (`nvidia-smi`), on every poll. Nothing to
  configure when GPUs are added or removed from the box.
- **Jobs are JSON files** in a shared spool directory (`$CORRAL_HOME/spool`).
  Submitting a job just writes a file to `spool/pending/`. A job's state is
  simply which subdirectory it's in (`pending/` -> `running/` -> `done/`),
  so an admin can inspect or hand-fix the whole queue with `ls` and `cat` --
  no database to open.
- **Job output is a plain log file**, tee'd to disk as the job runs (see
  `corral log`), so it isn't lost when the job's tmux window closes. By
  default it's written to a `.corral/logs/` folder inside the directory you
  ran `corral submit` from -- next to your own project, not buried in the
  shared spool -- but this is independently configurable via `CORRAL_LOG_DIR`
  (see [Configuration](#configuration)) if you'd rather centralize it.
- **Cancelling a job leaves a `CANCELLED` record**, same as `COMPLETED`/
  `FAILED`, visible in `corral queue`/`status` -- so you get confirmation the
  cancel actually registered instead of the job just silently vanishing. It's
  swept away automatically after `CORRAL_CANCELLED_RETENTION_SEC` (default 10
  minutes); unlike `COMPLETED`/`FAILED` records, which are kept forever.
- **One daemon process** (`corral daemon`) owns scheduling. It polls
  `nvidia-smi`, works out which GPUs are genuinely free, and launches the
  oldest pending job that fits into a dedicated tmux window, with
  `CUDA_VISIBLE_DEVICES` set to exactly the GPUs it assigned. Jobs survive
  the submitting user logging out, an SSH disconnect, or the daemon itself
  restarting.
- **Scheduling is FIFO with bounded backfill.** The oldest pending job (the
  "head") gets first priority, but if it can't run yet, a later, smaller job
  is allowed to run in its place using GPUs the head doesn't need -- for up
  to `CORRAL_BACKFILL_MAX_SKIPS` (default 5) consecutive scheduling passes.
  Past that, all newly-freed GPUs are reserved for the head alone until it
  can launch, so it's never starved indefinitely by a steady stream of
  smaller jobs behind it. Set `CORRAL_BACKFILL_MAX_SKIPS=0` for plain strict
  FIFO instead (the head always blocks everything immediately -- the
  original, simpler behavior, if you'd rather have that guarantee than the
  extra utilization).
- **A GPU only counts as free** if (a) `nvidia-smi` reports it under a
  memory threshold, (b) it isn't already assigned to a job the daemon is
  tracking as running, and (c) it has looked free for 2 consecutive polls in
  a row -- not just 1. That third rule exists because a freshly-started job
  (yours or someone else's, corral-managed or not) can take a few seconds to
  ramp up its memory usage while it loads weights, and can otherwise look
  "free" to a single memory-threshold check for a poll or two.

## Install (admin)

Requires Python >= 3.9, `nvidia-smi`, and `tmux` on the server. No other
dependencies -- stdlib only.

```bash
./scripts/install.sh
```

This does `pip install --user -e .` and prints the remaining steps. In short:

1. **Pick a shared directory** every user on the box can read and write
   (a group-writable spot on a shared mount is ideal):
   ```bash
   mkdir -p /path/to/shared/corral_home
   chmod 2775 /path/to/shared/corral_home   # setgid, so new files inherit the group
   ```
2. **Have every user set `CORRAL_HOME`** to that path -- e.g. a shared
   `/etc/profile.d/corral.sh`, or a line added to each person's shell rc:
   ```bash
   export CORRAL_HOME=/path/to/shared/corral_home
   ```
3. **Start the daemon once.** It coordinates scheduling for everyone; nobody
   else needs to run it.
   ```bash
   tmux new-session -d -s corral-daemon 'corral daemon'
   ```
   Or use the systemd unit template at `scripts/corral-daemon.service` if you'd
   rather it survive a reboot without a login session at all.
4. **Everyone can now use `corral submit` / `corral queue` / `corral gpus`.**
   No per-user setup beyond step 2 and `pip install --user -e .` (or a
   shared venv on the PATH) for the CLI itself.

## Usage

```bash
# See what's on the box right now, and who (if anyone, via corral) is using it.
corral gpus

# Submit a job. Everything after `--` is run verbatim as your job's command,
# in your current working directory, with CUDA_VISIBLE_DEVICES already set.
corral submit --gpus 1 -- python train.py --epochs 10
corral submit --gpus 2 --name my-70b-run -- bash run_70b.sh

# See the queue: what's running, what's waiting (in FIFO order), what recently finished.
corral queue

# Follow a job's live output (same as `tail -f` on its log file).
corral log 20260819-131055-ab12cd -f

# Full detail on one job, including the tail of its log.
corral status 20260819-131055-ab12cd

# Cancel a job that hasn't started yet, or Ctrl-C one that's running.
corral cancel 20260819-131055-ab12cd
```

Every job also gets its own tmux window in the shared `corral` session, so you
can always `tmux attach -t corral` and watch (or debug) it directly, exactly
as if you'd launched it by hand.

By default (see `CORRAL_LOG_DIR` below), job logs land in a `.corral/`
folder inside whatever directory you ran `corral submit` from -- add
`.corral/` to that project's `.gitignore` so job logs don't end up
accidentally committed.

## Configuration

All configuration is environment variables, read at CLI/daemon startup --
nothing to edit in the source.

| Variable | Default | Meaning |
|---|---|---|
| `CORRAL_HOME` | `~/.corral` | Root of the spool directory. **Must be a shared path for a multi-user install.** |
| `CORRAL_LOG_DIR` | *(unset)* | Where job `.log`/`.exitcode` files are written. If unset, each job's logs default to a `.corral/logs/` folder inside the directory it was submitted from. Set this to force every job's logs into one shared location instead. Read by the daemon, not the submitting client -- see [Design notes](#design-notes). |
| `CORRAL_TMUX_SESSION` | `corral` | tmux session jobs are launched into. |
| `CORRAL_POLL_INTERVAL` | `10` | Seconds between daemon scheduling passes. |
| `CORRAL_FREE_STREAK` | `2` | Consecutive free polls required before a GPU is scheduled onto. |
| `CORRAL_FREE_MEM_THRESHOLD_MIB` | `512` | Memory (MiB) below which a GPU counts as "free". |
| `CORRAL_CANCELLED_RETENTION_SEC` | `600` | How long a cancelled job's record is kept (with state `CANCELLED`) in `corral queue`/`status` before the daemon sweeps it away. Read by the daemon, same caveat as `CORRAL_LOG_DIR` above. |
| `CORRAL_BACKFILL_MAX_SKIPS` | `5` | Consecutive scheduling polls a blocked head-of-queue job may be jumped by smaller, later jobs before all newly-free GPUs are reserved for it instead. `0` disables backfill entirely (plain strict FIFO). Read by the daemon. |

## Design notes

**Jobs run as the OS user who started the daemon, not the user who submitted
them.** This avoids needing root/sudo or a sudoers policy: a true
per-submitter identity would need privilege escalation the daemon doesn't
have. Fine for a small trusted team sharing one server, but worth knowing
before relying on file permissions to separate users' outputs. For real
multi-tenant isolation, extend this with a `sudo -u <user> -- <cmd>` wrapper
and a narrow, audited sudoers rule.

**A consequence of the above:** the daemon's OS account needs write access
into wherever people run `corral submit` from -- by default that's each
job's `.corral/logs/` folder, created on demand. If a user's working
directory isn't writable by the daemon's account, that user's job launches
fail cleanly (marked `FAILED`, other users unaffected) until this is fixed,
two ways:

1. **Filesystem ACLs (recommended).** Grant the daemon's account a *default*
   ACL scoped to wherever each user keeps their GPU-job projects -- not
   their whole home directory:
   ```bash
   setfacl -R -d -m u:<daemon-account>:rwx ~alice/gpu-projects
   setfacl -R    -m u:<daemon-account>:rwx ~alice/gpu-projects  # covers what's already there
   ```
   The default ACL (`-d`) means any new project directory created under
   `gpu-projects/` later inherits the grant automatically -- set once per
   user, no daemon execution privilege involved, just a scoped write grant.
2. **`CORRAL_LOG_DIR`** pointed at one shared, already-writable path instead,
   if you'd rather skip ACLs -- simpler, at the cost of logs living in one
   shared folder rather than next to each user's project.

Test with a real second OS user before rolling out either way.

**FIFO with bounded backfill.** See "How it works" above for the mechanism.
One flip side worth knowing: once the reservation kicks in, a big job that's
just slow to satisfy (not impossible, just larger than what's typically
free) blocks smaller jobs behind it the same way strict FIFO always did.

**No accounting, no priorities, no per-user quotas.** Anyone can submit as
many jobs, requesting as many GPUs, as they want. This is appropriate for a
small trusted group; it is not a substitute for Slurm on a large multi-team
cluster.

## Testing

Stdlib `unittest` only -- no test-time dependencies, consistent with the rest
of the project. Covers job persistence, GPU-output parsing, the tmux command
construction (including the Ctrl-C/`tee` interaction), and the scheduler's
backfill decision logic (`daemon._plan_schedule`) directly and exhaustively,
without needing real GPUs or a real tmux server:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## Repo layout

```
corral/
  corral/            the package
    gpu.py             GPU detection via nvidia-smi
    jobstore.py        job persistence (JSON files, atomic writes)
    models.py          the Job dataclass
    tmux_exec.py        launches a job's command in a tmux window
    daemon.py          the scheduler loop (FIFO + bounded backfill)
    cli.py             the `corral` command
  tests/             unit tests (stdlib unittest, no real GPU/tmux needed)
  scripts/
    install.sh          admin install helper
    corral-daemon.service   optional systemd unit template
```
