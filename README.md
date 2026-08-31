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
- **Scheduling and execution are two separate processes**, so the daemon
  never has to touch another user's files. One shared `corral daemon`
  (admin-run) polls `nvidia-smi`, works out which GPUs are genuinely free,
  and grants them to the oldest pending job that fits -- but it only ever
  writes JSON into its own spool, never launches anything. Each user also
  has their own `corral launcher`, auto-started by their first `corral
  submit` under their own OS account, which watches for jobs granted to
  them and does the actual launching -- into their own tmux window, with
  `CUDA_VISIBLE_DEVICES` set to exactly the GPUs it was granted. Because
  the job process is genuinely that user's own process, every file it
  creates -- logs, checkpoints, anything -- is owned by them, not by
  whoever happens to run the daemon. See [Design notes](#design-notes) for
  why this is split this way. Jobs survive the submitting user logging
  out, an SSH disconnect, or either process restarting.
- **Scheduling is FIFO with bounded backfill.** The oldest pending job (the
  "head") gets first priority, but if it can't run yet, a later, smaller job
  is allowed to run in its place using GPUs the head doesn't need -- for up
  to `CORRAL_BACKFILL_MAX_SKIPS` (default 5) consecutive scheduling passes.
  Past that, all newly-freed GPUs are reserved for the head alone until it
  can launch, so it's never starved indefinitely by a steady stream of
  smaller jobs behind it. Set `CORRAL_BACKFILL_MAX_SKIPS=0` for plain strict
  FIFO instead -- the head always blocks everything immediately, if you'd
  rather have that guarantee than the extra utilization.
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
   else needs to run it. tmux does **not** forward your shell's exported
   variables into a new session by default, so pass `CORRAL_HOME` explicitly
   with `-e` (requires tmux >= 3.2):
   ```bash
   tmux new-session -d -s corral-daemon -e CORRAL_HOME=/path/to/shared/corral_home 'corral daemon'
   ```
   Or use the systemd unit template at `scripts/corral-daemon.service` instead --
   its `Environment=` line sets this directly, sidesteps the tmux `-e` caveat
   entirely, and survives a reboot without a login session at all.
4. **Everyone can now use `corral submit` / `corral queue` / `corral gpus`.**
   No per-user setup beyond step 2 and `pip install --user -e .` (or a
   shared venv on the PATH) for the CLI itself. The first `corral submit`
   a user runs auto-starts their own `corral launcher` (see
   [Design notes](#design-notes)) -- nothing else to start by hand.

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

Every job also gets its own window in your personal `corral-<you>` tmux
session, so you can always `tmux attach -t corral-<you>` and watch (or debug)
it directly, exactly as if you'd launched it by hand. (Only your own jobs are
in there -- everyone's launcher, and tmux session, is separate.)

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
| `CORRAL_LOG_DIR` | *(unset)* | Where job `.log`/`.exitcode` files are written. If unset, each job's logs default to a `.corral/logs/` folder inside the directory it was submitted from. Set this to force every job's logs into one shared location instead. Read by each user's own launcher, not the central daemon -- see [Design notes](#design-notes). |
| `CORRAL_TMUX_SESSION` | `corral` | Base name for per-user launcher tmux sessions -- each user's jobs run in their own `<name>-<user>`. |
| `CORRAL_POLL_INTERVAL` | `10` | Seconds between daemon scheduling passes. |
| `CORRAL_FREE_STREAK` | `2` | Consecutive free polls required before a GPU is scheduled onto. |
| `CORRAL_FREE_MEM_THRESHOLD_MIB` | `512` | Memory (MiB) below which a GPU counts as "free". |
| `CORRAL_CANCELLED_RETENTION_SEC` | `600` | How long a cancelled job's record is kept (with state `CANCELLED`) in `corral queue`/`status` before the daemon sweeps it away. Read by the daemon, same caveat as `CORRAL_LOG_DIR` above. |
| `CORRAL_BACKFILL_MAX_SKIPS` | `5` | Consecutive scheduling polls a blocked head-of-queue job may be jumped by smaller, later jobs before all newly-free GPUs are reserved for it instead. `0` disables backfill entirely (plain strict FIFO). Read by the daemon. |

## Design notes

**Jobs run as the user who submitted them, not as the daemon's OS account.**
If jobs ran as whichever account started the daemon, that account would need
write access into every user's own project directories for logs, and -- more
fundamentally -- would end up owning every checkpoint and output file each
job creates, since a job can write anywhere. No single directory grant fixes
that. So the central `corral daemon` never launches anything: it only decides
which GPUs a pending job gets and writes that decision into the shared spool
(state `GRANTED`). Each user's own `corral launcher` -- running under their
own account, auto-started by their first `corral submit` into a
`corral-<user>` tmux session -- watches for jobs granted to them and does the
actual launching. The job process is then genuinely that user's own process,
so every file it creates is correctly owned, with zero ACLs, sudoers rules,
or admin per-user setup required. Read `corral/launcher.py` if you want the
full mechanism; it's under 100 lines.

**A consequence:** `corral cancel` on a running job sends Ctrl-C into that
job's own `corral-<owner>` tmux session. If you're cancelling your own job,
that's just your own tmux socket. Cancelling *someone else's* running job
from your account won't work (you don't have access to their tmux server) --
ask them, or use normal OS tools (`ps`/`kill`) if you have the access for it.
That's a deliberate trade: real isolation between users' jobs, at the cost of
any user being able to attach to and kill any other job.

**tmux does not forward your shell's exported environment into a new
session** (only `-e`, added in tmux 3.2, does) -- `corral submit` accounts
for this when auto-starting your launcher, forwarding every `CORRAL_*`
variable explicitly. If you ever start `corral daemon` or `corral launcher`
by hand instead of letting it auto-start, remember the same thing applies:
either pass `-e CORRAL_HOME=...` yourself, or use the systemd unit, whose
`Environment=` line doesn't have this problem.

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
construction (including the Ctrl-C/`tee` interaction), the scheduler's
backfill decision logic (`daemon._plan_schedule`) directly and exhaustively,
and the launcher's launch/reap logic, all without needing real GPUs or a
real tmux server:

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
    daemon.py          the scheduler (FIFO + bounded backfill) -- grants GPUs, never launches
    launcher.py        per-user launcher -- launches/reaps that user's own granted jobs
    cli.py             the `corral` command
  tests/             unit tests (stdlib unittest, no real GPU/tmux needed)
  scripts/
    install.sh          admin install helper
    corral-daemon.service   optional systemd unit template
```
