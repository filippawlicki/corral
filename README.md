# 🪸 Corral

A simple FIFO GPU job queue for a shared research server. No database, no
per-node daemon, no config DSL -- one shared spool of JSON files, plus tmux.

```
corral gpus                    # what's on the box, and who's using it?
corral submit --gpus 2 -- python train.py
corral queue                   # what's running / waiting / done
corral log <job-id> -f         # tail a job's output
corral cancel <job-id>         # cancel or interrupt a job
```

Run `corral --help`, or `corral <command> --help`, any time for the full
command reference -- this README won't try to duplicate it.

## Why

Everyone on a shared GPU box ends up hand-rolling the same thing: check
`nvidia-smi`, guess whether a GPU is really free, launch in tmux so it
survives a disconnect, hope nobody else grabs the same GPU. Corral is that
pattern, written once, shared by everyone on the machine -- **don't
double-book a GPU, run jobs in a sane order.** It's not trying to be Slurm:
no accounting, no cgroups, no multi-node support, just one shared machine.

## Getting started (users)

Ask whoever administers corral for the shared `CORRAL_HOME` path. Add to
your `.bashrc`/`.zshrc`:

```bash
export CORRAL_HOME=/path/to/shared/corral_home

# If there's a shared install, point at it instead of installing your own copy:
corral() { /path/to/shared/corral_venv/bin/python -m corral.cli "$@"; }
```

(No shared install? `pip install --user -e /path/to/corral` gives you your
own `corral` command instead -- skip the function above.)

Reload your shell, then:

```bash
corral submit --gpus 1 -- python train.py
corral queue
```

That's it -- your personal job launcher starts itself the first time you
submit, nothing else to run by hand. A few things worth knowing before you
rely on it:

- Logs default to `.corral/logs/` next to wherever you ran `corral submit`
  from -- add `.corral/` to that project's `.gitignore`.
- `corral cancel <id>` cancels a pending job or Ctrl-C's a running one; you
  can only cancel your own jobs.
- `corral submit --urgent` jumps your job ahead of normal-priority jobs
  already queued -- for something like a deadline, not routine use.
- Every job also runs in its own window in your personal `corral-<you>`
  tmux session -- `tmux attach -t corral-<you>` to watch one live.

## How it works

- **GPU detection** is automatic (`nvidia-smi`) -- nothing to configure.
- **Jobs are JSON files** in a shared spool; a job's state is simply which
  subdirectory it's in (`pending -> granted -> running -> done`), so anyone
  can inspect the whole queue with `ls`/`cat`.
- **Scheduling is FIFO with bounded backfill**: a smaller job can jump into
  GPUs a blocked head-of-queue job doesn't need, but only for a bounded
  number of scheduling passes, so the head can never be starved forever.
- **Cancelling leaves a short-lived `CANCELLED` record** instead of the job
  just vanishing.
- **A shared daemon decides *what* runs; your own personal launcher runs
  *your* jobs, under your own account.** That split means nobody else's
  account ever ends up owning your logs or checkpoints. See `CLAUDE.md` if
  you want the full mechanism.

## Admin setup

Requires Python >= 3.9, `nvidia-smi`, `tmux` >= 3.2, on the server.

```bash
./scripts/install.sh
```

1. **Shared spool directory**, writable by everyone:
   ```bash
   mkdir -p /path/to/shared/corral_home
   chmod 2775 /path/to/shared/corral_home   # setgid, so new files inherit the group
   ```
2. **Shared install** (recommended -- users need nothing installed locally):
   ```bash
   python3 -m venv /path/to/shared/corral_venv
   /path/to/shared/corral_venv/bin/pip install -e /path/to/this/repo
   ```
   Users point at it with the shell function in [Getting started](#getting-started-users).
3. **Start the daemon once** -- it coordinates scheduling for everyone.
   tmux does **not** forward your shell's exported variables into a new
   session, so pass `CORRAL_HOME` explicitly with `-e`:
   ```bash
   tmux new-session -d -s corral-daemon -e CORRAL_HOME=/path/to/shared/corral_home 'corral daemon'
   ```
   Or use `scripts/corral-daemon.service` (systemd) instead -- its
   `Environment=` line sets this directly and it survives a reboot.
4. **Tell users the `CORRAL_HOME` path** (and venv path, if you did step 2).
   That's the entire per-user setup.

## Configuration

Environment variables, read once at process startup -- nothing to edit in
the source. See `corral --help` for CLI flags (`--urgent`, `--gpus`, etc.).

| Variable | Default | Meaning |
|---|---|---|
| `CORRAL_HOME` | `~/.corral` | Root of the spool directory. **Must be a shared path for a multi-user install.** |
| `CORRAL_LOG_DIR` | *(unset)* | Force every job's logs into one shared location instead of the per-job default. |
| `CORRAL_TMUX_SESSION` | `corral` | Base name for per-user launcher tmux sessions (`<name>-<user>`). |
| `CORRAL_POLL_INTERVAL` | `10` | Seconds between daemon scheduling passes. |
| `CORRAL_FREE_STREAK` | `2` | Consecutive free polls required before a GPU is scheduled onto. |
| `CORRAL_FREE_MEM_THRESHOLD_MIB` | `512` | Memory (MiB) below which a GPU counts as "free". |
| `CORRAL_CANCELLED_RETENTION_SEC` | `600` | How long a `CANCELLED` record is kept before being swept. |
| `CORRAL_BACKFILL_MAX_SKIPS` | `5` | Polls a blocked head job may be skipped before GPUs are reserved for it. `0` = strict FIFO. |

GPU reservation (`corral reserve --gpus N`) is deliberately *not* an env
var -- it's a small file in the spool the daemon re-reads every poll, so an
admin can change it live without restarting anything.

## Notes

- **Cancelling someone else's running job** doesn't work from your account
  (each user's jobs run in their own tmux session) -- ask them, or use
  normal OS tools if you have the access.
- **No accounting or enforcement.** Anyone can submit as much as they want,
  and `--urgent` isn't gated -- it's a team norm, not a permission system.
- The full architecture, and the non-obvious bits worth knowing before
  changing the code, are in `CLAUDE.md`.

## Testing

Stdlib `unittest` only, no dependencies:

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
    daemon.py          the scheduler -- grants GPUs, never launches
    launcher.py        per-user launcher -- launches/reaps that user's own granted jobs
    cli.py             the `corral` command
  tests/             unit tests (stdlib unittest, no real GPU/tmux needed)
  scripts/
    install.sh          admin install helper
    corral-daemon.service   optional systemd unit template
```
