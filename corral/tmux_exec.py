"""Launches a job's command inside a dedicated tmux window so it survives the
launcher restarting, the submitting user logging out, or an SSH disconnect.
Every call takes an explicit `session`: each user's jobs run in their own
tmux session, under their own OS account (see `corral/launcher.py`).
"""
from __future__ import annotations

import shlex
import subprocess


def ensure_session(session: str) -> None:
    result = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if result.returncode != 0:
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", "_idle"], check=True)


def launch(
    job_id: str, gpu_ids: list[int], cmd: list[str], cwd: str, log_path: str, exitcode_path: str, session: str
) -> None:
    ensure_session(session)
    cuda_visible = ",".join(str(g) for g in gpu_ids)
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    # PIPESTATUS[0] is the job's real exit code, not tee's. `tee` ignores
    # SIGINT (the trap survives the exec) so `corral cancel`'s Ctrl-C -- which
    # hits every process in the pipeline at once -- doesn't kill tee before a
    # job's own `trap ... INT` cleanup finishes writing to it.
    inner = (
        f"cd {shlex.quote(cwd)} && "
        f"export CUDA_VISIBLE_DEVICES={cuda_visible} && "
        f"({cmd_str}) 2>&1 | (trap '' INT; exec tee {shlex.quote(log_path)}) ; "
        f"echo ${{PIPESTATUS[0]}} > {shlex.quote(exitcode_path)}"
    )
    window_name = job_id[:30]
    subprocess.run(
        ["tmux", "new-window", "-t", session, "-n", window_name, "bash", "-lc", inner],
        check=True,
    )


def interrupt(job_id: str, session: str) -> None:
    window_name = job_id[:30]
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{session}:{window_name}", "C-c"],
        check=False,
    )
