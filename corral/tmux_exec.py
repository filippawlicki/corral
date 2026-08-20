"""Launches a job's command inside a dedicated tmux window so it survives the
daemon restarting, the submitting user logging out, or an SSH disconnect --
the same pattern used for every long-running GPU job in this project.
"""
from __future__ import annotations

import shlex
import subprocess

from . import config


def ensure_session() -> None:
    result = subprocess.run(["tmux", "has-session", "-t", config.TMUX_SESSION], capture_output=True)
    if result.returncode != 0:
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", config.TMUX_SESSION, "-n", "_idle"], check=True
        )


def launch(job_id: str, gpu_ids: list[int], cmd: list[str], cwd: str, log_path: str, exitcode_path: str) -> None:
    ensure_session()
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
        ["tmux", "new-window", "-t", config.TMUX_SESSION, "-n", window_name, "bash", "-lc", inner],
        check=True,
    )


def interrupt(job_id: str) -> None:
    window_name = job_id[:30]
    subprocess.run(
        ["tmux", "send-keys", "-t", f"{config.TMUX_SESSION}:{window_name}", "C-c"],
        check=False,
    )
