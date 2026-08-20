"""GPU detection via nvidia-smi. No pynvml/torch dependency on purpose -- this
keeps `corral` installable and usable with nothing but the stdlib and whatever
NVIDIA driver tools are already on the machine.
"""
from __future__ import annotations

import dataclasses
import subprocess


@dataclasses.dataclass
class GpuInfo:
    index: int
    name: str
    mem_total_mib: int
    mem_used_mib: int


def _run_nvidia_smi(query: str) -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found -- is this a GPU server with NVIDIA drivers installed?")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out after 15s")
    return [line for line in out.stdout.strip().splitlines() if line.strip()]


def detect_gpus() -> list[GpuInfo]:
    """Auto-detects every GPU visible to nvidia-smi on this machine. Deliberately
    does NOT look at CUDA_VISIBLE_DEVICES -- the daemon needs the true physical
    GPU count and indices to schedule against, since it sets CUDA_VISIBLE_DEVICES
    itself for each job it launches.
    """
    lines = _run_nvidia_smi("index,name,memory.total,memory.used")
    gpus = []
    for line in lines:
        idx_s, name, total_s, used_s = [p.strip() for p in line.split(",")]
        gpus.append(GpuInfo(index=int(idx_s), name=name, mem_total_mib=int(total_s), mem_used_mib=int(used_s)))
    return sorted(gpus, key=lambda g: g.index)


def gpu_count() -> int:
    return len(detect_gpus())
