from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class Job:
    id: str
    name: str
    user: str
    n_gpus: int
    cmd: list[str]
    cwd: str
    state: str = "PENDING"  # PENDING, GRANTED, RUNNING, COMPLETED, FAILED, CANCELLED
    priority: str = "normal"  # normal, urgent
    submitted_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    gpu_ids: Optional[list[int]] = None
    exit_code: Optional[int] = None
    log_path: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(**d)
