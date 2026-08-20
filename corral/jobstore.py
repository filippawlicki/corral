"""Job persistence. Each job is one JSON file; its STATE is simply which
directory it currently lives in (pending/ -> running/ -> done/). Moving a job
between states is a write-then-unlink, which is as close to atomic as a
plain-files design gets and is trivial for an admin to inspect by hand
(`ls`, `cat`) without any database or extra tooling.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from .models import Job


def new_job_id() -> str:
    # Timestamp-prefixed so that sorting filenames lexicographically gives FIFO order.
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)  # atomic rename on the same filesystem


def write_job(job: Job, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / f"{job.id}.json", job.to_dict())


def read_job(path: Path) -> Job:
    return Job.from_dict(json.loads(path.read_text()))


def list_jobs(directory: Path) -> list[Job]:
    if not directory.exists():
        return []
    jobs = []
    for p in sorted(directory.glob("*.json")):
        try:
            jobs.append(read_job(p))
        except (json.JSONDecodeError, OSError):
            continue  # a job file mid-write from a concurrent submit -- picked up next poll
    return jobs


def move_job(job: Job, src_dir: Path, dst_dir: Path) -> None:
    write_job(job, dst_dir)
    src = src_dir / f"{job.id}.json"
    if src.exists():
        src.unlink()
