import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from corral import config, jobstore
from corral.daemon import _grant_assignments, _sweep_cancelled
from corral.models import Job


class TestGrantAssignments(unittest.TestCase):
    """The daemon only ever writes JSON into its own shared spool -- it never
    touches tmux or a user's own files, so granting a job GPUs has no failure
    mode tied to that job's owner (see corral/launcher.py for the code that
    actually launches jobs, and can fail per-user)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.pending_dir, self.granted_dir = base / "pending", base / "granted"
        self._patches = [
            patch.object(config, "PENDING_DIR", self.pending_dir),
            patch.object(config, "GRANTED_DIR", self.granted_dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_granted_job_moves_to_granted_dir_with_gpu_ids_and_state(self):
        job = Job(id="j1", name="j1", user="alice", n_gpus=1, cmd=["echo"], cwd="/tmp",
                   submitted_at="2026-01-01T00:00:00")
        jobstore.write_job(job, self.pending_dir)

        _grant_assignments([(job, [0, 1])])

        self.assertEqual(job.state, "GRANTED")
        self.assertEqual(job.gpu_ids, [0, 1])
        self.assertFalse((self.pending_dir / "j1.json").exists())
        granted = jobstore.read_job(self.granted_dir / "j1.json")
        self.assertEqual(granted.state, "GRANTED")
        self.assertEqual(granted.gpu_ids, [0, 1])


class TestSweepCancelled(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.done_dir = Path(self._tmp.name) / "done"
        self._patches = [
            patch.object(config, "DONE_DIR", self.done_dir),
            patch.object(config, "CANCELLED_RETENTION_SEC", 300),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _write_done(self, job_id, state, finished_at):
        jobstore.write_job(
            Job(id=job_id, name=job_id, user="alice", n_gpus=1, cmd=["echo"], cwd="/tmp",
                submitted_at="2026-01-01T00:00:00", state=state, finished_at=finished_at),
            self.done_dir,
        )

    def test_old_cancelled_record_is_swept(self):
        old_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 3600))
        self._write_done("j1", "CANCELLED", old_ts)
        _sweep_cancelled()
        self.assertFalse((self.done_dir / "j1.json").exists())

    def test_recent_cancelled_record_is_kept(self):
        recent_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_done("j1", "CANCELLED", recent_ts)
        _sweep_cancelled()
        self.assertTrue((self.done_dir / "j1.json").exists())

    def test_completed_record_is_never_swept_regardless_of_age(self):
        ancient_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 999999))
        self._write_done("j1", "COMPLETED", ancient_ts)
        _sweep_cancelled()
        self.assertTrue((self.done_dir / "j1.json").exists())


if __name__ == "__main__":
    unittest.main()
