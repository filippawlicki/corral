import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from corral import config, jobstore
from corral.daemon import _launch_assignments, _reap_running, _sweep_cancelled
from corral.models import Job


def make_running_job(job_id, log_path):
    return Job(
        id=job_id, name=job_id, user="alice", n_gpus=1, cmd=["sleep", "1"], cwd="/tmp",
        submitted_at="2026-01-01T00:00:00", started_at="2026-01-01T00:00:00",
        state="RUNNING", gpu_ids=[0], log_path=log_path,
    )


class TestReapRunning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.running_dir, self.done_dir, self.log_dir = base / "running", base / "done", base / "logs"
        self.log_dir.mkdir(parents=True)
        self._patches = [
            patch.object(config, "RUNNING_DIR", self.running_dir),
            patch.object(config, "DONE_DIR", self.done_dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_not_reaped_until_exitcode_file_appears(self):
        job = make_running_job("j1", str(self.log_dir / "j1.log"))
        running = {"j1": job}
        _reap_running(running)
        self.assertIn("j1", running)
        self.assertFalse((self.done_dir / "j1.json").exists())

    def test_reaped_as_completed_on_zero_exit(self):
        (self.log_dir / "j1.exitcode").write_text("0")
        job = make_running_job("j1", str(self.log_dir / "j1.log"))
        running = {"j1": job}
        _reap_running(running)
        self.assertNotIn("j1", running)
        self.assertEqual(job.state, "COMPLETED")
        self.assertTrue((self.done_dir / "j1.json").exists())

    def test_reaped_as_failed_on_nonzero_exit(self):
        (self.log_dir / "j1.exitcode").write_text("1")
        job = make_running_job("j1", str(self.log_dir / "j1.log"))
        running = {"j1": job}
        _reap_running(running)
        self.assertEqual(job.state, "FAILED")
        self.assertEqual(job.exit_code, 1)

    def test_cancel_sentinel_forces_cancelled_state_regardless_of_exit_code(self):
        (self.log_dir / "j1.exitcode").write_text("141")  # e.g. SIGPIPE from an interrupted job
        (self.log_dir / "j1.cancelled").touch()
        job = make_running_job("j1", str(self.log_dir / "j1.log"))
        running = {"j1": job}
        _reap_running(running)
        self.assertEqual(job.state, "CANCELLED")
        self.assertFalse((self.log_dir / "j1.cancelled").exists())  # sentinel cleaned up


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


class TestLaunchAssignmentsResilience(unittest.TestCase):
    """A single job's launch failure (e.g. the daemon's OS user can't write into
    that job's own submission directory) must not crash the daemon or block other
    jobs in the same batch from launching -- see README's admin setup notes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.pending_dir, self.done_dir = base / "pending", base / "done"
        self._patches = [
            patch.object(config, "PENDING_DIR", self.pending_dir),
            patch.object(config, "DONE_DIR", self.done_dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_failed_launch_marks_job_failed_without_raising(self):
        job = Job(id="j1", name="j1", user="alice", n_gpus=1, cmd=["echo"], cwd="/no/such/dir",
                   submitted_at="2026-01-01T00:00:00")
        running = {}
        with patch("corral.daemon._launch", side_effect=PermissionError("denied")):
            _launch_assignments([(job, [0])], running)  # must not raise
        self.assertNotIn("j1", running)
        self.assertEqual(job.state, "FAILED")
        self.assertTrue((self.done_dir / "j1.json").exists())

    def test_one_failure_does_not_block_other_jobs_in_the_same_batch(self):
        bad = Job(id="bad", name="bad", user="alice", n_gpus=1, cmd=["echo"], cwd="/tmp",
                   submitted_at="2026-01-01T00:00:00")
        good = Job(id="good", name="good", user="alice", n_gpus=1, cmd=["echo"], cwd="/tmp",
                    submitted_at="2026-01-01T00:00:00")
        running = {}

        def fake_launch(job, gpu_ids):
            if job.id == "bad":
                raise PermissionError("denied")
            job.state = "RUNNING"

        with patch("corral.daemon._launch", side_effect=fake_launch):
            _launch_assignments([(bad, [0]), (good, [1])], running)

        self.assertNotIn("bad", running)
        self.assertIn("good", running)
        self.assertEqual(bad.state, "FAILED")


if __name__ == "__main__":
    unittest.main()
