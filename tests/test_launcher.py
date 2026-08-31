import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corral import config, jobstore
from corral.launcher import _launch_granted, _reap_running
from corral.models import Job


class TestLaunchGranted(unittest.TestCase):
    """Runs as the job's owner, so a launch failure here is a real error
    (bad command, etc.), not a cross-user permission problem -- but it must
    still fail just the one job, not raise, so it can't wedge the launcher."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.granted_dir, self.running_dir, self.done_dir = base / "granted", base / "running", base / "done"
        self.log_dir = base / "logs"
        self._patches = [
            patch.object(config, "GRANTED_DIR", self.granted_dir),
            patch.object(config, "RUNNING_DIR", self.running_dir),
            patch.object(config, "DONE_DIR", self.done_dir),
            patch("corral.launcher.config.log_dir_for", return_value=self.log_dir),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def _write_granted(self, job_id, user="alice", gpu_ids=None):
        job = Job(id=job_id, name=job_id, user=user, n_gpus=1, cmd=["echo", "hi"], cwd="/tmp",
                   submitted_at="2026-01-01T00:00:00", state="GRANTED", gpu_ids=gpu_ids or [0])
        jobstore.write_job(job, self.granted_dir)
        return job

    def test_only_launches_jobs_for_the_given_user(self):
        self._write_granted("mine", user="alice")
        self._write_granted("not_mine", user="bob")

        with patch("corral.launcher.tmux_exec.launch") as mock_launch:
            _launch_granted("alice", "corral-alice")

        mock_launch.assert_called_once()
        self.assertFalse((self.granted_dir / "mine.json").exists())
        self.assertTrue((self.running_dir / "mine.json").exists())
        self.assertTrue((self.granted_dir / "not_mine.json").exists())

    def test_launch_failure_marks_job_failed_without_raising(self):
        self._write_granted("bad", user="alice")

        with patch("corral.launcher.tmux_exec.launch", side_effect=RuntimeError("tmux exploded")):
            _launch_granted("alice", "corral-alice")  # must not raise

        self.assertFalse((self.granted_dir / "bad.json").exists())
        done = jobstore.read_job(self.done_dir / "bad.json")
        self.assertEqual(done.state, "FAILED")

    def test_one_failure_does_not_block_other_jobs_in_the_same_batch(self):
        self._write_granted("bad", user="alice")
        self._write_granted("good", user="alice")

        def fake_launch(job_id, gpu_ids, cmd, cwd, log_path, exitcode_path, session):
            if job_id == "bad":
                raise RuntimeError("boom")

        with patch("corral.launcher.tmux_exec.launch", side_effect=fake_launch):
            _launch_granted("alice", "corral-alice")

        self.assertEqual(jobstore.read_job(self.done_dir / "bad.json").state, "FAILED")
        self.assertTrue((self.running_dir / "good.json").exists())


def make_running_job(job_id, log_path, user="alice"):
    return Job(
        id=job_id, name=job_id, user=user, n_gpus=1, cmd=["sleep", "1"], cwd="/tmp",
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
        jobstore.write_job(job, self.running_dir)
        _reap_running("alice")
        self.assertTrue((self.running_dir / "j1.json").exists())
        self.assertFalse((self.done_dir / "j1.json").exists())

    def test_reaped_as_completed_on_zero_exit(self):
        (self.log_dir / "j1.exitcode").write_text("0")
        jobstore.write_job(make_running_job("j1", str(self.log_dir / "j1.log")), self.running_dir)
        _reap_running("alice")
        self.assertFalse((self.running_dir / "j1.json").exists())
        done = jobstore.read_job(self.done_dir / "j1.json")
        self.assertEqual(done.state, "COMPLETED")

    def test_reaped_as_failed_on_nonzero_exit(self):
        (self.log_dir / "j1.exitcode").write_text("1")
        jobstore.write_job(make_running_job("j1", str(self.log_dir / "j1.log")), self.running_dir)
        _reap_running("alice")
        done = jobstore.read_job(self.done_dir / "j1.json")
        self.assertEqual(done.state, "FAILED")
        self.assertEqual(done.exit_code, 1)

    def test_cancel_sentinel_forces_cancelled_state_regardless_of_exit_code(self):
        (self.log_dir / "j1.exitcode").write_text("141")  # e.g. SIGPIPE from an interrupted job
        (self.log_dir / "j1.cancelled").touch()
        jobstore.write_job(make_running_job("j1", str(self.log_dir / "j1.log")), self.running_dir)
        _reap_running("alice")
        done = jobstore.read_job(self.done_dir / "j1.json")
        self.assertEqual(done.state, "CANCELLED")
        self.assertFalse((self.log_dir / "j1.cancelled").exists())  # sentinel cleaned up

    def test_only_reaps_jobs_for_the_given_user(self):
        (self.log_dir / "mine.exitcode").write_text("0")
        (self.log_dir / "not_mine.exitcode").write_text("0")
        jobstore.write_job(make_running_job("mine", str(self.log_dir / "mine.log"), user="alice"), self.running_dir)
        jobstore.write_job(make_running_job("not_mine", str(self.log_dir / "not_mine.log"), user="bob"), self.running_dir)

        _reap_running("alice")

        self.assertFalse((self.running_dir / "mine.json").exists())
        self.assertTrue((self.running_dir / "not_mine.json").exists())


if __name__ == "__main__":
    unittest.main()
