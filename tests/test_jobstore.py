import tempfile
import unittest
from pathlib import Path

from corral import jobstore
from corral.models import Job


def make_job(job_id, name="job"):
    return Job(
        id=job_id, name=name, user="alice", n_gpus=1, cmd=["echo", "hi"], cwd="/tmp",
        submitted_at="2026-01-01T00:00:00",
    )


class TestJobstore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_read_round_trip(self):
        job = make_job("20260101-000000-aaaaaa")
        jobstore.write_job(job, self.dir)
        path = self.dir / f"{job.id}.json"
        self.assertTrue(path.exists())
        self.assertEqual(jobstore.read_job(path), job)

    def test_write_leaves_no_tmp_file_behind(self):
        jobstore.write_job(make_job("20260101-000000-bbbbbb"), self.dir)
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_list_jobs_is_fifo_sorted_by_id(self):
        ids = ["20260101-000002-c", "20260101-000001-b", "20260101-000000-a"]
        for jid in ids:
            jobstore.write_job(make_job(jid), self.dir)
        jobs = jobstore.list_jobs(self.dir)
        self.assertEqual([j.id for j in jobs], sorted(ids))

    def test_list_jobs_skips_corrupt_file_instead_of_raising(self):
        jobstore.write_job(make_job("20260101-000000-good"), self.dir)
        (self.dir / "20260101-000001-bad.json").write_text("{not valid json")
        jobs = jobstore.list_jobs(self.dir)
        self.assertEqual([j.id for j in jobs], ["20260101-000000-good"])

    def test_list_jobs_missing_dir_returns_empty(self):
        self.assertEqual(jobstore.list_jobs(self.dir / "does-not-exist"), [])

    def test_move_job_between_directories(self):
        job = make_job("20260101-000000-move")
        src, dst = self.dir / "src", self.dir / "dst"
        jobstore.write_job(job, src)
        jobstore.move_job(job, src, dst)
        self.assertFalse((src / f"{job.id}.json").exists())
        self.assertTrue((dst / f"{job.id}.json").exists())

    def test_move_job_tolerates_missing_source(self):
        # e.g. concurrent cancel racing a move -- move_job must not raise.
        job = make_job("20260101-000000-nosrc")
        src, dst = self.dir / "src", self.dir / "dst"
        jobstore.move_job(job, src, dst)
        self.assertTrue((dst / f"{job.id}.json").exists())


if __name__ == "__main__":
    unittest.main()
