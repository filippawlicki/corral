import unittest

from corral.models import Job


class TestJobSerialization(unittest.TestCase):
    def test_round_trip(self):
        job = Job(
            id="20260101-000000-abcdef",
            name="train",
            user="alice",
            n_gpus=2,
            cmd=["python", "train.py"],
            cwd="/home/alice/proj",
            submitted_at="2026-01-01T00:00:00",
            gpu_ids=[0, 1],
        )
        restored = Job.from_dict(job.to_dict())
        self.assertEqual(job, restored)

    def test_defaults(self):
        job = Job(id="x", name="n", user="u", n_gpus=1, cmd=["echo"], cwd="/tmp")
        self.assertEqual(job.state, "PENDING")
        self.assertIsNone(job.started_at)
        self.assertIsNone(job.gpu_ids)
        self.assertIsNone(job.log_path)


if __name__ == "__main__":
    unittest.main()
