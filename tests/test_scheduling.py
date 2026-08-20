import unittest

from corral.daemon import _plan_schedule
from corral.models import Job


def make_job(job_id, n_gpus):
    return Job(
        id=job_id, name=job_id, user="alice", n_gpus=n_gpus, cmd=["sleep", "1"], cwd="/tmp",
        submitted_at="2026-01-01T00:00:00",
    )


class TestBasics(unittest.TestCase):
    def test_single_job_launches_on_lowest_index_gpus_first(self):
        job = make_job("j1", 2)
        assignments = _plan_schedule([job], eligible=[0, 1, 2], total_gpus=8, skip_counts={}, max_skips=5)
        self.assertEqual(assignments, [(job, [0, 1])])

    def test_multiple_jobs_bin_pack_in_fifo_order_when_all_fit(self):
        j1, j2 = make_job("j1", 1), make_job("j2", 1)
        assignments = _plan_schedule([j1, j2], eligible=[0, 1], total_gpus=8, skip_counts={}, max_skips=5)
        self.assertEqual(assignments, [(j1, [0]), (j2, [1])])

    def test_no_eligible_gpus_launches_nothing(self):
        job = make_job("j1", 1)
        assignments = _plan_schedule([job], eligible=[], total_gpus=8, skip_counts={}, max_skips=5)
        self.assertEqual(assignments, [])

    def test_impossible_job_never_blocks_and_is_never_the_head(self):
        impossible = make_job("j1", 9)  # more than the server's 8 total
        small = make_job("j2", 1)
        skip_counts = {}
        assignments = _plan_schedule(
            [impossible, small], eligible=[0], total_gpus=8, skip_counts=skip_counts, max_skips=5
        )
        self.assertEqual(assignments, [(small, [0])])
        self.assertNotIn(impossible.id, skip_counts)


class TestStrictFifoViaZeroMaxSkips(unittest.TestCase):
    """max_skips=0 must reproduce the original strict-FIFO behavior exactly:
    the head-of-queue job blocks everything behind it, immediately, always."""

    def test_big_head_blocks_smaller_job_even_with_idle_gpus(self):
        big, small = make_job("j1", 5), make_job("j2", 1)
        assignments = _plan_schedule([big, small], eligible=[0, 1, 2], total_gpus=8, skip_counts={}, max_skips=0)
        self.assertEqual(assignments, [])

    def test_head_launches_immediately_once_it_fits(self):
        big, small = make_job("j1", 3), make_job("j2", 1)
        assignments = _plan_schedule([big, small], eligible=[0, 1, 2], total_gpus=8, skip_counts={}, max_skips=0)
        self.assertEqual(assignments, [(big, [0, 1, 2])])


class TestBoundedBackfill(unittest.TestCase):
    def test_small_job_backfills_while_head_is_blocked(self):
        big, small = make_job("j1", 5), make_job("j2", 1)
        skip_counts = {}
        assignments = _plan_schedule(
            [big, small], eligible=[0, 1, 2], total_gpus=8, skip_counts=skip_counts, max_skips=5
        )
        self.assertEqual(assignments, [(small, [0])])
        self.assertEqual(skip_counts[big.id], 1)

    def test_multiple_small_jobs_all_backfill_if_they_collectively_fit(self):
        big, s1, s2 = make_job("j1", 5), make_job("j2", 1), make_job("j3", 1)
        assignments = _plan_schedule(
            [big, s1, s2], eligible=[0, 1, 2], total_gpus=8, skip_counts={}, max_skips=5
        )
        self.assertEqual(assignments, [(s1, [0]), (s2, [1])])

    def test_head_skip_count_increments_once_per_poll_while_blocked(self):
        big = make_job("j1", 5)
        skip_counts = {}
        for expected in (1, 2, 3):
            _plan_schedule([big], eligible=[0, 1, 2], total_gpus=8, skip_counts=skip_counts, max_skips=5)
            self.assertEqual(skip_counts[big.id], expected)

    def test_backfill_pauses_once_skip_budget_is_exhausted(self):
        big, small = make_job("j1", 5), make_job("j2", 1)
        skip_counts = {}
        max_skips = 3
        for _ in range(max_skips):
            assignments = _plan_schedule(
                [big, small], eligible=[0, 1, 2], total_gpus=8, skip_counts=skip_counts, max_skips=max_skips
            )
            self.assertEqual(assignments, [(small, [0])])
        # One more poll pushes the skip count past budget -- backfill must now pause,
        # reserving every eligible GPU for the head even though `small` would still fit.
        assignments = _plan_schedule(
            [big, small], eligible=[0, 1, 2], total_gpus=8, skip_counts=skip_counts, max_skips=max_skips
        )
        self.assertEqual(assignments, [])
        self.assertGreater(skip_counts[big.id], max_skips)

    def test_head_launches_and_clears_its_history_once_enough_gpus_accumulate(self):
        big = make_job("j1", 3)
        skip_counts = {big.id: 4}  # simulate an already-exhausted skip budget from prior polls
        assignments = _plan_schedule([big], eligible=[0, 1, 2], total_gpus=8, skip_counts=skip_counts, max_skips=5)
        self.assertEqual(assignments, [(big, [0, 1, 2])])
        self.assertNotIn(big.id, skip_counts)

    def test_skip_history_forgotten_once_job_leaves_the_pending_queue(self):
        big = make_job("j1", 5)
        skip_counts = {}
        _plan_schedule([big], eligible=[0], total_gpus=8, skip_counts=skip_counts, max_skips=5)
        self.assertIn(big.id, skip_counts)
        _plan_schedule([], eligible=[0], total_gpus=8, skip_counts=skip_counts, max_skips=5)  # cancelled elsewhere
        self.assertNotIn(big.id, skip_counts)


if __name__ == "__main__":
    unittest.main()
