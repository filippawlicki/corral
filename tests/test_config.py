import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from corral import config


class TestLogDirFor(unittest.TestCase):
    def test_default_is_relative_to_job_cwd(self):
        with patch.object(config, "LOG_DIR", None):
            result = config.log_dir_for("/home/alice/myproject")
        self.assertEqual(result, Path("/home/alice/myproject/.corral/logs"))

    def test_override_ignores_job_cwd(self):
        with patch.object(config, "LOG_DIR", Path("/shared/corral_logs")):
            result = config.log_dir_for("/home/alice/myproject")
        self.assertEqual(result, Path("/shared/corral_logs"))

    def test_two_different_jobs_get_different_default_dirs(self):
        with patch.object(config, "LOG_DIR", None):
            a = config.log_dir_for("/home/alice/proj")
            b = config.log_dir_for("/home/bob/proj")
        self.assertNotEqual(a, b)


class TestLauncherSession(unittest.TestCase):
    def test_different_users_get_different_sessions(self):
        with patch.object(config, "TMUX_SESSION", "corral"):
            self.assertEqual(config.launcher_session("alice"), "corral-alice")
            self.assertNotEqual(config.launcher_session("alice"), config.launcher_session("bob"))


class TestReservedGpus(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(config, "RESERVED_GPUS_PATH", Path(self._tmp.name) / "reserved_gpus")
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_defaults_to_zero_when_never_set(self):
        self.assertEqual(config.reserved_gpus(), 0)

    def test_round_trips_a_set_value(self):
        config.set_reserved_gpus(3)
        self.assertEqual(config.reserved_gpus(), 3)

    def test_can_be_changed_repeatedly_without_any_restart_concept(self):
        config.set_reserved_gpus(2)
        self.assertEqual(config.reserved_gpus(), 2)
        config.set_reserved_gpus(0)
        self.assertEqual(config.reserved_gpus(), 0)

    def test_garbage_file_contents_falls_back_to_zero(self):
        config.RESERVED_GPUS_PATH.write_text("not-a-number")
        self.assertEqual(config.reserved_gpus(), 0)


if __name__ == "__main__":
    unittest.main()
