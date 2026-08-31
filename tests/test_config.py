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


if __name__ == "__main__":
    unittest.main()
