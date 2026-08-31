import unittest
from unittest.mock import patch

from corral import tmux_exec


class TestLaunch(unittest.TestCase):
    @patch("corral.tmux_exec.subprocess.run")
    @patch("corral.tmux_exec.ensure_session")
    def test_sets_cuda_visible_devices_and_ignores_sigint_in_tee(self, mock_ensure, mock_run):
        tmux_exec.launch(
            job_id="20260101-000000-abcdef",
            gpu_ids=[0, 2],
            cmd=["python", "train.py"],
            cwd="/home/alice/proj",
            log_path="/home/alice/proj/.corral/logs/x.log",
            exitcode_path="/home/alice/proj/.corral/logs/x.exitcode",
            session="corral-alice",
        )
        mock_ensure.assert_called_once_with("corral-alice")
        argv = mock_run.call_args[0][0]
        self.assertEqual(argv[:3], ["tmux", "new-window", "-t"])
        self.assertEqual(argv[3], "corral-alice")
        inner_script = argv[-1]
        self.assertIn("CUDA_VISIBLE_DEVICES=0,2", inner_script)
        # The tee/SIGINT race fix: tee must ignore SIGINT so `corral cancel`'s
        # Ctrl-C doesn't kill it before a job's own trap handler finishes writing.
        self.assertIn("trap '' INT", inner_script)
        self.assertIn("tee", inner_script)
        self.assertIn("PIPESTATUS[0]", inner_script)

    @patch("corral.tmux_exec.subprocess.run")
    @patch("corral.tmux_exec.ensure_session")
    def test_window_name_is_truncated_job_id(self, mock_ensure, mock_run):
        long_id = "x" * 50
        tmux_exec.launch(long_id, [0], ["echo"], "/tmp", "/tmp/l.log", "/tmp/l.exitcode", session="corral-alice")
        argv = mock_run.call_args[0][0]
        window_name = argv[argv.index("-n") + 1]
        self.assertEqual(window_name, long_id[:30])


class TestInterrupt(unittest.TestCase):
    @patch("corral.tmux_exec.subprocess.run")
    def test_sends_ctrl_c_to_the_right_window(self, mock_run):
        tmux_exec.interrupt("20260101-000000-abcdef", session="corral-alice")
        argv = mock_run.call_args[0][0]
        self.assertIn("send-keys", argv)
        self.assertIn("C-c", argv)
        self.assertTrue(any("corral-alice" in a for a in argv))
        self.assertTrue(any("20260101-000000-abcdef" in a for a in argv))


if __name__ == "__main__":
    unittest.main()
