import subprocess
import unittest
from unittest.mock import MagicMock, patch

from corral import gpu


class TestDetectGpus(unittest.TestCase):
    @patch("subprocess.run")
    def test_parses_normal_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="0, NVIDIA B200, 183359, 0\n1, NVIDIA B200, 183359, 512\n",
        )
        gpus = gpu.detect_gpus()
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0].index, 0)
        self.assertEqual(gpus[0].name, "NVIDIA B200")
        self.assertEqual(gpus[0].mem_total_mib, 183359)
        self.assertEqual(gpus[1].mem_used_mib, 512)

    @patch("subprocess.run")
    def test_sorted_by_index_even_if_nvidia_smi_orders_differently(self, mock_run):
        mock_run.return_value = MagicMock(stdout="1, X, 100, 0\n0, X, 100, 0\n")
        gpus = gpu.detect_gpus()
        self.assertEqual([g.index for g in gpus], [0, 1])

    @patch("subprocess.run")
    def test_blank_lines_ignored(self, mock_run):
        mock_run.return_value = MagicMock(stdout="0, X, 100, 0\n\n\n")
        self.assertEqual(len(gpu.detect_gpus()), 1)

    @patch("subprocess.run", side_effect=FileNotFoundError())
    def test_missing_nvidia_smi_raises_runtime_error(self, mock_run):
        with self.assertRaises(RuntimeError):
            gpu.detect_gpus()

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi", stderr="boom"))
    def test_nvidia_smi_error_exit_raises_runtime_error(self, mock_run):
        with self.assertRaises(RuntimeError):
            gpu.detect_gpus()

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 15))
    def test_timeout_raises_runtime_error(self, mock_run):
        with self.assertRaises(RuntimeError):
            gpu.detect_gpus()


if __name__ == "__main__":
    unittest.main()
