from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class BrokenPipeSubprocessTests(unittest.TestCase):
    def invoke_with_closed_stdout(self, mode: str) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "cancellations"
            read_fd, write_fd = os.pipe()
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tests.v2.broken_pipe_child",
                        mode,
                        str(marker),
                    ],
                    cwd=REPO,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONWARNINGS": "error",
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=write_fd,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                os.close(write_fd)
                os.close(read_fd)

            try:
                _stdout, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                self.fail(f"{mode} child did not exit after its stdout pipe closed")

            cancellations = marker.read_text(encoding="utf-8").splitlines() if marker.exists() else []
            return process.returncode, stderr, cancellations

    def assert_clean_broken_pipe_exit(self, returncode: int, stderr: str) -> None:
        self.assertEqual(returncode, 141, stderr)
        self.assertNotIn("BrokenPipeError", stderr)
        self.assertNotIn("Exception ignored", stderr)

    def test_foreground_run_cancels_once_and_exits_cleanly(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_stdout("run")

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, ["cancel"])
        self.assertIn("output pipe closed", stderr)
        self.assertIn("cancelling foreground job", stderr)

    def test_following_logs_detaches_without_cancellation_and_exits_cleanly(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_stdout("logs")

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, [])
        self.assertIn("output pipe closed", stderr)
        self.assertIn("detached", stderr)


if __name__ == "__main__":
    unittest.main()
