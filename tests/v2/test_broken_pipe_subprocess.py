from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class BrokenPipeSubprocessTests(unittest.TestCase):
    def invoke_with_closed_pipe(
        self, mode: str, *, closed_streams: str
    ) -> tuple[int, str, list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "cancellations"
            read_fd, write_fd = os.pipe()
            stdout: int | object = (
                write_fd
                if closed_streams in {"stdout", "combined"}
                else subprocess.DEVNULL
            )
            stderr: int | object = (
                write_fd
                if closed_streams in {"stderr", "combined"}
                else subprocess.PIPE
            )
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
                        "HPC_ALLOC_BROKEN_PIPE_GATE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONWARNINGS": "error",
                    },
                    stdin=subprocess.PIPE,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
            finally:
                os.close(write_fd)
                os.close(read_fd)

            try:
                _stdout, captured_stderr = process.communicate("1", timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                self.fail(f"{mode} child did not exit after its stdout pipe closed")

            cancellations = (
                marker.read_text(encoding="utf-8").splitlines()
                if marker.exists()
                else []
            )
            return process.returncode, captured_stderr or "", cancellations

    def assert_clean_broken_pipe_exit(self, returncode: int, stderr: str) -> None:
        self.assertEqual(returncode, 141, stderr)
        self.assertNotIn("BrokenPipeError", stderr)
        self.assertNotIn("Exception ignored", stderr)

    def test_foreground_run_cancels_once_and_exits_cleanly(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "run", closed_streams="stdout"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, ["cancel"])
        self.assertIn("output pipe closed", stderr)
        self.assertIn("cancelled foreground job", stderr)

    def test_following_logs_detaches_without_cancellation_and_exits_cleanly(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "logs", closed_streams="stdout"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, [])
        self.assertIn("output pipe closed", stderr)
        self.assertIn("detached", stderr)

    def test_foreground_run_cancels_when_stderr_is_already_closed(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "run", closed_streams="stderr"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, ["cancel"])

    def test_foreground_run_cancels_with_closed_combined_output(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "run", closed_streams="combined"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, ["cancel"])

    def test_detached_run_does_not_cancel_with_closed_combined_output(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "run-detach", closed_streams="combined"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, [])

    def test_following_logs_detaches_with_closed_combined_output(self) -> None:
        returncode, stderr, cancellations = self.invoke_with_closed_pipe(
            "logs", closed_streams="combined"
        )

        self.assert_clean_broken_pipe_exit(returncode, stderr)
        self.assertEqual(cancellations, [])


if __name__ == "__main__":
    unittest.main()
