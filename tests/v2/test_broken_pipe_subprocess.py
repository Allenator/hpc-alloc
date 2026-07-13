from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.cli import main
from hpc_alloc.errors import ConfigInvalid


REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "hpc-alloc"


class CliExceptionBoundaryTests(unittest.TestCase):
    def test_generic_broken_pipe_neutralizes_both_output_streams(self) -> None:
        with (
            patch("hpc_alloc.commands.dispatch", side_effect=BrokenPipeError()),
            patch("hpc_alloc.cli.neutralize_stdout") as neutralize_stdout,
            patch("hpc_alloc.cli.neutralize_stderr") as neutralize_stderr,
        ):
            result = main(["status"], entrypoint=CLI)

        self.assertEqual(result, 141)
        neutralize_stdout.assert_called_once_with()
        neutralize_stderr.assert_called_once_with()

    def test_typed_error_keeps_its_exit_code_when_diagnostic_pipe_is_closed(
        self,
    ) -> None:
        broken_stderr = SimpleNamespace(
            write=Mock(side_effect=BrokenPipeError()),
            flush=Mock(),
        )
        with (
            patch(
                "hpc_alloc.commands.dispatch",
                side_effect=ConfigInvalid("configuration does not exist"),
            ),
            patch("hpc_alloc.cli.sys.stderr", broken_stderr),
            patch("hpc_alloc.cli.neutralize_stdout") as neutralize_stdout,
            patch("hpc_alloc.cli.neutralize_stderr") as neutralize_stderr,
        ):
            result = main(["status"], entrypoint=CLI)

        self.assertEqual(result, ConfigInvalid.exit_code)
        neutralize_stdout.assert_called_once_with()
        neutralize_stderr.assert_called_once_with()



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

    def invoke_launcher_with_closed_stderr(
        self, home: Path, *arguments: str, combined: bool = False
    ) -> subprocess.CompletedProcess[str]:
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        try:
            process = subprocess.Popen(
                [str(CLI), *arguments],
                cwd=REPO,
                env={
                    **os.environ,
                    "HOME": str(home),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONWARNINGS": "error",
                },
                stdin=subprocess.DEVNULL,
                stdout=write_fd if combined else subprocess.PIPE,
                stderr=write_fd,
                text=True,
            )
        finally:
            os.close(write_fd)

        try:
            stdout, _stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.fail(f"launcher did not exit with closed stderr: {arguments!r}")
        return subprocess.CompletedProcess(
            process.args,
            process.returncode,
            stdout or "",
            "",
        )

    def test_config_with_closed_stderr_exits_as_broken_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / ".config" / "hpc-alloc" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text(
                """\
[identity]
netid = "ab1234"

[defaults]
cluster = "grace"

[cluster.grace]
host = "grace.example.edu"
""",
                encoding="utf-8",
            )

            result = self.invoke_launcher_with_closed_stderr(home, "config")

        self.assertEqual(result.returncode, 141, result.stdout)

    def test_operational_missing_config_keeps_typed_exit_code_with_closed_stderr(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke_launcher_with_closed_stderr(
                Path(directory), "status"
            )

        self.assertEqual(result.returncode, ConfigInvalid.exit_code, result.stdout)

    def test_typed_error_keeps_its_exit_code_with_closed_combined_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.invoke_launcher_with_closed_stderr(
                Path(directory), "status", combined=True
            )

        self.assertEqual(result.returncode, ConfigInvalid.exit_code, result.stdout)

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
