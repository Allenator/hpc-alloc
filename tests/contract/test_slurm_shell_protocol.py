from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from hpc_alloc.errors import RemoteCommandFailed
from hpc_alloc.slurm import MAX_LOG_CHUNK_BYTES, LogSizeStatus, SlurmClient
from hpc_alloc.ssh import RemoteResult, RetryPolicy


class LocalShellTransport:
    """Execute the generated remote shell command locally, with a fake banner."""

    def __init__(self, *, startup_stderr: str = "") -> None:
        self.startup_stderr = startup_stderr

    def run(self, _cluster: str, command: str, **kwargs: object) -> RemoteResult:
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            timeout=kwargs.get("timeout", 60),
        )
        return RemoteResult(
            result.returncode,
            b"site login banner\n" + result.stdout,
            self.startup_stderr
            + result.stderr.decode("utf-8", errors="replace"),
        )


class TerminatingLocalShellTransport:
    """Terminate the wrapper after its nested command has started."""

    def __init__(self, directory: Path, ready: Path) -> None:
        self.directory = directory
        self.ready = ready

    def run(self, _cluster: str, command: str, **kwargs: object) -> RemoteResult:
        process = subprocess.Popen(
            ["sh", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "TMPDIR": str(self.directory)},
        )
        deadline = time.monotonic() + 5
        while not self.ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                raise AssertionError("nested shell command did not start")
            time.sleep(0.01)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        return RemoteResult(
            process.returncode,
            stdout,
            stderr.decode("utf-8", errors="replace"),
        )


class SlurmShellProtocolTests(unittest.TestCase):
    def test_command_stderr_is_framed_separately_from_startup_stderr(self) -> None:
        client = SlurmClient(
            LocalShellTransport(startup_stderr="site startup warning\n"),
            "local",
        )  # type: ignore[arg-type]

        result = client._framed(
            "printf '%s\\n' 'command diagnostic' >&2; exit 7",
            retry=RetryPolicy.SAFE_READ,
        )

        self.assertEqual(result.returncode, 7)
        self.assertEqual(result.payload, b"")
        self.assertEqual(result.stderr, "command diagnostic\n")

    def test_wrapper_signal_cleanup_terminates_without_emitting_a_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready"
            client = SlurmClient(
                TerminatingLocalShellTransport(root, ready),
                "local",
            )  # type: ignore[arg-type]

            with self.assertRaises(RemoteCommandFailed):
                client._framed(
                    f": > {shlex.quote(str(ready))}; sleep 0.2",
                    retry=RetryPolicy.SAFE_READ,
                )

            self.assertEqual(list(root.glob("hpc-alloc.*")), [])

    def test_binary_log_and_quoted_path_are_byte_exact(self) -> None:
        client = SlurmClient(LocalShellTransport(), "local")  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a log 'quoted'.bin"
            payload = b"prefix\x1eHPC_ALLOC_V2_not-a-header 0 3\n\xff\x00suffix"
            path.write_bytes(payload)
            size = client.log_size(str(path))
            self.assertEqual(size.status, LogSizeStatus.AVAILABLE)
            self.assertEqual(size.size, len(payload))
            self.assertEqual(client.read_log_chunk(str(path), 0), payload)
            self.assertEqual(client.read_log_chunk(str(path), 7), payload[7:])

    def test_missing_log_is_not_numeric_zero(self) -> None:
        client = SlurmClient(LocalShellTransport(), "local")  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            result = client.log_size(str(Path(directory) / "missing"))
        self.assertEqual(result.status, LogSizeStatus.MISSING)
        self.assertIsNone(result.size)

    def test_large_log_is_read_and_tailed_with_hard_byte_bounds(self) -> None:
        client = SlurmClient(LocalShellTransport(), "local")  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.bin"
            payload = (
                b"a" * MAX_LOG_CHUNK_BYTES
                + bytes(range(256))
                + b"\nlast line\x00\xff\n"
            )
            path.write_bytes(payload)
            first = client.read_log_chunk(str(path), 0)
            second = client.read_log_chunk(str(path), len(first))
            self.assertEqual(len(first), MAX_LOG_CHUNK_BYTES)
            self.assertEqual(first + second, payload)
            self.assertEqual(
                client.tail_log(str(path), 1_000_000),
                payload[-MAX_LOG_CHUNK_BYTES:],
            )


if __name__ == "__main__":
    unittest.main()
