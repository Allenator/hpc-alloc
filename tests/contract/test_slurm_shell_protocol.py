from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from hpc_alloc.slurm import MAX_LOG_CHUNK_BYTES, LogSizeStatus, SlurmClient
from hpc_alloc.ssh import RemoteResult


class LocalShellTransport:
    """Execute the generated remote shell command locally, with a fake banner."""

    def run(self, _cluster: str, command: str, **kwargs: object) -> RemoteResult:
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            timeout=kwargs.get("timeout", 60),
        )
        return RemoteResult(
            result.returncode,
            b"site login banner\n" + result.stdout,
            result.stderr.decode("utf-8", errors="replace"),
        )


class SlurmShellProtocolTests(unittest.TestCase):
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
