from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

from hpc_alloc.errors import RemoteCommandFailed
from hpc_alloc.slurm import MAX_LOG_CHUNK_BYTES, SlurmClient
from hpc_alloc.ssh import RemoteResult


class LocalShellTransport:
    """Execute generated shell commands with a controllable tail binary."""

    def __init__(
        self,
        path: str,
        *,
        source: str = "success",
        sink: str = "success",
        chunk_source: str = "success",
        chunk_sink: str = "success",
    ) -> None:
        self.env = os.environ.copy()
        self.env["PATH"] = path
        self.env["HPC_ALLOC_FAKE_TAIL_SOURCE"] = source
        self.env["HPC_ALLOC_FAKE_TAIL_SINK"] = sink
        self.env["HPC_ALLOC_FAKE_CHUNK_SOURCE"] = chunk_source
        self.env["HPC_ALLOC_FAKE_CHUNK_SINK"] = chunk_sink

    def run(self, _cluster: str, command: str, **kwargs: object) -> RemoteResult:
        result = subprocess.run(
            ["sh", "-c", command],
            capture_output=True,
            env=self.env,
            timeout=kwargs.get("timeout", 60),
        )
        return RemoteResult(
            result.returncode,
            b"site login banner\n" + result.stdout,
            result.stderr.decode("utf-8", errors="replace"),
        )


class TailPipelineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.bin_directory = root / "bin"
        self.bin_directory.mkdir()
        self.log_path = root / "log.bin"
        self.log_path.write_bytes(b"first line\nsecond line\n")

        real_tail = shutil.which("tail")
        real_head = shutil.which("head")
        if real_tail is None or real_head is None:
            self.skipTest("tail and head are required for shell protocol contract tests")
        fake_tail = self.bin_directory / "tail"
        fake_tail.write_text(
            textwrap.dedent(
                f"""\
                #!/bin/sh
                if [ "$1" = "-n" ]; then
                    case "$HPC_ALLOC_FAKE_TAIL_SOURCE" in
                        partial_failure)
                            printf 'partial source output'
                            printf 'simulated source read failure\\n' >&2
                            exit 23
                            ;;
                        empty_failure)
                            printf 'simulated empty source failure\\n' >&2
                            exit 24
                            ;;
                    esac
                elif [ "$1" = "-c" ]; then
                    case "$2" in
                        +*)
                            case "$HPC_ALLOC_FAKE_CHUNK_SOURCE" in
                                partial_failure)
                                    printf 'partial chunk output'
                                    printf 'simulated chunk source failure\\n' >&2
                                    exit 23
                                    ;;
                                empty_failure)
                                    printf 'simulated empty chunk failure\\n' >&2
                                    exit 24
                                    ;;
                                full_failure)
                                    {shlex.quote(real_tail)} "$@"
                                    printf 'simulated full chunk failure\\n' >&2
                                    exit 23
                                    ;;
                            esac
                            ;;
                    esac
                    if [ "$HPC_ALLOC_FAKE_TAIL_SINK" = "failure" ]; then
                        cat >/dev/null
                        printf 'simulated bounded-tail failure\\n' >&2
                        exit 25
                    fi
                fi
                exec {shlex.quote(real_tail)} "$@"
                """
            ),
            encoding="utf-8",
        )
        fake_tail.chmod(0o755)
        fake_head = self.bin_directory / "head"
        fake_head.write_text(
            textwrap.dedent(
                f"""\
                #!/bin/sh
                if [ "$HPC_ALLOC_FAKE_CHUNK_SINK" = "failure" ]; then
                    cat >/dev/null
                    printf 'simulated chunk sink failure\\n' >&2
                    exit 25
                fi
                exec {shlex.quote(real_head)} "$@"
                """
            ),
            encoding="utf-8",
        )
        fake_head.chmod(0o755)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def client(
        self,
        *,
        source: str = "success",
        sink: str = "success",
        chunk_source: str = "success",
        chunk_sink: str = "success",
    ) -> SlurmClient:
        path = f"{self.bin_directory}{os.pathsep}{os.environ.get('PATH', '')}"
        transport = LocalShellTransport(
            path,
            source=source,
            sink=sink,
            chunk_source=chunk_source,
            chunk_sink=chunk_sink,
        )
        return SlurmClient(transport, "local")  # type: ignore[arg-type]

    def test_partial_chunk_source_output_does_not_hide_failure(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated chunk source failure"):
            self.client(chunk_source="partial_failure").read_log_chunk(
                str(self.log_path), 0, limit=64
            )

    def test_empty_chunk_source_failure_is_not_successful_eof(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated empty chunk failure"):
            self.client(chunk_source="empty_failure").read_log_chunk(
                str(self.log_path), 0, limit=64
            )

    def test_full_chunk_from_a_failed_source_is_still_a_failure(self) -> None:
        self.log_path.write_bytes(b"x" * 64)
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated full chunk failure"):
            self.client(chunk_source="full_failure").read_log_chunk(
                str(self.log_path), 0, limit=64
            )

    def test_expected_source_sigpipe_is_accepted_at_the_byte_cap(self) -> None:
        payload = bytes(range(256)) * 8192
        self.log_path.write_bytes(payload)

        result = self.client().read_log_chunk(str(self.log_path), 0, limit=1024)

        self.assertEqual(result, payload[:1024])

    def test_chunk_sink_failure_is_preserved(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated chunk sink failure"):
            self.client(chunk_sink="failure").read_log_chunk(
                str(self.log_path), 0, limit=64
            )

    def test_partial_source_output_does_not_hide_source_failure(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated source read failure"):
            self.client(source="partial_failure").tail_log(str(self.log_path), 10)

    def test_empty_source_failure_is_not_reported_as_a_successful_empty_tail(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated empty source failure"):
            self.client(source="empty_failure").tail_log(str(self.log_path), 10)

    def test_successful_tail_remains_bounded_to_the_last_bytes(self) -> None:
        payload = b"a" * (MAX_LOG_CHUNK_BYTES + 257) + b"\nlast line\n"
        self.log_path.write_bytes(payload)

        result = self.client().tail_log(str(self.log_path), 1_000_000)

        self.assertEqual(result, payload[-MAX_LOG_CHUNK_BYTES:])

    def test_sink_failure_is_preserved_when_source_succeeds(self) -> None:
        with self.assertRaisesRegex(RemoteCommandFailed, "simulated bounded-tail failure"):
            self.client(sink="failure").tail_log(str(self.log_path), 10)


if __name__ == "__main__":
    unittest.main()
