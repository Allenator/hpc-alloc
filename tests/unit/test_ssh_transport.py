from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from hpc_alloc.errors import (
    AuthRequired,
    HostKeyChanged,
    LocalToolUnavailable,
    TransportLost,
)
from hpc_alloc.ssh import AuthMode, RetryPolicy, SshTransport


def completed(argv: object, rc: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


class SshTransportTests(unittest.TestCase):
    def transport(self, runner, root: Path) -> SshTransport:
        config = SimpleNamespace(clusters={"grace": object()}, ssh=SimpleNamespace(identity_file=None))
        paths = SimpleNamespace(ssh_dir=root)
        return SshTransport(config, paths, runner=runner)

    def test_noninteractive_auth_is_typed_and_never_prompts(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str], **_kwargs: object):
            calls.append(argv)
            if "check" in argv:
                return completed(argv, 1, stderr="no master")
            return completed(argv, 255, stderr="Permission denied (publickey,keyboard-interactive)")

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            with self.assertRaises(AuthRequired):
                transport.bootstrap("grace", AuthMode.NONINTERACTIVE)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("BatchMode=yes" in call for call in calls))

    def test_missing_local_ssh_is_not_misclassified_as_reconnectable(self) -> None:
        def runner(_argv: list[str], **_kwargs: object):
            raise FileNotFoundError("ssh")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(LocalToolUnavailable) as raised:
                self.transport(runner, Path(directory)).bootstrap(
                    "grace", AuthMode.NONINTERACTIVE
                )
        self.assertEqual(raised.exception.exit_code, 1)

    def test_host_key_change_is_not_reduced_to_network_failure(self) -> None:
        def runner(argv: list[str], **_kwargs: object):
            if "check" in argv:
                return completed(argv, 1)
            return completed(argv, 255, stderr="WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(HostKeyChanged):
                self.transport(runner, Path(directory)).bootstrap(
                    "grace", AuthMode.NONINTERACTIVE
                )

    def test_mutation_policy_never_retries_rc_255(self) -> None:
        commands = 0

        def runner(argv: list[str], **_kwargs: object):
            nonlocal commands
            if "check" in argv:
                return completed(argv, 0)
            commands += 1
            return completed(argv, 255, stderr="connection reset")

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            with self.assertRaises(TransportLost):
                transport.run("grace", "scancel 42", retry=RetryPolicy.NEVER)
        self.assertEqual(commands, 1)

    def test_safe_read_retries_once_after_healthy_probe(self) -> None:
        command_attempts = 0

        def runner(argv: list[str], **_kwargs: object):
            nonlocal command_attempts
            if "check" in argv:
                return completed(argv, 0)
            if argv[-1] == "true":
                return completed(argv, 0)
            command_attempts += 1
            if command_attempts == 1:
                return completed(argv, 255, stderr="mux reset")
            return completed(argv, 0, stdout="ok")

        with tempfile.TemporaryDirectory() as directory:
            result = self.transport(runner, Path(directory)).run(
                "grace", "squeue --me", retry=RetryPolicy.SAFE_READ
            )
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(command_attempts, 2)

    def test_require_node_preserves_each_typed_probe_failure(self) -> None:
        cases = (
            (
                "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!\nmore detail",
                HostKeyChanged,
                "host-key verification failed",
            ),
            (
                "Permission denied (publickey)",
                AuthRequired,
                "authentication failed",
            ),
            (
                "Connection timed out",
                TransportLost,
                "cannot reach",
            ),
        )
        for stderr, error_type, message in cases:
            with self.subTest(error=error_type.__name__):
                calls: list[list[str]] = []

                def runner(argv: list[str], **_kwargs: object):
                    calls.append(argv)
                    if "-O" in argv and "exit" in argv:
                        return completed(argv)
                    return completed(argv, 255, stderr=stderr)

                with tempfile.TemporaryDirectory() as directory:
                    transport = self.transport(runner, Path(directory))
                    with self.assertRaisesRegex(error_type, message) as raised:
                        transport.require_node("hpc-alloc-node-grace-dev")
                self.assertIn("hpc-alloc-node-grace-dev", str(raised.exception))
                self.assertIn(stderr.splitlines()[0], str(raised.exception))
                self.assertEqual(
                    sum(call[-1:] == ["true"] for call in calls),
                    2,
                    "the post-master-close probe must determine the returned status",
                )
                self.assertEqual(sum("exit" in call for call in calls), 1)

    def test_require_node_success_returns_without_closing_master(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str], **_kwargs: object):
            calls.append(argv)
            return completed(argv)

        with tempfile.TemporaryDirectory() as directory:
            result = self.transport(runner, Path(directory)).require_node("node-alias")
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("exit", calls[0])

    def test_run_raises_host_key_change_without_heal_or_retry(self) -> None:
        command_attempts = 0
        heal_attempts = 0

        def runner(argv: list[str], **_kwargs: object):
            nonlocal command_attempts, heal_attempts
            if "-O" in argv and "check" in argv:
                return completed(argv)
            if "-O" in argv and "exit" in argv:
                heal_attempts += 1
                return completed(argv)
            command_attempts += 1
            return completed(
                argv,
                255,
                stderr="WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!",
            )

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            with self.assertRaisesRegex(HostKeyChanged, "host-key verification failed"):
                transport.run("grace", "hostname", retry=RetryPolicy.SAFE_READ)
        self.assertEqual(command_attempts, 1)
        self.assertEqual(heal_attempts, 0)

    def test_run_raises_auth_required_without_generic_transport_retry(self) -> None:
        command_attempts = 0
        heal_attempts = 0

        def runner(argv: list[str], **_kwargs: object):
            nonlocal command_attempts, heal_attempts
            if "-O" in argv and "check" in argv:
                return completed(argv)
            if "-O" in argv and "exit" in argv:
                heal_attempts += 1
                return completed(argv)
            command_attempts += 1
            return completed(argv, 255, stderr="Permission denied (publickey)")

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            with self.assertRaisesRegex(AuthRequired, "authentication failed"):
                transport.run("grace", "hostname", retry=RetryPolicy.SAFE_READ)
        self.assertEqual(command_attempts, 1)
        self.assertEqual(heal_attempts, 0)

    def test_run_never_heals_host_key_discovered_by_retry_probe(self) -> None:
        command_attempts = 0
        heal_attempts = 0

        def runner(argv: list[str], **_kwargs: object):
            nonlocal command_attempts, heal_attempts
            if "-O" in argv and "check" in argv:
                return completed(argv)
            if "-O" in argv and "exit" in argv:
                heal_attempts += 1
                return completed(argv)
            if argv[-1] == "true":
                return completed(
                    argv,
                    255,
                    stderr="Host key verification failed.",
                )
            command_attempts += 1
            return completed(argv, 255, stderr="connection reset")

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            with self.assertRaisesRegex(HostKeyChanged, "host-key verification failed"):
                transport.run("grace", "hostname", retry=RetryPolicy.SAFE_READ)
        self.assertEqual(command_attempts, 1)
        self.assertEqual(heal_attempts, 0)


if __name__ == "__main__":
    unittest.main()
