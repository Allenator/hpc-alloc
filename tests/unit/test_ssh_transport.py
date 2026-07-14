from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.errors import (
    AuthRequired,
    HostKeyChanged,
    LocalToolUnavailable,
    TransportLost,
)
from hpc_alloc.ssh import (
    AuthMode,
    ProbeStatus,
    RetryPolicy,
    SshTransport,
    retire_compute_masters,
    ssh_argv,
)
from hpc_alloc.ssh_config import (
    ComputeMasterRetirement,
    compute_control_socket_prefix,
)


def completed(argv: object, rc: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(argv, rc, stdout, stderr)


class SshTransportTests(unittest.TestCase):
    def test_a_stalled_node_probe_does_not_retire_a_live_master(self) -> None:
        """A probe timeout means the node is slow, not that its master is dead.

        A loaded allocation -- the very thing an allocation is for -- or a
        briefly hung shared home can stall a new session past the probe timeout.
        Retiring the master on that evidence tears down every session
        multiplexed on the compute alias (another process's interactive shell,
        an in-flight rsync) even though its connection was healthy.
        """

        calls: list[list[str]] = []

        def runner(argv: list[str], **_kwargs: object):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, 15)

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            status = transport.probe_node("hpc-alloc-node-grace-dev")

        self.assertIs(status, ProbeStatus.NETWORK)
        self.assertEqual(
            sum("-O" in call for call in calls),
            0,
            "a stalled probe retired a possibly-live ControlMaster",
        )
        self.assertEqual(len(calls), 1, "a stalled probe must not re-probe")

    def test_remote_invocations_never_inherit_the_callers_stdin(self) -> None:
        """ssh forwards its own stdin to the remote command unless told not to.

        capture_output only redirects stdout/stderr, so an inherited fd 0 lets
        each polling ssh child drain the caller's stdin -- silently eating the
        remaining lines of a `while read ...; done < joblist` loop, or piped
        input.  These invocations never consume stdin, so they must get DEVNULL.
        """

        seen: list[dict[str, object]] = []

        def runner(argv: list[str], **kwargs: object):
            seen.append(kwargs)
            return completed(argv)

        with tempfile.TemporaryDirectory() as directory:
            transport = self.transport(runner, Path(directory))
            transport.probe_alias("hpc-grace.login")
            transport.close_master("hpc-grace.login")

        self.assertTrue(seen)
        for kwargs in seen:
            self.assertIs(kwargs.get("stdin"), subprocess.DEVNULL)

    def transport(self, runner, root: Path) -> SshTransport:
        config = SimpleNamespace(clusters={"grace": object()}, ssh=SimpleNamespace(identity_file=None))
        paths = SimpleNamespace(ssh_dir=root)
        return SshTransport(config, paths, runner=runner)

    def control_socket(self, root: Path, suffix: str) -> Path:
        return root / f"{compute_control_socket_prefix('grace')}{suffix}"

    def test_retirement_protects_shared_retained_path_and_closes_obsolete_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = self.control_socket(root, "shared")
            obsolete = self.control_socket(root, "obsolete")
            paths = {
                "hpc-grace.dev": shared,
                "hpc-grace.research": shared,
                "hpc-grace.old": obsolete,
            }
            closes: list[Path] = []

            def runner(argv: list[str], **_kwargs: object):
                if "-G" in argv:
                    alias = argv[-1]
                    return completed(
                        argv,
                        stdout=f"host {alias}\ncontrolpath {paths[alias]}\n",
                    )
                path = Path(argv[argv.index("-S") + 1])
                closes.append(path)
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(
                        (
                            "hpc-grace.dev",
                            "hpc-grace.old",
                            "hpc-grace.research",
                        ),
                        ("hpc-grace.research",),
                    ),
                    ssh_dir=root,
                    runner=runner,
                )

            self.assertEqual(warnings, ())
            self.assertEqual(closes, [obsolete])

    def test_retirement_deduplicates_two_aliases_for_one_obsolete_socket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obsolete = self.control_socket(root, "obsolete")
            closes = 0

            def runner(argv: list[str], **_kwargs: object):
                nonlocal closes
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {obsolete}\n")
                self.assertIn(argv[-1], {"hpc-grace.dev", "hpc-grace.research"})
                closes += 1
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(
                        ("hpc-grace.dev", "hpc-grace.research"), ()
                    ),
                    ssh_dir=root,
                    runner=runner,
                )

            self.assertEqual(warnings, ())
            self.assertEqual(closes, 1)

    def test_retirement_warns_and_skips_a_non_socket_managed_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / f"{compute_control_socket_prefix('grace')}regular"
            path.write_text("not a socket")
            close_calls = 0

            def runner(argv: list[str], **_kwargs: object):
                nonlocal close_calls
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {path}\n")
                close_calls += 1
                return completed(argv)

            warnings = retire_compute_masters(
                ComputeMasterRetirement(("hpc-grace.dev",), ()),
                ssh_dir=root,
                runner=runner,
            )

            self.assertEqual(close_calls, 0)
            self.assertEqual(len(warnings), 1)
            self.assertIn("non-socket", warnings[0])

    def test_retirement_warns_when_exit_fails_and_socket_remains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.control_socket(root, "stubborn")

            def runner(argv: list[str], **_kwargs: object):
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {path}\n")
                return completed(argv, 255, stderr="master refused exit")

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(("hpc-grace.dev",), ()),
                    ssh_dir=root,
                    runner=runner,
                )

            self.assertEqual(len(warnings), 1)
            self.assertIn("master refused exit", warnings[0])

    def test_retirement_refuses_a_socket_owned_by_another_user(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.control_socket(root, "foreign")
            close_calls = 0

            def runner(argv: list[str], **_kwargs: object):
                nonlocal close_calls
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {path}\n")
                close_calls += 1
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid() + 1,
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(("hpc-grace.dev",), ()),
                    ssh_dir=root,
                    runner=runner,
                )

            self.assertEqual(close_calls, 0)
            self.assertEqual(len(warnings), 1)
            self.assertIn("not owned by the current user", warnings[0])

    def test_retirement_aborts_when_a_retained_path_cannot_be_protected(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str], **_kwargs: object):
            calls.append(argv)
            return completed(argv, 255, stderr="invalid old projection")

        with tempfile.TemporaryDirectory() as directory:
            warnings = retire_compute_masters(
                ComputeMasterRetirement(
                    ("hpc-grace.dev", "hpc-grace.research"),
                    ("hpc-grace.research",),
                ),
                ssh_dir=Path(directory),
                runner=runner,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(warnings), 2)
        self.assertIn("could not inspect", warnings[0])
        self.assertIn("could not be protected", warnings[1])

    def test_retirement_bounds_inspection_and_exit_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.control_socket(root, "obsolete")
            calls: list[tuple[list[str], object]] = []

            def runner(argv: list[str], **kwargs: object):
                calls.append((argv, kwargs.get("timeout")))
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {path}\n")
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(("hpc-grace.dev",), ()),
                    ssh_dir=root,
                    runner=runner,
                )

        self.assertEqual(warnings, ())
        self.assertEqual(len(calls), 2)
        self.assertIn("-G", calls[0][0])
        self.assertIn("-O", calls[1][0])
        self.assertEqual([timeout for _argv, timeout in calls], [10, 10])

    def test_retirement_aborts_when_retained_alias_inspection_times_out(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str], **kwargs: object):
            calls.append(argv)
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as directory:
            warnings = retire_compute_masters(
                ComputeMasterRetirement(
                    ("hpc-grace.dev", "hpc-grace.old"),
                    ("hpc-grace.dev",),
                ),
                ssh_dir=Path(directory),
                runner=runner,
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(warnings), 2)
        self.assertIn("timed out after 10 seconds", warnings[0])
        self.assertIn("could not be protected", warnings[1])

    def test_retirement_skips_only_obsolete_alias_whose_inspection_times_out(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.control_socket(root, "closable")
            closes: list[str] = []

            def runner(argv: list[str], **kwargs: object):
                if "-G" in argv:
                    alias = argv[-1]
                    if alias == "hpc-grace.stuck":
                        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                    return completed(argv, stdout=f"controlpath {path}\n")
                closes.append(argv[-1])
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(
                        ("hpc-grace.stuck", "hpc-grace.closable"), ()
                    ),
                    ssh_dir=root,
                    runner=runner,
                )

        self.assertEqual(closes, ["hpc-grace.closable"])
        self.assertEqual(len(warnings), 1)
        self.assertIn("hpc-grace.stuck", warnings[0])
        self.assertIn("timed out after 10 seconds", warnings[0])

    def test_retirement_exit_timeout_is_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "hpc-grace.stuck": self.control_socket(root, "stuck"),
                "hpc-grace.closable": self.control_socket(root, "closable"),
            }
            exits: list[str] = []

            def runner(argv: list[str], **kwargs: object):
                if "-G" in argv:
                    return completed(
                        argv, stdout=f"controlpath {paths[argv[-1]]}\n"
                    )
                alias = argv[-1]
                exits.append(alias)
                if alias == "hpc-grace.stuck":
                    raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
                return completed(argv)

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ):
                warnings = retire_compute_masters(
                    ComputeMasterRetirement(tuple(paths), ()),
                    ssh_dir=root,
                    runner=runner,
                )

        self.assertCountEqual(exits, list(paths))
        self.assertEqual(len(warnings), 1)
        self.assertIn(str(paths["hpc-grace.stuck"]), warnings[0])
        self.assertIn("timed out after 10 seconds", warnings[0])

    def test_retirement_propagates_keyboard_interrupt(self) -> None:
        def runner(_argv: list[str], **_kwargs: object):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            KeyboardInterrupt
        ):
            retire_compute_masters(
                ComputeMasterRetirement(("hpc-grace.dev",), ()),
                ssh_dir=Path(directory),
                runner=runner,
            )

    def test_retirement_propagates_keyboard_interrupt_during_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.control_socket(root, "obsolete")

            def runner(argv: list[str], **_kwargs: object):
                if "-G" in argv:
                    return completed(argv, stdout=f"controlpath {path}\n")
                raise KeyboardInterrupt

            with patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFSOCK,
                    st_uid=os.geteuid(),
                ),
            ), self.assertRaises(KeyboardInterrupt):
                retire_compute_masters(
                    ComputeMasterRetirement(("hpc-grace.dev",), ()),
                    ssh_dir=root,
                    runner=runner,
                )

    def test_ssh_argv_sets_batch_mode_explicitly_for_both_policies(self) -> None:
        self.assertEqual(
            ssh_argv("hpc-grace.login"),
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                "hpc-grace.login",
            ],
        )
        self.assertEqual(
            ssh_argv(
                "hpc-grace.login",
                "true",
                batch=False,
                connect_timeout=30,
                extra_options=("NumberOfPasswordPrompts=1",),
            ),
            [
                "ssh",
                "-o",
                "BatchMode=no",
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "ConnectTimeout=30",
                "--",
                "hpc-grace.login",
                "true",
            ],
        )

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

    def test_interactive_bootstrap_explicitly_disables_batch_mode(self) -> None:
        calls: list[list[str]] = []

        def runner(argv: list[str], **_kwargs: object):
            calls.append(argv)
            if "check" in argv:
                return completed(argv, 1, stderr="no master")
            if "BatchMode=yes" in argv:
                return completed(
                    argv,
                    255,
                    stderr="Permission denied (publickey,keyboard-interactive)",
                )
            return completed(argv)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("hpc_alloc.ssh.can_prompt", return_value=True),
        ):
            self.transport(runner, Path(directory)).bootstrap(
                "grace", AuthMode.INTERACTIVE_BOOTSTRAP
            )

        self.assertEqual(
            calls[-1],
            [
                "ssh",
                "-o",
                "BatchMode=no",
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "ConnectTimeout=30",
                "--",
                "hpc-grace.login",
                "true",
            ],
        )
        self.assertEqual(
            calls[-2],
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "--",
                "hpc-grace.login",
                "true",
            ],
        )

    def test_push_login_explicitly_disables_batch_mode(self) -> None:
        calls: list[tuple[list[str], dict[str, object]]] = []

        def runner(argv: list[str], **kwargs: object):
            calls.append((argv, kwargs))
            return completed(argv)

        with tempfile.TemporaryDirectory() as directory:
            self.transport(runner, Path(directory)).push_login("grace", timeout=42)

        self.assertEqual(len(calls), 1)
        argv, kwargs = calls[0]
        self.assertEqual(
            argv,
            [
                "ssh",
                "-o",
                "BatchMode=no",
                "-o",
                "NumberOfPasswordPrompts=1",
                "-o",
                "ConnectTimeout=30",
                "--",
                "hpc-grace.login",
                "true",
            ],
        )
        self.assertEqual(kwargs["timeout"], 42)
        environment = kwargs["env"]
        self.assertIsInstance(environment, dict)
        self.assertEqual(environment["SSH_ASKPASS_REQUIRE"], "force")
        self.assertEqual(environment["HPC_ALLOC_ASKPASS"], "1")

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
                    if "-O" in argv:
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
                # The master is retired exactly once, and with the drain-safe
                # `-O stop`: `-O exit` would also kill every session multiplexed
                # on it, including another process's interactive shell or
                # in-flight rsync on this same compute alias.
                self.assertEqual(sum("stop" in call for call in calls), 1)
                self.assertEqual(sum("exit" in call for call in calls), 0)

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
