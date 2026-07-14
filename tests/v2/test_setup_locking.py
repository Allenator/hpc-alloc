from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc import commands
from hpc_alloc.commands import cmd_setup, dispatch
from hpc_alloc.errors import ConfigInvalid, StateConflict, StateInvalid
from hpc_alloc.locking import configuration_scope_lock, operation_scope_lock
from hpc_alloc.models import JobKind
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.state import StateRepository


def setup_config(netid: str = "ab1234", cluster: str = "grace", host: str = "grace.example.edu") -> str:
    return commands._render_initial_config(
        netid,
        cluster,
        host,
        "~/.ssh/id_ed25519",
    )


class ConfigurationScopeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "private" / ".scope.lock"

    def test_lock_is_secure_and_released_after_exception(self) -> None:
        self.path.parent.mkdir(mode=0o755)
        self.path.touch(mode=0o666)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            with configuration_scope_lock(self.path, exclusive=False):
                self.assertEqual(self.path.parent.stat().st_mode & 0o777, 0o700)
                self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
                raise RuntimeError("injected")

        with configuration_scope_lock(self.path, exclusive=True):
            pass

    def test_shared_holders_block_exclusive_holder(self) -> None:
        shared_acquired = threading.Event()
        release_shared = threading.Event()
        exclusive_started = threading.Event()
        exclusive_acquired = threading.Event()

        def shared() -> None:
            with configuration_scope_lock(self.path, exclusive=False):
                shared_acquired.set()
                self.assertTrue(release_shared.wait(5))

        def exclusive() -> None:
            exclusive_started.set()
            with configuration_scope_lock(self.path, exclusive=True):
                exclusive_acquired.set()

        shared_thread = threading.Thread(target=shared)
        exclusive_thread = threading.Thread(target=exclusive)
        shared_thread.start()
        self.assertTrue(shared_acquired.wait(5))
        exclusive_thread.start()
        self.assertTrue(exclusive_started.wait(5))
        self.assertFalse(exclusive_acquired.wait(0.1))
        release_shared.set()
        self.assertTrue(exclusive_acquired.wait(5))
        shared_thread.join(5)
        exclusive_thread.join(5)
        self.assertFalse(shared_thread.is_alive())
        self.assertFalse(exclusive_thread.is_alive())

    def test_a_bounded_exclusive_acquire_fails_fast_and_explains(self) -> None:
        """`setup` must never hang silently behind a long-running command.

        Every other command holds this lock *shared* for its entire lifetime --
        hours, for a `run` or `logs -f` -- so `setup`'s blocking exclusive
        acquire produced a dead terminal with no output and no explanation for
        as long as the job ran.  flock also grants no writer preference, so a
        steady trickle of short `status` polls (exactly what an agent driver
        produces) could starve it indefinitely.
        """

        with configuration_scope_lock(self.path, exclusive=False):
            started = time.monotonic()
            with self.assertRaises(StateConflict) as raised:
                with configuration_scope_lock(
                    self.path, exclusive=True, timeout=0.5
                ):
                    raise AssertionError("the exclusive lock must not be granted")
            elapsed = time.monotonic() - started

        self.assertIn("another hpc-alloc command", str(raised.exception))
        self.assertIn(str(self.path), str(raised.exception))
        # Bounded: it gives up promptly rather than waiting out the whole job.
        self.assertLess(elapsed, 5)

    def test_a_bounded_acquire_still_succeeds_once_the_holder_exits(self) -> None:
        release = threading.Event()
        acquired = threading.Event()

        def hold() -> None:
            with configuration_scope_lock(self.path, exclusive=False):
                acquired.set()
                release.wait(5)

        holder = threading.Thread(target=hold)
        holder.start()
        self.addCleanup(holder.join)
        self.assertTrue(acquired.wait(5))

        release.set()
        with configuration_scope_lock(self.path, exclusive=True, timeout=5):
            pass

    def test_symlink_and_special_file_are_rejected(self) -> None:
        self.path.parent.mkdir()
        target = self.path.parent / "target"
        target.touch()
        self.path.symlink_to(target)
        with self.assertRaisesRegex(ConfigInvalid, "configuration-scope lock"):
            with configuration_scope_lock(self.path, exclusive=True):
                pass

        self.path.unlink()
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ConfigInvalid, "not a regular file"):
            with configuration_scope_lock(self.path, exclusive=False):
                pass

    def test_multiply_linked_lock_file_is_rejected(self) -> None:
        self.path.parent.mkdir()
        self.path.touch()
        os.link(self.path, self.path.parent / "second-link")

        with self.assertRaisesRegex(ConfigInvalid, "exactly one hard link"):
            with configuration_scope_lock(self.path, exclusive=True):
                pass


class OperationScopeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "private" / "operation-locks"
        self.operation_id = "a" * 32

    def test_lock_is_secure_stable_and_released_after_exception(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with operation_scope_lock(
                self.path,
                self.operation_id,
                blocking=True,
            ):
                lock_file = self.path / f"{self.operation_id}.lock"
                self.assertEqual(self.path.stat().st_mode & 0o777, 0o700)
                self.assertEqual(lock_file.stat().st_mode & 0o777, 0o600)
                raise RuntimeError("injected")

        lock_file = self.path / f"{self.operation_id}.lock"
        self.assertTrue(lock_file.exists())
        with operation_scope_lock(self.path, self.operation_id, blocking=True):
            pass
        self.assertTrue(lock_file.exists())

    def test_nonblocking_contender_fails_until_owner_releases(self) -> None:
        with operation_scope_lock(self.path, self.operation_id, blocking=True):
            with self.assertRaisesRegex(StateConflict, "active in another"):
                with operation_scope_lock(
                    self.path,
                    self.operation_id,
                    blocking=False,
                ):
                    pass

        with operation_scope_lock(
            self.path,
            self.operation_id,
            blocking=False,
        ):
            pass

    def test_process_crash_releases_lock_without_unlinking_file(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        child = "\n".join(
            (
                "import os",
                "import sys",
                "from pathlib import Path",
                "from hpc_alloc.locking import operation_scope_lock",
                "with operation_scope_lock(Path(sys.argv[1]), sys.argv[2], blocking=True):",
                "    print('locked', flush=True)",
                "    os.read(0, 1)",
            )
        )
        process = subprocess.Popen(
            [sys.executable, "-c", child, str(self.path), self.operation_id],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout is not None
            self.assertEqual(process.stdout.readline().strip(), "locked")
            with self.assertRaisesRegex(StateConflict, "active in another"):
                with operation_scope_lock(
                    self.path,
                    self.operation_id,
                    blocking=False,
                ):
                    pass
        finally:
            process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

        lock_file = self.path / f"{self.operation_id}.lock"
        self.assertTrue(lock_file.exists())
        with operation_scope_lock(
            self.path,
            self.operation_id,
            blocking=False,
        ):
            pass

    def test_invalid_operation_ids_are_rejected_before_filesystem_changes(self) -> None:
        for operation_id in ("A" * 32, "a" * 31, "../" + "a" * 29):
            with self.subTest(operation_id=operation_id):
                with self.assertRaisesRegex(StateConflict, "32 lowercase hexadecimal"):
                    with operation_scope_lock(
                        self.path,
                        operation_id,
                        blocking=False,
                    ):
                        pass
        self.assertFalse(self.path.exists())

    def test_symlink_and_multiply_linked_lock_files_are_rejected(self) -> None:
        self.path.mkdir(parents=True)
        lock_file = self.path / f"{self.operation_id}.lock"
        target = self.path / "target"
        target.touch()
        lock_file.symlink_to(target)
        with self.assertRaises(StateInvalid):
            with operation_scope_lock(
                self.path,
                self.operation_id,
                blocking=False,
            ):
                pass

        lock_file.unlink()
        lock_file.touch()
        os.link(lock_file, self.path / "second-link")
        with self.assertRaisesRegex(StateInvalid, "exactly one hard link"):
            with operation_scope_lock(
                self.path,
                self.operation_id,
                blocking=False,
            ):
                pass


class DispatchConfigurationLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = AppPaths.for_home(Path(self.directory.name))

    def test_stateful_dispatch_locks_before_context_and_through_handler(self) -> None:
        events: list[str] = []

        @contextmanager
        def fake_lock(path: Path, *, exclusive: bool):
            self.assertEqual(path, self.paths.config_scope_lock)
            self.assertFalse(exclusive)
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")

        def load(_args: object, _paths: object) -> object:
            events.append("load")
            return object()

        def handler(_args: object, **_kwargs: object) -> int:
            events.append("handler")
            return 7

        args = SimpleNamespace(command_name="status")
        with (
            patch.object(commands.AppPaths, "for_home", return_value=self.paths),
            patch.object(commands, "configuration_scope_lock", side_effect=fake_lock),
            patch.object(commands, "_load_context", side_effect=load),
            patch.dict(commands.HANDLERS, {"status": handler}),
        ):
            self.assertEqual(dispatch(args, entrypoint=Path("/tmp/hpc-alloc")), 7)
        self.assertEqual(events, ["lock-enter", "load", "handler", "lock-exit"])

    def test_config_and_dry_runs_skip_scope_lock(self) -> None:
        def handler(_args: object, **_kwargs: object) -> int:
            return 0

        for args in (
            SimpleNamespace(command_name="config", cluster=None),
            SimpleNamespace(command_name="up", cluster=None, dry_run=True),
            SimpleNamespace(command_name="run", cluster=None, dry_run=True),
        ):
            with self.subTest(command=args.command_name):
                with (
                    patch.object(commands.AppPaths, "for_home", return_value=self.paths),
                    patch.object(commands, "configuration_scope_lock") as lock,
                    patch.object(commands, "_load_context", return_value=object()),
                    patch.dict(commands.HANDLERS, {args.command_name: handler}),
                ):
                    self.assertEqual(dispatch(args, entrypoint=Path("/tmp/hpc-alloc")), 0)
                lock.assert_not_called()


class SetupScopeSafetyTests(unittest.TestCase):
    def make_paths(self) -> AppPaths:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paths = AppPaths.for_home(Path(directory.name))
        paths.config_dir.mkdir(parents=True)
        return paths

    @staticmethod
    def args(**changes: object) -> SimpleNamespace:
        values = {
            "netid": "ab1234",
            "cluster": "grace",
            "host": "grace.example.edu",
            "force": True,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    @staticmethod
    def repository(jobs: list[object], operations: list[object]) -> Mock:
        repository = Mock()
        repository.initialize.return_value = repository
        repository.snapshot_setup_scope_blockers.return_value = (jobs, operations)
        return repository

    def test_invalid_candidate_does_not_create_application_files(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paths = AppPaths.for_home(Path(directory.name))

        with self.assertRaises(ConfigInvalid):
            cmd_setup(
                self.args(netid="!"),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertFalse(paths.config_dir.exists())
        self.assertFalse(paths.ssh_dir.exists())

    def test_blockers_reject_scope_changes_before_setup_mutations(self) -> None:
        cases = (
            (self.args(netid="cd5678"), "NetID", "valid"),
            (self.args(host="other.example.edu"), "host", "valid"),
            (
                self.args(cluster="beta", host="beta.example.edu"),
                "remove",
                "valid",
            ),
            (self.args(), "missing or invalid", "invalid"),
            (self.args(), "missing or invalid", "missing"),
        )
        for args, message, prior_kind in cases:
            with self.subTest(message=message, prior=prior_kind):
                paths = self.make_paths()
                if prior_kind == "valid":
                    paths.config_file.write_text(setup_config(), encoding="utf-8")
                elif prior_kind == "invalid":
                    paths.config_file.write_text("invalid = [", encoding="utf-8")
                original_config = (
                    paths.config_file.read_bytes()
                    if paths.config_file.exists()
                    else None
                )
                paths.state_db.write_bytes(b"unchanged-state")
                paths.managed_ssh_config.write_text("unchanged projection\n")
                paths.ssh_dir.mkdir()
                paths.user_ssh_config.write_text("unchanged ssh config\n")
                job = SimpleNamespace(operation_id="a" * 32, cluster="grace")
                operation = SimpleNamespace(operation_id="b" * 32, cluster="grace")
                repository = self.repository([job], [operation])

                with (
                    patch("hpc_alloc.state.StateRepository", return_value=repository),
                    patch.object(commands, "_find_or_create_ssh_key") as key,
                ):
                    with self.assertRaisesRegex(StateConflict, message) as raised:
                        cmd_setup(
                            args,
                            paths=paths,
                            entrypoint=Path("/tmp/hpc-alloc"),
                        )

                key.assert_not_called()
                self.assertIn("a" * 32, str(raised.exception))
                self.assertIn("b" * 32, str(raised.exception))
                self.assertIn("hpc-alloc recover", str(raised.exception))
                self.assertEqual(
                    paths.config_file.read_bytes()
                    if paths.config_file.exists()
                    else None,
                    original_config,
                )
                self.assertEqual(paths.state_db.read_bytes(), b"unchanged-state")
                self.assertEqual(
                    paths.managed_ssh_config.read_text(),
                    "unchanged projection\n",
                )
                self.assertEqual(
                    paths.user_ssh_config.read_text(),
                    "unchanged ssh config\n",
                )

    def test_real_active_job_blocks_scope_change_without_mutating_setup_artifacts(self) -> None:
        paths = self.make_paths()
        paths.config_file.write_text(setup_config(), encoding="utf-8")
        paths.ssh_dir.mkdir()
        private = paths.ssh_dir / "id_ed25519"
        public = paths.ssh_dir / "id_ed25519.pub"
        private.write_text("private\n")
        public.write_text("ssh-ed25519 AAAATEST user@test\n")
        paths.managed_ssh_config.write_text("unchanged projection\n")
        paths.user_ssh_config.write_text("unchanged ssh config\n")
        repository = StateRepository(
            paths.state_db,
            machine_id_factory=lambda: "deadbeef1234",
        ).initialize()
        owner = repository.get_or_create_machine_id("laptop")
        operation_id = "a" * 32
        repository.reserve_submission(
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=slurm_job_name("allocation", operation_id),
            slurm_comment=format_tag(
                owner, operation_id, "laptop", "allocation", "dev"
            ),
            operation_id=operation_id,
        )
        repository.acknowledge_submission(operation_id, "12345")
        artifacts = {
            path: path.read_bytes()
            for path in (
                paths.config_file,
                paths.state_db,
                paths.managed_ssh_config,
                paths.user_ssh_config,
                private,
                public,
            )
        }

        with self.assertRaisesRegex(StateConflict, "NetID"):
            cmd_setup(
                self.args(netid="cd5678"),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        for path, content in artifacts.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_jobs_and_unresolved_operations_independently_protect_scope(self) -> None:
        blockers = (
            ([SimpleNamespace(operation_id="a" * 32, cluster="grace")], []),
            ([], [SimpleNamespace(operation_id="b" * 32, cluster="grace")]),
        )
        for jobs, operations in blockers:
            with self.subTest(jobs=bool(jobs), operations=bool(operations)):
                paths = self.make_paths()
                paths.config_file.write_text(setup_config(), encoding="utf-8")
                repository = self.repository(jobs, operations)
                with patch(
                    "hpc_alloc.state.StateRepository",
                    return_value=repository,
                ):
                    with self.assertRaisesRegex(StateConflict, "NetID"):
                        cmd_setup(
                            self.args(netid="cd5678"),
                            paths=paths,
                            entrypoint=Path("/tmp/hpc-alloc"),
                        )

    def test_scope_comparison_uses_normalized_ip_host(self) -> None:
        _, prior = commands._validated_initial_config(
            "ab1234",
            "grace",
            "[2001:db8::1]",
            None,
        )
        _, candidate = commands._validated_initial_config(
            "ab1234",
            "grace",
            "2001:db8::1",
            None,
        )
        job = SimpleNamespace(operation_id="a" * 32, cluster="grace")

        commands._validate_setup_scope(prior, candidate, [job], [])

    def test_scope_comparison_normalizes_dns_but_preserves_ipv6_scope_id(self) -> None:
        job = SimpleNamespace(operation_id="a" * 32, cluster="grace")
        _, dns_prior = commands._validated_initial_config(
            "ab1234", "grace", "Grace.Example.EDU.", None
        )
        _, dns_candidate = commands._validated_initial_config(
            "ab1234", "grace", "grace.example.edu", None
        )
        commands._validate_setup_scope(dns_prior, dns_candidate, [job], [])

        _, expanded = commands._validated_initial_config(
            "ab1234", "grace", "FE80:0:0:0:0:0:0:1%eth0", None
        )
        _, compressed = commands._validated_initial_config(
            "ab1234", "grace", "fe80::1%eth0", None
        )
        commands._validate_setup_scope(expanded, compressed, [job], [])

        different_scopes = (
            ("fe80::1%ETH0", "fe80::1%eth0"),
            ("fe80::1%eth0.", "fe80::1%eth0"),
        )
        for old_host, new_host in different_scopes:
            with self.subTest(old=old_host, new=new_host):
                _, scoped_prior = commands._validated_initial_config(
                    "ab1234", "grace", old_host, None
                )
                _, scoped_candidate = commands._validated_initial_config(
                    "ab1234", "grace", new_host, None
                )
                with self.assertRaisesRegex(StateConflict, "change the host"):
                    commands._validate_setup_scope(
                        scoped_prior, scoped_candidate, [job], []
                    )

    def test_same_scope_force_is_allowed_with_blockers(self) -> None:
        paths = self.make_paths()
        paths.config_file.write_text(setup_config(), encoding="utf-8")
        paths.ssh_dir.mkdir()
        private = paths.ssh_dir / "id_ed25519"
        public = paths.ssh_dir / "id_ed25519.pub"
        private.write_text("private\n")
        public.write_text("ssh-ed25519 AAAATEST user@test\n")
        job = SimpleNamespace(operation_id="a" * 32, cluster="grace")
        operation = SimpleNamespace(operation_id="b" * 32, cluster="grace")
        repository = self.repository([job], [operation])

        with (
            patch("hpc_alloc.state.StateRepository", return_value=repository),
            patch.object(commands, "sync_managed_config", return_value=True) as projection,
            patch.object(commands, "ensure_include", return_value=True) as include,
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = cmd_setup(
                self.args(),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        repository.get_or_create_machine_id.assert_called_once()
        projection.assert_called_once()
        include.assert_called_once_with(paths.user_ssh_config)
        self.assertIn('netid = "ab1234"', paths.config_file.read_text())

    def test_cleared_blockers_allow_scope_change(self) -> None:
        paths = self.make_paths()
        paths.config_file.write_text(setup_config(), encoding="utf-8")
        paths.ssh_dir.mkdir()
        private = paths.ssh_dir / "id_ed25519"
        public = paths.ssh_dir / "id_ed25519.pub"
        private.write_text("private\n")
        public.write_text("ssh-ed25519 AAAATEST user@test\n")
        repository = self.repository([], [])

        with (
            patch("hpc_alloc.state.StateRepository", return_value=repository),
            patch.object(commands, "sync_managed_config", return_value=True),
            patch.object(commands, "ensure_include", return_value=True),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            result = cmd_setup(
                self.args(
                    netid="cd5678",
                    cluster="beta",
                    host="beta.example.edu",
                ),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        configured = paths.config_file.read_text()
        self.assertIn('netid = "cd5678"', configured)
        self.assertIn("[cluster.beta]", configured)


if __name__ == "__main__":
    unittest.main()
