from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.config import Config
from hpc_alloc.errors import ConfigInvalid
from hpc_alloc.models import JobKind
from hpc_alloc.ssh_config import (
    ComputeMasterRetirement,
    allocation_alias,
    atomic_write_600,
    compute_host_key_alias,
    render,
    sync_managed_config,
)


def stanza(text: str, alias: str) -> str:
    marker = f"Host {alias}\n"
    return text.split(marker, 1)[1].split("\n\n", 1)[0]


class SshConfigTests(unittest.TestCase):
    def test_alias_mapping_is_injective_at_cluster_name_boundary(self) -> None:
        self.assertNotEqual(
            allocation_alias("a-b", "c"),
            allocation_alias("a", "b-c"),
        )

    def test_render_quotes_paths_and_skips_jobs_for_removed_clusters(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={"grace": SimpleNamespace(host="grace.example.edu")},
        )
        jobs = [
            SimpleNamespace(
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            ),
            SimpleNamespace(
                cluster="removed",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node02",
            ),
        ]
        text = render(config, jobs, Path('/tmp/home with space/known"hosts'))
        self.assertIn(f"Host {allocation_alias('grace', 'dev')}", text)
        self.assertNotIn(allocation_alias("removed", "dev"), text)
        self.assertIn('UserKnownHostsFile "/tmp/home with space/known\\"hosts"', text)
        self.assertIn("HostKeyAlias hpc-alloc-node.grace.node01", text)
        self.assertIn("ControlPath ~/.ssh/hpc-alloc-e010fd1c-%C", text)

    def test_render_projects_bracketed_config_ip_literals_without_brackets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                '[identity]\nnetid = "ab1234"\n'
                '[cluster.ipv4]\nhost = "[192.0.2.7]"\n'
                '[cluster.ipv6]\nhost = "[2001:0DB8:0000::7]"\n'
                '[cluster.scoped]\nhost = "[FE80:0::1%eth 0]"\n'
            )
            config = Config.load(path)
            text = render(config, [], Path(directory) / "known_hosts")

        ipv4 = stanza(text, "hpc-ipv4.login")
        ipv6 = stanza(text, "hpc-ipv6.login")
        scoped = stanza(text, "hpc-scoped.login")
        self.assertIn("HostName 192.0.2.7", ipv4)
        self.assertIn("HostName 2001:0DB8:0000::7", ipv6)
        self.assertIn('HostName "FE80:0::1%%eth 0"', scoped)
        self.assertNotIn("HostName [", text)

    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH client is unavailable")
    def test_openssh_accepts_rendered_scoped_ipv6_hostname(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={
                "scoped": SimpleNamespace(host="fe80::1%eth 0"),
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ssh_config"
            path.write_text(render(config, [], Path(directory) / "known_hosts"))
            result = subprocess.run(
                ["ssh", "-G", "-F", str(path), "hpc-scoped.login"],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("percent_expand", result.stderr)
        self.assertNotIn("unknown key %", result.stderr)

    def test_compute_identity_uses_cluster_and_physical_node_not_allocation(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={
                "alpha": SimpleNamespace(host="alpha.example.edu"),
                "beta": SimpleNamespace(host="beta.example.edu"),
            },
        )
        jobs = [
            SimpleNamespace(
                cluster="alpha",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            ),
            SimpleNamespace(
                cluster="alpha",
                logical_name="research",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            ),
            SimpleNamespace(
                cluster="beta",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            ),
        ]

        text = render(config, jobs, Path("/tmp/shared-known-hosts"))
        alpha_dev = stanza(text, "hpc-alpha.dev")
        alpha_research = stanza(text, "hpc-alpha.research")
        beta_dev = stanza(text, "hpc-beta.dev")

        self.assertIn("HostKeyAlias hpc-alloc-node.alpha.node01", alpha_dev)
        self.assertIn("HostKeyAlias hpc-alloc-node.alpha.node01", alpha_research)
        self.assertIn("HostKeyAlias hpc-alloc-node.beta.node01", beta_dev)
        self.assertNotEqual(
            compute_host_key_alias("alpha", "node01"),
            compute_host_key_alias("beta", "node01"),
        )
        self.assertIn("ControlPath ~/.ssh/hpc-alloc-8ed3f6ad-%C", alpha_dev)
        self.assertIn("ControlPath ~/.ssh/hpc-alloc-8ed3f6ad-%C", alpha_research)
        self.assertIn("ControlPath ~/.ssh/hpc-alloc-f44e64e7-%C", beta_dev)
        self.assertNotEqual(
            next(line for line in alpha_dev.splitlines() if "ControlPath" in line),
            next(line for line in beta_dev.splitlines() if "ControlPath" in line),
        )

    def test_requeue_moves_physical_host_identity_but_keeps_cluster_socket_namespace(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={"alpha": SimpleNamespace(host="alpha.example.edu")},
        )

        def rendered(node: str) -> str:
            return stanza(
                render(
                    config,
                    [
                        SimpleNamespace(
                            cluster="alpha",
                            logical_name="dev",
                            kind=JobKind.ALLOCATION,
                            current_node=node,
                        )
                    ],
                    Path("/tmp/known-hosts"),
                ),
                "hpc-alpha.dev",
            )

        before, after = rendered("node01"), rendered("node02")
        self.assertIn("HostKeyAlias hpc-alloc-node.alpha.node01", before)
        self.assertIn("HostKeyAlias hpc-alloc-node.alpha.node02", after)
        before_control = next(
            line for line in before.splitlines() if "ControlPath" in line
        )
        after_control = next(
            line for line in after.splitlines() if "ControlPath" in line
        )
        self.assertEqual(before_control, after_control)

    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH client is unavailable")
    def test_effective_control_path_tracks_node_and_is_shared_on_same_node(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={"alpha": SimpleNamespace(host="alpha.example.edu")},
        )

        def job(name: str, node: str) -> object:
            return SimpleNamespace(
                cluster="alpha",
                logical_name=name,
                kind=JobKind.ALLOCATION,
                current_node=node,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def effective(jobs: list[object], alias: str) -> dict[str, str]:
                path = root / f"{alias.rsplit('.', 1)[-1]}.config"
                path.write_text(render(config, jobs, root / "known_hosts"))
                result = subprocess.run(
                    ["ssh", "-G", "-F", str(path), alias],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return {
                    fields[0]: fields[1]
                    for line in result.stdout.splitlines()
                    if len(fields := line.split(None, 1)) == 2
                }

            node01_jobs = [job("dev", "node01"), job("research", "node01")]
            dev = effective(node01_jobs, "hpc-alpha.dev")
            research = effective(node01_jobs, "hpc-alpha.research")
            moved = effective([job("dev", "node02")], "hpc-alpha.dev")

        self.assertEqual(dev["controlpath"], research["controlpath"])
        self.assertNotEqual(dev["controlpath"], moved["controlpath"])

    def test_render_rejects_unsafe_node_before_host_identity_interpolation(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={"alpha": SimpleNamespace(host="alpha.example.edu")},
        )
        job = SimpleNamespace(
            cluster="alpha",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            current_node="node01 HostKeyAlias forged",
        )
        with self.assertRaisesRegex(ConfigInvalid, "unsafe compute-node"):
            render(config, [job], Path("/tmp/known-hosts"))

    @unittest.skipUnless(shutil.which("ssh"), "OpenSSH client is unavailable")
    def test_openssh_effective_config_keeps_cross_cluster_keys_and_masters_distinct(self) -> None:
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            ssh=SimpleNamespace(identity_file=None),
            clusters={
                "alpha": SimpleNamespace(host="alpha.example.edu"),
                "beta": SimpleNamespace(host="beta.example.edu"),
            },
        )
        jobs = [
            SimpleNamespace(
                cluster=cluster,
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            )
            for cluster in ("alpha", "beta")
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ssh_config"
            path.write_text(render(config, jobs, Path(directory) / "known_hosts"))

            def effective(alias: str) -> dict[str, str]:
                result = subprocess.run(
                    ["ssh", "-G", "-F", str(path), alias],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return {
                    fields[0]: fields[1]
                    for line in result.stdout.splitlines()
                    if len(fields := line.split(None, 1)) == 2
                }

            alpha = effective("hpc-alpha.dev")
            beta = effective("hpc-beta.dev")

        self.assertEqual(alpha["hostname"], "node01")
        self.assertEqual(beta["hostname"], "node01")
        self.assertEqual(alpha["hostkeyalias"], "hpc-alloc-node.alpha.node01")
        self.assertEqual(beta["hostkeyalias"], "hpc-alloc-node.beta.node01")
        self.assertNotEqual(alpha["controlpath"], beta["controlpath"])
        self.assertIn("hpc-alloc-8ed3f6ad-", alpha["controlpath"])
        self.assertIn("hpc-alloc-f44e64e7-", beta["controlpath"])

    def test_projection_reloads_authoritative_inputs_under_secure_lock(self) -> None:
        class Repository:
            def __init__(self) -> None:
                self.jobs: list[object] = []

            def list_jobs(self, *, include_final: bool) -> list[object]:
                self.assertion = include_final
                return list(self.jobs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            managed_path = root / "ssh_config"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[ssh]\n[defaults]\ncluster = "grace"\n'
                '[cluster.grace]\nhost = "grace.example.edu"\n'
            )
            repository = Repository()
            repository.jobs.append(
                SimpleNamespace(
                    cluster="grace",
                    logical_name="dev",
                    kind=JobKind.ALLOCATION,
                    current_node="node01",
                )
            )
            changed = sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            self.assertTrue(changed)
            self.assertIn("Host hpc-grace.dev", managed_path.read_text())
            self.assertFalse(repository.assertion)
            self.assertEqual(os.stat(managed_path).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(lock_path).st_mode & 0o777, 0o600)

    def test_projection_retires_old_aliases_before_replacing_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            managed_path = root / "ssh_config"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            job = SimpleNamespace(
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            )
            repository = SimpleNamespace(
                list_jobs=lambda **_kwargs: [job],
            )
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            job.current_node = "node02"
            observed: list[ComputeMasterRetirement] = []

            def retire(plan: ComputeMasterRetirement) -> None:
                observed.append(plan)
                prior = managed_path.read_text()
                self.assertIn("HostName node01", prior)
                self.assertNotIn("HostName node02", prior)

            changed = sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
                before_replace=retire,
            )

            self.assertTrue(changed)
            self.assertEqual(
                observed,
                [ComputeMasterRetirement(("hpc-grace.dev",), ())],
            )
            replacement = managed_path.read_text()
            self.assertIn("HostName node02", replacement)
            self.assertNotIn("HostName node01", replacement)

    def test_projection_replaces_equal_content_symlink_without_touching_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: [])
            expected_path = root / "expected"
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=expected_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            expected = expected_path.read_text()
            target = root / "untrusted-target"
            target.write_text(expected)
            target.chmod(0o640)
            target_before = (
                target.read_bytes(),
                target.stat().st_mode,
                target.stat().st_ino,
            )
            managed_path = root / "ssh_config"
            managed_path.symlink_to(target)
            observed: list[ComputeMasterRetirement] = []

            changed = sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
                before_replace=observed.append,
            )

            self.assertTrue(changed)
            self.assertFalse(managed_path.is_symlink())
            self.assertEqual(managed_path.read_text(), expected)
            self.assertEqual(
                (target.read_bytes(), target.stat().st_mode, target.stat().st_ino),
                target_before,
            )
            self.assertEqual(observed, [])

    def test_projection_does_not_authorize_retirement_from_symlink_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            jobs = [
                SimpleNamespace(
                    cluster="grace",
                    logical_name="dev",
                    kind=JobKind.ALLOCATION,
                    current_node="node01",
                )
            ]
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: list(jobs))
            old_path = root / "old-projection"
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=old_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            target = root / "untrusted-target"
            target.write_text(old_path.read_text())
            target.chmod(0o640)
            target_before = (
                target.read_bytes(),
                target.stat().st_mode,
                target.stat().st_ino,
            )
            jobs.clear()
            managed_path = root / "ssh_config"
            managed_path.symlink_to(target)
            observed: list[ComputeMasterRetirement] = []

            changed = sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
                before_replace=observed.append,
            )

            self.assertTrue(changed)
            self.assertFalse(managed_path.is_symlink())
            self.assertNotIn("Host hpc-grace.dev", managed_path.read_text())
            self.assertEqual(
                (target.read_bytes(), target.stat().st_mode, target.stat().st_ino),
                target_before,
            )
            self.assertEqual(observed, [])

    def test_equal_content_atomic_write_replaces_hardlink_without_chmodding_peer(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "untrusted-target"
            target.write_text("same content\n")
            target.chmod(0o640)
            managed_path = root / "ssh_config"
            os.link(target, managed_path)
            target_before = (
                target.read_bytes(),
                target.stat().st_mode,
                target.stat().st_ino,
            )

            changed = atomic_write_600(managed_path, "same content\n")

            self.assertTrue(changed)
            self.assertNotEqual(managed_path.stat().st_ino, target.stat().st_ino)
            self.assertEqual(managed_path.read_text(), "same content\n")
            self.assertEqual(managed_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                (target.read_bytes(), target.stat().st_mode, target.stat().st_ino),
                target_before,
            )

    def test_projection_does_not_authorize_retirement_from_hardlinked_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            jobs = [
                SimpleNamespace(
                    cluster="grace",
                    logical_name="dev",
                    kind=JobKind.ALLOCATION,
                    current_node="node01",
                )
            ]
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: list(jobs))
            old_path = root / "old-projection"
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=old_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            old_path.chmod(0o640)
            target_before = (
                old_path.read_bytes(),
                old_path.stat().st_mode,
                old_path.stat().st_ino,
            )
            managed_path = root / "ssh_config"
            os.link(old_path, managed_path)
            jobs.clear()
            observed: list[ComputeMasterRetirement] = []

            changed = sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
                before_replace=observed.append,
            )

            self.assertTrue(changed)
            self.assertNotEqual(managed_path.stat().st_ino, old_path.stat().st_ino)
            self.assertNotIn("Host hpc-grace.dev", managed_path.read_text())
            self.assertEqual(
                (
                    old_path.read_bytes(),
                    old_path.stat().st_mode,
                    old_path.stat().st_ino,
                ),
                target_before,
            )
            self.assertEqual(observed, [])

    def test_projection_plan_preserves_unchanged_alias_that_may_share_a_master(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            managed_path = root / "ssh_config"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            jobs = [
                SimpleNamespace(
                    cluster="grace",
                    logical_name=name,
                    kind=JobKind.ALLOCATION,
                    current_node="node01",
                )
                for name in ("dev", "research")
            ]
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: list(jobs))
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            jobs.pop(0)
            observed: list[ComputeMasterRetirement] = []
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
                before_replace=observed.append,
            )

        self.assertEqual(
            observed,
            [
                ComputeMasterRetirement(
                    ("hpc-grace.dev", "hpc-grace.research"),
                    ("hpc-grace.research",),
                )
            ],
        )

    def test_projection_does_not_trust_duplicate_or_tampered_old_stanzas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            managed_path = root / "ssh_config"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            job = SimpleNamespace(
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            )
            jobs = [job]
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: list(jobs))
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            valid = managed_path.read_text()
            compute_block = (
                "Host hpc-grace.dev\n" + stanza(valid, "hpc-grace.dev")
            )
            cases = {
                "duplicate-directive": valid.replace(
                    compute_block,
                    compute_block.replace(
                        "    ControlMaster auto\n",
                        "    ControlMaster auto\n    ControlMaster auto\n",
                    ),
                ),
                "duplicate-stanza": f"{valid}\n{compute_block}\n",
            }
            jobs.clear()

            for label, prior in cases.items():
                with self.subTest(label=label):
                    managed_path.write_text(prior)
                    observed: list[ComputeMasterRetirement] = []
                    sync_managed_config(
                        config_path=config_path,
                        repository=repository,
                        managed_path=managed_path,
                        lock_path=lock_path,
                        known_hosts=root / "known_hosts",
                        before_replace=observed.append,
                    )
                    self.assertEqual(observed, [])
                    self.assertNotIn("Host hpc-grace.dev", managed_path.read_text())

    def test_retirement_interrupt_leaves_old_projection_for_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            managed_path = root / "ssh_config"
            lock_path = root / ".ssh_config.lock"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            job = SimpleNamespace(
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                current_node="node01",
            )
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: [job])
            sync_managed_config(
                config_path=config_path,
                repository=repository,
                managed_path=managed_path,
                lock_path=lock_path,
                known_hosts=root / "known_hosts",
            )
            job.current_node = None

            with self.assertRaises(KeyboardInterrupt):
                sync_managed_config(
                    config_path=config_path,
                    repository=repository,
                    managed_path=managed_path,
                    lock_path=lock_path,
                    known_hosts=root / "known_hosts",
                    before_replace=lambda _plan: (_ for _ in ()).throw(
                        KeyboardInterrupt()
                    ),
                )

            self.assertIn("Host hpc-grace.dev", managed_path.read_text())

    def test_projection_rejects_a_symlinked_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[ssh]\n[defaults]\ncluster = "grace"\n'
                '[cluster.grace]\nhost = "grace.example.edu"\n'
            )
            target = root / "target"
            target.write_text("")
            lock = root / ".ssh_config.lock"
            lock.symlink_to(target)
            repository = SimpleNamespace(list_jobs=lambda **_kwargs: [])
            with self.assertRaises(ConfigInvalid):
                sync_managed_config(
                    config_path=config_path,
                    repository=repository,
                    managed_path=root / "ssh_config",
                    lock_path=lock,
                    known_hosts=root / "known_hosts",
                )

    def test_projection_rejects_a_hardlinked_lock_before_reading_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            peer = root / "lock-peer"
            peer.write_bytes(b"lock peer bytes")
            peer.chmod(0o640)
            lock = root / ".ssh_config.lock"
            os.link(peer, lock)
            managed = root / "ssh_config"
            managed.write_text("existing projection\n")
            peer_before = (peer.read_bytes(), peer.stat().st_mode, peer.stat().st_ino)
            managed_before = (
                managed.read_bytes(),
                managed.stat().st_mode,
                managed.stat().st_ino,
            )
            list_jobs = Mock(return_value=[])
            repository = SimpleNamespace(list_jobs=list_jobs)

            with self.assertRaisesRegex(ConfigInvalid, "exactly one hard link"):
                sync_managed_config(
                    config_path=config_path,
                    repository=repository,
                    managed_path=managed,
                    lock_path=lock,
                    known_hosts=root / "known_hosts",
                )

            list_jobs.assert_not_called()
            self.assertEqual(
                (peer.read_bytes(), peer.stat().st_mode, peer.stat().st_ino),
                peer_before,
            )
            self.assertEqual(
                (managed.read_bytes(), managed.stat().st_mode, managed.stat().st_ino),
                managed_before,
            )

    def test_projection_rejects_a_foreign_lock_before_reading_or_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.toml"
            config_path.write_text(
                '[identity]\nnetid = "ab1234"\n[cluster.grace]\n'
                'host = "grace.example.edu"\n'
            )
            lock = root / ".ssh_config.lock"
            lock.write_bytes(b"foreign lock bytes")
            lock.chmod(0o640)
            managed = root / "ssh_config"
            managed.write_text("existing projection\n")
            lock_before = (lock.read_bytes(), lock.stat().st_mode, lock.stat().st_ino)
            managed_before = (
                managed.read_bytes(),
                managed.stat().st_mode,
                managed.stat().st_ino,
            )
            list_jobs = Mock(return_value=[])
            repository = SimpleNamespace(list_jobs=list_jobs)
            effective_uid = os.geteuid()

            with (
                patch(
                    "hpc_alloc.locking.os.geteuid",
                    return_value=effective_uid + 1,
                ),
                self.assertRaisesRegex(ConfigInvalid, "owned by the current user"),
            ):
                sync_managed_config(
                    config_path=config_path,
                    repository=repository,
                    managed_path=managed,
                    lock_path=lock,
                    known_hosts=root / "known_hosts",
                )

            list_jobs.assert_not_called()
            self.assertEqual(
                (lock.read_bytes(), lock.stat().st_mode, lock.stat().st_ino),
                lock_before,
            )
            self.assertEqual(
                (managed.read_bytes(), managed.stat().st_mode, managed.stat().st_ino),
                managed_before,
            )

    def test_projection_lock_prevents_a_stale_writer_from_winning(self) -> None:
        snapshot_taken = threading.Event()
        release_first = threading.Event()

        class BlockingSnapshot(list[object]):
            def __iter__(self):
                snapshot_taken.set()
                self.assert_release()
                return super().__iter__()

            @staticmethod
            def assert_release() -> None:
                if not release_first.wait(5):
                    raise AssertionError("timed out waiting to release first projection")

        class Repository:
            def __init__(self, jobs: list[object]) -> None:
                self.jobs = jobs
                self.calls = 0
                self.guard = threading.Lock()

            def list_jobs(self, *, include_final: bool) -> list[object]:
                self.assert_not_final(include_final)
                with self.guard:
                    self.calls += 1
                    snapshot = list(self.jobs)
                    return BlockingSnapshot(snapshot) if self.calls == 1 else snapshot

            @staticmethod
            def assert_not_final(include_final: bool) -> None:
                if include_final:
                    raise AssertionError("projection requested final jobs")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                '[identity]\nnetid = "ab1234"\n[ssh]\n[defaults]\ncluster = "grace"\n'
                '[cluster.grace]\nhost = "grace.example.edu"\n'
            )
            first = SimpleNamespace(
                cluster="grace", logical_name="a", kind=JobKind.ALLOCATION, current_node="node01"
            )
            second = SimpleNamespace(
                cluster="grace", logical_name="b", kind=JobKind.ALLOCATION, current_node="node02"
            )
            repository = Repository([first])
            errors: list[BaseException] = []

            def writer() -> None:
                try:
                    sync_managed_config(
                        config_path=config,
                        repository=repository,
                        managed_path=root / "ssh_config",
                        lock_path=root / ".ssh_config.lock",
                        known_hosts=root / "known_hosts",
                    )
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            old_writer = threading.Thread(target=writer)
            old_writer.start()
            self.assertTrue(snapshot_taken.wait(5))
            repository.jobs.append(second)
            fresh_writer = threading.Thread(target=writer)
            fresh_writer.start()
            time.sleep(0.05)
            self.assertTrue(fresh_writer.is_alive(), "fresh writer did not wait for projection lock")
            release_first.set()
            old_writer.join(5)
            fresh_writer.join(5)
            self.assertFalse(errors)
            text = (root / "ssh_config").read_text()
            self.assertIn("Host hpc-grace.a", text)
            self.assertIn("Host hpc-grace.b", text)


if __name__ == "__main__":
    unittest.main()
