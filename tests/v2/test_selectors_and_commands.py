from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.commands import _job_log_path, _remote_command, machine_host
from hpc_alloc.errors import IdentityMismatch
from hpc_alloc.models import JobKind, JobPhase
from hpc_alloc.selectors import (
    Selector,
    SelectorKind,
    canonical_job_selector,
    parse_selector,
    unique_job,
)


class SelectorAndCommandTests(unittest.TestCase):
    def test_qualified_selector_must_agree_with_explicit_cluster(self) -> None:
        self.assertEqual(parse_selector("grace:dev"), Selector("grace", "dev"))
        with self.assertRaisesRegex(IdentityMismatch, "conflicting"):
            parse_selector("grace:dev", "bouchet")

    def test_operation_selector_is_explicit_and_strict(self) -> None:
        operation_id = "a" * 32
        selector = parse_selector(f"grace:@{operation_id}")
        self.assertEqual(selector.kind, SelectorKind.OPERATION_ID)
        self.assertEqual(selector.value, f"@{operation_id}")
        with self.assertRaisesRegex(IdentityMismatch, "32 lowercase"):
            parse_selector("@ABC")

    def test_cross_cluster_and_same_cluster_ambiguity_have_actionable_remedies(self) -> None:
        jobs = [
            SimpleNamespace(cluster="grace", logical_name="dev", job_id="1", operation_id="a" * 32),
            SimpleNamespace(cluster="bouchet", logical_name="dev", job_id="2", operation_id="b" * 32),
        ]
        with self.assertRaisesRegex(IdentityMismatch, "grace:@"):
            unique_job(jobs, Selector(None, "dev"))

        same_cluster = [
            SimpleNamespace(cluster="grace", logical_name="run", job_id="1", operation_id="c" * 32),
            SimpleNamespace(cluster="grace", logical_name="run", job_id="2", operation_id="d" * 32),
        ]
        with self.assertRaisesRegex(IdentityMismatch, "grace:@"):
            unique_job(same_cluster, Selector("grace", "run"))

    def test_numeric_prefers_live_and_operation_selector_addresses_history(self) -> None:
        old = SimpleNamespace(
            cluster="grace",
            logical_name="run",
            job_id="111",
            operation_id="a" * 32,
            phase=JobPhase.FINAL,
        )
        live = SimpleNamespace(
            cluster="grace",
            logical_name="run",
            job_id="111",
            operation_id="b" * 32,
            phase=JobPhase.ACTIVE,
        )
        self.assertIs(unique_job([old, live], parse_selector("grace:111")), live)
        self.assertIs(unique_job([old, live], parse_selector("grace:@" + "a" * 32)), old)
        self.assertEqual(canonical_job_selector(old), "grace:@" + "a" * 32)

    def test_remote_command_preserves_shell_string_or_quotes_exact_argv(self) -> None:
        self.assertEqual(_remote_command(["cd ~/project && make test"]), "cd ~/project && make test")
        self.assertEqual(
            _remote_command(["python", "-c", "print('a b')"]),
            "python -c 'print('\"'\"'a b'\"'\"')'",
        )

    def test_machine_host_always_satisfies_ownership_grammar(self) -> None:
        with patch("hpc_alloc.commands.platform.node", return_value="_💻.example"):
            label = machine_host()
        self.assertRegex(label, r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

    def test_log_paths_are_derived_from_operation_not_recyclable_job_id(self) -> None:
        first = SimpleNamespace(kind=JobKind.RUN, operation_id="a" * 32, job_id="111")
        second = SimpleNamespace(kind=JobKind.RUN, operation_id="b" * 32, job_id="111")
        self.assertNotEqual(_job_log_path(first), _job_log_path(second))
        self.assertEqual(_job_log_path(first), ".hpc-alloc/run-" + "a" * 32 + ".log")


if __name__ == "__main__":
    unittest.main()
