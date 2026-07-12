from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from hpc_alloc.config import Config
from hpc_alloc.context import RuntimeContext
from hpc_alloc.errors import ConfigInvalid, StateConflict, StateInvalid, TransportLost
from hpc_alloc.models import FinalSource, JobKind, JobPhase, OperationPhase
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.state import SCHEMA_VERSION, StateRepository


VALID_CONFIG = """\
[identity]
netid = "ab1234"

[ssh]
identity_file = "~/.ssh/id_ed25519"

[defaults]
cluster = "alpha"
partition = "day"
cpus = 2
idle_timeout = 0

[cluster.alpha]
host = "alpha.example.edu"

[cluster.beta]
host = "10.0.0.2"
partition = "week"
"""

SUBMIT_ID = "a" * 32
SECOND_ID = "b" * 32
THIRD_ID = "c" * 32
FOURTH_ID = "d" * 32
CANCEL_ID = "e" * 32
SECOND_CANCEL_ID = "f" * 32


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.toml"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, text: str = VALID_CONFIG) -> None:
        self.path.write_text(text)

    def test_authoritative_config_and_precedence(self) -> None:
        self.write()
        config = Config.load(self.path)
        self.assertEqual(config.identity.netid, "ab1234")
        self.assertEqual(config.resolve_cluster(), "alpha")
        self.assertEqual(config.resolve_cluster("beta"), "beta")
        self.assertEqual(config.resolve_option("partition", "alpha"), "day")
        self.assertEqual(config.resolve_option("partition", "beta"), "week")
        self.assertEqual(config.resolve_option("time", "beta", fallback="4:00:00"), "4:00:00")

    def test_sole_cluster_is_primary_without_default(self) -> None:
        self.write(
            """[identity]\nnetid='ab1234'\n[cluster.notbouchet]\nhost='h.example.edu'\n"""
        )
        self.assertEqual(Config.load(self.path).resolve_cluster(), "notbouchet")

    def test_multiple_clusters_without_default_are_ambiguous(self) -> None:
        self.write(VALID_CONFIG.replace('cluster = "alpha"\n', ""))
        with self.assertRaisesRegex(ConfigInvalid, "multiple clusters"):
            Config.load(self.path).resolve_cluster()

    def test_missing_or_unregistered_values_fail_closed(self) -> None:
        with self.assertRaises(ConfigInvalid):
            Config.load(self.path)
        self.write(VALID_CONFIG.replace('cluster = "alpha"', 'cluster = "missing"'))
        with self.assertRaisesRegex(ConfigInvalid, "unconfigured"):
            Config.load(self.path)

    def test_unknown_keys_and_control_characters_are_rejected(self) -> None:
        self.write(VALID_CONFIG + "\nunknown = true\n")
        with self.assertRaises(ConfigInvalid):
            Config.load(self.path)
        self.write(VALID_CONFIG.replace("~/.ssh/id_ed25519", "~/.ssh/key\\nHost evil"))
        with self.assertRaises(ConfigInvalid):
            Config.load(self.path)


class StateRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "private" / "state.db"
        self.now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
        self.repo = StateRepository(
            self.path,
            clock=lambda: self.now,
            machine_id_factory=lambda: "owner0001",
        ).initialize()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reserve(self, operation_id: str = SUBMIT_ID, **changes: object):
        values = {
            "cluster": "alpha",
            "logical_name": "dev",
            "kind": JobKind.ALLOCATION,
            "owner_id": "owner0001",
            "resources": {"cpus": 2, "partition": "day"},
            "operation_id": operation_id,
        }
        values.update(changes)
        kind = JobKind(values["kind"])
        values["slurm_job_name"] = slurm_job_name(kind.value, operation_id)
        values["slurm_comment"] = format_tag(
            str(values["owner_id"]),
            operation_id,
            "laptop",
            kind.value,
            str(values["logical_name"]),
        )
        return self.repo.reserve_submission(**values)

    def test_schema_security_and_machine_identity(self) -> None:
        self.assertEqual(stat_mode(self.path), 0o600)
        self.assertEqual(stat_mode(self.path.parent), 0o700)
        self.assertEqual(self.repo.get_or_create_machine_id("laptop"), "owner0001")
        self.assertEqual(self.repo.get_or_create_machine_id("renamed"), "owner0001")
        self.assertEqual(self.repo.get_machine().hostname, "renamed")
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("SELECT schema_version FROM metadata").fetchone()[0], SCHEMA_VERSION)

    def test_submission_ack_and_lifecycle_update(self) -> None:
        operation = self.reserve()
        self.assertEqual(operation.phase, OperationPhase.PREPARED)
        self.assertEqual(len(self.repo.list_unresolved_operations()), 1)
        job = self.repo.acknowledge_submission(operation.operation_id, "1234")
        self.assertEqual(job.phase, JobPhase.QUEUED)
        self.assertEqual(job.ref.job_id, "1234")
        self.assertEqual(job.resources["cpus"], 2)
        self.assertEqual(self.repo.list_unresolved_operations(), [])
        job = self.repo.update_job(
            operation.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )
        self.assertTrue(job.ever_started)
        self.assertEqual(job.current_node, "node01")
        self.assertEqual(job.last_node, "node01")
        # Reprocessing the same acknowledgement must not rewind lifecycle state.
        self.assertEqual(
            self.repo.acknowledge_submission(operation.operation_id, "1234").phase,
            JobPhase.ACTIVE,
        )
        job = self.repo.update_job(operation.operation_id, phase=JobPhase.REQUEUEING)
        self.assertIsNone(job.current_node)
        self.assertEqual(job.last_node, "node01")
        with self.assertRaisesRegex(StateConflict, "monotonic"):
            self.repo.update_job(operation.operation_id, ever_started=False)

    def test_allocation_reservation_is_atomic_and_released_only_when_final(self) -> None:
        barrier = threading.Barrier(2)

        def attempt(operation_id: str) -> str:
            barrier.wait()
            try:
                self.reserve(operation_id)
                return "reserved"
            except StateConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(attempt, (SECOND_ID, THIRD_ID)))
        self.assertCountEqual(outcomes, ["reserved", "conflict"])
        winner = self.repo.find_jobs(include_final=False)[0]
        self.repo.abandon_operation(winner.operation_id, "operator released ambiguous intent")
        self.reserve(FOURTH_ID)

    def test_run_names_do_not_share_allocation_reservation(self) -> None:
        self.reserve(SECOND_ID, kind=JobKind.RUN, logical_name="run")
        self.reserve(THIRD_ID, kind=JobKind.RUN, logical_name="run")
        self.assertEqual(len(self.repo.find_jobs(kind=JobKind.RUN)), 2)

    def test_ambiguous_submission_remains_durable_until_reconciled(self) -> None:
        self.reserve()
        operation = self.repo.mark_submission_ambiguous(SUBMIT_ID, "SSH reply lost")
        self.assertEqual(operation.phase, OperationPhase.AMBIGUOUS)
        self.assertEqual(self.repo.get_job(SUBMIT_ID).phase, JobPhase.SUBMITTING)
        with self.assertRaises(StateConflict):
            self.reserve(SECOND_ID)
        adopted = self.repo.acknowledge_submission(SUBMIT_ID, "777")
        self.assertEqual(adopted.job_id, "777")
        self.assertEqual(self.repo.get_operation(SUBMIT_ID).phase, OperationPhase.ACKNOWLEDGED)

    def test_cancel_intent_is_single_and_dispatch_is_durably_ambiguous(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(SUBMIT_ID, "1234")
        cancel = self.repo.begin_cancel(SUBMIT_ID, operation_id=CANCEL_ID)
        self.assertEqual(cancel.phase, OperationPhase.CANCEL_PENDING)
        with self.assertRaises(StateConflict):
            self.repo.begin_cancel(SUBMIT_ID, operation_id=SECOND_CANCEL_ID)
        cancel = self.repo.mark_cancel_dispatching(CANCEL_ID)
        self.assertEqual(cancel.phase, OperationPhase.AMBIGUOUS)
        with self.assertRaises(StateConflict):
            self.repo.begin_cancel(SUBMIT_ID, operation_id=SECOND_CANCEL_ID)
        cancel = self.repo.mark_cancel_ambiguous(CANCEL_ID, "connection lost")
        self.assertEqual(cancel.phase, OperationPhase.AMBIGUOUS)
        self.repo.resolve_operation(
            CANCEL_ID,
            final_source=FinalSource.ACCOUNTING,
            terminal_state="CANCELLED",
            exit_code="0:15",
        )
        job = self.repo.get_job(SUBMIT_ID)
        self.assertEqual(job.phase, JobPhase.FINAL)
        self.assertEqual(job.terminal_state, "CANCELLED")
        self.assertEqual(job.exit_code, "0:15")

    def test_find_jobs_supports_selector_resolution(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(SUBMIT_ID, "1234")
        self.assertEqual(self.repo.find_jobs(cluster="alpha", logical_name="dev")[0].operation_id, SUBMIT_ID)
        self.assertEqual(self.repo.find_jobs(job_id="1234")[0].operation_id, SUBMIT_ID)
        self.assertEqual(self.repo.find_jobs(cluster="beta"), [])

    def test_cluster_cache_expires_without_conflating_zero_and_missing(self) -> None:
        self.repo.set_cluster_cache(
            "alpha", "remote_home", {"path": "/home/me", "size": 0}, expires_at=self.now + timedelta(minutes=5)
        )
        self.assertEqual(self.repo.get_cluster_cache("alpha", "remote_home")["size"], 0)
        self.now += timedelta(minutes=6)
        self.assertIsNone(self.repo.get_cluster_cache("alpha", "remote_home"))

    def test_wrong_schema_and_symlink_are_rejected(self) -> None:
        bad = Path(self.tmp.name) / "bad.db"
        with closing(sqlite3.connect(bad)) as connection:
            connection.execute("CREATE TABLE metadata(schema_version INTEGER)")
            connection.execute("INSERT INTO metadata VALUES(1)")
            connection.commit()
        with self.assertRaises(StateInvalid):
            StateRepository(bad).initialize()
        target = Path(self.tmp.name) / "target.db"
        link = Path(self.tmp.name) / "link.db"
        target.touch()
        link.symlink_to(target)
        with self.assertRaises(StateInvalid):
            StateRepository(link).initialize()


class RuntimeContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.paths = AppPaths.for_home(self.home)
        self.paths.config_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_operational_context_resolves_primary_and_initializes_state(self) -> None:
        self.paths.config_file.write_text(VALID_CONFIG)
        context = RuntimeContext.load(command="connect", paths=self.paths)
        self.assertEqual(context.cluster_name, "alpha")
        self.assertEqual(context.cluster.host, "alpha.example.edu")
        self.assertTrue(self.paths.state_db.exists())

    def test_recovery_command_exposes_config_error_without_creating_state(self) -> None:
        self.paths.config_file.write_text("invalid = [")
        context = RuntimeContext.load(command="config", paths=self.paths)
        self.assertIsNone(context.config)
        self.assertIsInstance(context.config_error, ConfigInvalid)
        self.assertFalse(self.paths.state_db.exists())
        with self.assertRaises(ConfigInvalid):
            RuntimeContext.load(command="status", paths=self.paths)

    def test_transport_errors_have_reconnect_exit_code(self) -> None:
        self.assertEqual(TransportLost.exit_code, 3)


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
