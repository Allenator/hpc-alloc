from __future__ import annotations

import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hpc_alloc.errors import (
    JobIdReused,
    LifecycleRevisionConflict,
    StateConflict,
    StateInvalid,
    TransportLost,
)
from hpc_alloc.lifecycle import AssessmentPhase, EvidenceEvent, EvidenceTracker
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    OperationPhase,
)
from hpc_alloc.monitor import JobMonitor, persist_assessment
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.state import SCHEMA_VERSION, StateRepository


class StateRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.db"
        self.repo = StateRepository(
            self.path,
            machine_id_factory=lambda: "deadbeef1234",
        ).initialize()
        self.owner = self.repo.get_or_create_machine_id("laptop")
        self.operation_id = "a" * 32

    def reserve(self, *, operation_id: str | None = None, name: str = "dev"):
        operation_id = operation_id or self.operation_id
        return self.repo.reserve_submission(
            cluster="grace",
            logical_name=name,
            kind=JobKind.ALLOCATION,
            owner_id=self.owner,
            slurm_job_name=slurm_job_name("allocation", operation_id),
            slurm_comment=format_tag(
                self.owner, operation_id, "laptop", "allocation", name
            ),
            resources={"cpus": 2, "partition": "day"},
            operation_id=operation_id,
        )

    def test_get_or_create_machine_id_rejects_corrupt_stored_identity(self) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE machine SET machine_id = ? WHERE singleton = 1",
                ("invalid machine ID",),
            )
            connection.commit()

        with self.assertRaisesRegex(StateInvalid, "machine record is malformed"):
            self.repo.get_or_create_machine_id("renamed-laptop")

    def test_ambiguous_submission_remains_durable_and_blocks_duplicate(self) -> None:
        prepared = self.reserve()
        self.assertEqual(prepared.phase, OperationPhase.PREPARED)

        ambiguous = self.repo.mark_submission_ambiguous(
            self.operation_id,
            "sbatch reply lost after possible commit",
        )
        self.assertEqual(ambiguous.phase, OperationPhase.AMBIGUOUS)
        self.assertIn("possible commit", ambiguous.detail or "")
        self.assertEqual(self.repo.get_job(self.operation_id).phase, JobPhase.SUBMITTING)
        self.assertEqual(
            [op.operation_id for op in self.repo.list_unresolved_operations()],
            [self.operation_id],
        )

        with self.assertRaisesRegex(StateConflict, "already has a non-final job"):
            self.reserve(operation_id="b" * 32)

        self.repo.abandon_operation(self.operation_id, "operator verified no remote job")
        self.assertEqual(self.repo.get_job(self.operation_id).phase, JobPhase.FINAL)
        replacement = self.reserve(operation_id="b" * 32)
        self.assertEqual(replacement.phase, OperationPhase.PREPARED)

    def test_remote_multiline_detail_is_safely_persisted_not_rejected(self) -> None:
        self.reserve()
        operation = self.repo.mark_submission_ambiguous(
            self.operation_id,
            "connection lost\nREMOTE HOST\x1b[31m changed",
        )
        self.assertEqual(
            operation.detail,
            "connection lost REMOTE HOST [31m changed",
        )

    def test_acknowledgement_is_idempotent_only_for_same_job_id(self) -> None:
        self.reserve()
        first = self.repo.acknowledge_submission(self.operation_id, "12345")
        self.assertEqual(first.job_id, "12345")
        self.assertEqual(first.phase, JobPhase.QUEUED)
        again = self.repo.acknowledge_submission(self.operation_id, "12345")
        self.assertEqual(again.job_id, "12345")
        with self.assertRaisesRegex(StateConflict, "different Slurm job"):
            self.repo.acknowledge_submission(self.operation_id, "54321")

    def test_setup_scope_snapshot_excludes_final_jobs_and_resolved_operations(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        blocker_id = "b" * 32
        self.reserve(operation_id=blocker_id, name="next")

        jobs, operations = self.repo.snapshot_setup_scope_blockers()

        self.assertEqual([job.operation_id for job in jobs], [blocker_id])
        self.assertEqual([operation.operation_id for operation in operations], [blocker_id])

    def test_submission_identity_must_be_internally_exact(self) -> None:
        with self.assertRaisesRegex(StateConflict, "identity metadata is inconsistent"):
            self.repo.reserve_submission(
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                owner_id=self.owner,
                slurm_job_name=slurm_job_name("allocation", self.operation_id),
                slurm_comment=format_tag(
                    self.owner,
                    self.operation_id,
                    "laptop",
                    "allocation",
                    "different-name",
                ),
                operation_id=self.operation_id,
            )

    def test_cancel_is_marked_ambiguous_before_dispatch_and_stays_unique(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        cancel = self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.assertEqual(cancel.phase, OperationPhase.CANCEL_PENDING)
        with self.assertRaisesRegex(StateConflict, "pending cancellation"):
            self.repo.begin_cancel(self.operation_id, operation_id="d" * 32)

        with self.assertRaisesRegex(StateConflict, "not dispatched ambiguously"):
            self.repo.mark_cancel_ambiguous(cancel_id, "cannot skip dispatch boundary")

        dispatching = self.repo.mark_cancel_dispatching(cancel_id)
        self.assertEqual(dispatching.phase, OperationPhase.AMBIGUOUS)
        self.assertIn("dispatch started", dispatching.detail or "")
        with self.assertRaisesRegex(StateConflict, "pending cancellation"):
            self.repo.begin_cancel(self.operation_id, operation_id="d" * 32)

        retained = self.repo.mark_cancel_ambiguous(
            cancel_id, "connection dropped mid-scancel"
        )
        self.assertEqual(retained.phase, OperationPhase.AMBIGUOUS)
        self.assertIn("mid-scancel", retained.detail or "")
        self.assertEqual(self.repo.get_job(self.operation_id).phase, JobPhase.QUEUED)

    def test_cancel_dispatch_atomically_persists_live_start_evidence(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)

        dispatching = self.repo.mark_cancel_dispatching(
            cancel_id,
            ever_started=True,
            last_node="node01",
            observation_epoch=7,
        )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(dispatching.phase, OperationPhase.AMBIGUOUS)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node01")
        self.assertEqual(stored.observation_epoch, 7)

    def test_cancel_dispatch_atomically_persists_non_live_provenance(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)

        self.repo.mark_cancel_dispatching(
            cancel_id,
            ever_started=True,
            last_node="node02",
            observation_epoch=8,
            evidence_provenance=EvidenceProvenance.QUEUE_TERMINAL,
            evidence_detail="exact queue row was terminal-looking",
        )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node02")
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.QUEUE_TERMINAL,
        )
        self.assertEqual(
            stored.evidence_detail,
            "exact queue row was terminal-looking",
        )

    def test_cancel_dispatch_rolls_back_operation_when_evidence_merge_fails(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """CREATE TRIGGER inject_cancel_dispatch_failure
                   BEFORE UPDATE OF ever_started ON jobs
                   BEGIN SELECT RAISE(ABORT, 'injected dispatch failure'); END"""
            )
            connection.commit()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected dispatch failure"):
            self.repo.mark_cancel_dispatching(
                cancel_id,
                ever_started=True,
                last_node="node01",
            )

        self.assertEqual(
            self.repo.get_operation(cancel_id).phase,
            OperationPhase.CANCEL_PENDING,
        )
        stored = self.repo.get_job(self.operation_id)
        self.assertFalse(stored.ever_started)
        self.assertIsNone(stored.last_node)

    def test_cancel_dispatch_revision_conflict_rolls_back_and_preserves_newer_node(self) -> None:
        self.reserve()
        queued = self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        newer = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )

        with self.assertRaises(LifecycleRevisionConflict):
            self.repo.mark_cancel_dispatching(
                cancel_id,
                expected_target_updated_at=queued.updated_at,
                ever_started=True,
                last_node="node01",
            )

        self.assertEqual(
            self.repo.get_operation(cancel_id).phase,
            OperationPhase.CANCEL_PENDING,
        )
        self.assertEqual(self.repo.get_job(self.operation_id), newer)

    def test_ever_started_is_monotonic_and_inactive_phase_clears_current_node(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        active = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )
        self.assertTrue(active.ever_started)
        self.assertEqual(active.current_node, "node01")
        self.assertEqual(active.last_node, "node01")

        suspended = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.STARTED_INACTIVE,
        )
        self.assertIsNone(suspended.current_node)
        self.assertEqual(suspended.last_node, "node01")
        with self.assertRaisesRegex(StateConflict, "monotonic"):
            self.repo.update_job(self.operation_id, ever_started=False)

    def test_update_job_rejects_a_stale_expected_timestamp(self) -> None:
        self.reserve()
        queued = self.repo.acknowledge_submission(self.operation_id, "12345")
        active = self.repo.update_job(
            self.operation_id,
            expected_updated_at=queued.updated_at,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )

        with self.assertRaisesRegex(LifecycleRevisionConflict, "changed while.*rerun"):
            self.repo.update_job(
                self.operation_id,
                expected_updated_at=queued.updated_at,
                phase=JobPhase.REQUEUEING,
            )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored, active)

    def test_semantic_noop_preserves_revision_token(self) -> None:
        self.reserve()
        queued = self.repo.acknowledge_submission(self.operation_id, "12345")
        active = self.repo.update_job(
            self.operation_id,
            expected_updated_at=queued.updated_at,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )

        unchanged = self.repo.update_job(
            self.operation_id,
            expected_updated_at=active.updated_at,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )

        self.assertEqual(unchanged, active)
        updated = self.repo.update_job(
            self.operation_id,
            expected_updated_at=active.updated_at,
            phase=JobPhase.REQUEUEING,
        )
        self.assertGreater(updated.updated_at, active.updated_at)

    def test_non_database_bytes_raise_typed_state_error(self) -> None:
        path = Path(self.directory.name) / "corrupt.db"
        path.write_bytes(b"\xffnot-a-sqlite-database")
        with self.assertRaises(StateInvalid):
            StateRepository(path).initialize()

    def test_wrong_database_and_schema_versions_are_rejected(self) -> None:
        unrelated = Path(self.directory.name) / "unrelated.db"
        with closing(sqlite3.connect(unrelated)) as connection:
            connection.execute("CREATE TABLE somebody_elses_state(value TEXT)")
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "not an hpc-alloc"):
            StateRepository(unrelated).initialize()

        old = Path(self.directory.name) / "old.db"
        with closing(sqlite3.connect(old)) as connection:
            connection.execute("CREATE TABLE metadata(schema_version INTEGER NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES (?)", (SCHEMA_VERSION - 1,))
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "unsupported state schema"):
            StateRepository(old).initialize()

    def test_symlink_and_nonregular_state_paths_are_rejected(self) -> None:
        target = Path(self.directory.name) / "target.db"
        target.write_bytes(b"")
        link = Path(self.directory.name) / "linked.db"
        link.symlink_to(target)
        with self.assertRaisesRegex(StateInvalid, "symbolic link"):
            StateRepository(link).initialize()

        directory_path = Path(self.directory.name) / "directory.db"
        directory_path.mkdir()
        with self.assertRaisesRegex(StateInvalid, "not a regular file"):
            StateRepository(directory_path).initialize()

    def test_corrupt_nested_json_is_reported_as_state_corruption(self) -> None:
        self.reserve()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE jobs SET resources_json = '[]' WHERE operation_id = ?",
                (self.operation_id,),
            )
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "resources contain invalid JSON"):
            self.repo.get_job(self.operation_id)

    def test_malformed_typed_rows_and_duplicate_metadata_are_rejected(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE jobs SET job_id = 'not-numeric' WHERE operation_id = ?",
                (self.operation_id,),
            )
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "job record is malformed"):
            self.repo.get_job(self.operation_id)

        duplicate = Path(self.directory.name) / "duplicate-metadata.db"
        StateRepository(duplicate).initialize()
        with closing(sqlite3.connect(duplicate)) as connection:
            connection.execute("INSERT INTO metadata VALUES (?)", (SCHEMA_VERSION,))
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "unsupported state schema"):
            StateRepository(duplicate).initialize()

    def test_corrupt_persisted_identity_is_never_reconstituted_as_a_job_ref(self) -> None:
        self.reserve()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE jobs SET slurm_comment = 'hpc-alloc:v2:forged' "
                "WHERE operation_id = ?",
                (self.operation_id,),
            )
            connection.commit()
        with self.assertRaisesRegex(StateInvalid, "job record is malformed"):
            self.repo.get_job(self.operation_id)

    def test_fail_submission_is_atomic_and_releases_allocation_name(self) -> None:
        self.reserve()
        operation = self.repo.fail_submission(self.operation_id, "sbatch rejected request")
        job = self.repo.get_job(self.operation_id)

        self.assertEqual(operation.phase, OperationPhase.FAILED)
        self.assertEqual(job.phase, JobPhase.FINAL)
        self.assertEqual(job.terminal_state, "SUBMIT_FAILED")
        self.assertEqual(job.final_source, FinalSource.SUBMIT_FAILED)
        self.assertIsNotNone(job.finalized_at)

        replacement = self.reserve(operation_id="b" * 32)
        self.assertEqual(replacement.phase, OperationPhase.PREPARED)

    def test_fail_submission_rolls_back_operation_job_and_name_release_on_fault(self) -> None:
        self.reserve()
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """CREATE TRIGGER inject_final_failure
                   BEFORE UPDATE OF phase ON jobs
                   WHEN OLD.phase <> 'FINAL' AND NEW.phase = 'FINAL'
                   BEGIN SELECT RAISE(ABORT, 'injected final failure'); END"""
            )
            connection.commit()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected final failure"):
            self.repo.fail_submission(self.operation_id, "sbatch rejected request")

        self.assertEqual(
            self.repo.get_operation(self.operation_id).phase,
            OperationPhase.PREPARED,
        )
        self.assertEqual(self.repo.get_job(self.operation_id).phase, JobPhase.SUBMITTING)
        with self.assertRaisesRegex(StateConflict, "already has a non-final job"):
            self.reserve(operation_id="b" * 32)

        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TRIGGER inject_final_failure")
            connection.commit()
        self.repo.fail_submission(self.operation_id, "sbatch rejected request")
        self.reserve(operation_id="b" * 32)

    def test_cancel_failure_policy_never_changes_target_job(self) -> None:
        self.reserve()
        before = self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)

        failed = self.repo.fail_cancel_operation(cancel_id, "identity guard rejected row")
        after = self.repo.get_job(self.operation_id)

        self.assertEqual(failed.phase, OperationPhase.FAILED)
        self.assertEqual(after.phase, before.phase)
        self.assertEqual(after.job_id, before.job_id)
        self.assertIsNone(after.final_source)
        # A definitive failure closes only the intent, so a later explicit
        # attempt may reserve a new cancellation operation.
        retry = self.repo.begin_cancel(self.operation_id, operation_id="d" * 32)
        self.assertEqual(retry.phase, OperationPhase.CANCEL_PENDING)

    def test_exact_guard_failure_can_close_an_ambiguous_dispatch(self) -> None:
        self.reserve()
        before = self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(cancel_id)

        failed = self.repo.fail_cancel_operation(
            cancel_id, "identity guard proved scancel was not reached"
        )
        after = self.repo.get_job(self.operation_id)
        self.assertEqual(failed.phase, OperationPhase.FAILED)
        self.assertEqual(after, before)
        self.assertEqual(after.phase, before.phase)
        self.assertEqual(after.job_id, before.job_id)
        retry = self.repo.begin_cancel(self.operation_id, operation_id="d" * 32)
        self.assertEqual(retry.phase, OperationPhase.CANCEL_PENDING)

    def test_resolve_cancel_departed_rolls_back_both_rows_on_fault(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(cancel_id)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                """CREATE TRIGGER inject_departure_failure
                   BEFORE UPDATE OF phase ON jobs
                   WHEN NEW.phase = 'TERMINAL_CANDIDATE'
                   BEGIN SELECT RAISE(ABORT, 'injected departure failure'); END"""
            )
            connection.commit()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected departure failure"):
            self.repo.resolve_cancel_departed(cancel_id, "scancel acknowledged")

        self.assertEqual(
            self.repo.get_operation(cancel_id).phase,
            OperationPhase.AMBIGUOUS,
        )
        self.assertEqual(self.repo.get_job(self.operation_id).phase, JobPhase.QUEUED)

        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("DROP TRIGGER inject_departure_failure")
            connection.commit()
        resolved = self.repo.resolve_cancel_departed(cancel_id, "scancel acknowledged")
        self.assertEqual(resolved.phase, OperationPhase.RESOLVED)
        self.assertEqual(
            self.repo.get_job(self.operation_id).phase,
            JobPhase.TERMINAL_CANDIDATE,
        )

    def test_cancel_resolution_retains_exact_non_live_evidence(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(cancel_id)

        self.repo.resolve_operation(
            cancel_id,
            final_source=FinalSource.CONFIRMED_QUEUE,
            detail="read-only recovery confirmed finality",
            terminal_state="COMPLETING",
            ever_started=True,
            last_node="node07",
            observation_epoch=11,
            evidence_provenance=EvidenceProvenance.QUEUE_TERMINAL,
            evidence_detail="job remained terminal in exact queue observations",
        )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.phase, JobPhase.FINAL)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node07")
        self.assertEqual(stored.observation_epoch, 11)
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.QUEUE_TERMINAL,
        )
        self.assertEqual(
            stored.evidence_detail,
            "job remained terminal in exact queue observations",
        )

    def test_cancel_resolution_merges_start_history_into_concurrent_final(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(cancel_id)
        finalized = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state="BOOT_FAIL",
            exit_code="1:0",
            final_source=FinalSource.ACCOUNTING,
        )

        self.repo.resolve_cancel_departed(
            cancel_id,
            "cancellation request acknowledged",
            ever_started=True,
            last_node="node09",
            observation_epoch=13,
        )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(stored.terminal_state, "BOOT_FAIL")
        self.assertEqual(stored.exit_code, "1:0")
        self.assertEqual(stored.finalized_at, finalized.finalized_at)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node09")
        self.assertEqual(stored.observation_epoch, 13)

    def test_cancel_ack_uses_fresh_live_node_without_replaying_preflight(self) -> None:
        self.reserve()
        queued = self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(
            cancel_id,
            expected_target_updated_at=queued.updated_at,
            ever_started=True,
            last_node="node01",
        )
        self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )

        self.repo.resolve_cancel_departed(
            cancel_id,
            "cancellation request acknowledged",
        )

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertTrue(stored.ever_started)
        self.assertIsNone(stored.current_node)
        self.assertEqual(stored.last_node, "node02")
        self.assertIsNone(stored.terminal_state)
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.CANCELLATION,
        )

    def test_cancel_ack_preserves_concurrent_final_without_preflight_replay(self) -> None:
        self.reserve()
        queued = self.repo.acknowledge_submission(self.operation_id, "12345")
        cancel_id = "c" * 32
        self.repo.begin_cancel(self.operation_id, operation_id=cancel_id)
        self.repo.mark_cancel_dispatching(
            cancel_id,
            expected_target_updated_at=queued.updated_at,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )
        final = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            ever_started=True,
            last_node="node02",
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )

        self.repo.resolve_cancel_departed(
            cancel_id,
            "cancellation request acknowledged",
        )

        self.assertEqual(self.repo.get_job(self.operation_id), final)
        self.assertEqual(
            self.repo.get_operation(cancel_id).phase,
            OperationPhase.RESOLVED,
        )

    def test_final_merge_retains_missing_values_and_obeys_authority(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )
        self.repo.update_job(
            self.operation_id,
            phase=JobPhase.TERMINAL_CANDIDATE,
            terminal_state="COMPLETING",
            evidence_provenance=EvidenceProvenance.QUEUE_TERMINAL,
        )

        queue_final = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state=None,
            exit_code=None,
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        self.assertEqual(queue_final.terminal_state, "COMPLETING")
        self.assertIsNone(queue_final.exit_code)
        self.assertEqual(queue_final.last_node, "node01")
        self.assertEqual(queue_final.resources, {"cpus": 2, "partition": "day"})
        finalized_at = queue_final.finalized_at

        accounting_final = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        self.assertEqual(accounting_final.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(accounting_final.terminal_state, "COMPLETED")
        self.assertEqual(accounting_final.exit_code, "0:0")
        self.assertEqual(accounting_final.finalized_at, finalized_at)

        lower_authority = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state="CANCELLED",
            exit_code="1:0",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        self.assertEqual(lower_authority.terminal_state, "COMPLETED")
        self.assertEqual(lower_authority.exit_code, "0:0")
        self.assertEqual(lower_authority.final_source, FinalSource.ACCOUNTING)

        retained = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            terminal_state=None,
            exit_code=None,
            final_source=FinalSource.ACCOUNTING,
        )
        self.assertEqual(retained.terminal_state, "COMPLETED")
        self.assertEqual(retained.exit_code, "0:0")
        with self.assertRaisesRegex(StateConflict, "equal-authority"):
            self.repo.update_job(
                self.operation_id,
                phase=JobPhase.FINAL,
                terminal_state="FAILED",
                final_source=FinalSource.ACCOUNTING,
            )
        self.assertEqual(self.repo.get_job(self.operation_id).terminal_state, "COMPLETED")

    def test_local_final_verdicts_are_immutable(self) -> None:
        self.reserve()
        self.repo.fail_submission(self.operation_id, "sbatch rejected request")
        original = self.repo.get_job(self.operation_id)

        merged = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.FINAL,
            ever_started=True,
            last_node="node99",
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        self.assertEqual(merged.final_source, FinalSource.SUBMIT_FAILED)
        self.assertEqual(merged.terminal_state, "SUBMIT_FAILED")
        self.assertIsNone(merged.exit_code)
        self.assertFalse(merged.ever_started)
        self.assertIsNone(merged.last_node)
        self.assertEqual(merged.finalized_at, original.finalized_at)
        self.assertEqual(merged.updated_at, original.updated_at)

        abandoned_id = "b" * 32
        self.reserve(operation_id=abandoned_id)
        self.repo.mark_submission_ambiguous(abandoned_id, "reply lost")
        self.repo.abandon_operation(abandoned_id, "operator abandoned intent")
        abandoned = self.repo.get_job(abandoned_id)
        ignored = self.repo.update_job(
            abandoned_id,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        self.assertEqual(ignored.final_source, FinalSource.ABANDONED)
        self.assertEqual(ignored.terminal_state, "ABANDONED")
        self.assertEqual(ignored.updated_at, abandoned.updated_at)

    def test_concurrent_final_sources_converge_on_accounting(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        peer = StateRepository(self.path).initialize()
        barrier = threading.Barrier(2)

        def persist(source: FinalSource) -> FinalSource:
            barrier.wait()
            if source is FinalSource.ACCOUNTING:
                state, exit_code = "COMPLETED", "0:0"
            else:
                state, exit_code = "COMPLETING", None
            return peer.update_job(
                self.operation_id,
                phase=JobPhase.FINAL,
                terminal_state=state,
                exit_code=exit_code,
                evidence_provenance=(
                    EvidenceProvenance.QUEUE_TERMINAL
                    if source is FinalSource.CONFIRMED_QUEUE
                    else None
                ),
                final_source=source,
            ).final_source

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    persist,
                    (FinalSource.CONFIRMED_QUEUE, FinalSource.ACCOUNTING),
                )
            )
        self.assertIn(FinalSource.ACCOUNTING, outcomes)
        final = self.repo.get_job(self.operation_id)
        self.assertEqual(final.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(final.terminal_state, "COMPLETED")
        self.assertEqual(final.exit_code, "0:0")

    def test_concurrent_equal_authority_conflict_has_one_winner(self) -> None:
        self.reserve()
        self.repo.acknowledge_submission(self.operation_id, "12345")
        barrier = threading.Barrier(2)
        peers = (StateRepository(self.path).initialize(), StateRepository(self.path).initialize())

        def persist(item: tuple[StateRepository, str]) -> str:
            repository, state = item
            barrier.wait()
            try:
                repository.update_job(
                    self.operation_id,
                    phase=JobPhase.FINAL,
                    terminal_state=state,
                    exit_code="0:0",
                    final_source=FinalSource.ACCOUNTING,
                )
                return "stored"
            except StateConflict:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(persist, zip(peers, ("COMPLETED", "FAILED"), strict=True))
            )
        self.assertCountEqual(outcomes, ["stored", "conflict"])
        self.assertIn(
            self.repo.get_job(self.operation_id).terminal_state,
            {"COMPLETED", "FAILED"},
        )

    def test_persist_assessment_rejects_stale_lifecycle_evidence(self) -> None:
        self.reserve()
        stale = self.repo.acknowledge_submission(self.operation_id, "12345")
        self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )
        tracker = EvidenceTracker()
        tracker.accept(EvidenceEvent.absent())
        assessment = tracker.accept(EvidenceEvent.absent())

        with self.assertRaisesRegex(LifecycleRevisionConflict, "changed while.*rerun"):
            persist_assessment(self.repo, stale, assessment)

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.phase, JobPhase.ACTIVE)
        self.assertIsNone(stored.final_source)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.current_node, "node01")
        self.assertEqual(stored.resources, {"cpus": 2, "partition": "day"})

    def test_uncertain_assessment_returns_its_source_snapshot_without_mixing_versions(self) -> None:
        self.reserve()
        source = self.repo.acknowledge_submission(self.operation_id, "12345")
        durable = self.repo.update_job(
            self.operation_id,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node04",
        )
        tracker = EvidenceTracker()
        uncertain = tracker.accept(EvidenceEvent.transport_lost("VPN unavailable"))

        returned = persist_assessment(self.repo, source, uncertain)

        self.assertIs(returned, source)
        self.assertEqual(self.repo.get_job(self.operation_id), durable)

    def test_separate_observer_cannot_finalize_after_requeue_becomes_live(self) -> None:
        self.reserve()
        stale = self.repo.acknowledge_submission(self.operation_id, "12345")
        tracker = EvidenceTracker()
        tracker.accept(EvidenceEvent.absent())
        stale_final = tracker.accept(EvidenceEvent.absent())
        observer = StateRepository(self.path).initialize()
        writer = StateRepository(self.path).initialize()
        live_written = threading.Event()

        def persist_live() -> None:
            writer.update_job(
                self.operation_id,
                phase=JobPhase.ACTIVE,
                ever_started=True,
                current_node="node02",
            )
            live_written.set()

        def persist_stale() -> str:
            if not live_written.wait(timeout=5):
                raise AssertionError("live requeue observation was not persisted")
            try:
                persist_assessment(observer, stale, stale_final)
            except StateConflict:
                return "conflict"
            return "stored"

        with ThreadPoolExecutor(max_workers=2) as executor:
            live_future = executor.submit(persist_live)
            stale_future = executor.submit(persist_stale)
            live_future.result()
            self.assertEqual(stale_future.result(), "conflict")

        stored = self.repo.get_job(self.operation_id)
        self.assertEqual(stored.phase, JobPhase.ACTIVE)
        self.assertEqual(stored.current_node, "node02")
        self.assertIsNone(stored.final_source)

    def test_restart_error_cannot_bridge_a_durable_recycled_id_candidate(self) -> None:
        self.reserve()
        job = self.repo.acknowledge_submission(self.operation_id, "12345")
        detail = "job grace:12345 now belongs to a different operation"
        tracker = EvidenceTracker(observation_epoch=7)
        candidate = tracker.accept(EvidenceEvent.id_reused(detail))
        stored = persist_assessment(self.repo, job, candidate)

        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertEqual(stored.observation_epoch, 7)
        self.assertEqual(stored.evidence_provenance, EvidenceProvenance.ID_REUSED)
        self.assertEqual(stored.evidence_detail, detail)

        reopened = StateRepository(self.path).initialize().get_job(self.operation_id)

        class FailingClient:
            def observe(self, *_args: object, **_kwargs: object) -> None:
                raise TransportLost("VPN dropped")

        with self.assertRaisesRegex(TransportLost, "VPN dropped"):
            JobMonitor(FailingClient()).assess(reopened, confirm=False)

        # The raised error cannot be persisted by the command, but the next
        # process still treats restart itself as a new observation epoch.
        still_stored = self.repo.get_job(self.operation_id)
        self.assertEqual(still_stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertEqual(still_stored.observation_epoch, 7)

        class ReusedClient:
            def observe(self, *_args: object, **_kwargs: object) -> None:
                raise JobIdReused(detail)

            def final(self, *_args: object, **_kwargs: object) -> None:
                return None

        after_error = JobMonitor(ReusedClient()).assess(
            still_stored,
            confirm=False,
        ).assessment
        self.assertEqual(after_error.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertFalse(after_error.final)
        self.assertEqual(after_error.terminal_evidence, 1)
        self.assertEqual(after_error.observation_epoch, 8)
        self.assertEqual(after_error.evidence_provenance, EvidenceProvenance.ID_REUSED)

    def test_recycled_id_final_detail_survives_repository_restart(self) -> None:
        self.reserve()
        job = self.repo.acknowledge_submission(self.operation_id, "12345")
        detail = "job grace:12345 now belongs to a different operation"
        tracker = EvidenceTracker(observation_epoch=11)
        tracker.accept(EvidenceEvent.id_reused(detail))
        final = tracker.accept(EvidenceEvent.id_reused(detail))

        persisted = persist_assessment(self.repo, job, final)
        reopened = StateRepository(self.path).initialize().get_job(self.operation_id)
        restored = JobMonitor.tracker(reopened).assessment

        self.assertEqual(persisted.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(reopened.observation_epoch, 11)
        self.assertEqual(reopened.evidence_provenance, EvidenceProvenance.ID_REUSED)
        self.assertEqual(reopened.evidence_detail, detail)
        self.assertTrue(restored.final)
        self.assertEqual(restored.evidence_provenance, EvidenceProvenance.ID_REUSED)
        self.assertEqual(restored.detail, detail)


if __name__ == "__main__":
    unittest.main()
