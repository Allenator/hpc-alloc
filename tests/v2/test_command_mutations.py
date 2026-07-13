from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import (
    _cancel_record,
    _recover_cancel,
    _recover_submission,
    _sync_ssh_projection,
    _submit_job,
    cmd_cancel,
    cmd_down,
    cmd_recover,
)
from hpc_alloc.config import Config
from hpc_alloc.errors import (
    AmbiguousSubmission,
    AuthRequired,
    HostKeyChanged,
    IdentityMismatch,
    SchedulerUnavailable,
    StateConflict,
    TransportLost,
)
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    OperationPhase,
)
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.paths import AppPaths
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.slurm import (
    AccountingRecord,
    CancellationInspection,
    CancellationInspectionStatus,
    CancellationResult,
    CancellationStatus,
    QueueRow,
    QueueSnapshot,
    RawQueueRow,
    RawQueueScan,
    SubmissionResult,
    SubmissionSpec,
)
from hpc_alloc.ssh import AuthMode
from hpc_alloc.state import StateRepository

from .fakes import ExpectedCall, StrictProxy, StrictScript


OPERATION_ID = "a" * 32


class CommandMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = AppPaths.for_home(Path(self.directory.name))
        self.repository = StateRepository(
            self.paths.state_db,
            machine_id_factory=lambda: "deadbeef1234",
        ).initialize()
        self.context = SimpleNamespace(state=self.repository)
        self.resources = {
            "partition": "day",
            "time": "1:00:00",
            "cpus": 2,
            "mem": None,
            "gpus": None,
            "constraint": None,
            "chdir": None,
            "idle_timeout": None,
        }

    def invoke(self, transport: object, client: object):
        fake_uuid = SimpleNamespace(hex=OPERATION_ID)
        with (
            patch("hpc_alloc.commands.uuid.uuid4", return_value=fake_uuid),
            patch("hpc_alloc.commands.machine_host", return_value="laptop"),
            patch("hpc_alloc.commands._services", return_value=(transport, client)),
        ):
            return _submit_job(
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
                cluster="grace",
                kind=JobKind.ALLOCATION,
                logical_name="dev",
                resources=self.resources,
                wrap="sleep infinity",
                logfile_template=".hpc-alloc/alloc-{operation_id}.log",
                dry_run=False,
            )

    def transport(self) -> tuple[StrictProxy, StrictScript]:
        script = StrictScript(
            [
                ExpectedCall(
                    "bootstrap",
                    args=("grace", AuthMode.INTERACTIVE_BOOTSTRAP),
                )
            ]
        )
        return StrictProxy(script), script

    def acknowledged_job(self):
        owner = self.repository.get_or_create_machine_id("laptop")
        self.repository.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=slurm_job_name("allocation", OPERATION_ID),
            slurm_comment=format_tag(
                owner, OPERATION_ID, "laptop", "allocation", "dev"
            ),
            resources=self.resources,
        )
        return self.repository.acknowledge_submission(OPERATION_ID, "12345")

    def configure_clusters(self) -> None:
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.config_file.write_text(
            """\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
[cluster.grace]
host = "grace.example.edu"
[cluster.secondary]
host = "secondary.example.edu"
"""
        )
        self.context.config = Config.load(self.paths.config_file)

    def active_allocation_with_projection(self):
        self.configure_clusters()
        job = self.acknowledged_job()
        job = self.repository.update_job(
            OPERATION_ID,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )
        _sync_ssh_projection(self.context, self.paths)
        self.assertIn(
            "Host hpc-grace.dev",
            self.paths.managed_ssh_config.read_text(),
        )
        return job

    def test_prepared_intent_exists_before_the_only_submit_call(self) -> None:
        transport, transport_script = self.transport()

        def prepare(spec: SubmissionSpec, **kwargs: object) -> None:
            self.assertIsInstance(spec, SubmissionSpec)
            self.assertNotIn("sbatch", spec.preparation_command())
            self.assertEqual(kwargs, {"auth": AuthMode.NONINTERACTIVE})
            self.assertEqual(self.repository.list_operations(), [])

        def acknowledge(spec: SubmissionSpec, **kwargs: object) -> SubmissionResult:
            self.assertIsInstance(spec, SubmissionSpec)
            self.assertIn("sbatch --parsable", spec.command())
            self.assertEqual(kwargs, {"auth": AuthMode.NONINTERACTIVE})
            operation = self.repository.get_operation(OPERATION_ID)
            job = self.repository.get_job(OPERATION_ID)
            self.assertEqual(operation.phase, OperationPhase.PREPARED)
            self.assertEqual(job.phase, JobPhase.SUBMITTING)
            self.assertIsNone(job.job_id)
            return SubmissionResult("12345", "12345")

        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission", result=prepare),
                ExpectedCall("submit", result=acknowledge),
            ]
        )
        job = self.invoke(transport, StrictProxy(client_script))
        self.assertEqual(job.phase, JobPhase.QUEUED)
        self.assertEqual(job.job_id, "12345")
        self.assertEqual(client_script.count("submit"), 1)
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.ACKNOWLEDGED,
        )
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_submission_preparation_failure_creates_no_intent_or_remote_job(self) -> None:
        transport, transport_script = self.transport()

        def fail_preparation(_spec: SubmissionSpec, **_kwargs: object) -> None:
            raise SchedulerUnavailable("log directory preparation failed")

        client_script = StrictScript(
            [ExpectedCall("prepare_submission", result=fail_preparation)]
        )
        with self.assertRaisesRegex(SchedulerUnavailable, "preparation failed"):
            self.invoke(transport, StrictProxy(client_script))

        self.assertEqual(self.repository.list_operations(), [])
        self.assertEqual(self.repository.list_jobs(), [])
        self.assertEqual(client_script.count("submit"), 0)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_preparation_host_key_failure_is_preserved_without_creating_intent(self) -> None:
        transport, transport_script = self.transport()
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace")
        client_script = StrictScript(
            [ExpectedCall("prepare_submission", result=failure)]
        )

        with self.assertRaises(HostKeyChanged) as raised:
            self.invoke(transport, StrictProxy(client_script))

        self.assertIs(raised.exception, failure)
        self.assertEqual(self.repository.list_operations(), [])
        self.assertEqual(self.repository.list_jobs(), [])
        self.assertEqual(client_script.count("submit"), 0)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_submit_host_key_failure_closes_prepared_intent_and_is_preserved(self) -> None:
        transport, transport_script = self.transport()
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace")
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=failure),
            ]
        )

        with self.assertRaises(HostKeyChanged) as raised:
            self.invoke(transport, StrictProxy(client_script))

        self.assertIs(raised.exception, failure)
        operation = self.repository.get_operation(OPERATION_ID)
        job = self.repository.get_job(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.FAILED)
        self.assertEqual(job.phase, JobPhase.FINAL)
        self.assertEqual(job.terminal_state, "SUBMIT_FAILED")
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_identity_mismatch_fails_cancel_operation_without_finalizing_job(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        client_script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.IDENTITY_MISMATCH,
                        "live job belongs to another operation",
                    ),
                    args=(job.ref,),
                )
            ]
        )
        with self.assertRaisesRegex(IdentityMismatch, "another operation"):
            _cancel_record(self.context, StrictProxy(client_script), job)

        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        cancels = [
            operation
            for operation in self.repository.list_operations()
            if operation.operation_id != OPERATION_ID
        ]
        self.assertEqual(len(cancels), 1)
        self.assertEqual(cancels[0].phase, OperationPhase.FAILED)
        client_script.assert_complete()

    def test_ambiguous_cancel_remains_pending_for_recovery(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        client_script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(CancellationInspectionStatus.READY),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(
                        CancellationStatus.MUTATION_AMBIGUOUS,
                        "connection dropped mid-scancel",
                    ),
                    args=(job.ref,),
                )
            ]
        )
        with self.assertRaisesRegex(TransportLost, "cancellation .* is ambiguous"):
            _cancel_record(self.context, StrictProxy(client_script), job)

        unresolved = self.repository.list_unresolved_operations()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].phase, OperationPhase.AMBIGUOUS)
        self.assertIn("mid-scancel", unresolved[0].detail or "")
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        client_script.assert_complete()

    def test_running_cancel_preflight_persists_start_proof_before_dispatch(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        running = QueueRow(
            job_id=job.job_id or "",
            state="RUNNING",
            node="node01",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=running,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(
                        CancellationStatus.MUTATION_AMBIGUOUS,
                        "connection dropped mid-scancel",
                    ),
                    args=(job.ref,),
                ),
            ]
        )

        with self.assertRaises(TransportLost):
            _cancel_record(self.context, StrictProxy(script), job)

        stored = self.repository.get_job(OPERATION_ID)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node01")
        self.assertEqual(stored.observation_epoch, 1)
        self.assertEqual(
            self.repository.list_unresolved_operations()[0].phase,
            OperationPhase.AMBIGUOUS,
        )
        script.assert_complete()

    def test_cancel_reinspects_after_revision_race_and_never_replays_stale_node(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        running = QueueRow(
            job_id=job.job_id or "",
            state="RUNNING",
            node="node01",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        pending = QueueRow(
            job_id=job.job_id or "",
            state="PENDING",
            node=None,
            reason="Requeued",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=running,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=pending,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=pending,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(CancellationStatus.CANCELLED),
                    args=(job.ref,),
                ),
            ]
        )
        real_mark = self.repository.mark_cancel_dispatching
        mark_calls = 0

        def racing_mark(*args: object, **kwargs: object) -> object:
            nonlocal mark_calls
            mark_calls += 1
            if mark_calls in {1, 2}:
                node = "node02" if mark_calls == 1 else "node03"
                self.repository.update_job(
                    OPERATION_ID,
                    phase=JobPhase.ACTIVE,
                    ever_started=True,
                    current_node=node,
                    last_node=node,
                )
            return real_mark(*args, **kwargs)

        with patch.object(
            self.repository,
            "mark_cancel_dispatching",
            side_effect=racing_mark,
        ):
            _cancel_record(self.context, StrictProxy(script), job)

        stored = self.repository.get_job(OPERATION_ID)
        self.assertEqual(mark_calls, 3)
        self.assertEqual(script.count("inspect_cancel"), 3)
        self.assertEqual(script.count("execute_cancel"), 1)
        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertEqual(stored.last_node, "node03")
        self.assertNotEqual(stored.last_node, "node01")
        script.assert_complete()

    def test_pending_cancel_preflight_does_not_fabricate_start_history(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        pending = QueueRow(
            job_id=job.job_id or "",
            state="PENDING",
            node=None,
            reason="Resources",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=pending,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(CancellationStatus.CANCELLED),
                    args=(job.ref,),
                ),
            ]
        )

        _cancel_record(self.context, StrictProxy(script), job)

        stored = self.repository.get_job(OPERATION_ID)
        self.assertFalse(stored.ever_started)
        self.assertIsNone(stored.last_node)
        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.CANCELLATION,
        )
        script.assert_complete()

    def test_queue_terminal_cancel_preflight_retains_exact_provenance(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        terminal = QueueRow(
            job_id=job.job_id or "",
            state="COMPLETED",
            node="node07",
            reason="None",
            time_left="0:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=terminal,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(CancellationStatus.CANCELLED),
                    args=(job.ref,),
                ),
            ]
        )

        _cancel_record(self.context, StrictProxy(script), job)

        stored = self.repository.get_job(OPERATION_ID)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node07")
        self.assertEqual(stored.terminal_state, "COMPLETED")
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.QUEUE_TERMINAL,
        )
        script.assert_complete()

    def test_ambiguous_terminal_cancel_retains_pre_dispatch_provenance(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        terminal = QueueRow(
            job_id=job.job_id or "",
            state="COMPLETED",
            node="node08",
            reason="None",
            time_left="0:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=terminal,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(
                        CancellationStatus.MUTATION_AMBIGUOUS,
                        "reply lost after guarded dispatch",
                    ),
                    args=(job.ref,),
                ),
            ]
        )

        with self.assertRaises(TransportLost):
            _cancel_record(self.context, StrictProxy(script), job)

        stored = self.repository.get_job(OPERATION_ID)
        self.assertEqual(stored.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertTrue(stored.ever_started)
        self.assertEqual(stored.last_node, "node08")
        self.assertEqual(
            stored.evidence_provenance,
            EvidenceProvenance.QUEUE_TERMINAL,
        )
        self.assertEqual(
            self.repository.list_unresolved_operations()[0].phase,
            OperationPhase.AMBIGUOUS,
        )
        script.assert_complete()

    def test_already_final_cancel_preflight_uses_terminal_start_policy(self) -> None:
        for index, (state, proves_started) in enumerate(
            (("BOOT_FAIL", False), ("COMPLETED", True)),
            start=1,
        ):
            with self.subTest(state=state):
                home = Path(self.directory.name) / f"terminal-{index}"
                paths = AppPaths.for_home(home)
                repository = StateRepository(paths.state_db).initialize()
                context = SimpleNamespace(state=repository)
                operation_id = f"{index:032x}"
                owner = repository.get_or_create_machine_id("laptop")
                repository.reserve_submission(
                    operation_id=operation_id,
                    cluster="grace",
                    logical_name=f"dev{index}",
                    kind=JobKind.ALLOCATION,
                    owner_id=owner,
                    slurm_job_name=slurm_job_name("allocation", operation_id),
                    slurm_comment=format_tag(
                        owner,
                        operation_id,
                        "laptop",
                        "allocation",
                        f"dev{index}",
                    ),
                    resources=self.resources,
                )
                job = repository.acknowledge_submission(operation_id, str(12340 + index))
                assert job.ref is not None
                record = AccountingRecord(
                    job_id=job.job_id or "",
                    state=state,
                    exit_code="0:0" if state == "COMPLETED" else "1:0",
                    job_name=job.slurm_job_name,
                    comment=job.slurm_comment,
                )
                script = StrictScript(
                    [
                        ExpectedCall(
                            "inspect_cancel",
                            result=CancellationInspection(
                                CancellationInspectionStatus.ALREADY_FINAL,
                                f"job ended as {state}",
                                record,
                            ),
                            args=(job.ref,),
                        )
                    ]
                )

                _cancel_record(context, StrictProxy(script), job)

                stored = repository.get_job(operation_id)
                self.assertEqual(stored.phase, JobPhase.FINAL)
                self.assertEqual(stored.final_source, FinalSource.ACCOUNTING)
                self.assertEqual(stored.terminal_state, state)
                self.assertEqual(stored.ever_started, proves_started)
                script.assert_complete()

    def test_ambiguous_running_cancel_recovery_keeps_start_proof_from_preflight(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        running = QueueRow(
            job_id=job.job_id or "",
            state="RUNNING",
            node="node03",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        dispatch = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.READY,
                        queue_row=running,
                    ),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(
                        CancellationStatus.MUTATION_AMBIGUOUS,
                        "reply lost",
                    ),
                    args=(job.ref,),
                ),
            ]
        )
        with self.assertRaises(TransportLost):
            _cancel_record(self.context, StrictProxy(dispatch), job)
        dispatch.assert_complete()

        stored = self.repository.get_job(OPERATION_ID)
        cancel = self.repository.list_unresolved_operations()[0]
        assert stored.ref is not None
        record = AccountingRecord(
            job_id=stored.job_id or "",
            state="CANCELLED",
            exit_code="0:15",
            job_name=stored.slurm_job_name,
            comment=stored.slurm_comment,
        )
        recovery = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(stored.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=record,
                    args=(stored.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )

        with patch("hpc_alloc.commands.info"):
            self.assertTrue(
                _recover_cancel(
                    self.context,
                    StrictProxy(recovery),
                    cancel,
                    stored,
                )
            )

        final = self.repository.get_job(OPERATION_ID)
        self.assertEqual(final.phase, JobPhase.FINAL)
        self.assertEqual(final.final_source, FinalSource.ACCOUNTING)
        self.assertTrue(final.ever_started)
        self.assertEqual(final.last_node, "node03")
        recovery.assert_complete()

    def test_scancel_acknowledgement_does_not_fabricate_final_state(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        client_script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(CancellationInspectionStatus.READY),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(CancellationStatus.CANCELLED),
                    args=(job.ref,),
                )
            ]
        )

        outcome = _cancel_record(self.context, StrictProxy(client_script), job)

        self.assertEqual(outcome.status, CancellationStatus.CANCELLED)
        updated = self.repository.get_job(OPERATION_ID)
        self.assertEqual(updated.phase, JobPhase.TERMINAL_CANDIDATE)
        self.assertIsNone(updated.terminal_state)
        cancel = [
            operation
            for operation in self.repository.list_operations()
            if operation.operation_id != OPERATION_ID
        ][0]
        self.assertEqual(cancel.phase, OperationPhase.RESOLVED)
        client_script.assert_complete()

    def test_confirmed_preflight_absence_resolves_without_dispatch(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        client_script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(
                        CancellationInspectionStatus.CONFIRMED_ABSENT,
                        "two exact queue observations were absent",
                    ),
                    args=(job.ref,),
                )
            ]
        )

        outcome = _cancel_record(self.context, StrictProxy(client_script), job)

        self.assertEqual(
            outcome.status, CancellationInspectionStatus.CONFIRMED_ABSENT
        )
        self.assertEqual(
            self.repository.get_job(OPERATION_ID).phase,
            JobPhase.TERMINAL_CANDIDATE,
        )
        cancellation = [
            operation
            for operation in self.repository.list_operations()
            if operation.kind.value == "cancel"
        ][0]
        self.assertEqual(cancellation.phase, OperationPhase.RESOLVED)
        self.assertEqual(client_script.count("execute_cancel"), 0)
        client_script.assert_complete()

    def test_preflight_failure_closes_cancel_before_reraising(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None

        def unavailable(_ref: object) -> CancellationInspection:
            raise SchedulerUnavailable("squeue unavailable before dispatch")

        client_script = StrictScript(
            [ExpectedCall("inspect_cancel", result=unavailable, args=(job.ref,))]
        )
        with self.assertRaisesRegex(SchedulerUnavailable, "before dispatch"):
            _cancel_record(self.context, StrictProxy(client_script), job)

        cancellation = [
            operation
            for operation in self.repository.list_operations()
            if operation.kind.value == "cancel"
        ][0]
        self.assertEqual(cancellation.phase, OperationPhase.FAILED)
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        retry = self.repository.begin_cancel(OPERATION_ID, operation_id="d" * 32)
        self.assertEqual(retry.phase, OperationPhase.CANCEL_PENDING)
        client_script.assert_complete()

    def test_preflight_host_key_failure_closes_cancel_and_preserves_type(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace")
        client_script = StrictScript(
            [ExpectedCall("inspect_cancel", result=failure, args=(job.ref,))]
        )

        with self.assertRaises(HostKeyChanged) as raised:
            _cancel_record(self.context, StrictProxy(client_script), job)

        self.assertIs(raised.exception, failure)
        cancellation = [
            operation
            for operation in self.repository.list_operations()
            if operation.kind.value == "cancel"
        ][0]
        self.assertEqual(cancellation.phase, OperationPhase.FAILED)
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        self.assertEqual(client_script.count("execute_cancel"), 0)
        client_script.assert_complete()

    def test_execute_access_failures_close_dispatch_intent_and_preserve_type(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        failures = (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                existing_cancel_ids = {
                    operation.operation_id
                    for operation in self.repository.list_operations()
                    if operation.kind.value == "cancel"
                }
                current = self.repository.get_job(OPERATION_ID)
                assert current.ref is not None
                client_script = StrictScript(
                    [
                        ExpectedCall(
                            "inspect_cancel",
                            result=CancellationInspection(
                                CancellationInspectionStatus.READY
                            ),
                            args=(current.ref,),
                        ),
                        ExpectedCall(
                            "execute_cancel",
                            result=failure,
                            args=(current.ref,),
                        ),
                    ]
                )
                with self.assertRaises(type(failure)) as raised:
                    _cancel_record(
                        self.context,
                        StrictProxy(client_script),
                        current,
                    )
                self.assertIs(raised.exception, failure)
                new_cancellations = [
                    operation
                    for operation in self.repository.list_operations()
                    if operation.kind.value == "cancel"
                    and operation.operation_id not in existing_cancel_ids
                ]
                self.assertEqual(len(new_cancellations), 1)
                cancellation = new_cancellations[0]
                self.assertEqual(cancellation.phase, OperationPhase.FAILED)
                self.assertEqual(
                    self.repository.get_job(OPERATION_ID).phase,
                    JobPhase.QUEUED,
                )
                client_script.assert_complete()

    def test_guard_failure_closes_dispatched_intent_without_finalizing_job(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        client_script = StrictScript(
            [
                ExpectedCall(
                    "inspect_cancel",
                    result=CancellationInspection(CancellationInspectionStatus.READY),
                    args=(job.ref,),
                ),
                ExpectedCall(
                    "execute_cancel",
                    result=CancellationResult(
                        CancellationStatus.GUARD_FAILED,
                        "exact guard failed before scancel",
                    ),
                    args=(job.ref,),
                ),
            ]
        )
        with self.assertRaisesRegex(SchedulerUnavailable, "guard failed"):
            _cancel_record(self.context, StrictProxy(client_script), job)

        cancellation = [
            operation
            for operation in self.repository.list_operations()
            if operation.kind.value == "cancel"
        ][0]
        self.assertEqual(cancellation.phase, OperationPhase.FAILED)
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        client_script.assert_complete()

    def test_cancel_recovery_with_live_exact_job_is_observation_only(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        live = QueueRow(
            job_id=job.job_id or "",
            state="RUNNING",
            node="node01",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-10T11:00:00",
            comment=job.slurm_comment,
        )
        client_script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=live,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                )
            ]
        )

        self.assertFalse(
            _recover_cancel(self.context, StrictProxy(client_script), cancel, job)
        )
        self.assertEqual(
            self.repository.get_operation(cancel.operation_id).phase,
            OperationPhase.AMBIGUOUS,
        )
        client_script.assert_complete()

    def test_cancel_recovery_resolves_authoritative_final_accounting(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        record = AccountingRecord(
            job_id=job.job_id or "",
            state="CANCELLED",
            exit_code="0:15",
            job_name=job.slurm_job_name,
            comment=job.slurm_comment,
        )
        client_script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=record,
                    args=(job.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )

        with patch("hpc_alloc.commands.info"):
            self.assertTrue(
                _recover_cancel(self.context, StrictProxy(client_script), cancel, job)
            )
        self.assertEqual(
            self.repository.get_operation(cancel.operation_id).phase,
            OperationPhase.RESOLVED,
        )
        final = self.repository.get_job(OPERATION_ID)
        self.assertEqual(final.phase, JobPhase.FINAL)
        self.assertEqual(final.terminal_state, "CANCELLED")
        client_script.assert_complete()

    def test_cancel_recovery_reassesses_instead_of_retrying_stale_final(self) -> None:
        job = self.acknowledged_job()
        assert job.ref is not None
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        record = AccountingRecord(
            job_id=job.job_id or "",
            state="CANCELLED",
            exit_code="0:15",
            job_name=job.slurm_job_name,
            comment=job.slurm_comment,
        )
        running = QueueRow(
            job_id=job.job_id or "",
            state="RUNNING",
            node="node02",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job.slurm_job_name,
            submitted_at="2026-07-12T11:00:00",
            comment=job.slurm_comment,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=record,
                    args=(job.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=running,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        real_resolve = self.repository.resolve_operation
        resolve_calls = 0

        def racing_resolve(*args: object, **kwargs: object) -> object:
            nonlocal resolve_calls
            resolve_calls += 1
            if resolve_calls == 1:
                self.repository.update_job(
                    OPERATION_ID,
                    phase=JobPhase.ACTIVE,
                    ever_started=True,
                    current_node="node02",
                    last_node="node02",
                )
            return real_resolve(*args, **kwargs)

        with (
            patch.object(
                self.repository,
                "resolve_operation",
                side_effect=racing_resolve,
            ),
            patch("hpc_alloc.commands.info"),
        ):
            self.assertFalse(
                _recover_cancel(self.context, StrictProxy(script), cancel, job)
            )

        stored = self.repository.get_job(OPERATION_ID)
        self.assertEqual(resolve_calls, 1)
        self.assertEqual(stored.phase, JobPhase.ACTIVE)
        self.assertEqual(stored.current_node, "node02")
        self.assertEqual(stored.last_node, "node02")
        self.assertEqual(
            self.repository.get_operation(cancel.operation_id).phase,
            OperationPhase.AMBIGUOUS,
        )
        script.assert_complete()

    def test_recover_closes_undispatched_cancel_without_constructing_services(self) -> None:
        self.acknowledged_job()
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        args = SimpleNamespace(
            operation_id=cancel.operation_id,
            cluster=None,
            abandon=False,
            yes=False,
        )

        with (
            patch(
                "hpc_alloc.commands._services",
                side_effect=AssertionError("undispatched recovery contacted the cluster"),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=False),
            patch("hpc_alloc.commands.info"),
        ):
            self.assertEqual(
                cmd_recover(
                    args,
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                ),
                0,
            )

        self.assertEqual(
            self.repository.get_operation(cancel.operation_id).phase,
            OperationPhase.FAILED,
        )
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)

    def test_cancel_recovery_resolves_two_confirmed_absences_without_mutation(self) -> None:
        from hpc_alloc.monitor import JobMonitor

        class ImmediateMonitor(JobMonitor):
            def __init__(self, client: object) -> None:
                super().__init__(client, confirmation_delay=0)

        job = self.acknowledged_job()
        assert job.ref is not None
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        client_script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(job.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(job.ref,),
                    kwargs={
                        "attempts": (0, 2, 2),
                        "auth": AuthMode.NONINTERACTIVE,
                    },
                ),
            ]
        )

        with (
            patch("hpc_alloc.monitor.JobMonitor", ImmediateMonitor),
            patch("hpc_alloc.commands.info"),
        ):
            self.assertTrue(
                _recover_cancel(self.context, StrictProxy(client_script), cancel, job)
            )

        final = self.repository.get_job(OPERATION_ID)
        self.assertEqual(final.phase, JobPhase.FINAL)
        self.assertEqual(final.final_source.value, "confirmed-queue")
        self.assertEqual(final.evidence_provenance, EvidenceProvenance.ABSENT)
        client_script.assert_complete()

    def test_cancel_recovery_preserves_id_reuse_provenance_and_detail(self) -> None:
        from hpc_alloc.lifecycle import EvidenceEvent

        job = self.acknowledged_job()
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        tracker = JobMonitor.tracker(job)
        tracker.begin_observation_epoch()
        tracker.accept(EvidenceEvent.id_reused("numeric ID belongs to a replacement"))
        assessment = tracker.accept(
            EvidenceEvent.id_reused("numeric ID belongs to a replacement")
        )
        monitor = SimpleNamespace(
            assess=lambda *_args, **_kwargs: SimpleNamespace(assessment=assessment)
        )

        with (
            patch("hpc_alloc.monitor.JobMonitor", return_value=monitor),
            patch("hpc_alloc.commands.info"),
        ):
            self.assertTrue(
                _recover_cancel(self.context, object(), cancel, job)
            )

        final = self.repository.get_job(OPERATION_ID)
        self.assertEqual(final.phase, JobPhase.FINAL)
        self.assertEqual(final.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(final.evidence_provenance, EvidenceProvenance.ID_REUSED)
        self.assertEqual(final.evidence_detail, "numeric ID belongs to a replacement")

    def test_cancel_recovery_error_after_one_absence_stays_ambiguous(self) -> None:
        from hpc_alloc.monitor import JobMonitor

        class ImmediateMonitor(JobMonitor):
            def __init__(self, client: object) -> None:
                super().__init__(client, confirmation_delay=0)

        def lose_transport(_ref: object, **_kwargs: object) -> None:
            raise TransportLost("connection lost during confirmation")

        job = self.acknowledged_job()
        assert job.ref is not None
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        cancel = self.repository.mark_cancel_dispatching(cancel.operation_id)
        client_script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(job.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(job.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall("observe", result=lose_transport),
            ]
        )

        with patch("hpc_alloc.monitor.JobMonitor", ImmediateMonitor):
            with self.assertRaisesRegex(TransportLost, "during confirmation"):
                _recover_cancel(self.context, StrictProxy(client_script), cancel, job)

        self.assertEqual(
            self.repository.get_operation(cancel.operation_id).phase,
            OperationPhase.AMBIGUOUS,
        )
        self.assertEqual(self.repository.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
        client_script.assert_complete()

    def test_ambiguous_submit_is_journaled_and_never_reissued(self) -> None:
        transport, transport_script = self.transport()

        def prepare(_spec: SubmissionSpec, **kwargs: object) -> None:
            self.assertEqual(kwargs, {"auth": AuthMode.NONINTERACTIVE})
            self.assertEqual(self.repository.list_operations(), [])

        def lose_reply(spec: SubmissionSpec, **kwargs: object) -> SubmissionResult:
            self.assertIsInstance(spec, SubmissionSpec)
            self.assertIn("sbatch --parsable", spec.command())
            self.assertEqual(kwargs, {"auth": AuthMode.NONINTERACTIVE})
            self.assertEqual(
                self.repository.get_operation(OPERATION_ID).phase,
                OperationPhase.PREPARED,
            )
            raise AmbiguousSubmission("reply lost after possible commit")

        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission", result=prepare),
                ExpectedCall("submit", result=lose_reply),
            ]
        )
        with self.assertRaisesRegex(AmbiguousSubmission, f"submission {OPERATION_ID} may have committed"):
            self.invoke(transport, StrictProxy(client_script))

        operation = self.repository.get_operation(OPERATION_ID)
        job = self.repository.get_job(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.AMBIGUOUS)
        self.assertIn("possible commit", operation.detail or "")
        self.assertEqual(job.phase, JobPhase.SUBMITTING)
        self.assertIsNone(job.job_id)
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_submit_interrupt_prints_recovery_guidance_before_reraising(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=KeyboardInterrupt()),
            ]
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(KeyboardInterrupt):
            self.invoke(transport, StrictProxy(client_script))

        operation = self.repository.get_operation(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.AMBIGUOUS)
        self.assertEqual(client_script.count("submit"), 1)
        self.assertIn(f"submission {OPERATION_ID} may have committed", stderr.getvalue())
        self.assertIn("do not resubmit", stderr.getvalue())
        self.assertIn(f"`hpc-alloc recover {OPERATION_ID}`", stderr.getvalue())
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_interrupt_between_submit_reply_and_acknowledgement_is_guarded(self) -> None:
        class InterruptingReply:
            accesses = 0

            @property
            def job_id(self) -> str:
                self.accesses += 1
                if self.accesses == 1:
                    raise KeyboardInterrupt
                return "12345"

        transport, transport_script = self.transport()
        reply = InterruptingReply()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=reply),
            ]
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(KeyboardInterrupt):
            self.invoke(transport, StrictProxy(client_script))

        operation = self.repository.get_operation(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.AMBIGUOUS)
        self.assertIn("trusted Slurm job ID 12345", stderr.getvalue())
        self.assertIn("do not resubmit", stderr.getvalue())
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_submit_interrupt_reaches_cli_as_exit_130_without_replay(self) -> None:
        from hpc_alloc.cli import main

        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=KeyboardInterrupt()),
            ]
        )

        def dispatch(_args: object, *, entrypoint: Path) -> object:
            self.assertEqual(entrypoint, Path("/tmp/hpc-alloc"))
            return self.invoke(transport, StrictProxy(client_script))

        stderr = io.StringIO()
        with (
            patch("hpc_alloc.commands.dispatch", side_effect=dispatch),
            redirect_stderr(stderr),
        ):
            self.assertEqual(
                main(["status"], entrypoint=Path("/tmp/hpc-alloc")),
                130,
            )

        self.assertEqual(client_script.count("submit"), 1)
        self.assertIn("do not resubmit", stderr.getvalue())
        self.assertIn(f"`hpc-alloc recover {OPERATION_ID}`", stderr.getvalue())
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.AMBIGUOUS,
        )
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_cli_interrupt_keeps_exit_130_when_stderr_is_broken(self) -> None:
        from hpc_alloc.cli import main

        broken_stderr = SimpleNamespace(
            write=Mock(side_effect=BrokenPipeError()),
            flush=Mock(),
        )
        with (
            patch("hpc_alloc.commands.dispatch", side_effect=KeyboardInterrupt()),
            patch("hpc_alloc.cli.sys.stderr", broken_stderr),
            patch("hpc_alloc.cli.neutralize_stderr") as neutralize,
        ):
            result = main(["status"], entrypoint=Path("/tmp/hpc-alloc"))

        self.assertEqual(result, 130)
        neutralize.assert_called_once_with()

    def test_broken_guidance_cannot_skip_ambiguous_interrupt_journal(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=KeyboardInterrupt()),
            ]
        )

        with (
            patch("hpc_alloc.commands.info", side_effect=BrokenPipeError()),
            patch("hpc_alloc.commands.neutralize_stderr") as neutralize,
            self.assertRaises(KeyboardInterrupt),
        ):
            self.invoke(transport, StrictProxy(client_script))

        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.AMBIGUOUS,
        )
        self.assertEqual(client_script.count("submit"), 1)
        neutralize.assert_called_once_with()
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_acknowledgement_interrupt_reports_known_job_and_preserves_exit_130(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=SubmissionResult("12345", "12345")),
            ]
        )
        stderr = io.StringIO()
        with (
            patch.object(
                self.repository,
                "acknowledge_submission",
                side_effect=KeyboardInterrupt(),
            ),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.invoke(transport, StrictProxy(client_script))

        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.AMBIGUOUS,
        )
        self.assertIn("trusted Slurm job ID 12345", stderr.getvalue())
        self.assertIn("do not resubmit", stderr.getvalue())
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_ambiguous_journal_failure_keeps_prepared_operation_actionable(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall(
                    "submit",
                    result=AmbiguousSubmission("reply lost after possible commit"),
                ),
            ]
        )
        with (
            patch.object(
                self.repository,
                "mark_submission_ambiguous",
                side_effect=StateConflict("journal unavailable; retry the command"),
            ),
            self.assertRaises(AmbiguousSubmission) as raised,
        ):
            self.invoke(transport, StrictProxy(client_script))

        message = str(raised.exception)
        self.assertIn("do not resubmit", message)
        self.assertIn(f"`hpc-alloc recover {OPERATION_ID}`", message)
        self.assertNotIn("retry the command", message)
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.PREPARED,
        )
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_acknowledgement_failure_reports_trusted_job_without_replay_advice(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=SubmissionResult("12345", "12345")),
            ]
        )
        with (
            patch.object(
                self.repository,
                "acknowledge_submission",
                side_effect=StateConflict("database busy; retry the command"),
            ),
            self.assertRaises(AmbiguousSubmission) as raised,
        ):
            self.invoke(transport, StrictProxy(client_script))

        message = str(raised.exception)
        self.assertIn("trusted Slurm job ID 12345", message)
        self.assertIn("do not resubmit", message)
        self.assertIn(f"`hpc-alloc recover {OPERATION_ID}`", message)
        self.assertNotIn("retry the command", message)
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.AMBIGUOUS,
        )
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_acknowledgement_and_ambiguity_journal_failures_keep_prepared_recovery(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=SubmissionResult("12345", "12345")),
            ]
        )
        with (
            patch.object(
                self.repository,
                "acknowledge_submission",
                side_effect=StateConflict("acknowledgement journal unavailable"),
            ),
            patch.object(
                self.repository,
                "mark_submission_ambiguous",
                side_effect=StateConflict("ambiguity journal unavailable"),
            ),
            self.assertRaises(AmbiguousSubmission) as raised,
        ):
            self.invoke(transport, StrictProxy(client_script))

        message = str(raised.exception)
        self.assertIn("trusted Slurm job ID 12345", message)
        self.assertIn("do not resubmit", message)
        self.assertIn(f"`hpc-alloc recover {OPERATION_ID}`", message)
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.PREPARED,
        )
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_interrupt_after_acknowledgement_commit_keeps_durable_resolution(self) -> None:
        transport, transport_script = self.transport()
        client_script = StrictScript(
            [
                ExpectedCall("prepare_submission"),
                ExpectedCall("submit", result=SubmissionResult("12345", "12345")),
            ]
        )
        acknowledge = self.repository.acknowledge_submission

        def commit_then_interrupt(operation_id: str, job_id: str) -> object:
            acknowledge(operation_id, job_id)
            raise KeyboardInterrupt

        stderr = io.StringIO()
        with (
            patch.object(
                self.repository,
                "acknowledge_submission",
                side_effect=commit_then_interrupt,
            ),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.invoke(transport, StrictProxy(client_script))

        operation = self.repository.get_operation(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.ACKNOWLEDGED)
        self.assertEqual(operation.job_id, "12345")
        self.assertIn("trusted Slurm job ID 12345", stderr.getvalue())
        self.assertIn("do not resubmit", stderr.getvalue())
        self.assertEqual(client_script.count("submit"), 1)
        transport_script.assert_complete()
        client_script.assert_complete()

    def test_abandon_requires_an_explicit_operation_id(self) -> None:
        owner = self.repository.get_or_create_machine_id("laptop")
        self.repository.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=slurm_job_name("allocation", OPERATION_ID),
            slurm_comment=format_tag(
                owner, OPERATION_ID, "laptop", "allocation", "dev"
            ),
            resources=self.resources,
        )
        self.repository.mark_submission_ambiguous(OPERATION_ID, "reply lost")
        args = SimpleNamespace(
            operation_id=None,
            cluster=None,
            abandon=True,
            yes=True,
        )
        with self.assertRaisesRegex(StateConflict, "explicit operation ID"):
            cmd_recover(
                args,
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.AMBIGUOUS,
        )

    def test_submission_recovery_requires_the_complete_persisted_comment(self) -> None:
        owner = self.repository.get_or_create_machine_id("laptop")
        job_name = slurm_job_name("allocation", OPERATION_ID)
        comment = format_tag(owner, OPERATION_ID, "laptop", "allocation", "dev")
        self.repository.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=job_name,
            slurm_comment=comment,
            resources=self.resources,
        )
        self.repository.mark_submission_ambiguous(OPERATION_ID, "reply lost")
        operation = self.repository.get_operation(OPERATION_ID)
        job = self.repository.get_job(OPERATION_ID)

        row = QueueRow(
            job_id="12345",
            state="RUNNING",
            node="node01",
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=job_name,
            submitted_at="2026-07-10T11:00:00",
            comment=format_tag(
                owner, OPERATION_ID, "different-host", "allocation", "dev"
            ),
        )
        mismatch = StrictScript(
            [
                ExpectedCall(
                    "scan",
                    result=RawQueueScan(
                        (
                            RawQueueRow(
                                row.job_id,
                                row.state,
                                row.node or "",
                                row.reason,
                                row.time_left,
                                row.partition,
                                row.name,
                                row.submitted_at,
                                row.comment,
                            ),
                        )
                    ),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "find_accounting_by_name",
                    result=None,
                    args=(job_name,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        with patch("hpc_alloc.commands.info"):
            self.assertFalse(
                _recover_submission(
                    self.context, StrictProxy(mismatch), operation, job
                )
            )
        self.assertIsNone(self.repository.get_job(OPERATION_ID).job_id)
        mismatch.assert_complete()

        exact_row = QueueRow(
            job_id=row.job_id,
            state=row.state,
            node=row.node,
            reason=row.reason,
            time_left=row.time_left,
            partition=row.partition,
            name=row.name,
            submitted_at=row.submitted_at,
            comment=comment,
        )
        exact = StrictScript(
            [
                ExpectedCall(
                    "scan",
                    result=RawQueueScan(
                        (
                            RawQueueRow(
                                exact_row.job_id,
                                exact_row.state,
                                exact_row.node or "",
                                exact_row.reason,
                                exact_row.time_left,
                                exact_row.partition,
                                exact_row.name,
                                exact_row.submitted_at,
                                exact_row.comment,
                            ),
                        )
                    ),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                )
            ]
        )
        with patch("hpc_alloc.commands.info"):
            self.assertTrue(
                _recover_submission(self.context, StrictProxy(exact), operation, job)
            )
        self.assertEqual(self.repository.get_job(OPERATION_ID).job_id, "12345")
        exact.assert_complete()

    def test_submission_recovery_accepts_canonical_name_with_omitted_accounting_comment(self) -> None:
        owner = self.repository.get_or_create_machine_id("laptop")
        job_name = slurm_job_name("allocation", OPERATION_ID)
        comment = format_tag(owner, OPERATION_ID, "laptop", "allocation", "dev")
        self.repository.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=job_name,
            slurm_comment=comment,
            resources=self.resources,
        )
        self.repository.mark_submission_ambiguous(OPERATION_ID, "reply lost")
        operation = self.repository.get_operation(OPERATION_ID)
        job = self.repository.get_job(OPERATION_ID)
        name_only = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=job_name,
            comment="",
        )

        def accept_omitted_comment(
            _ref: object, _job_name: str, accounting_comment: str
        ) -> None:
            self.assertEqual(accounting_comment, "")

        script = StrictScript(
            [
                ExpectedCall(
                    "scan",
                    result=RawQueueScan(()),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "find_accounting_by_name",
                    result=name_only,
                    args=(job_name,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "verify_accounting_identity",
                    result=accept_omitted_comment,
                ),
            ]
        )
        with patch("hpc_alloc.commands.info"):
            self.assertTrue(
                _recover_submission(
                    self.context,
                    StrictProxy(script),
                    operation,
                    job,
                )
            )
        recovered = self.repository.get_job(OPERATION_ID)
        self.assertEqual(recovered.job_id, "12345")
        self.assertEqual(recovered.phase, JobPhase.FINAL)
        self.assertEqual(recovered.terminal_state, "COMPLETED")
        self.assertEqual(
            recovered.slurm_comment,
            comment,
            "weak accounting evidence must never replace the full persisted live guard",
        )
        self.assertEqual(
            self.repository.get_operation(OPERATION_ID).phase,
            OperationPhase.ACKNOWLEDGED,
        )
        script.assert_complete()

    def test_submission_recovery_uses_lifecycle_start_evidence_for_final_accounting(self) -> None:
        cases = (
            ("BOOT_FAIL", False),
            ("CANCELLED", False),
            ("DEADLINE", False),
            ("REVOKED", False),
            ("COMPLETED", True),
            ("FAILED", True),
        )
        owner = self.repository.get_or_create_machine_id("laptop")
        for index, (state, proves_started) in enumerate(cases, start=1):
            with self.subTest(state=state):
                operation_id = f"{index:032x}"
                logical_name = f"recovered{index}"
                job_name = slurm_job_name("allocation", operation_id)
                comment = format_tag(
                    owner,
                    operation_id,
                    "laptop",
                    "allocation",
                    logical_name,
                )
                self.repository.reserve_submission(
                    operation_id=operation_id,
                    cluster="grace",
                    logical_name=logical_name,
                    kind=JobKind.ALLOCATION,
                    owner_id=owner,
                    slurm_job_name=job_name,
                    slurm_comment=comment,
                    resources=self.resources,
                )
                self.repository.mark_submission_ambiguous(operation_id, "reply lost")
                operation = self.repository.get_operation(operation_id)
                job = self.repository.get_job(operation_id)
                record = AccountingRecord(
                    job_id=str(12000 + index),
                    state=state,
                    exit_code="0:0" if state == "COMPLETED" else "1:0",
                    job_name=job_name,
                    comment=comment,
                )
                script = StrictScript(
                    [
                        ExpectedCall(
                            "scan",
                            result=RawQueueScan(()),
                            kwargs={"auth": AuthMode.NONINTERACTIVE},
                        ),
                        ExpectedCall(
                            "find_accounting_by_name",
                            result=record,
                            args=(job_name,),
                            kwargs={"auth": AuthMode.NONINTERACTIVE},
                        ),
                        ExpectedCall("verify_accounting_identity"),
                    ]
                )

                with patch("hpc_alloc.commands.info"):
                    self.assertTrue(
                        _recover_submission(
                            self.context,
                            StrictProxy(script),
                            operation,
                            job,
                        )
                    )

                recovered = self.repository.get_job(operation_id)
                self.assertEqual(recovered.phase, JobPhase.FINAL)
                self.assertEqual(recovered.final_source, FinalSource.ACCOUNTING)
                self.assertEqual(recovered.terminal_state, state)
                self.assertEqual(recovered.exit_code, record.exit_code)
                self.assertEqual(recovered.ever_started, proves_started)
                self.assertEqual(
                    JobMonitor.tracker(recovered).assessment.log_eligible,
                    proves_started,
                )
                self.assertEqual(
                    self.repository.get_operation(operation_id).phase,
                    OperationPhase.ACKNOWLEDGED,
                )
                script.assert_complete()

    def test_explicit_recovery_cluster_mismatch_precedes_projection_and_mutation(self) -> None:
        self.configure_clusters()
        self.acknowledged_job()
        cancel = self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)

        for abandon in (False, True):
            with self.subTest(abandon=abandon):
                with (
                    patch("hpc_alloc.commands._services") as services,
                    patch("hpc_alloc.commands._sync_ssh_projection") as project,
                ):
                    with self.assertRaises(IdentityMismatch):
                        cmd_recover(
                            SimpleNamespace(
                                operation_id=cancel.operation_id,
                                cluster="secondary",
                                abandon=abandon,
                                yes=True,
                            ),
                            ctx=self.context,
                            paths=self.paths,
                            entrypoint=Path("/tmp/hpc-alloc"),
                        )
                services.assert_not_called()
                project.assert_not_called()
                self.assertEqual(
                    self.repository.get_operation(cancel.operation_id).phase,
                    OperationPhase.CANCEL_PENDING,
                )

    def test_cmd_cancel_repairs_real_projection_when_cancellation_unwinds(self) -> None:
        job = self.active_allocation_with_projection()
        failure = TransportLost("guarded cancellation reply was lost")
        transport = SimpleNamespace(bootstrap=lambda _cluster: None)

        def finalize_then_fail(_ctx: object, _client: object, target: object) -> object:
            self.repository.update_job(
                target.operation_id,
                phase=JobPhase.FINAL,
                terminal_state="CANCELLED",
                exit_code="0:15",
                final_source=FinalSource.ACCOUNTING,
            )
            raise failure

        with (
            patch("hpc_alloc.commands._resolve_managed_job", return_value=job),
            patch("hpc_alloc.commands._services", return_value=(transport, object())),
            patch("hpc_alloc.commands._cancel_record", side_effect=finalize_then_fail),
        ):
            with self.assertRaises(TransportLost) as raised:
                cmd_cancel(
                    SimpleNamespace(target="12345", cluster=None),
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

        self.assertIs(raised.exception, failure)
        self.assertNotIn(
            "Host hpc-grace.dev",
            self.paths.managed_ssh_config.read_text(),
        )

    def test_single_target_down_repairs_real_projection_when_cancellation_unwinds(self) -> None:
        job = self.active_allocation_with_projection()
        failure = TransportLost("guarded cancellation reply was lost")
        transport = SimpleNamespace(bootstrap=lambda _cluster: None)

        def finalize_then_fail(_ctx: object, _client: object, target: object) -> object:
            self.repository.update_job(
                target.operation_id,
                phase=JobPhase.FINAL,
                terminal_state="CANCELLED",
                exit_code="0:15",
                final_source=FinalSource.ACCOUNTING,
            )
            raise failure

        with (
            patch("hpc_alloc.commands._resolve_managed_job", return_value=job),
            patch("hpc_alloc.commands._services", return_value=(transport, object())),
            patch("hpc_alloc.commands._cancel_record", side_effect=finalize_then_fail),
        ):
            with self.assertRaises(TransportLost) as raised:
                cmd_down(
                    SimpleNamespace(all=False, target="dev", cluster=None),
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

        self.assertIs(raised.exception, failure)
        self.assertNotIn(
            "Host hpc-grace.dev",
            self.paths.managed_ssh_config.read_text(),
        )

    def test_explicit_resolved_recovery_reports_phase_and_rejects_abandon(self) -> None:
        self.configure_clusters()
        self.acknowledged_job()
        args = SimpleNamespace(
            operation_id=OPERATION_ID,
            cluster=None,
            abandon=False,
            yes=False,
        )
        with (
            patch("hpc_alloc.commands._services") as services,
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=False) as project,
            patch("hpc_alloc.commands.info") as report,
        ):
            self.assertEqual(
                cmd_recover(
                    args,
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                ),
                0,
            )
        services.assert_not_called()
        project.assert_called_once_with(self.context, self.paths)
        self.assertIn("ACKNOWLEDGED", report.call_args.args[0])

        args.abandon = True
        with patch("hpc_alloc.commands._sync_ssh_projection", return_value=False):
            with self.assertRaisesRegex(StateConflict, "durable phase ACKNOWLEDGED"):
                cmd_recover(
                    args,
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

    def test_abandonment_projects_after_the_state_transition(self) -> None:
        self.configure_clusters()
        owner = self.repository.get_or_create_machine_id("laptop")
        self.repository.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=slurm_job_name("allocation", OPERATION_ID),
            slurm_comment=format_tag(
                owner, OPERATION_ID, "laptop", "allocation", "dev"
            ),
            resources=self.resources,
        )
        self.repository.mark_submission_ambiguous(OPERATION_ID, "reply lost")

        def projection(ctx: object, _paths: object) -> bool:
            self.assertEqual(
                ctx.state.get_operation(OPERATION_ID).phase,
                OperationPhase.ABANDONED,
            )
            return False

        with patch(
            "hpc_alloc.commands._sync_ssh_projection", side_effect=projection
        ) as project:
            self.assertEqual(
                cmd_recover(
                    SimpleNamespace(
                        operation_id=OPERATION_ID,
                        cluster=None,
                        abandon=True,
                        yes=True,
                    ),
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                ),
                0,
            )
        project.assert_called_once_with(self.context, self.paths)

    def test_multi_operation_recovery_projects_early_change_on_later_failure(self) -> None:
        self.configure_clusters()
        owner = self.repository.get_or_create_machine_id("laptop")
        second = "b" * 32
        for operation_id, cluster, name, job_id, node in (
            (OPERATION_ID, "grace", "dev", "12345", "node01"),
            (second, "secondary", "viz", "23456", "node02"),
        ):
            self.repository.reserve_submission(
                operation_id=operation_id,
                cluster=cluster,
                logical_name=name,
                kind=JobKind.ALLOCATION,
                owner_id=owner,
                slurm_job_name=slurm_job_name("allocation", operation_id),
                slurm_comment=format_tag(
                    owner, operation_id, "laptop", "allocation", name
                ),
                resources=self.resources,
            )
            self.repository.acknowledge_submission(operation_id, job_id)
            self.repository.update_job(
                operation_id,
                phase=JobPhase.ACTIVE,
                ever_started=True,
                current_node=node,
                last_node=node,
            )
        self.repository.begin_cancel(OPERATION_ID, operation_id="c" * 32)
        self.repository.begin_cancel(second, operation_id="d" * 32)
        _sync_ssh_projection(self.context, self.paths)
        failure = TransportLost("later recovery observation failed")
        calls = 0

        def recover(ctx: object, _client: object, _operation: object, job: object) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                ctx.state.update_job(
                    job.operation_id,
                    phase=JobPhase.FINAL,
                    terminal_state="CANCELLED",
                    exit_code="0:15",
                    final_source=FinalSource.ACCOUNTING,
                )
                return True
            raise failure

        with patch("hpc_alloc.commands._recover_cancel", side_effect=recover):
            with self.assertRaises(TransportLost) as raised:
                cmd_recover(
                    SimpleNamespace(
                        operation_id=None,
                        cluster=None,
                        abandon=False,
                        yes=False,
                    ),
                    ctx=self.context,
                    paths=self.paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

        self.assertIs(raised.exception, failure)
        projected = self.paths.managed_ssh_config.read_text()
        self.assertNotIn("Host hpc-grace.dev", projected)
        self.assertIn("Host hpc-secondary.viz", projected)


if __name__ == "__main__":
    unittest.main()
