from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.commands import (
    _cancel_record,
    _recover_cancel,
    _recover_submission,
    _submit_job,
    cmd_recover,
)
from hpc_alloc.errors import (
    AmbiguousSubmission,
    AuthRequired,
    HostKeyChanged,
    IdentityMismatch,
    SchedulerUnavailable,
    StateConflict,
    TransportLost,
)
from hpc_alloc.models import FinalSource, JobKind, JobPhase, OperationPhase
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
        client_script.assert_complete()

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


if __name__ == "__main__":
    unittest.main()
