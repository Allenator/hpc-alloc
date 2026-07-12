from __future__ import annotations

import unittest

from hpc_alloc.errors import (
    AmbiguousSubmission,
    AuthRequired,
    HostKeyChanged,
    JobIdReused,
    RemoteCommandFailed,
    TransportLost,
)
from hpc_alloc.models import JobKind, JobRef
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.slurm import (
    CancellationInspectionStatus,
    CancellationStatus,
    SlurmClient,
    SubmissionSpec,
)
from hpc_alloc.ssh import AuthMode, RemoteResult, RetryPolicy

from .fakes import ExpectedCall, StrictProxy, StrictScript


NONCE = "strict"
DELIMITER = f"__HPC_{NONCE}__"
TIME_MARKER = f"__HPC_TIME_{NONCE}__"
REMOTE_TIME = "2026-07-10T12:00:00"


def framed(payload: bytes | str, *, command_rc: int = 0, stderr: str = "") -> RemoteResult:
    """Build the exact byte envelope returned by ``SlurmClient._framed``."""

    body = payload.encode() if isinstance(payload, str) else payload
    marker = f"\x1eHPC_ALLOC_V2_{NONCE} {command_rc} {len(body)}\n".encode()
    # Startup noise before the marker must never affect protocol parsing.
    return RemoteResult(0, b"login banner\n" + marker + body, stderr)


def queue_payload(
    ref: JobRef,
    comment: str,
    *,
    state: str = "RUNNING",
    job_name: str | None = None,
) -> str:
    fields = (
        ref.job_id,
        state,
        "node01" if state == "RUNNING" else "",
        "None",
        "1:00:00",
        "day",
        job_name or ref.slurm_job_name,
        "2026-07-10T11:00:00",
        comment,
    )
    return DELIMITER.join(fields) + f"\n{TIME_MARKER}{REMOTE_TIME}\n"


def empty_queue_payload() -> str:
    return f"\n{TIME_MARKER}{REMOTE_TIME}\n"


class SlurmMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.owner_id = "deadbeef1234"
        self.operation_id = "a" * 32
        self.job_name = slurm_job_name("run", self.operation_id)
        self.comment = format_tag(
            self.owner_id,
            self.operation_id,
            "laptop",
            "run",
            None,
        )
        self.ref = JobRef(
            cluster="grace",
            job_id="12345",
            owner_id=self.owner_id,
            operation_id=self.operation_id,
            slurm_job_name=self.job_name,
            slurm_comment=self.comment,
        )

    def client(self, script: StrictScript) -> SlurmClient:
        return SlurmClient(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref.cluster,
            sleeper=lambda _seconds: None,
            token_factory=lambda _size: NONCE,
        )

    def submission_spec(self) -> SubmissionSpec:
        return SubmissionSpec(
            operation_id=self.operation_id,
            owner_id=self.owner_id,
            owner_host="laptop",
            kind=JobKind.RUN,
            logical_name="run",
            partition="day",
            walltime="1:00:00",
            cpus=2,
            logfile=".hpc-alloc/run.log",
            wrap="echo ok",
        )

    def test_submit_transport_loss_is_ambiguous_and_never_retried(self) -> None:
        def lose_once(cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertEqual(cluster, "grace")
            self.assertIn("sbatch --parsable", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            self.assertEqual(kwargs["auth"], AuthMode.NONINTERACTIVE)
            raise TransportLost("reply lost after possible commit")

        script = StrictScript([ExpectedCall("run", result=lose_once)])
        with self.assertRaisesRegex(AmbiguousSubmission, "may have committed"):
            self.client(script).submit(["sbatch", "--parsable", "--wrap", "echo ok"])
        self.assertEqual(script.count("run"), 1, "a non-idempotent sbatch must not be retried")
        script.assert_complete()

    def test_submit_preserves_definitive_pre_dispatch_ssh_failures(self) -> None:
        failures = (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                script = StrictScript([ExpectedCall("run", result=failure)])
                with self.assertRaises(type(failure)) as raised:
                    self.client(script).submit(self.submission_spec())
                self.assertIs(raised.exception, failure)
                self.assertEqual(script.count("run"), 1)
                script.assert_complete()

    def test_submit_malformed_ack_is_ambiguous_not_resubmitted(self) -> None:
        def malformed_ack(_cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertIn("sbatch --parsable", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            return framed("shell-noise\n12345")

        script = StrictScript([ExpectedCall("run", result=malformed_ack)])
        with self.assertRaisesRegex(AmbiguousSubmission, "untrustworthy reply"):
            self.client(script).submit("sbatch --parsable --wrap true")
        self.assertEqual(script.count("run"), 1)
        script.assert_complete()

    def test_submission_preparation_is_retry_safe_and_contains_no_sbatch(self) -> None:
        def prepared(_cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertIn("mkdir -p", command)
            self.assertIn("find", command)
            self.assertNotIn("sbatch", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.SAFE_READ)
            return framed(b"")

        script = StrictScript([ExpectedCall("run", result=prepared)])
        self.client(script).prepare_submission(self.submission_spec())
        script.assert_complete()

    def test_submission_preparation_failure_is_definitive_and_never_dispatches(self) -> None:
        script = StrictScript(
            [
                ExpectedCall(
                    "run",
                    result=framed(
                        b"", command_rc=1, stderr="mkdir: quota exceeded\n"
                    ),
                )
            ]
        )
        with self.assertRaisesRegex(RemoteCommandFailed, "quota exceeded"):
            self.client(script).prepare_submission(self.submission_spec())
        remote_commands = [args[1] for _name, args, _kwargs in script.ledger]
        self.assertFalse(any("sbatch" in command for command in remote_commands))
        script.assert_complete()

    def test_submission_spec_dispatch_contains_only_the_one_shot_sbatch(self) -> None:
        def acknowledged(
            _cluster: str, command: str, **kwargs: object
        ) -> RemoteResult:
            self.assertTrue(command.find("sbatch --parsable") >= 0)
            self.assertNotIn("mkdir", command)
            self.assertNotIn("find", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            return framed("12345;grace\n")

        script = StrictScript([ExpectedCall("run", result=acknowledged)])
        result = self.client(script).submit(self.submission_spec())
        self.assertEqual(result.job_id, "12345")
        script.assert_complete()

    def test_every_nonzero_sbatch_result_is_ambiguous_after_dispatch(self) -> None:
        cases = {
            "empty": framed(b"", command_rc=1),
            "stderr": framed(
                b"", command_rc=1, stderr="sbatch: invalid partition\n"
            ),
            "numeric-output": framed("12345\n", command_rc=1),
        }
        for label, reply in cases.items():
            with self.subTest(result=label):
                script = StrictScript([ExpectedCall("run", result=reply)])
                with self.assertRaisesRegex(AmbiguousSubmission, "may have committed"):
                    self.client(script).submit(self.submission_spec())
                self.assertEqual(script.count("run"), 1)
                script.assert_complete()

    def test_cancel_requires_exact_owner_operation_and_job_name(self) -> None:
        wrong_owner = format_tag(
            "cafebabefeed", self.operation_id, "desktop", "run", None
        )
        wrong_operation = format_tag(
            self.owner_id, "b" * 32, "laptop", "run", None
        )
        cases = {
            "empty-live-comment": ("", self.job_name),
            "owner": (wrong_owner, self.job_name),
            "operation": (wrong_operation, self.job_name),
            "full-comment": (
                format_tag(
                    self.owner_id, self.operation_id, "desktop", "run", None
                ),
                self.job_name,
            ),
            "job-name": (self.comment, slurm_job_name("run", "b" * 32)),
        }

        for dimension, (comment, job_name) in cases.items():
            with self.subTest(dimension=dimension):
                def foreign_row(
                    _cluster: str,
                    command: str,
                    **kwargs: object,
                ) -> RemoteResult:
                    self.assertIn("squeue --me", command)
                    self.assertEqual(kwargs["retry"], RetryPolicy.SAFE_READ)
                    return framed(queue_payload(self.ref, comment, job_name=job_name))

                script = StrictScript([ExpectedCall("run", result=foreign_row)])
                result = self.client(script).inspect_cancel(self.ref)
                self.assertEqual(
                    result.status,
                    CancellationInspectionStatus.IDENTITY_MISMATCH,
                )
                self.assertEqual(
                    script.count("run"),
                    1,
                    "identity mismatch must stop before scancel",
                )
                script.assert_complete()

    def test_live_name_and_comment_mismatch_is_typed_as_recycled_id(self) -> None:
        foreign_name = slurm_job_name("run", "b" * 32)
        foreign_comment = format_tag(
            "cafebabefeed", "b" * 32, "desktop", "run", None
        )
        reply = framed(
            queue_payload(
                self.ref,
                foreign_comment,
                job_name=foreign_name,
            )
        )
        observe_script = StrictScript([ExpectedCall("run", result=reply)])
        client = self.client(observe_script)
        with self.assertRaises(JobIdReused):
            client.observe(self.ref)
        observe_script.assert_complete()

        script = StrictScript([ExpectedCall("run", result=reply)])
        inspection = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            inspection.status, CancellationInspectionStatus.IDENTITY_MISMATCH
        )
        self.assertEqual(script.count("run"), 1)
        script.assert_complete()

    def test_owned_cancel_uses_safe_read_then_single_nonretrying_scancel(self) -> None:
        def owned_row(_cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertIn("squeue --me", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.SAFE_READ)
            return framed(queue_payload(self.ref, self.comment))

        def scancel(_cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertIn("row=$(squeue --me", command)
            self.assertIn(self.comment, command)
            self.assertIn(f"scancel -- {self.ref.job_id}", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            return framed(b"")

        script = StrictScript(
            [ExpectedCall("run", result=owned_row), ExpectedCall("run", result=scancel)]
        )
        client = self.client(script)
        inspection = client.inspect_cancel(self.ref)
        self.assertEqual(inspection.status, CancellationInspectionStatus.READY)
        result = client.execute_cancel(self.ref)
        self.assertEqual(result.status, CancellationStatus.CANCELLED)
        self.assertEqual(script.count("run"), 2)
        script.assert_complete()

    def test_mid_scancel_transport_loss_remains_typed_and_is_not_retried(self) -> None:
        def owned_row(_cluster: str, _command: str, **_kwargs: object) -> RemoteResult:
            return framed(queue_payload(self.ref, self.comment))

        def lose_scancel(_cluster: str, command: str, **kwargs: object) -> RemoteResult:
            self.assertIn("row=$(squeue --me", command)
            self.assertIn("scancel --", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            raise TransportLost("connection dropped mid-scancel")

        script = StrictScript(
            [ExpectedCall("run", result=owned_row), ExpectedCall("run", result=lose_scancel)]
        )
        client = self.client(script)
        inspection = client.inspect_cancel(self.ref)
        self.assertEqual(inspection.status, CancellationInspectionStatus.READY)
        result = client.execute_cancel(self.ref)
        self.assertEqual(result.status, CancellationStatus.MUTATION_AMBIGUOUS)
        self.assertIn("mid-scancel", result.detail)
        self.assertEqual(script.count("run"), 2, "scancel must be issued at most once")
        script.assert_complete()

    def test_execute_cancel_preserves_definitive_pre_dispatch_ssh_failures(self) -> None:
        failures = (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                script = StrictScript([ExpectedCall("run", result=failure)])
                with self.assertRaises(type(failure)) as raised:
                    self.client(script).execute_cancel(self.ref)
                self.assertIs(raised.exception, failure)
                self.assertEqual(script.count("run"), 1)
                script.assert_complete()

    def test_preflight_transport_failure_propagates_before_any_mutation(self) -> None:
        script = StrictScript(
            [ExpectedCall("run", result=TransportLost("login node unreachable"))]
        )
        with self.assertRaisesRegex(TransportLost, "unreachable"):
            self.client(script).inspect_cancel(self.ref)
        remote_commands = [args[1] for _name, args, _kwargs in script.ledger]
        self.assertEqual(len(remote_commands), 1)
        self.assertFalse(any("scancel" in command for command in remote_commands))
        script.assert_complete()

    def test_untrusted_post_dispatch_replies_are_mutation_ambiguous(self) -> None:
        cases = {
            "missing-frame": RemoteResult(0, b"no trusted frame", ""),
            "success-with-output": framed(b"unexpected", command_rc=0),
            "unexpected-status": framed(b"", command_rc=42),
        }
        for label, reply in cases.items():
            with self.subTest(reply=label):
                script = StrictScript([ExpectedCall("run", result=reply)])
                result = self.client(script).execute_cancel(self.ref)
                self.assertEqual(
                    result.status, CancellationStatus.MUTATION_AMBIGUOUS
                )
                self.assertEqual(script.count("run"), 1)
                script.assert_complete()

    def test_identity_change_at_mutation_boundary_prevents_cancellation(self) -> None:
        def owned_row(_cluster: str, _command: str, **_kwargs: object) -> RemoteResult:
            return framed(queue_payload(self.ref, self.comment))

        def guarded_mismatch(
            _cluster: str, command: str, **kwargs: object
        ) -> RemoteResult:
            self.assertIn("row=$(squeue --me", command)
            self.assertIn("|| exit 45", command)
            self.assertEqual(kwargs["retry"], RetryPolicy.NEVER)
            return framed(b"", command_rc=45)

        script = StrictScript(
            [
                ExpectedCall("run", result=owned_row),
                ExpectedCall("run", result=guarded_mismatch),
            ]
        )
        client = self.client(script)
        inspection = client.inspect_cancel(self.ref)
        self.assertEqual(inspection.status, CancellationInspectionStatus.READY)
        result = client.execute_cancel(self.ref)
        self.assertEqual(result.status, CancellationStatus.IDENTITY_MISMATCH)
        self.assertIn("changed identity", result.detail)
        script.assert_complete()

    def test_guard_absence_with_accounting_lag_is_left_queue_not_cancelled(self) -> None:
        def owned_row(_cluster: str, _command: str, **_kwargs: object) -> RemoteResult:
            return framed(queue_payload(self.ref, self.comment))

        for label, guard in (
            ("empty-row", framed(b"", command_rc=44)),
            (
                "invalid-id-no-newline",
                framed(
                    b"",
                    command_rc=46,
                    stderr="slurm_load_jobs error: Invalid job id specified",
                ),
            ),
            (
                "invalid-id-newline",
                framed(
                    b"",
                    command_rc=46,
                    stderr="slurm_load_jobs error: Invalid job id specified\n",
                ),
            ),
        ):
            with self.subTest(guard=label):
                script = StrictScript(
                    [
                        ExpectedCall("run", result=owned_row),
                        ExpectedCall("run", result=guard),
                    ]
                )
                client = self.client(script)
                inspection = client.inspect_cancel(self.ref)
                self.assertEqual(
                    inspection.status, CancellationInspectionStatus.READY
                )
                result = client.execute_cancel(self.ref)
                self.assertEqual(result.status, CancellationStatus.LEFT_QUEUE)
                self.assertIn("scancel was not issued", result.detail)
                script.assert_complete()

    def test_scancel_failure_is_ambiguous_and_never_followed_by_accounting(self) -> None:
        script = StrictScript(
            [
                ExpectedCall(
                    "run",
                    result=framed(
                        b"", command_rc=47, stderr="scancel: Invalid job id\n"
                    ),
                )
            ]
        )
        result = self.client(script).execute_cancel(self.ref)
        self.assertEqual(result.status, CancellationStatus.MUTATION_AMBIGUOUS)
        self.assertIn("Invalid job id", result.detail)
        remote_commands = [args[1] for _name, args, _kwargs in script.ledger]
        self.assertEqual(len(remote_commands), 1)
        self.assertFalse(any("sacct" in command for command in remote_commands))
        script.assert_complete()

    def test_guard_failure_shapes_are_never_normalized_to_absence(self) -> None:
        exact_error = "slurm_load_jobs error: Invalid job id specified\n"
        cases = {
            "invalid-id-with-output": framed(
                b"unexpected", command_rc=46, stderr=exact_error
            ),
            "different-squeue-error": framed(
                b"", command_rc=46, stderr="slurm_load_jobs error: Socket timed out\n"
            ),
            "empty-row-with-stderr": framed(
                b"", command_rc=44, stderr="warning: controller failover\n"
            ),
            "empty-row-with-output": framed(b"unexpected", command_rc=44),
        }
        for label, guard in cases.items():
            with self.subTest(shape=label):
                script = StrictScript([ExpectedCall("run", result=guard)])
                result = self.client(script).execute_cancel(self.ref)
                self.assertEqual(result.status, CancellationStatus.GUARD_FAILED)
                script.assert_complete()

    def test_running_accounting_is_not_misreported_as_already_final(self) -> None:
        def absent_row(_cluster: str, command: str, **_kwargs: object) -> RemoteResult:
            self.assertIn("squeue --me", command)
            return framed(empty_queue_payload())

        def running_record(_cluster: str, command: str, **_kwargs: object) -> RemoteResult:
            self.assertIn("sacct -j 12345", command)
            return framed(
                f"{self.ref.job_id}|RUNNING|0:0|{self.job_name}|{self.comment}\n"
            )

        script = StrictScript(
            [
                ExpectedCall("run", result=absent_row),
                ExpectedCall("run", result=running_record),
                ExpectedCall("run", result=absent_row),
                ExpectedCall("run", result=running_record),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.CONFIRMED_ABSENT
        )
        self.assertNotEqual(
            result.status, CancellationInspectionStatus.ALREADY_FINAL
        )
        script.assert_complete()

    def test_single_absence_then_live_row_is_ready_for_guarded_cancel(self) -> None:
        sleeps: list[float] = []
        script = StrictScript(
            [
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(b"")),
                ExpectedCall(
                    "run", result=framed(queue_payload(self.ref, self.comment))
                ),
            ]
        )
        client = SlurmClient(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref.cluster,
            sleeper=sleeps.append,
            token_factory=lambda _size: NONCE,
        )
        result = client.inspect_cancel(self.ref)
        self.assertEqual(result.status, CancellationInspectionStatus.READY)
        self.assertEqual(sleeps, [3])
        script.assert_complete()

    def test_failure_between_absence_observations_never_confirms_departure(self) -> None:
        sleeps: list[float] = []
        script = StrictScript(
            [
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(b"")),
                ExpectedCall("run", result=TransportLost("controller unavailable")),
            ]
        )
        client = SlurmClient(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref.cluster,
            sleeper=sleeps.append,
            token_factory=lambda _size: NONCE,
        )
        with self.assertRaisesRegex(TransportLost, "unavailable"):
            client.inspect_cancel(self.ref)
        self.assertEqual(sleeps, [3])
        script.assert_complete()

    def test_cancel_confirmation_delay_validation_performs_no_remote_work(self) -> None:
        for delay in (-1, float("nan"), float("inf"), True):
            with self.subTest(delay=delay):
                script = StrictScript([])
                with self.assertRaises(ValueError):
                    self.client(script).inspect_cancel(
                        self.ref, confirmation_delay=delay  # type: ignore[arg-type]
                    )
                script.assert_complete()

    def test_final_accounting_after_second_absence_short_circuits(self) -> None:
        final = (
            f"{self.ref.job_id}|COMPLETED|0:0|{self.job_name}|{self.comment}\n"
        )
        script = StrictScript(
            [
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(b"")),
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(final)),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.ALREADY_FINAL
        )
        self.assertIsNotNone(result.final_record)
        script.assert_complete()

    def test_exact_invalid_singleton_must_also_be_observed_twice(self) -> None:
        invalid = framed(
            b"",
            command_rc=1,
            stderr="slurm_load_jobs error: Invalid job id specified\n",
        )
        script = StrictScript(
            [
                ExpectedCall("run", result=invalid),
                ExpectedCall("run", result=framed(b"")),
                ExpectedCall("run", result=invalid),
                ExpectedCall("run", result=framed(b"")),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.CONFIRMED_ABSENT
        )
        script.assert_complete()

    def test_empty_comment_accounting_can_finalize_locally_but_never_issues_scancel(self) -> None:
        def absent_row(_cluster: str, command: str, **_kwargs: object) -> RemoteResult:
            self.assertIn("squeue --me", command)
            return framed(empty_queue_payload())

        def name_only_record(
            _cluster: str, command: str, **_kwargs: object
        ) -> RemoteResult:
            self.assertIn("sacct -j 12345", command)
            return framed(
                f"{self.ref.job_id}|COMPLETED|0:0|{self.job_name}|\n"
            )

        script = StrictScript(
            [
                ExpectedCall("run", result=absent_row),
                ExpectedCall("run", result=name_only_record),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.ALREADY_FINAL
        )
        self.assertIsNotNone(result.final_record)
        assert result.final_record is not None
        self.assertEqual(result.final_record.comment, "")
        remote_commands = [args[1] for _name, args, _kwargs in script.ledger]
        self.assertFalse(any("scancel" in command for command in remote_commands))
        script.assert_complete()

    def test_foreign_accounting_name_is_missing_not_an_identity_failure(self) -> None:
        foreign = (
            f"{self.ref.job_id}|COMPLETED|0:0|"
            f"{slurm_job_name('run', 'b' * 32)}|foreign-comment\n"
        )
        script = StrictScript(
            [
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(foreign)),
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(foreign)),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.CONFIRMED_ABSENT
        )
        script.assert_complete()

    def test_exact_accounting_name_with_wrong_comment_is_identity_mismatch(self) -> None:
        wrong_comment = format_tag(
            self.owner_id,
            self.operation_id,
            "desktop",
            "run",
            None,
        )
        record = (
            f"{self.ref.job_id}|COMPLETED|0:0|{self.job_name}|{wrong_comment}\n"
        )
        script = StrictScript(
            [
                ExpectedCall("run", result=framed(empty_queue_payload())),
                ExpectedCall("run", result=framed(record)),
            ]
        )
        result = self.client(script).inspect_cancel(self.ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.IDENTITY_MISMATCH
        )
        remote_commands = [args[1] for _name, args, _kwargs in script.ledger]
        self.assertFalse(any("scancel" in command for command in remote_commands))
        script.assert_complete()


if __name__ == "__main__":
    unittest.main()
