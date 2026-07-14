from __future__ import annotations

import shlex
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.commands import _recover_submission
from hpc_alloc.errors import (
    IdentityMismatch,
    ProtocolViolation,
    RemoteCommandFailed,
    SchedulerUnavailable,
)
from hpc_alloc.models import JobKind, JobPhase, JobRecord, JobRef
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.slurm import (
    AccountingRecord,
    CancellationInspectionStatus,
    MAX_LOG_CHUNK_BYTES,
    RawQueueScan,
    SlurmClient,
    SubmissionSpec,
    _LOG_CHUNK_SIGPIPE_STATUS,
)
from hpc_alloc.ssh import RemoteResult, RetryPolicy


NONCE = "fixed"


def framed(
    payload: bytes | str,
    *,
    rc: int = 0,
    declared: int | None = None,
    stderr: bytes | str = "",
    declared_stderr: int | None = None,
    startup_stderr: str = "",
) -> RemoteResult:
    body = payload.encode() if isinstance(payload, str) else payload
    stderr_body = stderr.encode() if isinstance(stderr, str) else stderr
    length = len(body) if declared is None else declared
    stderr_length = (
        len(stderr_body) if declared_stderr is None else declared_stderr
    )
    header = (
        f"\x1eHPC_ALLOC_V2_{NONCE} {rc} {length} {stderr_length}\n".encode()
    )
    return RemoteResult(
        0,
        b"banner before command\n" + header + body + stderr_body,
        startup_stderr,
    )


class FakeTransport:
    def __init__(self, replies: list[RemoteResult]) -> None:
        self.replies = replies
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def run(self, cluster: str, command: str, **kwargs: object) -> RemoteResult:
        self.calls.append((cluster, command, dict(kwargs)))
        return self.replies.pop(0)


class SlurmProtocolTests(unittest.TestCase):
    def client(self, replies: list[RemoteResult]) -> tuple[SlurmClient, FakeTransport]:
        transport = FakeTransport(replies)
        return (
            SlurmClient(transport, "grace", token_factory=lambda _size: NONCE),  # type: ignore[arg-type]
            transport,
        )

    @staticmethod
    def managed_ref() -> JobRef:
        operation_id = "a" * 32
        comment = format_tag("deadbeef1234", operation_id, "laptop", "run", None)
        return JobRef(
            cluster="grace",
            job_id="123",
            owner_id="deadbeef1234",
            operation_id=operation_id,
            slurm_job_name=slurm_job_name("run", operation_id),
            slurm_comment=comment,
        )

    @staticmethod
    def queue_payload(ref: JobRef, *, comment: str) -> str:
        delimiter = f"__HPC_{NONCE}__"
        fields = (
            ref.job_id,
            "RUNNING",
            "node01",
            "None",
            "1:00:00",
            "day",
            ref.slurm_job_name,
            "2026-07-10T11:00:00",
            comment,
        )
        return delimiter.join(fields) + f"\n__HPC_TIME_{NONCE}__2026-07-10T12:00:00\n"

    @staticmethod
    def empty_queue_payload() -> str:
        return f"\n__HPC_TIME_{NONCE}__2026-07-10T12:00:00\n"

    def test_declared_length_mismatch_is_not_treated_as_valid_output(self) -> None:
        client, _transport = self.client([framed(b"MISSING", declared=99)])
        with self.assertRaisesRegex(ProtocolViolation, "length mismatch"):
            client.log_size("log")

    def test_declared_stderr_length_mismatch_is_not_a_valid_frame(self) -> None:
        client, _transport = self.client(
            [framed(b"MISSING", stderr="scheduler failed\n", declared_stderr=99)]
        )
        with self.assertRaisesRegex(ProtocolViolation, "length mismatch"):
            client.log_size("log")

    def test_malformed_dual_stream_header_is_rejected(self) -> None:
        header = f"\x1eHPC_ALLOC_V2_{NONCE} 0 0\n".encode()
        client, _transport = self.client(
            [RemoteResult(0, b"banner\n" + header, "startup warning\n")]
        )
        with self.assertRaisesRegex(ProtocolViolation, "malformed frame header"):
            client.log_size("log")

    def test_missing_frame_preserves_transport_stderr_diagnostic(self) -> None:
        client, _transport = self.client(
            [RemoteResult(0, b"no frame", "login shell failed\n")]
        )
        with self.assertRaisesRegex(ProtocolViolation, "login shell failed"):
            client.log_size("log")

    def test_payload_containing_exact_marker_is_not_reframed(self) -> None:
        marker = b"\x1eHPC_ALLOC_V2_fixed 0 1 0\n"
        payload = b"before" + marker + b"after\xff"
        client, _transport = self.client([framed(payload)])
        self.assertEqual(client.read_log_chunk("log", 0), payload)

    def test_log_chunk_rejects_oversized_declared_frame_before_payload(self) -> None:
        client, transport = self.client([framed(b"", declared=11)])
        with self.assertRaisesRegex(ProtocolViolation, "payload limit"):
            client.read_log_chunk("log", 7, limit=10)
        command = transport.calls[0][1]
        self.assertIn("tail -c +8 -- log; source_rc=$?", command)
        self.assertIn("| head -c 10", command)
        self.assertIn('kill -l "$source_rc"', command)
        self.assertIn(f"exit {_LOG_CHUNK_SIGPIPE_STATUS}", command)
        self.assertIn('[ "$n" -le 10 ]', command)

    def test_log_chunk_accepts_source_sigpipe_only_at_the_byte_cap(self) -> None:
        client, _transport = self.client(
            [framed(b"0123456789", rc=_LOG_CHUNK_SIGPIPE_STATUS)]
        )
        self.assertEqual(client.read_log_chunk("log", 0, limit=10), b"0123456789")

        client, _transport = self.client(
            [framed(b"short", rc=_LOG_CHUNK_SIGPIPE_STATUS)]
        )
        with self.assertRaisesRegex(RemoteCommandFailed, "SIGPIPE.*byte limit"):
            client.read_log_chunk("log", 0, limit=10)

    def test_log_chunk_limit_validation_is_strict(self) -> None:
        client, transport = self.client([])
        for limit in (0, MAX_LOG_CHUNK_BYTES + 1, True, 1.5):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    client.read_log_chunk("log", 0, limit=limit)  # type: ignore[arg-type]
        self.assertEqual(transport.calls, [])

    def test_tail_is_byte_bounded_even_when_line_count_is_large(self) -> None:
        client, transport = self.client(
            [framed(b"", declared=MAX_LOG_CHUNK_BYTES + 1)]
        )
        with self.assertRaisesRegex(ProtocolViolation, "payload limit"):
            client.tail_log("log", 999_999)
        command = transport.calls[0][1]
        source = f"tail -c {MAX_LOG_CHUNK_BYTES} -- log; source_rc=$?"
        sink = "| tail -n 999999; sink_rc=$?"
        self.assertIn(source, command)
        self.assertIn(sink, command)
        self.assertLess(command.index(source), command.index(sink))
        self.assertIn(f'[ "$n" -le {MAX_LOG_CHUNK_BYTES} ]', command)

    def test_nonzero_scheduler_status_is_typed_not_empty_snapshot(self) -> None:
        client, _transport = self.client([framed(b"", rc=1)])
        with self.assertRaises(SchedulerUnavailable):
            client.snapshot()

    def test_only_framed_command_stderr_is_reported(self) -> None:
        client, _transport = self.client(
            [
                framed(
                    b"",
                    rc=1,
                    stderr=b"scheduler failed: \xff\n",
                    startup_stderr="site startup warning\n",
                )
            ]
        )
        with self.assertRaises(SchedulerUnavailable) as raised:
            client.snapshot()
        self.assertIn("scheduler failed: \ufffd", str(raised.exception))
        self.assertNotIn("site startup warning", str(raised.exception))

    def test_singleton_invalid_job_response_is_normalized_to_absence(self) -> None:
        error = "slurm_load_jobs error: Invalid job id specified\n"
        client, transport = self.client(
            [
                framed(
                    b"",
                    rc=1,
                    stderr=error,
                    startup_stderr="site startup warning\n",
                )
            ]
        )
        snapshot = client.snapshot(["123"])
        self.assertEqual(dict(snapshot.rows), {})
        self.assertIn(" -j 123 ", transport.calls[0][1])

    def test_invalid_job_normalization_rejects_every_broader_shape(self) -> None:
        exact_error = "slurm_load_jobs error: Invalid job id specified\n"
        cases = (
            ("unfiltered", None, framed(b"", rc=1, stderr=exact_error)),
            ("multiple IDs", ["123", "124"], framed(b"", rc=1, stderr=exact_error)),
            ("output bearing", ["123"], framed(b"\n", rc=1, stderr=exact_error)),
            ("wrong rc", ["123"], framed(b"", rc=2, stderr=exact_error)),
            (
                "different error",
                ["123"],
                framed(b"", rc=1, stderr="slurm_load_jobs error: Socket timed out\n"),
            ),
            (
                "extra stderr",
                ["123"],
                framed(b"", rc=1, stderr="warning\n" + exact_error),
            ),
            (
                "extra trailing line",
                ["123"],
                framed(b"", rc=1, stderr=exact_error + "\n"),
            ),
        )
        for label, selected, reply in cases:
            with self.subTest(label=label):
                client, _transport = self.client([reply])
                with self.assertRaises(SchedulerUnavailable):
                    client.snapshot(selected)

    def test_broad_scan_preserves_valid_arrays_and_multi_node_expressions(self) -> None:
        delimiter = f"__HPC_{NONCE}__"
        rows = [
            delimiter.join(
                (
                    "123_4",
                    "RUNNING",
                    "node[01-04]",
                    "None",
                    "1:00:00",
                    "day",
                    "foreign-array",
                    "2026-07-10T11:00:00",
                    "foreign-comment",
                )
            ),
            delimiter.join(
                (
                    "987",
                    "PENDING",
                    "",
                    "Resources",
                    "2:00:00",
                    "gpu*",
                    "foreign-singleton",
                    "2026-07-10T11:05:00",
                    "",
                )
            ),
        ]
        payload = (
            "\n".join(rows)
            + f"\n__HPC_TIME_{NONCE}__2026-07-10T12:00:00\n"
        )
        client, transport = self.client([framed(payload)])
        scan = client.scan()
        self.assertEqual(
            [candidate.job_id for candidate in scan.rows], ["123_4", "987"]
        )
        self.assertEqual(scan.rows[0].node, "node[01-04]")
        self.assertEqual(scan.rows[1].partition, "gpu*")
        self.assertNotIn(" -j ", transport.calls[0][1])

    def test_targeted_observation_rejects_broad_only_shapes(self) -> None:
        ref = self.managed_ref()
        delimiter = f"__HPC_{NONCE}__"
        array_row = delimiter.join(
            (
                "123_4",
                "RUNNING",
                "node01",
                "None",
                "1:00:00",
                "day",
                ref.slurm_job_name,
                "2026-07-10T11:00:00",
                ref.slurm_comment,
            )
        )
        payload = (
            array_row + f"\n__HPC_TIME_{NONCE}__2026-07-10T12:00:00\n"
        )
        client, _transport = self.client([framed(payload)])
        with self.assertRaisesRegex(ProtocolViolation, "unexpected job IDs"):
            client.observe(ref)

        multi_node = self.queue_payload(ref, comment=ref.slurm_comment).replace(
            "node01", "node[01-04]", 1
        )
        client, _transport = self.client([framed(multi_node)])
        with self.assertRaisesRegex(ProtocolViolation, "compute-node name"):
            client.observe(ref)

    def test_broad_scan_fails_closed_on_malformed_or_control_bearing_rows(self) -> None:
        delimiter = f"__HPC_{NONCE}__"
        clock = f"\n__HPC_TIME_{NONCE}__2026-07-10T12:00:00\n"
        valid = [
            "123_4",
            "RUNNING",
            "node[01-04]",
            "None",
            "1:00:00",
            "day",
            "foreign",
            "2026-07-10T11:00:00",
            "comment",
        ]
        cases = {
            "field count": delimiter.join(valid[:-1]) + clock,
            "control": delimiter.join((*valid[:-1], "bad\tcomment")) + clock,
            "oversized": delimiter.join(
                (*valid[:-1], "x" * (64 * 1024 + 1))
            )
            + clock,
        }
        for label, payload in cases.items():
            with self.subTest(label=label):
                client, _transport = self.client([framed(payload)])
                with self.assertRaises(ProtocolViolation):
                    client.scan()

        client, _transport = self.client(
            [framed(b"123\xff\n__HPC_TIME_fixed__2026-07-10T12:00:00\n")]
        )
        with self.assertRaisesRegex(ProtocolViolation, "non-UTF-8"):
            client.scan()

    def test_broad_scan_frame_has_a_hard_payload_ceiling(self) -> None:
        client, transport = self.client(
            [framed(b"", declared=32 * 1024 * 1024 + 1)]
        )
        with self.assertRaisesRegex(ProtocolViolation, "payload limit"):
            client.scan()
        self.assertIn('[ "$n" -le 33554432 ]', transport.calls[0][1])

    def test_singleton_absence_can_fall_through_to_exact_accounting(self) -> None:
        ref = self.managed_ref()
        invalid = framed(
            b"",
            rc=1,
            stderr="slurm_load_jobs error: Invalid job id specified\n",
        )
        accounting = framed(
            f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|\n"
        )
        client, _transport = self.client([invalid, accounting])
        self.assertIsNone(client.observe(ref))
        record = client.final(ref)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.state_code, "COMPLETED")

    def test_special_exit_accounting_is_not_a_final_verdict(self) -> None:
        ref = self.managed_ref()
        accounting = framed(
            f"{ref.job_id}|SPECIAL_EXIT|0:0|{ref.slurm_job_name}|\n"
        )
        client, _transport = self.client([accounting])

        self.assertIsNone(client.final(ref))

    def test_accounting_requires_exact_job_id_not_first_line(self) -> None:
        payload = "999|COMPLETED|0:0|other|tag\n"
        client, _transport = self.client([framed(payload)])
        with self.assertRaisesRegex(ProtocolViolation, "no trustworthy exact record"):
            client.accounting("123")

    def test_accounting_requests_full_identity_widths_before_extra_fields(self) -> None:
        operation_id = "a" * 32
        job_name = slurm_job_name("allocation", operation_id)
        comment = format_tag(
            "o" * 63,
            operation_id,
            "h" * 63,
            "allocation",
            "n" * 63,
        )
        self.assertEqual(len(job_name), 50)
        self.assertEqual(len(comment), 248)
        ref = JobRef(
            cluster="grace",
            job_id="123",
            owner_id="o" * 63,
            operation_id=operation_id,
            slurm_job_name=job_name,
            slurm_comment=comment,
        )
        client, transport = self.client(
            [framed(f"123|COMPLETED|0:0|{job_name}|{comment}|1:02|node01\n")]
        )

        record = client.accounting(ref, extra_fields=("Elapsed", "NodeList"))

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.job_name, job_name)
        self.assertEqual(record.comment, comment)
        self.assertEqual(record.extra, ("1:02", "node01"))
        self.assertIn(
            "-o JobIDRaw,State,ExitCode,JobName%255,Comment%255,Elapsed,NodeList",
            transport.calls[0][1],
        )

    def test_accounting_rejects_a_truncated_identity(self) -> None:
        ref = self.managed_ref()
        truncated_comment = ref.slurm_comment[:-1] + "+"
        client, _transport = self.client(
            [
                framed(
                    f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|"
                    f"{truncated_comment}\n"
                )
            ]
        )

        with self.assertRaises(IdentityMismatch):
            client.accounting(ref)

    def test_accounting_accepts_only_exact_derived_name_when_comment_is_omitted(self) -> None:
        ref = self.managed_ref()
        payload = f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|\n"
        client, _transport = self.client([framed(payload)])
        record = client.accounting(ref)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.comment, "")

        wrong_name = slurm_job_name("run", "b" * 32)
        client, _transport = self.client(
            [framed(f"{ref.job_id}|COMPLETED|0:0|{wrong_name}|\n")]
        )
        self.assertIsNone(client.accounting(ref))

        corrupt_ref = JobRef(
            cluster=ref.cluster,
            job_id=ref.job_id,
            owner_id=ref.owner_id,
            operation_id=ref.operation_id,
            slurm_job_name="manually-chosen-name",
            slurm_comment=ref.slurm_comment,
        )
        client, _transport = self.client(
            [framed(f"{ref.job_id}|COMPLETED|0:0|manually-chosen-name|\n")]
        )
        with self.assertRaises(IdentityMismatch):
            client.accounting(corrupt_ref)

    def test_accounting_rejects_every_nonblank_comment_mismatch(self) -> None:
        ref = self.managed_ref()
        wrong_comment = format_tag(
            "deadbeef1234", ref.operation_id, "other-host", "run", None
        )
        client, _transport = self.client(
            [
                framed(
                    f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|{wrong_comment}\n"
                )
            ]
        )
        with self.assertRaises(IdentityMismatch):
            client.accounting(ref)

        client, _transport = self.client(
            [framed(f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}| \n")]
        )
        with self.assertRaises(IdentityMismatch):
            client.accounting(ref)

    def test_accounting_filters_recycled_ids_by_operation_name_before_duplicates(self) -> None:
        ref = self.managed_ref()
        old_name = slurm_job_name("run", "b" * 32)
        old = f"{ref.job_id}|COMPLETED|0:0|{old_name}|old-comment\n"
        current = (
            f"{ref.job_id}|RUNNING|0:0|{ref.slurm_job_name}|{ref.slurm_comment}\n"
        )
        for payload in (old + current, current + old):
            with self.subTest(old_record_first=payload.startswith(old)):
                client, _transport = self.client([framed(payload)])
                record = client.accounting(ref)
                self.assertIsNotNone(record)
                assert record is not None
                self.assertEqual(record.job_name, ref.slurm_job_name)
                self.assertEqual(record.state_code, "RUNNING")

        client, _transport = self.client([framed(old)])
        self.assertIsNone(client.accounting(ref))

        client, _transport = self.client([framed(old + old)])
        self.assertIsNone(
            client.accounting(ref),
            "duplicate records for foreign operations remain irrelevant",
        )

    def test_accounting_rejects_foreign_name_with_exact_operation_comment(self) -> None:
        ref = self.managed_ref()
        foreign_name = slurm_job_name("run", "b" * 32)
        client, _transport = self.client(
            [
                framed(
                    f"{ref.job_id}|COMPLETED|0:0|{foreign_name}|"
                    f"{ref.slurm_comment}\n"
                )
            ]
        )

        with self.assertRaises(IdentityMismatch):
            client.accounting(ref)

    def test_accounting_still_rejects_ambiguous_current_or_raw_id_records(self) -> None:
        ref = self.managed_ref()
        current = (
            f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|{ref.slurm_comment}\n"
        )
        client, _transport = self.client([framed(current + current)])
        with self.assertRaisesRegex(ProtocolViolation, "duplicate parent"):
            client.accounting(ref)

        client, _transport = self.client([framed(current + current)])
        with self.assertRaisesRegex(ProtocolViolation, "duplicate parent"):
            client.accounting(ref.job_id)

    def test_live_queue_and_mutation_still_require_the_full_comment(self) -> None:
        ref = self.managed_ref()
        client, _transport = self.client([framed(self.queue_payload(ref, comment=""))])
        with self.assertRaises(IdentityMismatch):
            client.observe(ref)

        client, transport = self.client([framed(self.queue_payload(ref, comment=""))])
        result = client.inspect_cancel(ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.IDENTITY_MISMATCH
        )
        self.assertEqual(len(transport.calls), 1, "blank live comment must stop before scancel")

    def test_absent_cancel_may_use_exact_final_accounting_with_omitted_comment(self) -> None:
        ref = self.managed_ref()
        accounting = f"{ref.job_id}|COMPLETED|0:0|{ref.slurm_job_name}|\n"
        client, transport = self.client(
            [framed(self.empty_queue_payload()), framed(accounting)]
        )
        result = client.inspect_cancel(ref)
        self.assertEqual(
            result.status, CancellationInspectionStatus.ALREADY_FINAL
        )
        self.assertEqual(len(transport.calls), 2)
        self.assertNotIn("scancel", transport.calls[-1][1])

    def test_recovery_uses_accounting_specific_identity_rule(self) -> None:
        ref = self.managed_ref()
        record = AccountingRecord(
            ref.job_id, "COMPLETED", "0:0", ref.slurm_job_name, ""
        )

        class State:
            def __init__(self) -> None:
                self.acknowledged: tuple[str, str] | None = None
                self.updated = False

            def acknowledge_submission(self, operation_id: str, job_id: str):
                self.acknowledged = (operation_id, job_id)
                return JobRecord(
                    operation_id=operation_id,
                    cluster=ref.cluster,
                    logical_name="run",
                    kind=JobKind.RUN,
                    owner_id=ref.owner_id,
                    slurm_job_name=ref.slurm_job_name,
                    slurm_comment=ref.slurm_comment,
                    phase=JobPhase.QUEUED,
                    job_id=job_id,
                )

            def update_job(self, _operation_id: str, **_changes: object) -> JobRecord:
                self.updated = True
                return self.acknowledge_submission(ref.operation_id, ref.job_id)

        class Client:
            def __init__(self) -> None:
                self.accounting_verified = False

            def scan(self, **_kwargs: object) -> RawQueueScan:
                return RawQueueScan(())

            def find_accounting_by_name(self, _job_name: str, **_kwargs: object):
                return record

            def verify_accounting_identity(
                self, candidate: JobRef, job_name: str, comment: str
            ) -> None:
                SlurmClient.verify_accounting_identity(candidate, job_name, comment)
                self.accounting_verified = True

            def verify_live_identity(self, *_args: object) -> None:
                raise AssertionError("recovery must not apply the live queue comment rule")

        state = State()
        client = Client()
        context = SimpleNamespace(state=state)
        operation = SimpleNamespace(operation_id=ref.operation_id, cluster=ref.cluster)
        job = SimpleNamespace(
            owner_id=ref.owner_id,
            slurm_job_name=ref.slurm_job_name,
            slurm_comment=ref.slurm_comment,
        )
        with patch("hpc_alloc.commands.info"):
            recovered = _recover_submission(context, client, operation, job)
        self.assertTrue(recovered)
        self.assertTrue(client.accounting_verified)
        self.assertEqual(state.acknowledged, (ref.operation_id, ref.job_id))
        self.assertTrue(state.updated)

    def test_recovery_lookup_filters_exact_name_and_refuses_duplicates(self) -> None:
        name = "hpcalloc-v2-run-" + "a" * 32
        one = f"123|COMPLETED|0:0|{name}|tag\n"
        client, transport = self.client([framed(one)])
        record = client.find_accounting_by_name(name)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.job_id, "123")
        self.assertIn("sacct -X", transport.calls[0][1])
        self.assertIn(
            "-o JobIDRaw,State,ExitCode,JobName%255,Comment%255",
            transport.calls[0][1],
        )
        self.assertEqual(transport.calls[0][2]["retry"], RetryPolicy.SAFE_READ)

        duplicate = one + f"124|FAILED|1:0|{name}|tag\n"
        client, _transport = self.client([framed(duplicate)])
        with self.assertRaisesRegex(ProtocolViolation, "multiple records"):
            client.find_accounting_by_name(name)

        truncated = f"123|COMPLETED|0:0|{name[:-1]}+|tag\n"
        client, _transport = self.client([framed(truncated)])
        self.assertIsNone(client.find_accounting_by_name(name))

    def test_submission_spec_owns_quoting_and_identity_metadata(self) -> None:
        spec = SubmissionSpec(
            operation_id="a" * 32,
            owner_id="deadbeef1234",
            owner_host="laptop",
            kind=JobKind.RUN,
            logical_name="run",
            partition="day",
            walltime="1:00:00",
            cpus=2,
            logfile="/home/me/a log-%j.txt",
            wrap="python -c 'print(1)'",
        )
        command = spec.command()
        self.assertIn("sbatch --parsable", command)
        self.assertIn(spec.job_name, command)
        self.assertIn("hpc-alloc:v2:deadbeef1234", command)
        self.assertIn("'/home/me/a log-%j.txt'", command)
        self.assertIn("mkdir -p", spec.preparation_command())
        self.assertNotIn("sbatch", spec.preparation_command())
        self.assertTrue(spec.sbatch_command().startswith("sbatch --parsable"))
        self.assertNotIn("mkdir", spec.sbatch_command())

    def test_dry_run_paths_use_symbolic_home_without_changing_live_commands(self) -> None:
        spec = SubmissionSpec(
            operation_id="a" * 32,
            owner_id="deadbeef1234",
            owner_host="laptop",
            kind=JobKind.RUN,
            logical_name="run",
            partition="day",
            walltime="1:00:00",
            cpus=2,
            logfile=".hpc-alloc/run.log",
            wrap="true",
            chdir="~/project dir/it's;$(touch INJECTED)",
        )

        live_preparation = spec.preparation_command()
        live_submission = spec.sbatch_command()
        dry_run = spec.command()

        self.assertEqual(
            live_preparation,
            "mkdir -p .hpc-alloc && "
            "(find .hpc-alloc -name '*.log' -mtime +30 -delete "
            "2>/dev/null || true)",
        )
        self.assertIn("--output .hpc-alloc/run.log", live_submission)
        self.assertIn(
            "--chdir=~/project dir/it's;$(touch INJECTED)",
            shlex.split(live_submission),
        )
        self.assertNotIn("${HOME", live_submission)
        self.assertIn('mkdir -p "${HOME:?}"/.hpc-alloc', dry_run)
        self.assertIn('--output "${HOME:?}"/.hpc-alloc/run.log', dry_run)
        self.assertIn('--chdir="${HOME:?}"', dry_run)
        self.assertIn(
            "--chdir=${HOME:?}/project dir/it's;$(touch INJECTED)",
            shlex.split(dry_run),
        )


if __name__ == "__main__":
    unittest.main()
