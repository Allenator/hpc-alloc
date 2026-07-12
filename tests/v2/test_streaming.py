from __future__ import annotations

import io
import unittest

from hpc_alloc.errors import (
    JobIdReused,
    ProtocolViolation,
    SchedulerUnavailable,
    TransportLost,
)
from hpc_alloc.lifecycle import AssessmentPhase, EvidenceTracker
from hpc_alloc.models import FinalSource, JobRef
from hpc_alloc.slurm import (
    MAX_LOG_CHUNK_BYTES,
    AccountingRecord,
    LogSizeResult,
    LogSizeStatus,
    QueueRow,
)
from hpc_alloc.streaming import LogFollower

from .fakes import ExpectedCall, StrictProxy, StrictScript, VirtualClock


LOG_PATH = ".hpc-alloc/run-12345.log"


def row(state: str, *, node: str | None = None, reason: str = "") -> QueueRow:
    return QueueRow(
        job_id="12345",
        state=state,
        node=node,
        reason=reason,
        time_left="1:00:00",
        partition="day",
        name="hpcalloc-v2-run-" + "a" * 32,
        submitted_at="2026-07-10T11:00:00",
        comment="hpc-alloc:v2:deadbeef1234:" + "a" * 32 + ":laptop:run:-",
    )


class StreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ref = JobRef(
            cluster="grace",
            job_id="12345",
            owner_id="deadbeef1234",
            operation_id="a" * 32,
            slurm_job_name="hpcalloc-v2-run-" + "a" * 32,
            slurm_comment="hpc-alloc:v2:deadbeef1234:"
            + "a" * 32
            + ":laptop:run:-",
        )

    def follower(
        self,
        script: StrictScript,
        *,
        tracker: EvidenceTracker | None = None,
        output: io.BytesIO | None = None,
        notes: list[str] | None = None,
        clock: VirtualClock | None = None,
        final_attempts: tuple[float, ...] = (0, 9),
    ) -> LogFollower:
        clock = clock or VirtualClock()
        return LogFollower(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref,
            LOG_PATH,
            tracker=tracker,
            output=output or io.BytesIO(),
            info=(notes.append if notes is not None else None),
            sleeper=clock.sleep,
            clock=clock.monotonic,
            final_attempts=final_attempts,
        )

    def test_pending_job_performs_no_log_operations(self) -> None:
        script = StrictScript(
            [ExpectedCall("observe", result=row("PENDING", reason="Resources"), args=(self.ref,))]
        )
        result = self.follower(script).poll_once()
        self.assertEqual(result.assessment.phase, AssessmentPhase.QUEUED)
        self.assertFalse(result.assessment.ever_started)
        self.assertIsNone(result.log_status)
        self.assertEqual(script.count("log_size"), 0)
        self.assertEqual(script.count("read_log_chunk"), 0)
        script.assert_complete()

    def test_observation_failure_records_uncertainty_and_touches_no_log(self) -> None:
        script = StrictScript(
            [ExpectedCall("observe", result=TransportLost("VPN dropped"), args=(self.ref,))]
        )
        follower = self.follower(script)
        with self.assertRaisesRegex(TransportLost, "VPN dropped"):
            follower.poll_once()
        self.assertEqual(follower.tracker.assessment.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(script.count("log_size"), 0)
        self.assertEqual(script.count("read_log_chunk"), 0)
        script.assert_complete()

    def test_recycled_id_uses_two_observations_and_exact_accounting(self) -> None:
        reused = JobIdReused("job grace:12345 now belongs to another operation")
        script = StrictScript(
            [
                ExpectedCall("observe", result=reused, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(self.ref,),
                    kwargs={"attempts": (0,)},
                ),
                ExpectedCall("observe", result=reused, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(self.ref,),
                    kwargs={"attempts": (0, 9)},
                ),
            ]
        )
        follower = self.follower(script)

        candidate = follower.poll_once()
        self.assertEqual(candidate.assessment.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertEqual(candidate.assessment.detail, str(reused))

        confirmed = follower.poll_once()
        self.assertEqual(confirmed.assessment.phase, AssessmentPhase.FINAL)
        self.assertEqual(confirmed.assessment.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(script.count("log_size"), 0)
        script.assert_complete()

    def test_recycled_id_observation_error_breaks_confirmation(self) -> None:
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=JobIdReused("numeric ID was recycled"),
                    args=(self.ref,),
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(self.ref,),
                    kwargs={"attempts": (0,)},
                ),
                ExpectedCall(
                    "observe",
                    result=TransportLost("VPN dropped"),
                    args=(self.ref,),
                ),
            ]
        )
        follower = self.follower(script)

        follower.poll_once()
        with self.assertRaisesRegex(TransportLost, "VPN dropped"):
            follower.poll_once()

        self.assertEqual(follower.tracker.assessment.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(follower.tracker.assessment.terminal_evidence, 0)
        script.assert_complete()

    def test_never_started_final_follow_skips_size_read_and_drain(self) -> None:
        script = StrictScript(
            [
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(self.ref,),
                    kwargs={"attempts": (0,)},
                ),
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(self.ref,),
                    kwargs={"attempts": (0, 9)},
                ),
            ]
        )
        clock = VirtualClock()
        follower = self.follower(script, clock=clock)
        outcome = follower.follow(drain=True)
        self.assertTrue(outcome.assessment.final)
        self.assertFalse(outcome.assessment.ever_started)
        self.assertEqual(outcome.final_log_offset, 0)
        self.assertEqual(script.count("log_size"), 0)
        self.assertEqual(script.count("read_log_chunk"), 0)
        self.assertEqual(clock.sleeps, [5])
        script.assert_complete()

    def test_terminal_candidate_consults_accounting_before_any_log_touch(self) -> None:
        accounting = AccountingRecord(
            job_id=self.ref.job_id,
            state="CANCELLED",
            exit_code="0:15",
            job_name=self.ref.slurm_job_name,
            comment="hpc-alloc:v2:deadbeef1234:" + "a" * 32 + ":laptop:run:-",
        )
        script = StrictScript(
            [
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=accounting,
                    args=(self.ref,),
                    kwargs={"attempts": (0,)},
                ),
            ]
        )
        follower = self.follower(script)
        result = follower.poll_once()
        self.assertTrue(result.assessment.final)
        self.assertEqual(result.assessment.final_source, "accounting")
        self.assertEqual(result.assessment.terminal_state, "CANCELLED")
        self.assertFalse(result.assessment.ever_started)
        self.assertEqual(script.count("log_size"), 0)
        self.assertEqual(script.count("read_log_chunk"), 0)
        script.assert_complete()

    def test_cold_attach_to_completed_accounting_streams_existing_log(self) -> None:
        accounting = AccountingRecord(
            job_id=self.ref.job_id,
            state="COMPLETED",
            exit_code="0:0",
            job_name=self.ref.slurm_job_name,
            comment="hpc-alloc:v2:deadbeef1234:" + "a" * 32 + ":laptop:run:-",
        )
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=accounting,
                    args=(self.ref,),
                    kwargs={"attempts": (0,)},
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(5), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"done\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
            ]
        )
        result = self.follower(script, output=output).poll_once()
        self.assertTrue(result.assessment.final)
        self.assertTrue(result.assessment.ever_started)
        self.assertEqual(result.bytes_written, 5)
        self.assertEqual(output.getvalue(), b"done\n")
        script.assert_complete()

    def test_cold_attach_to_suspended_job_streams_prior_output(self) -> None:
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall("observe", result=row("SUSPENDED", node="node01"), args=(self.ref,)),
                ExpectedCall("log_size", result=LogSizeResult.available(5), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"prior",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
            ]
        )
        result = self.follower(script, output=output).poll_once()
        self.assertEqual(result.assessment.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertTrue(result.assessment.log_eligible)
        self.assertEqual(result.bytes_written, 5)
        self.assertEqual(output.getvalue(), b"prior")
        script.assert_complete()

    def test_present_completing_streams_without_terminal_accounting(self) -> None:
        script = StrictScript(
            [
                ExpectedCall("observe", result=row("COMPLETING", node="node01"), args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
                ),
            ]
        )

        result = self.follower(script).poll_once()

        self.assertEqual(result.assessment.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertEqual(result.assessment.terminal_evidence, 0)
        self.assertEqual(result.log_status, LogSizeStatus.MISSING)
        self.assertEqual(script.count("final"), 0)
        script.assert_complete()

    def test_unreadable_or_missing_size_never_resets_offset_or_reads(self) -> None:
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node01"),
                    args=(self.ref,),
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(5), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"hello",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
                ExpectedCall("observe", result=row("SUSPENDED", node="node01"), args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.UNREADABLE, detail="transient NFS error"),
                    args=(LOG_PATH,),
                ),
                ExpectedCall("observe", result=row("SUSPENDED", node="node01"), args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
                ),
                ExpectedCall("observe", result=row("SUSPENDED", node="node01"), args=(self.ref,)),
                ExpectedCall("log_size", result=LogSizeResult.available(10), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"world",
                    args=(LOG_PATH, 5),
                    kwargs={"limit": 5},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        follower.poll_once()

        unreadable = follower.poll_once()
        self.assertEqual(unreadable.log_status, LogSizeStatus.UNREADABLE)
        self.assertEqual(follower.offset, 5)
        self.assertFalse(unreadable.log_restarted)

        missing = follower.poll_once()
        self.assertEqual(missing.log_status, LogSizeStatus.MISSING)
        self.assertEqual(follower.offset, 5)
        self.assertFalse(missing.log_restarted)

        recovered = follower.poll_once()
        self.assertEqual(recovered.bytes_written, 5)
        self.assertFalse(recovered.log_restarted)
        self.assertEqual(follower.offset, 10)
        self.assertEqual(output.getvalue(), b"helloworld")
        self.assertEqual(script.count("read_log_chunk"), 2)
        script.assert_complete()

    def test_true_truncation_resets_to_zero_and_restreams_attempt(self) -> None:
        output = io.BytesIO()
        notes: list[str] = []
        script = StrictScript(
            [
                ExpectedCall("observe", result=row("RUNNING", node="node01"), args=(self.ref,)),
                ExpectedCall("log_size", result=LogSizeResult.available(10), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"old-output",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 10},
                ),
                ExpectedCall("observe", result=row("REQUEUED"), args=(self.ref,)),
                ExpectedCall("log_size", result=LogSizeResult.available(4), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"new\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 4},
                ),
            ]
        )
        follower = self.follower(script, output=output, notes=notes)
        follower.poll_once()
        restarted = follower.poll_once()
        self.assertTrue(restarted.log_restarted)
        self.assertEqual(restarted.bytes_written, 4)
        self.assertEqual(follower.offset, 4)
        self.assertEqual(output.getvalue(), b"old-outputnew\n")
        self.assertTrue(any("re-streaming from the top" in note for note in notes))
        script.assert_complete()

    def test_binary_chunks_are_written_byte_exact_and_offset_counts_bytes(self) -> None:
        chunk = b"\xff\r\n\x00\x1eOK"
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall("observe", result=row("RUNNING", node="node01"), args=(self.ref,)),
                ExpectedCall("log_size", result=LogSizeResult.available(len(chunk)), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=chunk,
                    args=(LOG_PATH, 0),
                    kwargs={"limit": len(chunk)},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        result = follower.poll_once()
        self.assertEqual(result.bytes_written, len(chunk))
        self.assertEqual(follower.offset, len(chunk))
        self.assertEqual(output.getvalue(), chunk)
        script.assert_complete()

    def test_large_snapshot_is_reconstructed_from_bounded_chunks(self) -> None:
        first = b"a" * MAX_LOG_CHUNK_BYTES
        second = bytes(range(256)) * (MAX_LOG_CHUNK_BYTES // 256)
        third = b"tail\x00\xff\n"
        total_size = len(first) + len(second) + len(third)
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe", result=row("RUNNING", node="node01"), args=(self.ref,)
                ),
                ExpectedCall(
                    "log_size", result=LogSizeResult.available(total_size), args=(LOG_PATH,)
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=first,
                    args=(LOG_PATH, 0),
                    kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=second,
                    args=(LOG_PATH, MAX_LOG_CHUNK_BYTES),
                    kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=third,
                    args=(LOG_PATH, 2 * MAX_LOG_CHUNK_BYTES),
                    kwargs={"limit": len(third)},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        result = follower.poll_once()
        self.assertEqual(result.bytes_written, total_size)
        self.assertEqual(follower.offset, total_size)
        self.assertEqual(output.getvalue(), first + second + third)
        self.assertEqual(script.count("read_log_chunk"), 3)
        script.assert_complete()

    def test_short_or_empty_chunk_stops_without_spinning_or_advancing_extra(self) -> None:
        for label, chunk in (("short", b"abc"), ("empty", b"")):
            with self.subTest(chunk=label):
                output = io.BytesIO()
                script = StrictScript(
                    [
                        ExpectedCall(
                            "observe",
                            result=row("RUNNING", node="node01"),
                            args=(self.ref,),
                        ),
                        ExpectedCall(
                            "log_size",
                            result=LogSizeResult.available(MAX_LOG_CHUNK_BYTES + 9),
                            args=(LOG_PATH,),
                        ),
                        ExpectedCall(
                            "read_log_chunk",
                            result=chunk,
                            args=(LOG_PATH, 0),
                            kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                        ),
                    ]
                )
                follower = self.follower(script, output=output)
                result = follower.poll_once()
                self.assertEqual(result.bytes_written, len(chunk))
                self.assertEqual(follower.offset, len(chunk))
                self.assertEqual(output.getvalue(), chunk)
                script.assert_complete()

    def test_later_chunk_failure_preserves_the_last_committed_offset(self) -> None:
        first = b"a" * MAX_LOG_CHUNK_BYTES
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe", result=row("RUNNING", node="node01"), args=(self.ref,)
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(MAX_LOG_CHUNK_BYTES + 1),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=first,
                    args=(LOG_PATH, 0),
                    kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=SchedulerUnavailable("slurmctld restarted"),
                    args=(LOG_PATH, MAX_LOG_CHUNK_BYTES),
                    kwargs={"limit": 1},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        with self.assertRaisesRegex(SchedulerUnavailable, "restarted"):
            follower.poll_once()
        self.assertEqual(follower.offset, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(follower.total_bytes_written, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(output.getvalue(), first)
        self.assertEqual(follower.tracker.assessment.phase, AssessmentPhase.UNCERTAIN)
        script.assert_complete()

    def test_oversized_chunk_is_rejected_before_any_output_is_written(self) -> None:
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe", result=row("RUNNING", node="node01"), args=(self.ref,)
                ),
                ExpectedCall(
                    "log_size", result=LogSizeResult.available(4), args=(LOG_PATH,)
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"12345",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 4},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        with self.assertRaises(ProtocolViolation):
            follower.poll_once()
        self.assertEqual(follower.offset, 0)
        self.assertEqual(follower.total_bytes_written, 0)
        self.assertEqual(output.getvalue(), b"")
        script.assert_complete()

    def test_broken_pipe_on_later_chunk_does_not_advance_remote_offset(self) -> None:
        class BreakOnSecondWrite(io.BytesIO):
            def __init__(self) -> None:
                super().__init__()
                self.write_count = 0

            def write(self, data: bytes) -> int:
                self.write_count += 1
                if self.write_count == 2:
                    raise BrokenPipeError("downstream closed")
                return super().write(data)

        first = b"a" * MAX_LOG_CHUNK_BYTES
        output = BreakOnSecondWrite()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe", result=row("RUNNING", node="node01"), args=(self.ref,)
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(MAX_LOG_CHUNK_BYTES + 1),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=first,
                    args=(LOG_PATH, 0),
                    kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"x",
                    args=(LOG_PATH, MAX_LOG_CHUNK_BYTES),
                    kwargs={"limit": 1},
                ),
            ]
        )
        follower = self.follower(script, output=output)
        with self.assertRaises(BrokenPipeError):
            follower.poll_once()
        self.assertEqual(follower.offset, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(follower.total_bytes_written, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(output.getvalue(), first)
        script.assert_complete()

    def test_drain_reads_a_fresh_size_and_keeps_partial_progress_on_failure(self) -> None:
        first = b"a" * MAX_LOG_CHUNK_BYTES
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(MAX_LOG_CHUNK_BYTES + 1),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=first,
                    args=(LOG_PATH, 0),
                    kwargs={"limit": MAX_LOG_CHUNK_BYTES},
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=SchedulerUnavailable("accounting lag"),
                    args=(LOG_PATH, MAX_LOG_CHUNK_BYTES),
                    kwargs={"limit": 1},
                ),
            ]
        )
        follower = self.follower(
            script,
            tracker=EvidenceTracker(ever_started=True),
            output=output,
        )
        self.assertEqual(follower.drain(), MAX_LOG_CHUNK_BYTES)
        self.assertEqual(follower.offset, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(output.getvalue(), first)
        script.assert_complete()

    def test_final_seeded_follow_skips_observation_and_drains_once(self) -> None:
        output = io.BytesIO()
        tracker = EvidenceTracker(
            ever_started=True,
            phase=AssessmentPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        script = StrictScript(
            [
                ExpectedCall("log_size", result=LogSizeResult.available(5), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"done\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
            ]
        )

        outcome = self.follower(script, tracker=tracker, output=output).follow()

        self.assertTrue(outcome.assessment.final)
        self.assertEqual(outcome.observed_terminal_state, "COMPLETED")
        self.assertEqual(outcome.final_log_offset, 5)
        self.assertEqual(outcome.bytes_written, 5)
        self.assertEqual(output.getvalue(), b"done\n")
        self.assertEqual(script.count("observe"), 0)
        self.assertEqual(script.count("final"), 0)
        self.assertEqual(script.count("log_size"), 1)
        script.assert_complete()

    def test_rebase_replaces_only_lifecycle_tracker_and_preserves_stream_progress(self) -> None:
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall("observe", result=row("RUNNING", node="node01"), args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(5),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"part\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
                ExpectedCall("observe", result=row("PENDING"), args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(5),
                    args=(LOG_PATH,),
                ),
            ]
        )
        follower = self.follower(script, output=output)
        follower.poll_once()
        replacement = EvidenceTracker(
            ever_started=True,
            last_node="node01",
            observation_epoch=4,
            restart_boundary=True,
            phase=AssessmentPhase.REQUEUEING,
        )

        follower.rebase(replacement)

        self.assertIs(follower.tracker, replacement)
        self.assertEqual(follower.offset, 5)
        self.assertEqual(follower.total_bytes_written, 5)
        self.assertEqual(output.getvalue(), b"part\n")
        result = follower.poll_once()
        self.assertEqual(result.assessment.phase, AssessmentPhase.REQUEUEING)
        self.assertEqual(result.assessment.observation_epoch, 5)
        self.assertEqual(result.bytes_written, 0)
        self.assertEqual(follower.offset, 5)
        self.assertEqual(follower.total_bytes_written, 5)
        self.assertEqual(output.getvalue(), b"part\n")
        script.assert_complete()


if __name__ == "__main__":
    unittest.main()
