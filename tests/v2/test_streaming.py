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
from hpc_alloc.models import EvidenceProvenance, FinalSource, JobRef
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
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
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
        self.assertEqual(script.count("log_size"), 1)
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

    def test_unknown_start_queue_final_streams_and_drains_operation_log(self) -> None:
        output = io.BytesIO()
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
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(5),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"done\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 5},
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(5),
                    args=(LOG_PATH,),
                ),
            ]
        )
        clock = VirtualClock()
        follower = self.follower(script, clock=clock, output=output)
        outcome = follower.follow(drain=True)
        self.assertTrue(outcome.assessment.final)
        self.assertFalse(outcome.assessment.ever_started)
        self.assertTrue(outcome.assessment.log_eligible)
        self.assertEqual(outcome.final_log_offset, 5)
        self.assertEqual(outcome.bytes_written, 5)
        self.assertEqual(output.getvalue(), b"done\n")
        self.assertEqual(script.count("log_size"), 2)
        self.assertEqual(script.count("read_log_chunk"), 1)
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
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
                ),
            ]
        )
        follower = self.follower(script)
        result = follower.poll_once()
        self.assertTrue(result.assessment.final)
        self.assertEqual(result.assessment.final_source, "accounting")
        self.assertEqual(result.assessment.terminal_state, "CANCELLED")
        self.assertFalse(result.assessment.ever_started)
        self.assertTrue(result.assessment.log_eligible)
        self.assertEqual(script.count("log_size"), 1)
        self.assertEqual(script.count("read_log_chunk"), 0)
        script.assert_complete()

    def test_publication_brackets_accounting_before_log_access(self) -> None:
        accounting = AccountingRecord(
            job_id=self.ref.job_id,
            state="COMPLETED",
            exit_code="0:0",
            job_name=self.ref.slurm_job_name,
            comment=self.ref.slurm_comment,
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
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
                ),
            ]
        )
        publications: list[tuple[AssessmentPhase, list[str]]] = []

        def publish(assessment):
            publications.append(
                (
                    assessment.phase,
                    [name for name, _args, _kwargs in script.ledger],
                )
            )
            return assessment, None

        result = self.follower(script).poll_once(publish_assessment=publish)

        self.assertTrue(result.assessment.final)
        self.assertEqual(
            publications,
            [
                (AssessmentPhase.TERMINAL_CANDIDATE, ["observe"]),
                (AssessmentPhase.FINAL, ["observe", "final"]),
            ],
        )
        script.assert_complete()

    def test_reconciled_live_authority_rebases_before_policy_and_sleep(self) -> None:
        stale_tracker = EvidenceTracker(
            ever_started=True,
            last_node="node01",
            phase=AssessmentPhase.TERMINAL_CANDIDATE,
            terminal_state="FAILED",
            evidence_provenance=EvidenceProvenance.QUEUE_TERMINAL,
        )
        replacement = EvidenceTracker(
            ever_started=True,
            last_node="node02",
            phase=AssessmentPhase.REQUEUEING,
        )
        script = StrictScript(
            [
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(37),
                    args=(LOG_PATH,),
                ),
            ]
        )
        stopped = RuntimeError("stop after the reconciled live poll")
        follower: LogFollower

        def sleep(_delay: float) -> None:
            self.assertIs(follower.tracker, replacement)
            self.assertEqual(follower.offset, 37)
            self.assertEqual(follower.total_bytes_written, 37)
            raise stopped

        follower = LogFollower(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref,
            LOG_PATH,
            tracker=stale_tracker,
            output=io.BytesIO(),
            sleeper=sleep,
        )
        follower.offset = 37
        follower.total_bytes_written = 37
        publications: list[AssessmentPhase] = []

        def publish(assessment):
            publications.append(assessment.phase)
            self.assertTrue(assessment.final)
            return replacement.assessment, replacement

        with self.assertRaises(RuntimeError) as raised:
            follower.follow(publish_assessment=publish)

        self.assertIs(raised.exception, stopped)
        self.assertEqual(publications, [AssessmentPhase.FINAL])
        self.assertEqual(script.count("final"), 0)
        self.assertEqual(script.count("log_size"), 1)
        self.assertEqual(follower.offset, 37)
        self.assertEqual(follower.total_bytes_written, 37)
        script.assert_complete()

    def test_log_size_failure_follows_successful_active_publication(self) -> None:
        failure = SchedulerUnavailable("log filesystem probe failed")
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node01"),
                    args=(self.ref,),
                ),
                ExpectedCall("log_size", result=failure, args=(LOG_PATH,)),
            ]
        )
        published: list[AssessmentPhase] = []

        def publish(assessment):
            published.append(assessment.phase)
            self.assertEqual(script.count("log_size"), 0)
            return assessment, None

        follower = self.follower(script)
        with self.assertRaises(SchedulerUnavailable) as raised:
            follower.poll_once(publish_assessment=publish)

        self.assertIs(raised.exception, failure)
        self.assertEqual(published, [AssessmentPhase.ACTIVE])
        self.assertEqual(follower.tracker.assessment.phase, AssessmentPhase.UNCERTAIN)
        script.assert_complete()

    def test_broken_output_pipe_occurs_after_successful_publication(self) -> None:
        failure = BrokenPipeError("downstream closed")
        published: list[AssessmentPhase] = []

        class BrokenOutput(io.BytesIO):
            def write(inner_self, data: bytes) -> int:
                self.assertEqual(published, [AssessmentPhase.ACTIVE])
                raise failure

        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node01"),
                    args=(self.ref,),
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(1),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"x",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 1},
                ),
            ]
        )

        def publish(assessment):
            published.append(assessment.phase)
            return assessment, None

        follower = self.follower(script, output=BrokenOutput())
        with self.assertRaises(BrokenPipeError) as raised:
            follower.poll_once(publish_assessment=publish)

        self.assertIs(raised.exception, failure)
        self.assertEqual(follower.offset, 0)
        self.assertEqual(follower.total_bytes_written, 0)
        script.assert_complete()

    def test_interrupt_at_sleep_occurs_after_successful_publication(self) -> None:
        interrupt = KeyboardInterrupt()
        published: list[tuple[AssessmentPhase, str | None, str | None]] = []
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node02"),
                    args=(self.ref,),
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult(LogSizeStatus.MISSING),
                    args=(LOG_PATH,),
                ),
            ]
        )

        def publish(assessment):
            published.append(
                (assessment.phase, assessment.current_node, assessment.last_node)
            )
            return assessment, None

        def sleep(_delay: float) -> None:
            self.assertEqual(
                published,
                [(AssessmentPhase.ACTIVE, "node02", "node02")],
            )
            raise interrupt

        follower = LogFollower(
            StrictProxy(script),  # type: ignore[arg-type]
            self.ref,
            LOG_PATH,
            tracker=EvidenceTracker(
                ever_started=True,
                last_node="node01",
                phase=AssessmentPhase.REQUEUEING,
            ),
            output=io.BytesIO(),
            sleeper=sleep,
        )
        with self.assertRaises(KeyboardInterrupt) as raised:
            follower.follow(publish_assessment=publish)

        self.assertIs(raised.exception, interrupt)
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

    def test_special_exit_requeue_preserves_stream_progress_and_can_run_again(self) -> None:
        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("SPECIAL_EXIT", node="node01"),
                    args=(self.ref,),
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(4), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"old\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 4},
                ),
                ExpectedCall(
                    "observe",
                    result=row("PENDING", reason="Held"),
                    args=(self.ref,),
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(4), args=(LOG_PATH,)),
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node02"),
                    args=(self.ref,),
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(8), args=(LOG_PATH,)),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"new\n",
                    args=(LOG_PATH, 4),
                    kwargs={"limit": 4},
                ),
            ]
        )
        follower = self.follower(script, output=output)

        special = follower.poll_once()
        self.assertEqual(special.assessment.phase, AssessmentPhase.REQUEUEING)
        self.assertFalse(special.assessment.final)
        self.assertEqual(follower.offset, 4)
        self.assertEqual(script.count("final"), 0)

        pending = follower.poll_once()
        self.assertEqual(pending.assessment.phase, AssessmentPhase.REQUEUEING)
        self.assertEqual(follower.offset, 4)

        running = follower.poll_once()
        self.assertEqual(running.assessment.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(running.assessment.current_node, "node02")
        self.assertEqual(follower.offset, 8)
        self.assertEqual(output.getvalue(), b"old\nnew\n")
        self.assertEqual(script.count("final"), 0)
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
        follower.drain_attempts = 1

        # The read failed part-way through.  drain must keep the bytes it did
        # deliver *and* report the shortfall: silently returning the truncated
        # output as if it were the whole log is how `run` used to print a partial
        # result and still report COMPLETED with exit 0.
        self.assertFalse(follower.drain())
        self.assertEqual(follower.offset, MAX_LOG_CHUNK_BYTES)
        self.assertEqual(output.getvalue(), first)
        script.assert_complete()

    def test_the_log_and_the_scheduler_run_on_separate_clocks(self) -> None:
        """The stream stays responsive without hammering the controller.

        poll_once used to bundle a scheduler query and a log read into one
        cadence, so keeping the stream snappy (3s) meant querying the scheduler
        3s apart for the entire life of the job -- ~1200 queries an hour, on a
        controller shared with the whole cluster, almost all of them reporting
        the same thing.  The log is a file read; the scheduler is not.  They now
        keep separate time: the log every `log_interval`, the scheduler on a
        backoff that widens while nothing changes.
        """

        clock = VirtualClock()
        running = row("RUNNING", node="node01")
        completed = AccountingRecord(
            job_id=self.ref.job_id,
            state="COMPLETED",
            exit_code="0:0",
            job_name=self.ref.slurm_job_name,
            comment=self.ref.slurm_comment,
        )

        class Client:
            """Counts calls; the job runs steadily, then ends."""

            def __init__(self) -> None:
                self.observations = 0
                self.log_reads = 0

            def observe(self, _ref, **_kwargs):
                self.observations += 1
                # Run steadily for the first several observations, then vanish
                # twice so the two-observation rule can finalize it.
                return running if self.observations <= 8 else None

            def final(self, _ref, **_kwargs):
                return completed if self.observations > 9 else None

            def log_size(self, _path, **_kwargs):
                self.log_reads += 1
                return LogSizeResult.available(0)

            def read_log_chunk(self, *_args, **_kwargs):  # pragma: no cover
                raise AssertionError("the log is empty")

        client = Client()
        follower = LogFollower(
            client,  # type: ignore[arg-type]
            self.ref,
            LOG_PATH,
            tracker=EvidenceTracker(ever_started=True),
            output=io.BytesIO(),
            sleeper=clock.sleep,
            clock=clock.monotonic,
        )

        outcome = follower.follow()

        self.assertTrue(outcome.assessment.final)
        # The job's situation never changed while it ran, so the scheduler
        # interval widened 5 -> 10 -> 20 -> 30 and the observations became rare,
        # while the log kept being read every 3s throughout.
        self.assertLess(
            client.observations,
            client.log_reads,
            "the scheduler was polled as often as the log",
        )
        self.assertGreater(clock.now, 60, "the job ran for a meaningful stretch")

    def test_follow_rides_out_a_transient_scheduler_failure(self) -> None:
        """A controller restart mid-stream must not kill the stream.

        `follow` had no exception handling at all, and the transport's single
        heal-and-retry never covers SchedulerUnavailable (a scheduler query that
        runs and exits nonzero leaves ssh itself exiting 0).  So one hiccup
        aborted `run` / `logs -f` outright while the GPU job kept running -- and
        the failed observation had already cleared the death candidate, so the
        stream was thrown away for nothing.
        """

        clock = VirtualClock()
        completed = AccountingRecord(
            job_id=self.ref.job_id,
            state="COMPLETED",
            exit_code="0:0",
            job_name=self.ref.slurm_job_name,
            comment=self.ref.slurm_comment,
        )
        script = StrictScript(
            [
                # The blip.  It used to end the stream right here.
                ExpectedCall(
                    "observe",
                    result=SchedulerUnavailable("controller is restarting"),
                    args=(self.ref,),
                ),
                # Absent once: a death candidate, not yet a death.
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final", result=None, args=(self.ref,), kwargs={"attempts": (0,)}
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(0), args=(LOG_PATH,)),
                # A log tick between the two observations: the stream keeps its
                # own cadence and does not touch the scheduler.
                ExpectedCall("log_size", result=LogSizeResult.available(0), args=(LOG_PATH,)),
                # Absent twice: confirmed departure, then accounting refines it.
                ExpectedCall("observe", result=None, args=(self.ref,)),
                ExpectedCall(
                    "final",
                    result=completed,
                    args=(self.ref,),
                    kwargs={"attempts": (0, 9)},
                ),
                ExpectedCall("log_size", result=LogSizeResult.available(0), args=(LOG_PATH,)),
                ExpectedCall("log_size", result=LogSizeResult.available(0), args=(LOG_PATH,)),
            ]
        )
        follower = self.follower(
            script,
            tracker=EvidenceTracker(ever_started=True),
            output=io.BytesIO(),
            clock=clock,
        )

        outcome = follower.follow()

        self.assertTrue(outcome.assessment.final)
        self.assertEqual(outcome.assessment.terminal_state, "COMPLETED")
        self.assertTrue(outcome.log_complete)
        # The blip was ridden out rather than aborting the stream.
        self.assertIn(15, clock.sleeps)
        script.assert_complete()

    def test_drain_retries_a_transient_failure_instead_of_truncating(self) -> None:
        """One transient ssh or shared-filesystem error must not lose the tail.

        The tail is exactly where a job's results and tracebacks land, and it is
        the only part `run` has not already streamed.  drain used to swallow the
        error and return 0, so the output was silently truncated while the
        command still reported COMPLETED and exited 0.
        """

        output = io.BytesIO()
        script = StrictScript(
            [
                ExpectedCall(
                    "log_size",
                    result=TransportLost("connection reset"),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(11),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"RESULT: 0.9",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 11},
                ),
            ]
        )
        follower = self.follower(
            script,
            tracker=EvidenceTracker(ever_started=True),
            output=output,
        )

        self.assertTrue(follower.drain())
        self.assertEqual(output.getvalue(), b"RESULT: 0.9")
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

    def test_unknown_start_final_seeded_follow_also_drains_once(self) -> None:
        output = io.BytesIO()
        tracker = EvidenceTracker(
            phase=AssessmentPhase.FINAL,
            terminal_state="CANCELLED",
            exit_code="0:15",
            final_source=FinalSource.ACCOUNTING,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "log_size",
                    result=LogSizeResult.available(10),
                    args=(LOG_PATH,),
                ),
                ExpectedCall(
                    "read_log_chunk",
                    result=b"cancelled\n",
                    args=(LOG_PATH, 0),
                    kwargs={"limit": 10},
                ),
            ]
        )

        outcome = self.follower(script, tracker=tracker, output=output).follow()

        self.assertFalse(outcome.assessment.ever_started)
        self.assertTrue(outcome.assessment.log_eligible)
        self.assertEqual(outcome.final_log_offset, 10)
        self.assertEqual(outcome.bytes_written, 10)
        self.assertEqual(output.getvalue(), b"cancelled\n")
        self.assertEqual(script.count("observe"), 0)
        self.assertEqual(script.count("final"), 0)
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
