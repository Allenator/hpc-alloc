from __future__ import annotations

import io
import unittest

from hpc_alloc.errors import SchedulerUnavailable
from hpc_alloc.lifecycle import AssessmentPhase, EvidenceTracker
from hpc_alloc.slurm import AccountingRecord, LogSizeResult, LogSizeStatus, QueueRow
from hpc_alloc.streaming import LogFollower


def row(state: str, node: str | None = None) -> QueueRow:
    return QueueRow("7", state, node, "Priority", "1:00:00", "day", "job", "now", "tag")


class FakeClient:
    def __init__(self, observations: list[object], *, final: AccountingRecord | None = None) -> None:
        self.observations = list(observations)
        self.final_record = final
        self.sizes: list[LogSizeResult] = []
        self.chunks: list[bytes] = []
        self.calls: list[tuple] = []

    def observe(self, ref: object):
        self.calls.append(("observe", ref))
        value = self.observations.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def final(self, ref: object, *, attempts: tuple[int, ...]):
        self.calls.append(("final", ref, attempts))
        return self.final_record

    def log_size(self, path: str):
        self.calls.append(("log_size", path))
        return self.sizes.pop(0)

    def read_log_chunk(self, path: str, offset: int, *, limit: int):
        self.calls.append(("read_log_chunk", path, offset, limit))
        return self.chunks.pop(0)


class LogFollowerTests(unittest.TestCase):
    def test_pending_job_never_touches_log(self) -> None:
        client = FakeClient([row("PENDING")])
        follower = LogFollower(client, "7", ".hpc/log", output=io.BytesIO())  # type: ignore[arg-type]
        result = follower.poll_once()
        self.assertEqual(result.assessment.phase, AssessmentPhase.QUEUED)
        self.assertEqual([call[0] for call in client.calls], ["observe"])
        # Nothing to read: the job never started, so no log exists and the drain
        # is trivially complete rather than a truncated read.
        self.assertTrue(follower.drain())
        self.assertEqual([call[0] for call in client.calls], ["observe"])

    def test_scheduler_failure_makes_no_file_call_and_breaks_evidence(self) -> None:
        client = FakeClient([None, SchedulerUnavailable("slurmctld"), None])
        follower = LogFollower(client, "7", ".hpc/log", output=io.BytesIO())  # type: ignore[arg-type]
        first = follower.poll_once()
        self.assertEqual(first.assessment.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        with self.assertRaises(SchedulerUnavailable):
            follower.poll_once()
        self.assertEqual(follower.tracker.assessment.phase, AssessmentPhase.UNCERTAIN)
        third = follower.poll_once()
        self.assertEqual(third.assessment.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertFalse(third.assessment.final)
        self.assertNotIn("log_size", [call[0] for call in client.calls])

    def test_unreadable_size_does_not_change_offset_or_read(self) -> None:
        client = FakeClient([row("SUSPENDED")])
        client.sizes = [LogSizeResult(LogSizeStatus.UNREADABLE)]
        follower = LogFollower(
            client,
            "7",
            ".hpc/log",
            tracker=EvidenceTracker(ever_started=True),
            output=io.BytesIO(),
        )  # type: ignore[arg-type]
        follower.offset = 19
        result = follower.poll_once()
        self.assertEqual(result.log_status, LogSizeStatus.UNREADABLE)
        self.assertEqual(follower.offset, 19)
        self.assertNotIn("read_log_chunk", [call[0] for call in client.calls])

    def test_truncation_resets_then_streams_from_zero(self) -> None:
        output = io.BytesIO()
        notes: list[str] = []
        client = FakeClient([row("REQUEUED")])
        client.sizes = [LogSizeResult.available(4)]
        client.chunks = [b"new\xff"]
        follower = LogFollower(
            client,
            "7",
            ".hpc/log",
            tracker=EvidenceTracker(ever_started=True),
            output=output,
            info=notes.append,
        )  # type: ignore[arg-type]
        follower.offset = 20
        result = follower.poll_once()
        self.assertTrue(result.log_restarted)
        self.assertEqual(output.getvalue(), b"new\xff")
        self.assertEqual(follower.offset, 4)
        self.assertIn(("read_log_chunk", ".hpc/log", 0, 4), client.calls)
        self.assertTrue(any("log restarted" in note for note in notes))

    def test_cold_completed_job_uses_accounting_then_reads_log(self) -> None:
        output = io.BytesIO()
        final = AccountingRecord("7", "COMPLETED", "0:0", "job", "tag")
        client = FakeClient([None], final=final)
        client.sizes = [LogSizeResult.available(3)]
        client.chunks = [b"end"]
        follower = LogFollower(client, "7", ".hpc/log", output=output)  # type: ignore[arg-type]
        result = follower.poll_once()
        self.assertTrue(result.assessment.final)
        self.assertTrue(result.assessment.ever_started)
        self.assertEqual(output.getvalue(), b"end")
        self.assertEqual(
            [call[0] for call in client.calls],
            ["observe", "final", "log_size", "read_log_chunk"],
        )


if __name__ == "__main__":
    unittest.main()
