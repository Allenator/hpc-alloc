"""Byte-exact log following driven by canonical lifecycle assessments."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import BinaryIO, Callable

from .errors import (
    HpcAllocError,
    ProtocolViolation,
)
from .lifecycle import AssessmentPhase, EvidenceEvent, EvidenceTracker, JobAssessment
from .models import JobRef
from .monitor import accept_observation, break_lifecycle_evidence
from .slurm import MAX_LOG_CHUNK_BYTES, LogSizeStatus, QueueRow, SlurmClient


@dataclass(frozen=True, slots=True)
class PollResult:
    assessment: JobAssessment
    bytes_written: int = 0
    log_restarted: bool = False
    log_status: LogSizeStatus | None = None


@dataclass(frozen=True, slots=True)
class FollowOutcome:
    assessment: JobAssessment
    observed_terminal_state: str | None
    final_log_offset: int
    bytes_written: int
    detach_reason: str | None = None


class LogFollower:
    """Follow one job without mixing scheduler metadata and arbitrary bytes.

    A poll always observes Slurm first.  Log size and log content are separate
    remote calls and are made only after ``ever_started`` is established.
    """

    def __init__(
        self,
        client: SlurmClient,
        ref: JobRef | str | int,
        log_path: str,
        *,
        tracker: EvidenceTracker | None = None,
        output: BinaryIO | None = None,
        info: Callable[[str], None] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        active_interval: float = 3,
        queued_interval: float = 5,
        note_interval: float = 30,
        final_attempts: tuple[float, ...] = (0, 2, 2, 3, 4, 5, 7, 7),
    ) -> None:
        self.client = client
        self.ref = ref
        self.job_id = str(ref.job_id if isinstance(ref, JobRef) else ref)
        self.log_path = log_path
        self.tracker = tracker or EvidenceTracker()
        self.output = output or sys.stdout.buffer
        self.info = info or (lambda _message: None)
        self._sleep = sleeper
        self._clock = clock
        self.active_interval = active_interval
        self.queued_interval = queued_interval
        self.note_interval = note_interval
        self.final_attempts = final_attempts
        self.offset = 0
        self.total_bytes_written = 0
        self._last_note = float("-inf")

    def _observe(self) -> tuple[QueueRow | None, JobAssessment]:
        return accept_observation(self.tracker, lambda: self.client.observe(self.ref))

    def _write(self, chunk: bytes) -> int:
        if not chunk:
            return 0
        written = self.output.write(chunk)
        if written is not None and written != len(chunk):
            raise OSError(f"short log write ({written} of {len(chunk)} bytes)")
        self.output.flush()
        self.offset += len(chunk)
        self.total_bytes_written += len(chunk)
        return len(chunk)

    def _state_note(self, row: QueueRow | None, assessment: JobAssessment) -> None:
        if row is None or assessment.phase == AssessmentPhase.ACTIVE:
            return
        now = self._clock()
        if now - self._last_note < self.note_interval:
            return
        message = f"job {self.job_id}: {row.state}"
        if row.state == "PENDING" and row.reason:
            message += f" (reason: {row.reason})"
        self.info(message)
        self._last_note = now

    def _stream_to_size(self, target_size: int, *, best_effort: bool = False) -> int:
        """Write chunks up to one captured size without chasing later growth."""

        total = 0
        while self.offset < target_size:
            limit = min(MAX_LOG_CHUNK_BYTES, target_size - self.offset)
            try:
                chunk = self.client.read_log_chunk(
                    self.log_path,
                    self.offset,
                    limit=limit,
                )
                if len(chunk) > limit:
                    raise ProtocolViolation(
                        f"log chunk exceeded requested limit ({len(chunk)} > {limit})"
                    )
            except HpcAllocError:
                if best_effort:
                    break
                raise
            if not chunk:
                break
            total += self._write(chunk)
            if len(chunk) < limit:
                break
        return total

    def poll_once(self) -> PollResult:
        """Observe once and stream currently available bytes.

        If scheduler observation fails, this method performs no log operation.
        Missing or unreadable size is a tagged condition and never changes the
        byte offset or fabricates truncation.
        """

        row, assessment = self._observe()
        if assessment.phase in (AssessmentPhase.TERMINAL_CANDIDATE, AssessmentPhase.FINAL):
            attempts = (
                self.final_attempts if assessment.phase == AssessmentPhase.FINAL else (0,)
            )
            try:
                record = self.client.final(self.ref, attempts=attempts)
            except HpcAllocError as exc:
                break_lifecycle_evidence(self.tracker, exc)
                raise
            if record is not None:
                assessment = self.tracker.accept(EvidenceEvent.final(record))
        self._state_note(row, assessment)
        if not assessment.ever_started:
            return PollResult(assessment)

        try:
            size = self.client.log_size(self.log_path)
        except HpcAllocError as exc:
            break_lifecycle_evidence(self.tracker, exc)
            raise
        if size.status != LogSizeStatus.AVAILABLE:
            return PollResult(assessment, log_status=size.status)
        assert size.size is not None

        restarted = size.size < self.offset
        if restarted:
            self.offset = 0
            self.info(
                f"job {self.job_id}: log restarted (requeued?) — re-streaming from the top"
            )
        try:
            count = self._stream_to_size(size.size)
        except HpcAllocError as exc:
            break_lifecycle_evidence(self.tracker, exc)
            raise
        return PollResult(
            assessment=assessment,
            bytes_written=count,
            log_restarted=restarted,
            log_status=LogSizeStatus.AVAILABLE,
        )

    def drain(self) -> int:
        """Best-effort final read, but only for a job known to have started."""

        if not self.tracker.assessment.ever_started:
            return 0
        try:
            size = self.client.log_size(self.log_path)
            if size.status != LogSizeStatus.AVAILABLE or size.size is None:
                return 0
            if size.size < self.offset:
                self.offset = 0
                self.info(
                    f"job {self.job_id}: log restarted (requeued?) — re-streaming from the top"
                )
            return self._stream_to_size(size.size, best_effort=True)
        except HpcAllocError:
            return 0

    def follow(self, *, drain: bool = True) -> FollowOutcome:
        """Poll until final evidence and return context for command-level policy."""

        seeded = self.tracker.assessment
        if seeded.final:
            if drain:
                self.drain()
            final = self.tracker.assessment
            return FollowOutcome(
                assessment=final,
                observed_terminal_state=final.terminal_state,
                final_log_offset=self.offset,
                bytes_written=self.total_bytes_written,
            )

        while True:
            result = self.poll_once()
            assessment = result.assessment
            if assessment.final:
                if drain:
                    self.drain()
                final = self.tracker.assessment
                return FollowOutcome(
                    assessment=final,
                    observed_terminal_state=final.terminal_state,
                    final_log_offset=self.offset,
                    bytes_written=self.total_bytes_written,
                )
            delay = self.active_interval if assessment.ever_started else self.queued_interval
            self._sleep(delay)


__all__ = ["FollowOutcome", "LogFollower", "PollResult"]
