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
from .lifecycle import (
    AssessmentPhase,
    EvidenceEvent,
    EvidenceTracker,
    JobAssessment,
    awaits_requeue_confirmation,
)
from .models import FinalSource, JobRef
from .monitor import accept_observation, break_lifecycle_evidence
from .retry import PollBackoff, RetryBudget, observation_signature
from .slurm import MAX_LOG_CHUNK_BYTES, LogSizeStatus, QueueRow, SlurmClient


AssessmentPublisher = Callable[
    [JobAssessment],
    tuple[JobAssessment, EvidenceTracker | None],
]


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
    # False when the final drain could not read the log to EOF, so the streamed
    # output is knowingly incomplete.  Callers must surface this: it is the
    # difference between "the job printed nothing more" and "we failed to read
    # what it printed".
    log_complete: bool = True


class LogFollower:
    """Follow one job without mixing scheduler metadata and arbitrary bytes.

    A poll always observes Slurm first.  Log size and log content are separate
    remote calls and are made only after start evidence or a scheduler-final
    verdict makes the operation-scoped log eligible.
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
        log_interval: float = 3,
        note_interval: float = 30,
        final_attempts: tuple[float, ...] = (0, 2, 2, 3, 4, 5, 7, 7),
        drain_attempts: int = 3,
        drain_retry_delay: float = 2,
        retry_budget: RetryBudget | None = None,
        backoff: PollBackoff | None = None,
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
        self.log_interval = log_interval
        self.note_interval = note_interval
        self.backoff = backoff or PollBackoff()
        self.final_attempts = final_attempts
        self.drain_attempts = drain_attempts
        self.drain_retry_delay = drain_retry_delay
        self.retry_budget = retry_budget or RetryBudget(
            sleeper=sleeper, clock=clock, info=self.info
        )
        self.offset = 0
        self.total_bytes_written = 0
        self._last_note = float("-inf")

    def rebase(self, tracker: EvidenceTracker) -> None:
        """Replace lifecycle evidence without restarting the byte stream.

        Reconciliation may discover that another process advanced the durable
        job while this follower was running.  The fresh tracker becomes the
        authority for later scheduler observations, while the existing byte
        offset, byte counter, output stream, and notification cadence remain
        attached to this follower.
        """

        self.tracker = tracker

    def _observe(self) -> tuple[QueueRow | None, JobAssessment]:
        return accept_observation(self.tracker, lambda: self.client.observe(self.ref))

    def _publish_assessment(
        self,
        assessment: JobAssessment,
        publish: AssessmentPublisher | None,
    ) -> JobAssessment:
        """Return reconciled authority and adopt any replacement evidence."""

        if publish is None:
            return assessment
        authoritative, replacement = publish(assessment)
        if replacement is not None:
            self.rebase(replacement)
        return authoritative

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

    def _state_note(self, assessment: JobAssessment) -> None:
        state = assessment.scheduler_state
        if not state or assessment.phase == AssessmentPhase.ACTIVE:
            return
        now = self._clock()
        if now - self._last_note < self.note_interval:
            return
        message = f"job {self.job_id}: {state}"
        if state == "PENDING" and assessment.detail:
            message += f" (reason: {assessment.detail})"
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

    def poll_once(
        self,
        *,
        publish_assessment: AssessmentPublisher | None = None,
    ) -> PollResult:
        """Observe once and stream currently available bytes.

        If scheduler observation fails, this method performs no log operation.
        Missing or unreadable size is a tagged condition and never changes the
        byte offset or fabricates truncation.  A successful observation is
        published before any diagnostic or log operation, so callers may make
        durable evidence authoritative for all subsequent policy.
        """

        _row, observed = self._observe()
        assessment = self._publish_assessment(observed, publish_assessment)
        # A NODE_FAIL / PREEMPTED candidate is deliberately NOT finalized here.
        # The accounting read below would be taken in the same cycle as the
        # observation that produced the candidate, so it describes the same
        # instant -- inside the window in which Slurm requeues the job -- and a
        # single requeue-eligible record locked the job to FINAL/ACCOUNTING
        # irreversibly.  Deferring lets the next poll decide: a requeued job
        # reappears as PENDING and clears the candidate, while a genuinely dead
        # one earns a second, independent terminal or absent observation.
        if not awaits_requeue_confirmation(assessment) and (
            assessment.phase is AssessmentPhase.TERMINAL_CANDIDATE
            or (
                assessment.phase is AssessmentPhase.FINAL
                and assessment.final_source is FinalSource.CONFIRMED_QUEUE
            )
        ):
            attempts = (
                self.final_attempts if assessment.phase == AssessmentPhase.FINAL else (0,)
            )
            try:
                record = self.client.final(self.ref, attempts=attempts)
            except HpcAllocError as exc:
                break_lifecycle_evidence(self.tracker, exc)
                raise
            if record is not None:
                enriched = self.tracker.accept(EvidenceEvent.final(record))
                assessment = self._publish_assessment(
                    enriched,
                    publish_assessment,
                )
        self._state_note(assessment)
        count, restarted, status = self._read_log()
        return PollResult(
            assessment=assessment,
            bytes_written=count,
            log_restarted=restarted,
            log_status=status,
        )

    def _read_log(self) -> tuple[int, bool, LogSizeStatus | None]:
        """Stream whatever the job has newly written.  Touches no scheduler."""

        if not self.tracker.assessment.log_eligible:
            return 0, False, None
        try:
            size = self.client.log_size(self.log_path)
        except HpcAllocError as exc:
            break_lifecycle_evidence(self.tracker, exc)
            raise
        if size.status != LogSizeStatus.AVAILABLE:
            return 0, False, size.status
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
        return count, restarted, LogSizeStatus.AVAILABLE

    def stream_once(self) -> int:
        """Read newly written log bytes without consulting the scheduler.

        The log is a file on a shared filesystem; the scheduler is a shared
        controller under real load from every user on the cluster.  Bundling both
        into one poll forced the cheap read and the expensive query to share a
        cadence, so keeping the stream responsive meant querying the scheduler
        every few seconds for the entire life of the job -- thousands of times,
        almost always to be told what we already knew.  They are separate
        concerns and now run on separate clocks.
        """

        return self._read_log()[0]

    def drain(self) -> bool:
        """Read the log to EOF after finality; report whether that succeeded.

        This is the read that decides whether `run` delivered the job's whole
        output: the tail written between the last poll and the job's exit, which
        is exactly where results and tracebacks land.  It used to swallow every
        failure and return 0, so one transient ssh or shared-filesystem error
        silently truncated stdout while the command still reported COMPLETED and
        exited 0 -- and on the seeded-final path, where this is the *only* read,
        the entire log could be lost that way.

        Retry a bounded number of times (most such failures are transient), then
        report the shortfall so the caller can warn.  The remote log itself is
        not lost, so the remedy is cheap: re-read it with `hpc-alloc logs`.
        """

        if not self.tracker.assessment.log_eligible:
            return True
        for attempt in range(self.drain_attempts):
            if attempt:
                self._sleep(self.drain_retry_delay)
            try:
                size = self.client.log_size(self.log_path)
                if size.status is LogSizeStatus.MISSING and self.offset == 0:
                    # No output file at finality, and nothing was ever streamed:
                    # the job produced no output (cancelled or failed before it
                    # wrote a byte).  That is a complete, empty result -- not a
                    # truncated read -- so it must not warn that output was lost.
                    # A job that did write still has its file here; if we had been
                    # streaming (offset > 0) and it has now vanished, that is a
                    # genuine shortfall and falls through to the retry/warn path.
                    return True
                if size.status is not LogSizeStatus.AVAILABLE or size.size is None:
                    continue
                if size.size < self.offset:
                    self.offset = 0
                    self.info(
                        f"job {self.job_id}: log restarted (requeued?) — re-streaming from the top"
                    )
                # Not best-effort: a mid-stream failure must be retried, not
                # silently accepted as the end of the output.  The offset keeps
                # whatever progress was made, so a retry resumes where it stopped.
                self._stream_to_size(size.size)
                return True
            except HpcAllocError:
                continue
        return False

    def follow(
        self,
        *,
        drain: bool = True,
        publish_assessment: AssessmentPublisher | None = None,
    ) -> FollowOutcome:
        """Poll until final evidence and return context for command-level policy."""

        seeded = self.tracker.assessment
        if seeded.final:
            complete = self.drain() if drain else True
            final = self.tracker.assessment
            return FollowOutcome(
                assessment=final,
                observed_terminal_state=final.terminal_state,
                final_log_offset=self.offset,
                bytes_written=self.total_bytes_written,
                log_complete=complete,
            )

        assessment = seeded
        observe_due = self._clock()
        while True:
            try:
                if self._clock() >= observe_due:
                    # Scheduler tick: the expensive one.  It also streams, since
                    # we are here anyway.
                    assessment = self.poll_once(
                        publish_assessment=publish_assessment
                    ).assessment
                    observe_due = self._clock() + self.backoff.interval(
                        observation_signature(assessment)
                    )
                else:
                    # Log tick: a file read, no scheduler query.
                    self.stream_once()
            except HpcAllocError as exc:
                # A controller restart, a VPN blip, or a closed laptop lid used
                # to abort the whole stream while the GPU job kept running.  Ride
                # it out within the budget; absorb() re-raises anything that is
                # not transient (auth, host key) or that outlives its patience.
                # A broken observation already cleared the tracker's death
                # candidate, so evidence from before the gap cannot combine with
                # evidence after it.
                self.retry_budget.absorb(exc)
                continue
            self.retry_budget.reset()
            if assessment.final:
                complete = self.drain() if drain else True
                return FollowOutcome(
                    assessment=assessment,
                    observed_terminal_state=assessment.terminal_state,
                    final_log_offset=self.offset,
                    bytes_written=self.total_bytes_written,
                    log_complete=complete,
                )
            self._sleep(self._next_tick(assessment, observe_due))

    def _next_tick(self, assessment: JobAssessment, observe_due: float) -> float:
        """How long to sleep before the next unit of work."""

        remaining = max(0.0, observe_due - self._clock())
        if not assessment.log_eligible:
            # Nothing has been written yet -- the job has not started -- so the
            # scheduler alone sets the pace.
            return remaining
        # Stream on a steady, responsive cadence, but never sleep past the next
        # scheduler observation.
        return min(self.log_interval, remaining)


__all__ = ["AssessmentPublisher", "FollowOutcome", "LogFollower", "PollResult"]
