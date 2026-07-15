"""Scheduler observation orchestration around the pure lifecycle engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .errors import (
    AuthRequired,
    HostKeyChanged,
    HpcAllocError,
    JobIdReused,
    TransportLost,
)
from .lifecycle import (
    AssessmentPhase,
    EvidenceEvent,
    EvidenceTracker,
    JobAssessment,
    awaits_requeue_confirmation,
)
from .models import FinalSource, JobPhase, JobRecord
from .slurm import REQUEUE_ELIGIBLE_FINAL, AccountingRecord, QueueRow
from .ssh import AuthMode


@dataclass(frozen=True, slots=True)
class MonitorResult:
    assessment: JobAssessment
    accounting_checked: bool = False
    # The accounting record this assessment consulted, if it consulted one.
    # `why` needs the same record for its own display, and was fetching it a
    # second time -- an entire retry ladder, the heaviest query the tool makes,
    # for a record whose content could not have changed in between.
    record: AccountingRecord | None = None


def break_lifecycle_evidence(tracker: EvidenceTracker, exc: HpcAllocError) -> JobAssessment:
    """Break consecutive observations after a typed operational failure."""

    if isinstance(exc, (TransportLost, AuthRequired, HostKeyChanged)):
        return tracker.accept(EvidenceEvent.transport_lost(str(exc)))
    return tracker.accept(EvidenceEvent.scheduler_error(str(exc)))


def accept_observation(
    tracker: EvidenceTracker,
    observe: Callable[[], QueueRow | None],
) -> tuple[QueueRow | None, JobAssessment]:
    """Convert one client observation into canonical lifecycle evidence.

    A recycled numeric locator is a successful non-live observation for the
    exact persisted identity.  Other typed failures break consecutiveness and
    remain failures for the caller to report.
    """

    tracker.begin_observation_epoch()
    try:
        row = observe()
    except JobIdReused as exc:
        return None, tracker.accept(EvidenceEvent.id_reused(str(exc)))
    except HpcAllocError as exc:
        break_lifecycle_evidence(tracker, exc)
        raise
    return row, tracker.accept(EvidenceEvent.queue(row))


class JobMonitor:
    def __init__(
        self,
        client: object,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        confirmation_delay: float = 3,
    ) -> None:
        self.client = client
        self._sleep = sleeper
        self.confirmation_delay = confirmation_delay

    @staticmethod
    def tracker(job: JobRecord) -> EvidenceTracker:
        return EvidenceTracker(
            ever_started=job.ever_started,
            current_node=job.current_node,
            last_node=job.last_node,
            phase=job.phase,
            terminal_state=job.terminal_state,
            exit_code=job.exit_code,
            observation_epoch=job.observation_epoch,
            evidence_provenance=job.evidence_provenance,
            evidence_detail=job.evidence_detail,
            restart_boundary=True,
            final_source=job.final_source,
        )

    def assess(
        self,
        job: JobRecord,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        confirm: bool = True,
        tracker: EvidenceTracker | None = None,
        extra_fields: tuple[str, ...] = (),
    ) -> MonitorResult:
        if job.ref is None:
            return MonitorResult((tracker or self.tracker(job)).assessment)
        tracker = tracker or self.tracker(job)
        accounting_checked = False
        consulted: AccountingRecord | None = None

        def exact_final(attempts: tuple[float, ...]) -> AccountingRecord | None:
            nonlocal accounting_checked, consulted
            accounting_checked = True
            # Only ask for the display-only columns when a caller wants them, so
            # the ordinary monitoring query keeps its exact shape.
            options: dict[str, object] = {"attempts": attempts, "auth": auth}
            if extra_fields:
                options["extra_fields"] = extra_fields
            try:
                consulted = self.client.final(job.ref, **options)
            except HpcAllocError as exc:
                break_lifecycle_evidence(tracker, exc)
                raise
            return consulted

        if job.phase is JobPhase.FINAL:
            if job.final_source is not FinalSource.CONFIRMED_QUEUE:
                return MonitorResult(tracker.assessment)
            record = exact_final((0,))
            if record is not None:
                return MonitorResult(
                    tracker.accept(EvidenceEvent.final(record)),
                    accounting_checked,
                    consulted,
                )
            return MonitorResult(tracker.assessment, accounting_checked, consulted)
        row, assessment = accept_observation(
            tracker,
            lambda: self.client.observe(job.ref, auth=auth),
        )
        if assessment.final:
            record = exact_final((0, 2, 2))
            if record is not None:
                assessment = tracker.accept(EvidenceEvent.final(record))
            return MonitorResult(assessment, accounting_checked, consulted)
        if assessment.phase == AssessmentPhase.TERMINAL_CANDIDATE:
            # A NODE_FAIL / PREEMPTED candidate must not be finalized by an
            # accounting read taken in this same cycle: that read describes the
            # same instant, inside the window in which Slurm requeues the job.
            # Fall through to the confirmation observation instead, which either
            # sees the job back in the queue (clearing the candidate) or confirms
            # the death with a genuinely independent second observation.
            if not awaits_requeue_confirmation(assessment):
                record = exact_final((0,))
                # awaits_requeue_confirmation keys on the candidate's own state,
                # but an absent (or otherwise state-less) candidate carries no
                # requeue-eligible terminal_state -- so a requeue-eligible
                # ACCOUNTING record slips past it into this same-cycle finalize.
                # Defer on the record too: a NODE_FAIL/PREEMPTED accounting row is
                # the same reap, and only the confirmation observation below (a
                # genuinely independent second look) may finalize it.
                if record is not None and record.state_code not in REQUEUE_ELIGIBLE_FINAL:
                    return MonitorResult(
                        tracker.accept(EvidenceEvent.final(record)),
                        accounting_checked,
                        consulted,
                    )
            if confirm:
                self._sleep(self.confirmation_delay)
                row, assessment = accept_observation(
                    tracker,
                    lambda: self.client.observe(job.ref, auth=auth),
                )
                if assessment.final:
                    record = exact_final((0, 2, 2))
                    if record is not None:
                        assessment = tracker.accept(EvidenceEvent.final(record))
                    return MonitorResult(assessment, accounting_checked, consulted)
        return MonitorResult(assessment, accounting_checked, consulted)


def persist_assessment(repository: object, job: JobRecord, assessment: JobAssessment) -> JobRecord:
    if assessment.phase == AssessmentPhase.UNCERTAIN:
        # No evidence is persisted.  Keep the returned row from the same
        # snapshot as ``assessment`` so callers never combine a fresh durable
        # job with stale process-local policy.
        return job
    phase = JobPhase(assessment.phase.value)
    return repository.update_job(
        job.operation_id,
        expected_updated_at=job.updated_at,
        phase=phase,
        # ``False`` means this tracker has not observed a start; it is not
        # evidence that another process's persisted start history is wrong.
        ever_started=True if assessment.ever_started else None,
        current_node=assessment.current_node,
        last_node=assessment.last_node,
        terminal_state=assessment.terminal_state,
        exit_code=assessment.exit_code,
        observation_epoch=assessment.observation_epoch,
        evidence_provenance=assessment.evidence_provenance,
        evidence_detail=(
            assessment.detail or None
            if assessment.evidence_provenance is not None
            else None
        ),
        final_source=(
            assessment.final_source if phase is JobPhase.FINAL else None
        ),
    )
