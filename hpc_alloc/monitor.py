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
from .lifecycle import AssessmentPhase, EvidenceEvent, EvidenceTracker, JobAssessment
from .models import FinalSource, JobPhase, JobRecord
from .slurm import AccountingRecord, QueueRow
from .ssh import AuthMode


@dataclass(frozen=True, slots=True)
class MonitorResult:
    assessment: JobAssessment
    accounting_checked: bool = False


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
    ) -> MonitorResult:
        if job.ref is None:
            return MonitorResult(self.tracker(job).assessment)
        tracker = self.tracker(job)
        accounting_checked = False

        def exact_final(attempts: tuple[float, ...]) -> AccountingRecord | None:
            nonlocal accounting_checked
            accounting_checked = True
            try:
                return self.client.final(job.ref, attempts=attempts, auth=auth)
            except HpcAllocError as exc:
                break_lifecycle_evidence(tracker, exc)
                raise

        if job.phase is JobPhase.FINAL:
            if job.final_source is not FinalSource.CONFIRMED_QUEUE:
                return MonitorResult(tracker.assessment)
            record = exact_final((0,))
            if record is not None:
                return MonitorResult(
                    tracker.accept(EvidenceEvent.final(record)), accounting_checked
                )
            return MonitorResult(tracker.assessment, accounting_checked)
        row, assessment = accept_observation(
            tracker,
            lambda: self.client.observe(job.ref, auth=auth),
        )
        if assessment.final:
            record = exact_final((0, 2, 2))
            if record is not None:
                assessment = tracker.accept(EvidenceEvent.final(record))
            return MonitorResult(assessment, accounting_checked)
        if assessment.phase == AssessmentPhase.TERMINAL_CANDIDATE:
            record = exact_final((0,))
            if record is not None:
                return MonitorResult(
                    tracker.accept(EvidenceEvent.final(record)), accounting_checked
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
                    return MonitorResult(assessment, accounting_checked)
        return MonitorResult(assessment, accounting_checked)


def persist_assessment(repository: object, job: JobRecord, assessment: JobAssessment) -> JobRecord:
    if assessment.phase == AssessmentPhase.UNCERTAIN:
        # The caller may hold a stale object while another process records
        # stronger evidence.  Never return that stale copy as if it were the
        # durable result of this persistence attempt.
        return repository.get_job(job.operation_id)
    phase = JobPhase(assessment.phase.value)
    return repository.update_job(
        job.operation_id,
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
