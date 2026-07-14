"""Typed values passed between hpc-alloc services."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class JobKind(StrEnum):
    ALLOCATION = "allocation"
    RUN = "run"


class JobPhase(StrEnum):
    SUBMITTING = "SUBMITTING"
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    STARTED_INACTIVE = "STARTED_INACTIVE"
    REQUEUEING = "REQUEUEING"
    TERMINAL_CANDIDATE = "TERMINAL_CANDIDATE"
    FINAL = "FINAL"


class FinalSource(StrEnum):
    """Durable provenance for a final job verdict.

    The values are part of the state schema.  Accounting is stronger than a
    conclusion drawn from confirmed queue departure.  The two local verdicts
    are explicit operator/mutation outcomes and must never be rewritten by a
    later scheduler observation.
    """

    ACCOUNTING = "accounting"
    CONFIRMED_QUEUE = "confirmed-queue"
    SUBMIT_FAILED = "submit-failed"
    ABANDONED = "abandoned"


class EvidenceProvenance(StrEnum):
    """Durable origin of non-live lifecycle evidence."""

    QUEUE_TERMINAL = "queue-terminal"
    ABSENT = "absent"
    ID_REUSED = "id-reused"
    CANCELLATION = "cancellation"


class OperationKind(StrEnum):
    SUBMIT = "submit"
    CANCEL = "cancel"


class OperationPhase(StrEnum):
    PREPARED = "PREPARED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    AMBIGUOUS = "AMBIGUOUS"
    CANCEL_PENDING = "CANCEL_PENDING"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


UNRESOLVED_OPERATION_PHASES = frozenset(
    {
        OperationPhase.PREPARED,
        OperationPhase.AMBIGUOUS,
        OperationPhase.CANCEL_PENDING,
    }
)


@dataclass(frozen=True, slots=True)
class JobRef:
    """Strong identity for a Slurm job managed by this machine."""

    cluster: str
    job_id: str
    owner_id: str
    operation_id: str
    slurm_job_name: str
    slurm_comment: str


@dataclass(frozen=True, slots=True)
class MachineRecord:
    machine_id: str
    hostname: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class JobRecord:
    operation_id: str
    cluster: str
    logical_name: str
    kind: JobKind
    owner_id: str
    slurm_job_name: str
    slurm_comment: str
    phase: JobPhase
    # Excluded from the generated __hash__, not from ==.  frozen+slots makes
    # dataclasses synthesize a __hash__, so this type advertises itself as
    # hashable exactly like its siblings -- but a dict field made that hash raise
    # TypeError, so the first set(jobs) or {job: alias} (the natural way to dedupe
    # the multi-cluster job lists built here) blew up with an *untyped* error that
    # escapes the CLI's error boundary.  Equal records have equal resources, so
    # equal records still hash equal: the hash/eq contract holds.
    resources: dict[str, Any] = field(default_factory=dict, hash=False)
    job_id: str | None = None
    ever_started: bool = False
    current_node: str | None = None
    last_node: str | None = None
    terminal_state: str | None = None
    exit_code: str | None = None
    observation_epoch: int = 0
    evidence_provenance: EvidenceProvenance | None = None
    evidence_detail: str | None = None
    final_source: FinalSource | None = None
    created_at: str = ""
    updated_at: str = ""
    finalized_at: str | None = None

    @property
    def ref(self) -> JobRef | None:
        """Return a strong reference once Slurm has acknowledged the job."""

        if self.job_id is None:
            return None
        return JobRef(
            cluster=self.cluster,
            job_id=self.job_id,
            owner_id=self.owner_id,
            operation_id=self.operation_id,
            slurm_job_name=self.slurm_job_name,
            slurm_comment=self.slurm_comment,
        )


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    kind: OperationKind
    phase: OperationPhase
    cluster: str
    logical_name: str
    target_job_operation_id: str
    job_id: str | None = None
    detail: str | None = None
    created_at: str = ""
    updated_at: str = ""
    resolved_at: str | None = None

    @property
    def unresolved(self) -> bool:
        return self.phase in UNRESOLVED_OPERATION_PHASES
