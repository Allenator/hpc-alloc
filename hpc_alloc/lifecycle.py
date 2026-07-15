"""Pure, requeue-aware lifecycle evidence tracking.

Commands feed successful queue observations, accounting records, and explicit
uncertainty events into :class:`EvidenceTracker`.  No command is allowed to
infer job death directly from a raw Slurm state or a single missing row.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import EvidenceProvenance, FinalSource, JobPhase
from .slurm import FINAL_STATES, REQUEUE_ELIGIBLE_FINAL, AccountingRecord, QueueRow


class AssessmentPhase(StrEnum):
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    STARTED_INACTIVE = "STARTED_INACTIVE"
    REQUEUEING = "REQUEUEING"
    TERMINAL_CANDIDATE = "TERMINAL_CANDIDATE"
    FINAL = "FINAL"
    UNCERTAIN = "UNCERTAIN"


class EvidenceKind(StrEnum):
    QUEUE = "queue"
    ABSENT = "absent"
    ID_REUSED = "id-reused"
    TRANSPORT_LOST = "transport-lost"
    SCHEDULER_ERROR = "scheduler-error"
    FINAL_ACCOUNTING = "final-accounting"


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    kind: EvidenceKind
    row: QueueRow | None = None
    accounting: AccountingRecord | None = None
    detail: str = ""

    @classmethod
    def queue(cls, row: QueueRow | None) -> "EvidenceEvent":
        return cls(EvidenceKind.QUEUE if row is not None else EvidenceKind.ABSENT, row=row)

    @classmethod
    def absent(cls) -> "EvidenceEvent":
        return cls(EvidenceKind.ABSENT)

    @classmethod
    def id_reused(cls, detail: str = "") -> "EvidenceEvent":
        return cls(EvidenceKind.ID_REUSED, detail=detail)

    @classmethod
    def transport_lost(cls, detail: str = "") -> "EvidenceEvent":
        return cls(EvidenceKind.TRANSPORT_LOST, detail=detail)

    @classmethod
    def scheduler_error(cls, detail: str = "") -> "EvidenceEvent":
        return cls(EvidenceKind.SCHEDULER_ERROR, detail=detail)

    @classmethod
    def final(cls, record: AccountingRecord) -> "EvidenceEvent":
        return cls(EvidenceKind.FINAL_ACCOUNTING, accounting=record)


@dataclass(frozen=True, slots=True)
class JobAssessment:
    phase: AssessmentPhase
    scheduler_state: str | None
    ever_started: bool
    current_node: str | None
    last_node: str | None
    terminal_state: str | None
    exit_code: str | None
    terminal_evidence: int
    absence_streak: int
    observation_epoch: int
    evidence_provenance: EvidenceProvenance | None = None
    final_source: FinalSource | None = None
    detail: str = ""

    @property
    def final(self) -> bool:
        return self.phase == AssessmentPhase.FINAL

    @property
    def uncertain(self) -> bool:
        return self.phase == AssessmentPhase.UNCERTAIN

    @property
    def log_eligible(self) -> bool:
        if self.final_source in {
            FinalSource.SUBMIT_FAILED,
            FinalSource.ABANDONED,
        }:
            return False
        return self.ever_started or self.final_source in {
            FinalSource.CONFIRMED_QUEUE,
            FinalSource.ACCOUNTING,
        }


_ACTIVE = frozenset({"RUNNING", "RESIZING", "SIGNALING"})
_QUEUED = frozenset({"PENDING", "CONFIGURING", "RESV_DEL_HOLD"})
_STARTED_INACTIVE = frozenset({"SUSPENDED", "STOPPED", "COMPLETING", "STAGE_OUT"})
_REQUEUEING = frozenset({"REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SPECIAL_EXIT"})
# The non-final states a landed cancellation can be observed in: while the kill
# signal is being delivered the job reports SIGNALING, then it drains through
# COMPLETING (or a stage-out epilog) to terminal.  Observing any OTHER non-final
# state proves a cancellation never arrived.  SIGNALING is also in _ACTIVE, so
# this set is consulted by scheduler state, not by phase.
_CANCELLATION_DRAINING = frozenset({"SIGNALING", "COMPLETING", "STAGE_OUT"})
_PROVES_STARTED = _ACTIVE | _STARTED_INACTIVE | _REQUEUEING | frozenset(
    {
        "COMPLETED",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)


def state_code(state: str) -> str:
    """Reduce a Slurm state phrase to its bare state code.

    ``CANCELLED by 12345`` becomes ``CANCELLED`` and a blank or whitespace-only
    phrase becomes ``""``.  Every consumer must route through this helper: an
    empty phrase has no first word, so hand-rolled ``split()[0]`` indexing
    raises instead of yielding the empty code its callers guard on.
    """

    words = state.upper().split()
    return words[0].rstrip("+") if words else ""


class EvidenceTracker:
    """Accumulate evidence without conflating absence, failure, and finality."""

    def __init__(
        self,
        *,
        ever_started: bool = False,
        current_node: str | None = None,
        last_node: str | None = None,
        observation_epoch: int = 0,
        evidence_provenance: EvidenceProvenance | str | None = None,
        evidence_detail: str | None = None,
        restart_boundary: bool = False,
        phase: AssessmentPhase | JobPhase | str | None = None,
        terminal_state: str | None = None,
        exit_code: str | None = None,
        final_source: FinalSource | str | None = None,
    ) -> None:
        if phase is None:
            initial_phase = AssessmentPhase.ACTIVE if current_node else AssessmentPhase.QUEUED
        elif JobPhase(phase) is JobPhase.SUBMITTING:
            initial_phase = AssessmentPhase.QUEUED
        else:
            initial_phase = AssessmentPhase(JobPhase(phase).value)
        source = FinalSource(final_source) if final_source is not None else None
        provenance = (
            EvidenceProvenance(evidence_provenance)
            if evidence_provenance is not None
            else None
        )
        if initial_phase is AssessmentPhase.FINAL and source is None:
            raise ValueError("a final tracker requires final-source provenance")
        if initial_phase is not AssessmentPhase.FINAL and source is not None:
            raise ValueError("only a final tracker may have final-source provenance")
        retains_non_live_evidence = (
            initial_phase is AssessmentPhase.TERMINAL_CANDIDATE
            or source is FinalSource.CONFIRMED_QUEUE
        )
        if provenance is not None and not retains_non_live_evidence:
            raise ValueError("only non-live candidate/final evidence may have provenance")
        if provenance is None and retains_non_live_evidence:
            provenance = (
                EvidenceProvenance.QUEUE_TERMINAL
                if terminal_state
                else EvidenceProvenance.ABSENT
            )
        if evidence_detail is not None and provenance is None:
            raise ValueError("evidence detail requires non-live provenance")

        self._phase = initial_phase
        self._scheduler_state: str | None = None
        self._ever_started = ever_started or current_node is not None
        self._current_node = current_node if initial_phase is AssessmentPhase.ACTIVE else None
        self._last_node = last_node or current_node
        self._terminal_state = terminal_state
        self._exit_code = exit_code
        self._terminal_evidence = (
            2
            if source is FinalSource.CONFIRMED_QUEUE
            else 1
            if initial_phase in {AssessmentPhase.TERMINAL_CANDIDATE, AssessmentPhase.FINAL}
            else 0
        )
        self._absence_streak = 0
        self._epoch = observation_epoch
        self._evidence_provenance = provenance
        self._final_source = source
        self._detail = evidence_detail or ""
        self._restart_boundary_pending = restart_boundary

    @property
    def assessment(self) -> JobAssessment:
        return JobAssessment(
            phase=self._phase,
            scheduler_state=self._scheduler_state,
            ever_started=self._ever_started,
            current_node=self._current_node,
            last_node=self._last_node,
            terminal_state=self._terminal_state,
            exit_code=self._exit_code,
            terminal_evidence=self._terminal_evidence,
            absence_streak=self._absence_streak,
            observation_epoch=self._epoch,
            evidence_provenance=self._evidence_provenance,
            final_source=self._final_source,
            detail=self._detail,
        )

    def begin_observation_epoch(self) -> JobAssessment:
        """Start one process-local observation session exactly once.

        A process restart is an uncertainty boundary: a persisted candidate is
        retained for diagnosis, but its count cannot combine with a later
        observation because an unrecorded failure may have intervened.
        """

        if not self._restart_boundary_pending:
            return self.assessment
        self._restart_boundary_pending = False
        self._epoch += 1
        if self._phase is AssessmentPhase.TERMINAL_CANDIDATE:
            self._clear_candidate()
            self._phase = AssessmentPhase.UNCERTAIN
            self._scheduler_state = None
            self._detail = "new observation session started"
        return self.assessment

    def _clear_candidate(self) -> None:
        self._terminal_evidence = 0
        self._absence_streak = 0
        self._terminal_state = None
        self._exit_code = None
        self._evidence_provenance = None
        self._final_source = None

    def _uncertain(self, detail: str) -> JobAssessment:
        # A failed observation boundary breaks consecutiveness.  Evidence from
        # before it may not combine with evidence after it.
        self._epoch += 1
        self._clear_candidate()
        self._phase = AssessmentPhase.UNCERTAIN
        self._scheduler_state = None
        self._detail = detail
        return self.assessment

    def _non_live(
        self,
        state: str | None,
        *,
        absent: bool,
        provenance: EvidenceProvenance,
        detail: str = "",
    ) -> JobAssessment:
        self._current_node = None
        self._terminal_evidence += 1
        self._absence_streak = self._absence_streak + 1 if absent else 0
        if state:
            self._terminal_state = state
        self._scheduler_state = state
        self._evidence_provenance = provenance
        self._detail = (
            "job absent from a successful queue snapshot" if absent else detail
        )
        if self._terminal_evidence >= 2:
            self._phase = AssessmentPhase.FINAL
            self._final_source = FinalSource.CONFIRMED_QUEUE
        else:
            self._phase = AssessmentPhase.TERMINAL_CANDIDATE
            self._final_source = None
        return self.assessment

    def _queue(self, row: QueueRow) -> JobAssessment:
        state = state_code(row.state)
        if not state:
            return self._uncertain("queue row had no scheduler state")
        if state in FINAL_STATES:
            if state in _PROVES_STARTED:
                self._ever_started = True
                if row.node:
                    self._last_node = row.node
            return self._non_live(
                state,
                absent=False,
                provenance=EvidenceProvenance.QUEUE_TERMINAL,
            )

        # A present, nonterminal row invalidates every prior death candidate.
        self._clear_candidate()
        self._scheduler_state = state
        self._detail = row.reason if state == "PENDING" else ""
        if state in _PROVES_STARTED:
            self._ever_started = True
            if row.node:
                self._last_node = row.node

        if state in _ACTIVE:
            self._ever_started = True
            self._current_node = row.node
            if row.node:
                self._last_node = row.node
            self._phase = AssessmentPhase.ACTIVE
        elif state in _STARTED_INACTIVE:
            self._ever_started = True
            self._current_node = None
            self._phase = AssessmentPhase.STARTED_INACTIVE
        elif state in _REQUEUEING:
            self._ever_started = True
            self._current_node = None
            self._phase = AssessmentPhase.REQUEUEING
        elif state in _QUEUED:
            self._current_node = None
            self._phase = (
                AssessmentPhase.REQUEUEING if self._ever_started else AssessmentPhase.QUEUED
            )
        else:
            # The row proves presence but an unknown Slurm state cannot safely
            # drive any command policy.  Break evidence like other uncertainty.
            return self._uncertain(f"unrecognized Slurm state {state!r}")
        return self.assessment

    def accept(self, event: EvidenceEvent) -> JobAssessment:
        if self._phase is AssessmentPhase.FINAL:
            # Finality is monotonic.  Only authoritative accounting may refine
            # a queue-derived final; local terminal verdicts are immutable.
            if not (
                event.kind is EvidenceKind.FINAL_ACCOUNTING
                and self._final_source is FinalSource.CONFIRMED_QUEUE
            ):
                return self.assessment
        if event.kind == EvidenceKind.QUEUE:
            if event.row is None:
                raise ValueError("queue evidence requires a QueueRow")
            return self._queue(event.row)
        if event.kind == EvidenceKind.ABSENT:
            return self._non_live(
                self._terminal_state,
                absent=True,
                provenance=EvidenceProvenance.ABSENT,
            )
        if event.kind == EvidenceKind.ID_REUSED:
            return self._non_live(
                self._terminal_state,
                absent=False,
                provenance=EvidenceProvenance.ID_REUSED,
                detail=event.detail
                or "numeric Slurm job ID now belongs to a different operation",
            )
        if event.kind == EvidenceKind.TRANSPORT_LOST:
            return self._uncertain(event.detail or "SSH transport was unavailable")
        if event.kind == EvidenceKind.SCHEDULER_ERROR:
            return self._uncertain(event.detail or "scheduler observation failed")
        if event.kind == EvidenceKind.FINAL_ACCOUNTING:
            record = event.accounting
            if record is None or not record.final:
                raise ValueError("final-accounting evidence requires a final record")
            if record.state_code in _PROVES_STARTED:
                self._ever_started = True
            self._scheduler_state = record.state_code
            self._terminal_state = record.state
            self._exit_code = record.exit_code
            self._current_node = None
            self._terminal_evidence = max(1, self._terminal_evidence)
            self._absence_streak = 0
            self._phase = AssessmentPhase.FINAL
            self._evidence_provenance = None
            self._final_source = FinalSource.ACCOUNTING
            self._detail = ""
            return self.assessment
        raise ValueError(f"unsupported evidence kind {event.kind!r}")


def awaits_requeue_confirmation(assessment: JobAssessment) -> bool:
    """True for a terminal candidate whose state Slurm may requeue it out of.

    Queue evidence already requires two independent observations before a job is
    declared dead.  Accounting evidence did not: a single accounting read, taken
    in the same poll cycle as the observation that produced the candidate,
    promoted the job straight to an immutable FINAL/ACCOUNTING verdict.  For
    NODE_FAIL and PREEMPTED -- the two states Slurm actually requeues on -- that
    read describes the same instant, inside the requeue window, so it is not
    independent evidence at all: the job was irreversibly reaped, `status` then
    hid it, and the requeued instance ran on holding its GPUs, untracked, for
    the full walltime.

    Callers must therefore defer the accounting read for such a candidate and
    let the next cycle's queue observation decide: a second terminal or absent
    row finalizes the job, while a requeued job reappears as PENDING and clears
    the candidate outright.
    """

    return (
        assessment.phase is AssessmentPhase.TERMINAL_CANDIDATE
        and state_code(assessment.terminal_state or "") in REQUEUE_ELIGIBLE_FINAL
    )


def proves_cancellation_did_not_land(assessment: JobAssessment) -> bool:
    """True when a read-only observation shows the job unambiguously alive in a
    state a landed cancellation could not have produced -- so an ambiguous
    cancellation whose reply was lost definitely never arrived, and its guard
    can be released for an idempotent retry.

    False for final, uncertain, and terminal-candidate assessments (their own
    paths handle finality, and a death candidate is provisional death -- often
    because the cancellation landed -- not proof of life), and for a job draining
    through the kill sequence ({SIGNALING, COMPLETING, STAGE_OUT}), which is
    exactly what a landed cancellation looks like.  This is self-contained: it
    does not rely on the caller having resolved candidates first."""

    if (
        assessment.final
        or assessment.uncertain
        or assessment.phase is AssessmentPhase.TERMINAL_CANDIDATE
    ):
        return False
    return state_code(assessment.scheduler_state or "") not in _CANCELLATION_DRAINING


__all__ = [
    "AssessmentPhase",
    "EvidenceEvent",
    "EvidenceKind",
    "EvidenceTracker",
    "JobAssessment",
    "awaits_requeue_confirmation",
    "proves_cancellation_did_not_land",
    "state_code",
]
