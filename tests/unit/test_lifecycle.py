from __future__ import annotations

import unittest

from hpc_alloc.lifecycle import AssessmentPhase, EvidenceEvent, EvidenceTracker
from hpc_alloc.slurm import AccountingRecord, QueueRow


def row(state: str, node: str | None = None) -> QueueRow:
    return QueueRow("42", state, node, "None", "1:00:00", "day", "job", "now", "tag")


class EvidenceTrackerTests(unittest.TestCase):
    def test_uncertainty_breaks_consecutive_absence(self) -> None:
        tracker = EvidenceTracker()
        self.assertEqual(
            tracker.accept(EvidenceEvent.absent()).phase,
            AssessmentPhase.TERMINAL_CANDIDATE,
        )
        uncertain = tracker.accept(EvidenceEvent.transport_lost("VPN dropped"))
        self.assertEqual(uncertain.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(uncertain.terminal_evidence, 0)
        after = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(after.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertFalse(after.final)
        self.assertEqual(after.observation_epoch, 1)

    def test_transient_preemption_can_return_to_running(self) -> None:
        tracker = EvidenceTracker(ever_started=True, current_node="old")
        candidate = tracker.accept(EvidenceEvent.queue(row("PREEMPTED", "old")))
        self.assertEqual(candidate.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertIsNone(candidate.current_node)
        running = tracker.accept(EvidenceEvent.queue(row("RUNNING", "new")))
        self.assertEqual(running.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(running.current_node, "new")
        self.assertEqual(running.last_node, "new")
        self.assertEqual(running.terminal_evidence, 0)

    def test_completing_requires_two_later_absences_to_confirm_departure(self) -> None:
        tracker = EvidenceTracker(ever_started=True)
        completing = tracker.accept(EvidenceEvent.queue(row("COMPLETING")))
        self.assertEqual(completing.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertEqual(completing.terminal_evidence, 0)
        candidate = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(candidate.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        final = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(final.phase, AssessmentPhase.FINAL)
        self.assertIsNone(final.terminal_state)
        self.assertEqual(final.final_source, "confirmed-queue")

    def test_suspended_job_remains_log_eligible_but_has_no_current_node(self) -> None:
        tracker = EvidenceTracker(ever_started=True, current_node="n1")
        result = tracker.accept(EvidenceEvent.queue(row("SUSPENDED", "n1")))
        self.assertEqual(result.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertTrue(result.log_eligible)
        self.assertIsNone(result.current_node)
        self.assertEqual(result.last_node, "n1")

    def test_pending_after_start_is_requeueing(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="n1")
        result = tracker.accept(EvidenceEvent.queue(row("PENDING")))
        self.assertEqual(result.phase, AssessmentPhase.REQUEUEING)
        self.assertTrue(result.ever_started)

    def test_final_accounting_is_conclusive_and_completed_proves_start(self) -> None:
        tracker = EvidenceTracker()
        record = AccountingRecord("42", "COMPLETED", "0:0", "job", "tag")
        result = tracker.accept(EvidenceEvent.final(record))
        self.assertEqual(result.phase, AssessmentPhase.FINAL)
        self.assertTrue(result.ever_started)
        self.assertEqual(result.exit_code, "0:0")
        self.assertEqual(result.final_source, "accounting")

    def test_cancelled_accounting_does_not_invent_start_history(self) -> None:
        tracker = EvidenceTracker()
        result = tracker.accept(
            EvidenceEvent.final(AccountingRecord("42", "CANCELLED", "0:15", "job", "tag"))
        )
        self.assertFalse(result.ever_started)


if __name__ == "__main__":
    unittest.main()
