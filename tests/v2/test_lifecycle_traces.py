from __future__ import annotations

import unittest

from hpc_alloc.lifecycle import AssessmentPhase, EvidenceEvent, EvidenceTracker
from hpc_alloc.models import FinalSource, JobPhase
from hpc_alloc.slurm import FINAL_STATES, AccountingRecord, QueueRow


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


def final_record(state: str = "COMPLETED", exit_code: str = "0:0") -> AccountingRecord:
    return AccountingRecord(
        job_id="12345",
        state=state,
        exit_code=exit_code,
        job_name="hpcalloc-v2-run-" + "a" * 32,
        comment="hpc-alloc:v2:deadbeef1234:" + "a" * 32 + ":laptop:run:-",
    )


class LifecycleTraceTests(unittest.TestCase):
    def test_every_squeue_state_has_an_explicit_safe_classification(self) -> None:
        expected = {
            "BOOT_FAIL": (AssessmentPhase.TERMINAL_CANDIDATE, False),
            "CANCELLED": (AssessmentPhase.TERMINAL_CANDIDATE, False),
            "COMPLETED": (AssessmentPhase.TERMINAL_CANDIDATE, True),
            "CONFIGURING": (AssessmentPhase.QUEUED, False),
            "COMPLETING": (AssessmentPhase.STARTED_INACTIVE, True),
            "DEADLINE": (AssessmentPhase.TERMINAL_CANDIDATE, False),
            "FAILED": (AssessmentPhase.TERMINAL_CANDIDATE, True),
            "NODE_FAIL": (AssessmentPhase.TERMINAL_CANDIDATE, True),
            "OUT_OF_MEMORY": (AssessmentPhase.TERMINAL_CANDIDATE, True),
            "PENDING": (AssessmentPhase.QUEUED, False),
            "PREEMPTED": (AssessmentPhase.TERMINAL_CANDIDATE, True),
            "RUNNING": (AssessmentPhase.ACTIVE, True),
            "RESV_DEL_HOLD": (AssessmentPhase.QUEUED, False),
            "REQUEUE_FED": (AssessmentPhase.REQUEUEING, True),
            "REQUEUE_HOLD": (AssessmentPhase.REQUEUEING, True),
            "REQUEUED": (AssessmentPhase.REQUEUEING, True),
            "RESIZING": (AssessmentPhase.ACTIVE, True),
            "REVOKED": (AssessmentPhase.TERMINAL_CANDIDATE, False),
            "SIGNALING": (AssessmentPhase.ACTIVE, True),
            "SPECIAL_EXIT": (AssessmentPhase.REQUEUEING, True),
            "STAGE_OUT": (AssessmentPhase.STARTED_INACTIVE, True),
            "STOPPED": (AssessmentPhase.STARTED_INACTIVE, True),
            "SUSPENDED": (AssessmentPhase.STARTED_INACTIVE, True),
            "TIMEOUT": (AssessmentPhase.TERMINAL_CANDIDATE, True),
        }

        self.assertEqual(
            FINAL_STATES,
            {
                state
                for state, (phase, _started) in expected.items()
                if phase is AssessmentPhase.TERMINAL_CANDIDATE
            },
        )
        for state, (phase, started) in expected.items():
            with self.subTest(state=state):
                assessment = EvidenceTracker().accept(
                    EvidenceEvent.queue(row(state, node="node01"))
                )
                self.assertEqual(assessment.phase, phase)
                self.assertEqual(assessment.ever_started, started)
                self.assertEqual(assessment.log_eligible, started)
                self.assertFalse(assessment.final)
                self.assertEqual(
                    assessment.current_node,
                    "node01" if phase is AssessmentPhase.ACTIVE else None,
                )
                self.assertEqual(assessment.last_node, "node01" if started else None)
                self.assertEqual(
                    assessment.terminal_evidence,
                    1 if phase is AssessmentPhase.TERMINAL_CANDIDATE else 0,
                )

    def test_requeue_trace_preserves_started_history_and_clears_stale_node(self) -> None:
        tracker = EvidenceTracker()

        pending = tracker.accept(EvidenceEvent.queue(row("PENDING", reason="Resources")))
        self.assertEqual(pending.phase, AssessmentPhase.QUEUED)
        self.assertFalse(pending.ever_started)
        self.assertFalse(pending.log_eligible)

        running = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node01")))
        self.assertEqual(running.phase, AssessmentPhase.ACTIVE)
        self.assertTrue(running.ever_started)
        self.assertEqual(running.current_node, "node01")
        self.assertEqual(running.last_node, "node01")

        suspended = tracker.accept(EvidenceEvent.queue(row("SUSPENDED", node="node01")))
        self.assertEqual(suspended.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertTrue(suspended.log_eligible)
        self.assertIsNone(suspended.current_node)
        self.assertEqual(suspended.last_node, "node01")

        requeued = tracker.accept(EvidenceEvent.queue(row("REQUEUED")))
        self.assertEqual(requeued.phase, AssessmentPhase.REQUEUEING)
        self.assertTrue(requeued.ever_started)
        self.assertIsNone(requeued.current_node)

        queued_again = tracker.accept(EvidenceEvent.queue(row("PENDING", reason="Priority")))
        self.assertEqual(queued_again.phase, AssessmentPhase.REQUEUEING)
        self.assertTrue(queued_again.log_eligible)
        self.assertEqual(queued_again.last_node, "node01")

        moved = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node02")))
        self.assertEqual(moved.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(moved.current_node, "node02")
        self.assertEqual(moved.last_node, "node02")

    def test_one_terminal_looking_row_is_not_final_and_requeue_clears_it(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")

        candidate = tracker.accept(EvidenceEvent.queue(row("PREEMPTED")))
        self.assertEqual(candidate.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertFalse(candidate.final)
        self.assertEqual(candidate.terminal_evidence, 1)

        bounce = tracker.accept(EvidenceEvent.queue(row("PENDING", reason="Requeued")))
        self.assertEqual(bounce.phase, AssessmentPhase.REQUEUEING)
        self.assertFalse(bounce.final)
        self.assertEqual(bounce.terminal_evidence, 0)
        self.assertIsNone(bounce.terminal_state)

    def test_special_exit_is_requeueing_and_can_return_to_running(self) -> None:
        tracker = EvidenceTracker()

        special = tracker.accept(EvidenceEvent.queue(row("SPECIAL_EXIT", node="node01")))
        self.assertEqual(special.phase, AssessmentPhase.REQUEUEING)
        self.assertTrue(special.ever_started)
        self.assertTrue(special.log_eligible)
        self.assertIsNone(special.current_node)
        self.assertEqual(special.last_node, "node01")
        self.assertEqual(special.terminal_evidence, 0)
        self.assertIsNone(special.terminal_state)

        pending = tracker.accept(EvidenceEvent.queue(row("PENDING", reason="Held")))
        self.assertEqual(pending.phase, AssessmentPhase.REQUEUEING)
        self.assertFalse(pending.final)

        running = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node02")))
        self.assertEqual(running.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(running.current_node, "node02")
        self.assertEqual(running.last_node, "node02")

    def test_reservation_deleted_hold_after_start_is_requeueing(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")

        assessment = tracker.accept(EvidenceEvent.queue(row("RESV_DEL_HOLD")))

        self.assertEqual(assessment.phase, AssessmentPhase.REQUEUEING)
        self.assertTrue(assessment.ever_started)
        self.assertTrue(assessment.log_eligible)
        self.assertIsNone(assessment.current_node)
        self.assertEqual(assessment.last_node, "node01")

    def test_special_exit_accounting_is_not_final_evidence(self) -> None:
        record = final_record("SPECIAL_EXIT", "0:0")
        self.assertFalse(record.final)
        with self.assertRaisesRegex(ValueError, "requires a final record"):
            EvidenceTracker().accept(EvidenceEvent.final(record))

    def test_present_completing_is_started_inactive_not_terminal_evidence(self) -> None:
        tracker = EvidenceTracker()

        first = tracker.accept(EvidenceEvent.queue(row("COMPLETING", node="node01")))
        self.assertEqual(first.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertTrue(first.ever_started)
        self.assertEqual(first.last_node, "node01")
        self.assertIsNone(first.current_node)
        self.assertEqual(first.terminal_evidence, 0)
        self.assertIsNone(first.terminal_state)

        repeated = tracker.accept(EvidenceEvent.queue(row("COMPLETING", node="node01")))
        self.assertEqual(repeated.phase, AssessmentPhase.STARTED_INACTIVE)
        self.assertFalse(repeated.final)
        self.assertEqual(repeated.terminal_evidence, 0)

    def test_recycled_id_is_detailed_non_live_evidence_with_normal_confirmation(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")
        detail = "job grace:12345 now belongs to another operation"

        candidate = tracker.accept(EvidenceEvent.id_reused(detail))
        self.assertEqual(candidate.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertEqual(candidate.terminal_evidence, 1)
        self.assertEqual(candidate.absence_streak, 0)
        self.assertEqual(candidate.detail, detail)

        confirmed = tracker.accept(EvidenceEvent.id_reused(detail))
        self.assertEqual(confirmed.phase, AssessmentPhase.FINAL)
        self.assertEqual(confirmed.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(confirmed.terminal_evidence, 2)
        self.assertEqual(confirmed.detail, detail)

    def test_exact_reappearance_clears_recycled_id_candidate(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")
        tracker.accept(EvidenceEvent.id_reused("numeric ID was recycled"))

        alive = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node02")))

        self.assertEqual(alive.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(alive.current_node, "node02")
        self.assertEqual(alive.terminal_evidence, 0)
        self.assertEqual(alive.detail, "")

    def test_error_breaks_recycled_id_consecutiveness(self) -> None:
        tracker = EvidenceTracker()
        tracker.accept(EvidenceEvent.id_reused("numeric ID was recycled"))

        uncertain = tracker.accept(EvidenceEvent.transport_lost("VPN dropped"))
        self.assertEqual(uncertain.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(uncertain.terminal_evidence, 0)

        candidate = tracker.accept(EvidenceEvent.id_reused("numeric ID was recycled"))
        self.assertEqual(candidate.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertEqual(candidate.terminal_evidence, 1)
        self.assertFalse(candidate.final)

    def test_transport_boundary_breaks_absence_consecutiveness(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")

        first_absence = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(first_absence.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertEqual(first_absence.absence_streak, 1)

        offline = tracker.accept(EvidenceEvent.transport_lost("VPN dropped"))
        self.assertEqual(offline.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(offline.absence_streak, 0)
        self.assertEqual(offline.terminal_evidence, 0)
        self.assertEqual(offline.observation_epoch, 1)

        post_reconnect_blip = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(post_reconnect_blip.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertFalse(post_reconnect_blip.final)
        self.assertEqual(post_reconnect_blip.absence_streak, 1)

        confirmed = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(confirmed.phase, AssessmentPhase.FINAL)
        self.assertEqual(confirmed.final_source, "confirmed-queue")

    def test_scheduler_error_also_breaks_death_evidence(self) -> None:
        tracker = EvidenceTracker()
        tracker.accept(EvidenceEvent.queue(row("COMPLETING")))
        uncertain = tracker.accept(EvidenceEvent.scheduler_error("controller failover"))
        self.assertEqual(uncertain.phase, AssessmentPhase.UNCERTAIN)
        self.assertEqual(uncertain.terminal_evidence, 0)

        alive = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node03")))
        self.assertEqual(alive.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(alive.current_node, "node03")

    def test_final_accounting_is_immediate_authoritative_evidence(self) -> None:
        tracker = EvidenceTracker(ever_started=True, current_node="node01")
        assessment = tracker.accept(EvidenceEvent.final(final_record("TIMEOUT", "0:0")))
        self.assertTrue(assessment.final)
        self.assertEqual(assessment.final_source, "accounting")
        self.assertEqual(assessment.terminal_state, "TIMEOUT")
        self.assertEqual(assessment.exit_code, "0:0")
        self.assertIsNone(assessment.current_node)

    def test_only_scheduler_finals_are_log_eligible_without_start_proof(self) -> None:
        for state, exit_code in (("CANCELLED", "0:15"), ("BOOT_FAIL", "1:0")):
            with self.subTest(accounting_state=state):
                accounting = EvidenceTracker().accept(
                    EvidenceEvent.final(final_record(state, exit_code))
                )
                self.assertFalse(accounting.ever_started)
                self.assertTrue(accounting.log_eligible)

        queue_tracker = EvidenceTracker()
        queue_tracker.accept(EvidenceEvent.absent())
        queue_final = queue_tracker.accept(EvidenceEvent.absent())
        self.assertFalse(queue_final.ever_started)
        self.assertEqual(queue_final.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertTrue(queue_final.log_eligible)

        for source, state in (
            (FinalSource.SUBMIT_FAILED, "SUBMIT_FAILED"),
            (FinalSource.ABANDONED, "ABANDONED"),
        ):
            with self.subTest(source=source):
                local = EvidenceTracker(
                    phase=JobPhase.FINAL,
                    ever_started=True,
                    terminal_state=state,
                    final_source=source,
                ).assessment
                self.assertTrue(local.ever_started)
                self.assertFalse(local.log_eligible)

    def test_confirmed_queue_death_retains_the_observed_terminal_state(self) -> None:
        tracker = EvidenceTracker(ever_started=True, last_node="node01")
        candidate = tracker.accept(EvidenceEvent.queue(row("COMPLETED")))
        self.assertFalse(candidate.final)
        self.assertEqual(candidate.terminal_state, "COMPLETED")

        confirmed = tracker.accept(EvidenceEvent.absent())
        self.assertTrue(confirmed.final)
        self.assertEqual(confirmed.final_source, "confirmed-queue")
        self.assertEqual(confirmed.terminal_state, "COMPLETED")

    def test_nonfinal_accounting_cannot_be_smuggled_in_as_final(self) -> None:
        tracker = EvidenceTracker()
        with self.assertRaisesRegex(ValueError, "requires a final record"):
            tracker.accept(EvidenceEvent.final(final_record("RUNNING")))

    def test_unknown_scheduler_state_is_uncertainty_not_death(self) -> None:
        assessment = EvidenceTracker().accept(EvidenceEvent.queue(row("FUTURE_STATE")))
        self.assertEqual(assessment.phase, AssessmentPhase.UNCERTAIN)
        self.assertFalse(assessment.final)
        self.assertIn("unrecognized", assessment.detail)

    def test_persisted_candidate_seeds_evidence_and_terminal_metadata(self) -> None:
        tracker = EvidenceTracker(
            phase=JobPhase.TERMINAL_CANDIDATE,
            ever_started=True,
            last_node="node01",
            terminal_state="COMPLETING",
        )
        seeded = tracker.assessment
        self.assertEqual(seeded.phase, AssessmentPhase.TERMINAL_CANDIDATE)
        self.assertEqual(seeded.terminal_evidence, 1)
        self.assertEqual(seeded.terminal_state, "COMPLETING")

        final = tracker.accept(EvidenceEvent.absent())
        self.assertEqual(final.phase, AssessmentPhase.FINAL)
        self.assertEqual(final.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(final.terminal_state, "COMPLETING")

    def test_persisted_final_is_monotonic_and_accounting_can_upgrade_queue_source(self) -> None:
        tracker = EvidenceTracker(
            phase=JobPhase.FINAL,
            ever_started=True,
            last_node="node01",
            terminal_state="COMPLETING",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        still_final = tracker.accept(EvidenceEvent.queue(row("RUNNING", node="node02")))
        self.assertEqual(still_final.phase, AssessmentPhase.FINAL)
        self.assertEqual(still_final.terminal_state, "COMPLETING")
        self.assertEqual(still_final.last_node, "node01")

        upgraded = tracker.accept(EvidenceEvent.final(final_record("COMPLETED", "0:0")))
        self.assertEqual(upgraded.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(upgraded.terminal_state, "COMPLETED")
        self.assertEqual(upgraded.exit_code, "0:0")


if __name__ == "__main__":
    unittest.main()
