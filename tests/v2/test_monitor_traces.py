from __future__ import annotations

import unittest
from dataclasses import replace

from hpc_alloc.errors import JobIdReused, TransportLost
from hpc_alloc.lifecycle import AssessmentPhase
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    JobRecord,
)
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.slurm import AccountingRecord, QueueRow
from hpc_alloc.ssh import AuthMode

from .fakes import ExpectedCall, StrictProxy, StrictScript, VirtualClock


OPERATION_ID = "a" * 32
OWNER_ID = "deadbeef1234"
JOB_NAME = slurm_job_name("run", OPERATION_ID)
COMMENT = format_tag(OWNER_ID, OPERATION_ID, "laptop", "run", None)


def job(*, ever_started: bool = False) -> JobRecord:
    return JobRecord(
        operation_id=OPERATION_ID,
        cluster="grace",
        logical_name="run",
        kind=JobKind.RUN,
        owner_id=OWNER_ID,
        slurm_job_name=JOB_NAME,
        slurm_comment=COMMENT,
        phase=JobPhase.ACTIVE if ever_started else JobPhase.QUEUED,
        job_id="12345",
        ever_started=ever_started,
        current_node="node01" if ever_started else None,
        last_node="node01" if ever_started else None,
    )


def row(state: str, *, node: str | None = None) -> QueueRow:
    return QueueRow(
        job_id="12345",
        state=state,
        node=node,
        reason="Requeued" if state == "PENDING" else "",
        time_left="1:00:00",
        partition="day",
        name=JOB_NAME,
        submitted_at="2026-07-10T11:00:00",
        comment=COMMENT,
    )


class MonitorTraceTests(unittest.TestCase):
    def monitor(self, script: StrictScript, clock: VirtualClock) -> JobMonitor:
        return JobMonitor(
            StrictProxy(script),
            sleeper=clock.sleep,
            confirmation_delay=3,
        )

    def test_preempted_then_pending_is_requeueing_not_final(self) -> None:
        # No accounting read happens here at all: a PREEMPTED candidate is
        # requeue-eligible, so consulting accounting in this same cycle would
        # only re-describe the instant already observed, from inside the window
        # in which Slurm requeues the job.  The confirmation observation decides.
        managed = job(ever_started=True)
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("PREEMPTED"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=row("PENDING"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()
        result = self.monitor(script, clock).assess(managed)
        self.assertEqual(result.assessment.phase, AssessmentPhase.REQUEUEING)
        self.assertFalse(result.assessment.final)
        self.assertEqual(result.assessment.terminal_evidence, 0)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_a_requeued_job_survives_even_when_accounting_reports_the_failure(
        self,
    ) -> None:
        """The case the old code actually got wrong.

        The previous trace only stayed alive because the fake's accounting read
        returned None.  In reality slurmdbd *does* report NODE_FAIL for the
        failed attempt, and one such record promoted the job straight to an
        immutable FINAL/ACCOUNTING verdict -- so a job Slurm then requeued was
        irreversibly reaped: `status` hid it, `cancel` refused it, and it held
        its GPUs for the full walltime.  No accounting read may be taken for a
        requeue-eligible candidate in the cycle that produced it.
        """

        managed = job(ever_started=True)
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("NODE_FAIL"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=row("RUNNING"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()
        result = self.monitor(script, clock).assess(managed)

        self.assertFalse(result.assessment.final)
        self.assertEqual(result.assessment.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(result.assessment.terminal_evidence, 0)
        self.assertFalse(result.accounting_checked)
        script.assert_complete()

    def test_two_absences_are_confirmed_but_accounting_is_still_consulted(self) -> None:
        managed = job(ever_started=True)
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0, 2, 2), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()
        result = self.monitor(script, clock).assess(managed)
        self.assertEqual(result.assessment.phase, AssessmentPhase.FINAL)
        self.assertEqual(result.assessment.final_source, "confirmed-queue")
        self.assertTrue(result.accounting_checked)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_recycled_id_needs_confirmation_and_exact_accounting_is_consulted(self) -> None:
        managed = job(ever_started=True)
        assert managed.ref is not None
        reused = JobIdReused("job grace:12345 now belongs to another operation")
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=reused,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=reused,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0, 2, 2), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()

        result = self.monitor(script, clock).assess(managed)

        self.assertEqual(result.assessment.phase, AssessmentPhase.FINAL)
        self.assertEqual(result.assessment.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(result.assessment.detail, str(reused))
        self.assertTrue(result.accounting_checked)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_exact_live_reappearance_clears_recycled_id_candidate(self) -> None:
        managed = job(ever_started=True)
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=JobIdReused("numeric ID was recycled"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=row("RUNNING", node="node02"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()

        result = self.monitor(script, clock).assess(managed)

        self.assertEqual(result.assessment.phase, AssessmentPhase.ACTIVE)
        self.assertEqual(result.assessment.current_node, "node02")
        self.assertEqual(result.assessment.terminal_evidence, 0)
        self.assertEqual(result.assessment.detail, "")
        self.assertTrue(result.accounting_checked)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_reconnect_failure_between_observations_never_returns_final(self) -> None:
        managed = job(ever_started=True)
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=TransportLost("VPN dropped"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()
        with self.assertRaisesRegex(TransportLost, "VPN dropped"):
            self.monitor(script, clock).assess(managed)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_final_accounting_short_circuits_confirmation_and_preserves_verdict(self) -> None:
        managed = job(ever_started=True)
        assert managed.ref is not None
        record = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=JOB_NAME,
            comment=COMMENT,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=row("COMPLETED"),
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=record,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()
        result = self.monitor(script, clock).assess(managed)
        self.assertTrue(result.assessment.final)
        self.assertEqual(result.assessment.final_source, "accounting")
        self.assertEqual(result.assessment.terminal_state, "COMPLETED")
        self.assertEqual(result.assessment.exit_code, "0:0")
        self.assertEqual(clock.sleeps, [])
        script.assert_complete()

    def test_restart_boundary_requires_two_fresh_nonlive_observations(self) -> None:
        managed = replace(
            job(ever_started=True),
            phase=JobPhase.TERMINAL_CANDIDATE,
            current_node=None,
            terminal_state="COMPLETING",
        )
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "observe",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"auth": AuthMode.NONINTERACTIVE},
                ),
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0, 2, 2), "auth": AuthMode.NONINTERACTIVE},
                ),
            ]
        )
        clock = VirtualClock()

        result = self.monitor(script, clock).assess(managed)

        self.assertEqual(result.assessment.phase, AssessmentPhase.FINAL)
        self.assertEqual(result.assessment.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertIsNone(result.assessment.terminal_state)
        self.assertEqual(
            result.assessment.evidence_provenance,
            EvidenceProvenance.ABSENT,
        )
        self.assertEqual(result.assessment.observation_epoch, 1)
        self.assertTrue(result.accounting_checked)
        self.assertEqual(clock.sleeps, [3])
        script.assert_complete()

    def test_tracker_preserves_persisted_inactive_phase_and_final_monitor_is_quiet(self) -> None:
        inactive = replace(
            job(ever_started=True),
            phase=JobPhase.REQUEUEING,
            current_node=None,
        )
        self.assertEqual(
            JobMonitor.tracker(inactive).assessment.phase,
            AssessmentPhase.REQUEUEING,
        )

        final = replace(
            inactive,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
            finalized_at="2026-07-10T12:00:00+00:00",
        )
        script = StrictScript([])
        result = self.monitor(script, VirtualClock()).assess(final)
        self.assertEqual(result.assessment.phase, AssessmentPhase.FINAL)
        self.assertEqual(result.assessment.terminal_state, "COMPLETED")
        self.assertEqual(result.assessment.final_source, FinalSource.ACCOUNTING)
        script.assert_complete()

    def test_persisted_queue_final_is_enriched_by_later_accounting(self) -> None:
        managed = replace(
            job(ever_started=True),
            phase=JobPhase.FINAL,
            current_node=None,
            terminal_state="COMPLETING",
            final_source=FinalSource.CONFIRMED_QUEUE,
            finalized_at="2026-07-10T12:00:00+00:00",
        )
        assert managed.ref is not None
        record = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=JOB_NAME,
            comment=COMMENT,
        )
        script = StrictScript(
            [
                ExpectedCall(
                    "final",
                    result=record,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                )
            ]
        )

        result = self.monitor(script, VirtualClock()).assess(managed)

        self.assertTrue(result.accounting_checked)
        self.assertEqual(result.assessment.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(result.assessment.terminal_state, "COMPLETED")
        self.assertEqual(result.assessment.exit_code, "0:0")
        script.assert_complete()

    def test_persisted_queue_final_is_retained_while_accounting_lags(self) -> None:
        managed = replace(
            job(ever_started=True),
            phase=JobPhase.FINAL,
            current_node=None,
            terminal_state="COMPLETING",
            final_source=FinalSource.CONFIRMED_QUEUE,
            finalized_at="2026-07-10T12:00:00+00:00",
        )
        assert managed.ref is not None
        script = StrictScript(
            [
                ExpectedCall(
                    "final",
                    result=None,
                    args=(managed.ref,),
                    kwargs={"attempts": (0,), "auth": AuthMode.NONINTERACTIVE},
                )
            ]
        )

        result = self.monitor(script, VirtualClock()).assess(managed)

        self.assertTrue(result.accounting_checked)
        self.assertEqual(result.assessment.final_source, FinalSource.CONFIRMED_QUEUE)
        self.assertEqual(result.assessment.terminal_state, "COMPLETING")
        script.assert_complete()


if __name__ == "__main__":
    unittest.main()
