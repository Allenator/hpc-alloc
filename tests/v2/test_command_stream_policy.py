from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import _follow_with_reconciliation, _persist_reconciled_assessment, cmd_run
from hpc_alloc.errors import LifecycleRevisionConflict, StateConflict, TransportLost
from hpc_alloc.lifecycle import EvidenceEvent, EvidenceTracker
from hpc_alloc.models import FinalSource, JobKind, JobPhase, JobRecord
from hpc_alloc.paths import AppPaths
from hpc_alloc.slurm import AccountingRecord, QueueRow
from hpc_alloc.streaming import FollowOutcome


class BrokenFollower:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self) -> object:
        raise BrokenPipeError


class InterruptedFollower:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self) -> object:
        raise KeyboardInterrupt


class CommandStreamPolicyTests(unittest.TestCase):
    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            command=["--", "true"],
            cluster=None,
            partition=None,
            time=None,
            cpus=None,
            mem=None,
            gpus=None,
            constraint=None,
            chdir=None,
            dry_run=False,
            detach=False,
        )

    def context(self) -> SimpleNamespace:
        config = SimpleNamespace(
            resolve_cluster=lambda _cluster: "grace",
            resolve_option=lambda key, _cluster, fallback=None: fallback,
        )
        return SimpleNamespace(config=config, state=object())

    def job(self) -> SimpleNamespace:
        return SimpleNamespace(
            ref=object(),
            cluster="grace",
            job_id="12345",
            operation_id="a" * 32,
            kind=JobKind.RUN,
            ever_started=False,
            current_node=None,
            last_node=None,
        )

    def invoke(self, follower: type, cancel_result: object = None):
        paths = AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test"))
        transport = SimpleNamespace(bootstrap=Mock())
        client = object()
        cancel = Mock(side_effect=cancel_result if isinstance(cancel_result, BaseException) else None)
        with (
            patch("hpc_alloc.commands._services", return_value=(transport, client)),
            patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
            patch("hpc_alloc.commands._submit_job", return_value=self.job()),
            patch("hpc_alloc.commands._cancel_record", cancel),
            patch("hpc_alloc.streaming.LogFollower", follower),
            patch("hpc_alloc.commands.neutralize_stdout"),
            patch("hpc_alloc.commands.info"),
        ):
            result = cmd_run(
                self.args(),
                ctx=self.context(),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
        return result, cancel

    def test_broken_pipe_cancels_foreground_run_and_returns_141(self) -> None:
        result, cancel = self.invoke(BrokenFollower)
        self.assertEqual(result, 141)
        cancel.assert_called_once()

    def test_interrupt_still_returns_to_cli_as_interrupt_when_cancel_is_uncertain(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(InterruptedFollower, TransportLost("reply lost"))

    def test_run_exit_code_comes_from_reconciled_final_authority(self) -> None:
        job = self.job()
        tracker = EvidenceTracker(ever_started=True)
        assessment = tracker.accept(
            EvidenceEvent.final(
                AccountingRecord(
                    job_id="12345",
                    state="FAILED",
                    exit_code="7:0",
                    job_name="hpcalloc-v2-run-" + "a" * 32,
                    comment="hpc-alloc:v2:deadbeef1234:" + "a" * 32 + ":laptop:run:-",
                )
            )
        )
        paths = AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test"))
        transport = SimpleNamespace(bootstrap=Mock())
        with (
            patch("hpc_alloc.commands._services", return_value=(transport, object())),
            patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch(
                "hpc_alloc.commands._follow_with_reconciliation",
                return_value=(job, assessment),
            ),
            patch("hpc_alloc.commands.info"),
        ):
            result = cmd_run(
                self.args(),
                ctx=self.context(),
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 7)


class LifecycleFollowerReconciliationTests(unittest.TestCase):
    operation_id = "a" * 32

    def job(self, *, phase: JobPhase, updated_at: str, **changes: object) -> JobRecord:
        values = {
            "operation_id": self.operation_id,
            "cluster": "grace",
            "logical_name": "run",
            "kind": JobKind.RUN,
            "owner_id": "deadbeef1234",
            "slurm_job_name": "hpcalloc-v2-run-" + self.operation_id,
            "slurm_comment": (
                "hpc-alloc:v2:deadbeef1234:"
                + self.operation_id
                + ":laptop:run:-"
            ),
            "phase": phase,
            "job_id": "12345",
            "updated_at": updated_at,
        }
        values.update(changes)
        return JobRecord(**values)

    def accounting_assessment(self, state: str, exit_code: str):
        tracker = EvidenceTracker(ever_started=True, last_node="node01")
        return tracker.accept(
            EvidenceEvent.final(
                AccountingRecord(
                    job_id="12345",
                    state=state,
                    exit_code=exit_code,
                    job_name="hpcalloc-v2-run-" + self.operation_id,
                    comment=(
                        "hpc-alloc:v2:deadbeef1234:"
                        + self.operation_id
                        + ":laptop:run:-"
                    ),
                )
            )
        )

    def test_stale_terminal_reloads_live_state_rebases_and_preserves_stream_progress(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        durable = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v2",
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )
        live = replace(
            durable,
            phase=JobPhase.REQUEUEING,
            updated_at="v3",
            current_node=None,
        )
        final = replace(
            live,
            phase=JobPhase.FINAL,
            updated_at="v4",
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        stale_final = self.accounting_assessment("FAILED", "9:0")
        authoritative_final = self.accounting_assessment("COMPLETED", "0:0")

        class Client:
            observations = 0

            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                self.observations += 1
                return QueueRow(
                    job_id="12345",
                    state="PENDING",
                    node=None,
                    reason="Resources",
                    time_left="1:00:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

        class Follower:
            def __init__(self) -> None:
                self.offset = 37
                self.total_bytes_written = 37
                self.outcomes = [stale_final, authoritative_final]
                self.rebases: list[object] = []
                self.drains = 0

            def follow(self) -> FollowOutcome:
                assessment = self.outcomes.pop(0)
                return FollowOutcome(assessment, assessment.terminal_state, 37, 37)

            def rebase(self, tracker: object) -> None:
                self.rebases.append(tracker)
                self.assert_progress()

            def drain(self) -> int:
                self.drains += 1
                self.assert_progress()
                return 0

            def assert_progress(self) -> None:
                if self.offset != 37 or self.total_bytes_written != 37:
                    raise AssertionError("stream progress was reset")

        client = Client()
        follower = Follower()
        state = SimpleNamespace(get_job=Mock(return_value=durable))
        persisted = 0

        def persist(_ctx: object, _paths: object, job: JobRecord, assessment: object):
            nonlocal persisted
            persisted += 1
            if persisted == 1:
                raise LifecycleRevisionConflict("stale")
            if persisted == 2:
                self.assertEqual(assessment.phase.value, JobPhase.REQUEUEING.value)
                self.assertEqual(assessment.last_node, "node02")
                return live
            self.assertEqual(job.updated_at, "v3")
            return final

        with patch("hpc_alloc.commands._persist_and_render", side_effect=persist):
            updated, assessment = _follow_with_reconciliation(
                ctx=SimpleNamespace(state=state),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=client,
                job=source,
                follower=follower,
            )

        self.assertIs(updated, final)
        self.assertEqual(assessment.terminal_state, "COMPLETED")
        self.assertEqual(client.observations, 1)
        self.assertEqual(len(follower.rebases), 1)
        self.assertEqual(follower.drains, 0)
        follower.assert_progress()

    def test_non_revision_state_conflict_is_not_retried(self) -> None:
        job = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        assessment = self.accounting_assessment("COMPLETED", "0:0")
        state = SimpleNamespace(get_job=Mock())
        conflict = StateConflict("semantic conflict")

        with (
            patch("hpc_alloc.commands._persist_and_render", side_effect=conflict),
            self.assertRaises(StateConflict) as raised,
        ):
            _persist_reconciled_assessment(
                ctx=SimpleNamespace(state=state),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=object(),
                job=job,
                assessment=assessment,
            )

        self.assertIs(raised.exception, conflict)
        state.get_job.assert_not_called()

    def test_repeated_revision_conflicts_take_a_new_exact_assessment_each_time(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        first_durable = self.job(phase=JobPhase.QUEUED, updated_at="v2")
        second_durable = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v3",
            ever_started=True,
            current_node="node03",
            last_node="node03",
        )
        stored = replace(
            second_durable,
            updated_at="v4",
            current_node="node04",
            last_node="node04",
        )
        stale = self.accounting_assessment("FAILED", "1:0")

        class Client:
            def __init__(self) -> None:
                self.nodes = iter(("node02", "node04"))
                self.observations = 0

            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                self.observations += 1
                node = next(self.nodes)
                return QueueRow(
                    job_id="12345",
                    state="RUNNING",
                    node=node,
                    reason="",
                    time_left="1:00:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

        state = SimpleNamespace(
            get_job=Mock(side_effect=(first_durable, second_durable))
        )
        phases: list[object] = []

        def persist(_ctx: object, _paths: object, _job: JobRecord, assessment: object):
            phases.append(assessment.phase)
            if len(phases) < 3:
                raise LifecycleRevisionConflict("stale")
            return stored

        client = Client()
        with patch("hpc_alloc.commands._persist_and_render", side_effect=persist):
            updated, assessment, tracker = _persist_reconciled_assessment(
                ctx=SimpleNamespace(state=state),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=client,
                job=source,
                assessment=stale,
            )

        self.assertIs(updated, stored)
        self.assertEqual(
            [phase.value for phase in phases],
            [JobPhase.FINAL.value, JobPhase.ACTIVE.value, JobPhase.ACTIVE.value],
        )
        self.assertEqual(client.observations, 2)
        self.assertEqual(assessment.current_node, "node04")
        self.assertIsNotNone(tracker)

    def test_fresh_final_after_revision_race_rebases_and_drains_again(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        durable = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v2",
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )
        stored = replace(
            durable,
            phase=JobPhase.FINAL,
            updated_at="v3",
            current_node=None,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        stale = self.accounting_assessment("FAILED", "1:0")
        accounting = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=source.slurm_job_name,
            comment=source.slurm_comment,
        )

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                return QueueRow(
                    job_id="12345",
                    state="COMPLETED",
                    node="node02",
                    reason="",
                    time_left="0:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

            def final(self, *_args: object, **_kwargs: object) -> AccountingRecord:
                return accounting

        class Follower:
            offset = 23
            total_bytes_written = 23

            def __init__(self) -> None:
                self.follow_calls = 0
                self.rebases = 0
                self.drains = 0

            def follow(self) -> FollowOutcome:
                self.follow_calls += 1
                return FollowOutcome(stale, stale.terminal_state, 23, 23)

            def rebase(self, _tracker: object) -> None:
                self.rebases += 1

            def drain(self) -> int:
                self.drains += 1
                self.assert_progress()
                return 5

            def assert_progress(self) -> None:
                if self.offset != 23 or self.total_bytes_written != 23:
                    raise AssertionError("stream progress was reset")

        writes = 0

        def persist(_ctx: object, _paths: object, _job: JobRecord, _assessment: object):
            nonlocal writes
            writes += 1
            if writes == 1:
                raise LifecycleRevisionConflict("stale")
            return stored

        follower = Follower()
        with patch("hpc_alloc.commands._persist_and_render", side_effect=persist):
            updated, assessment = _follow_with_reconciliation(
                ctx=SimpleNamespace(
                    state=SimpleNamespace(get_job=Mock(return_value=durable))
                ),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=Client(),
                job=source,
                follower=follower,
            )

        self.assertIs(updated, stored)
        self.assertEqual(assessment.terminal_state, "COMPLETED")
        self.assertEqual(follower.follow_calls, 1)
        self.assertEqual(follower.rebases, 1)
        self.assertEqual(follower.drains, 1)
        follower.assert_progress()


if __name__ == "__main__":
    unittest.main()
