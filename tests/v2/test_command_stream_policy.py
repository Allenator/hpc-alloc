from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import (
    _follow_with_reconciliation,
    _persist_reconciled_assessment,
    cmd_run,
    cmd_up,
)
from hpc_alloc.config import Config
from hpc_alloc.errors import (
    HpcAllocError,
    LifecycleRevisionConflict,
    SchedulerUnavailable,
    StateConflict,
    TransportLost,
)
from hpc_alloc.lifecycle import EvidenceEvent, EvidenceTracker
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    JobRecord,
)
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.retry import RetryBudget
from hpc_alloc.slurm import AccountingRecord, QueueRow
from hpc_alloc.ssh_config import sync_managed_config
from hpc_alloc.state import StateRepository
from hpc_alloc.streaming import FollowOutcome, LogFollower


def _no_patience() -> RetryBudget:
    """A retry budget that absorbs nothing.

    These tests assert that a failure still surfaces, and that the durable
    checkpoint and the SSH projection both precede it.  Riding out a transient
    failure is a separate behaviour, covered in test_streaming.py; zero patience
    keeps these tests about ordering rather than about waiting.
    """

    return RetryBudget(scheduler_patience=0, transport_patience=0)


class BrokenFollower:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self, *, publish_assessment: object | None = None) -> object:
        raise BrokenPipeError


class InterruptedFollower:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self, *, publish_assessment: object | None = None) -> object:
        raise KeyboardInterrupt


class FailedFollower:
    failure: BaseException

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self, *, publish_assessment: object | None = None) -> object:
        raise self.failure


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
            phase=JobPhase.QUEUED,
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

    def test_run_without_a_final_state_raises_a_typed_error_not_indexerror(self) -> None:
        """A FINAL assessment can legitimately carry no terminal state.

        Two absent queue observations finalize a job as confirmed-queue with
        terminal_state=None, and the accounting ladder returns nothing while
        slurmdbd lags.  `"".split()` is `[]`, so indexing [0] raised IndexError
        -- which cli.main does not catch -- instead of reaching the typed guard
        written for exactly this case.
        """

        job = self.job()
        assessment = SimpleNamespace(terminal_state=None, exit_code=None)
        paths = AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test"))

        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(bootstrap=Mock()), object()),
            ),
            patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch(
                "hpc_alloc.commands._follow_with_reconciliation",
                return_value=(job, assessment),
            ),
            patch("hpc_alloc.commands._pipe_aware_info"),
            patch("hpc_alloc.commands.info"),
        ):
            with self.assertRaises(HpcAllocError) as raised:
                cmd_run(
                    self.args(),
                    ctx=self.context(),
                    paths=paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

        self.assertIn("left the queue without a final state", str(raised.exception))

    def test_broken_pipe_cancels_foreground_run_and_returns_141(self) -> None:
        result, cancel = self.invoke(BrokenFollower)
        self.assertEqual(result, 141)
        cancel.assert_called_once()

    def test_broken_pipe_keeps_141_when_cancellation_is_uncertain(self) -> None:
        result, cancel = self.invoke(
            BrokenFollower, TransportLost("cancellation reply lost")
        )
        self.assertEqual(result, 141)
        cancel.assert_called_once()

    def test_interrupt_still_returns_to_cli_as_interrupt_when_cancel_is_uncertain(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(InterruptedFollower, TransportLost("reply lost"))

    def test_follower_errors_preserve_identity_and_report_live_job_context(self) -> None:
        for failure in (
            SchedulerUnavailable("squeue unavailable"),
            TransportLost("SSH transport dropped"),
        ):
            with self.subTest(failure=type(failure).__name__):
                FailedFollower.failure = failure
                paths = AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test"))
                transport = SimpleNamespace(bootstrap=Mock())
                cancel = Mock()
                stderr = io.StringIO()
                with (
                    patch(
                        "hpc_alloc.commands._services",
                        return_value=(transport, object()),
                    ),
                    patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
                    patch("hpc_alloc.commands._submit_job", return_value=self.job()),
                    patch("hpc_alloc.commands._cancel_record", cancel),
                    patch("hpc_alloc.streaming.LogFollower", FailedFollower),
                    redirect_stderr(stderr),
                    self.assertRaises(type(failure)) as raised,
                ):
                    cmd_run(
                        self.args(),
                        ctx=self.context(),
                        paths=paths,
                        entrypoint=Path("/tmp/hpc-alloc"),
                    )

                self.assertIs(raised.exception, failure)
                cancel.assert_not_called()
                output = stderr.getvalue()
                selector = "grace:@" + "a" * 32
                self.assertIn("was not cancelled and may continue", output)
                self.assertIn(f"hpc-alloc logs {selector} -f", output)
                self.assertIn(f"hpc-alloc cancel {selector}", output)

    def test_follower_error_after_durable_final_avoids_live_guidance(self) -> None:
        failure = SchedulerUnavailable("log drain failed")
        FailedFollower.failure = failure
        job = self.job()
        final = SimpleNamespace(**{**vars(job), "phase": JobPhase.FINAL})
        context = self.context()
        context.state = SimpleNamespace(get_job=Mock(return_value=final))
        paths = AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test"))
        stderr = io.StringIO()
        cancel = Mock()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(bootstrap=Mock()), object()),
            ),
            patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch("hpc_alloc.commands._cancel_record", cancel),
            patch("hpc_alloc.streaming.LogFollower", FailedFollower),
            redirect_stderr(stderr),
            self.assertRaises(SchedulerUnavailable) as raised,
        ):
            cmd_run(
                self.args(),
                ctx=context,
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, failure)
        cancel.assert_not_called()
        output = stderr.getvalue()
        self.assertIn("reached a durable final state", output)
        self.assertNotIn("may continue", output)
        self.assertNotIn("hpc-alloc cancel", output)

    def test_follower_context_failure_cannot_replace_primary_error(self) -> None:
        failure = SchedulerUnavailable("squeue unavailable")
        FailedFollower.failure = failure
        job = self.job()

        def report(message: str) -> None:
            if "foreground follow stopped" in message:
                raise RuntimeError("secondary diagnostic failed")

        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(bootstrap=Mock()), object()),
            ),
            patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch("hpc_alloc.streaming.LogFollower", FailedFollower),
            patch("hpc_alloc.commands.info", side_effect=report),
            self.assertRaises(SchedulerUnavailable) as raised,
        ):
            cmd_run(
                self.args(),
                ctx=self.context(),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test")),
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, failure)

    def test_up_interrupt_reports_uncancelled_allocation_and_preserves_identity(
        self,
    ) -> None:
        interrupt = KeyboardInterrupt()
        job = SimpleNamespace(
            cluster="grace",
            job_id="12345",
            operation_id="b" * 32,
            phase=JobPhase.QUEUED,
        )
        context = self.context()
        context.state = SimpleNamespace(get_job=Mock(return_value=job))
        resources = {
            "partition": "day",
            "time": "1:00:00",
            "cpus": 2,
            "mem": None,
            "gpus": None,
            "constraint": None,
            "chdir": None,
            "idle_timeout": None,
        }
        cancel = Mock()
        stderr = io.StringIO()
        monitor = SimpleNamespace(assess=Mock(side_effect=interrupt))
        with (
            patch("hpc_alloc.commands._resource_values", return_value=resources),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(), object()),
            ),
            patch("hpc_alloc.monitor.JobMonitor", return_value=monitor),
            patch("hpc_alloc.commands._cancel_record", cancel),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            cmd_up(
                SimpleNamespace(
                    cluster=None,
                    name="dev",
                    wait_timeout=30,
                    dry_run=False,
                    no_wait=False,
                ),
                ctx=context,
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test")),
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, interrupt)
        cancel.assert_not_called()
        selector = "grace:@" + "b" * 32
        output = stderr.getvalue()
        self.assertIn("was not cancelled and may remain queued or running", output)
        self.assertIn("hpc-alloc status", output)
        self.assertIn(f"hpc-alloc down {selector}", output)

    def test_up_interrupt_after_durable_final_avoids_down_guidance(self) -> None:
        interrupt = KeyboardInterrupt()
        job = SimpleNamespace(
            cluster="grace",
            job_id="12345",
            operation_id="b" * 32,
            phase=JobPhase.QUEUED,
        )
        final = SimpleNamespace(**{**vars(job), "phase": JobPhase.FINAL})
        context = self.context()
        context.state = SimpleNamespace(get_job=Mock(return_value=final))
        resources = {
            "partition": "day",
            "time": "1:00:00",
            "cpus": 2,
            "mem": None,
            "gpus": None,
            "constraint": None,
            "chdir": None,
            "idle_timeout": None,
        }
        stderr = io.StringIO()
        with (
            patch("hpc_alloc.commands._resource_values", return_value=resources),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(), object()),
            ),
            patch(
                "hpc_alloc.monitor.JobMonitor",
                return_value=SimpleNamespace(assess=Mock(side_effect=interrupt)),
            ),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            cmd_up(
                SimpleNamespace(
                    cluster=None,
                    name="dev",
                    wait_timeout=30,
                    dry_run=False,
                    no_wait=False,
                ),
                ctx=context,
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test")),
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, interrupt)
        output = stderr.getvalue()
        self.assertIn("reached a durable final state", output)
        self.assertIn("hpc-alloc why grace:@" + "b" * 32, output)
        self.assertNotIn("hpc-alloc down", output)

    def test_up_sleep_interrupt_follows_successful_pending_checkpoint(self) -> None:
        interrupt = KeyboardInterrupt()
        job = SimpleNamespace(
            cluster="grace",
            job_id="12345",
            operation_id="b" * 32,
            phase=JobPhase.QUEUED,
        )
        context = self.context()
        context.state = SimpleNamespace(get_job=Mock(return_value=job))
        resources = {
            "partition": "day",
            "time": "1:00:00",
            "cpus": 2,
            "mem": None,
            "gpus": None,
            "constraint": None,
            "chdir": None,
            "idle_timeout": None,
        }
        assessment = SimpleNamespace(
            phase=JobPhase.QUEUED,
            current_node=None,
            final=False,
            scheduler_state="PENDING",
        )
        stderr = io.StringIO()
        with (
            patch("hpc_alloc.commands._resource_values", return_value=resources),
            patch("hpc_alloc.commands._submit_job", return_value=job),
            patch(
                "hpc_alloc.commands._services",
                return_value=(SimpleNamespace(), object()),
            ),
            patch(
                "hpc_alloc.monitor.JobMonitor",
                return_value=SimpleNamespace(
                    assess=Mock(return_value=SimpleNamespace(assessment=assessment))
                ),
            ),
            patch(
                "hpc_alloc.commands._persist_reconciled_assessment",
                return_value=(job, assessment, None),
            ) as persist,
            patch("hpc_alloc.commands.time.sleep", side_effect=interrupt),
            redirect_stderr(stderr),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            cmd_up(
                SimpleNamespace(
                    cluster=None,
                    name="dev",
                    wait_timeout=30,
                    dry_run=False,
                    no_wait=False,
                ),
                ctx=context,
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-policy-test")),
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, interrupt)
        persist.assert_called_once()
        self.assertIn("was not cancelled and may remain queued or running", stderr.getvalue())

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

    def test_running_checkpoint_and_projection_precede_log_size_failure(self) -> None:
        source = replace(
            self.job(phase=JobPhase.QUEUED, updated_at="v1"),
            logical_name="dev",
            kind=JobKind.ALLOCATION,
        )
        active = replace(
            source,
            phase=JobPhase.ACTIVE,
            updated_at="v2",
            ever_started=True,
            current_node="node01",
            last_node="node01",
            observation_epoch=1,
        )
        failure = SchedulerUnavailable("log size failed after RUNNING")
        events: list[str] = []

        class Client:
            def observe(inner_self, ref: object) -> QueueRow:
                events.append("observe")
                if ref != source.ref:
                    raise AssertionError("unexpected job identity")
                return QueueRow(
                    job_id="12345",
                    state="RUNNING",
                    node="node01",
                    reason="",
                    time_left="1:00:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

            def log_size(inner_self, path: str):
                events.append("log-size")
                if path != ".hpc-alloc/allocation.log":
                    raise AssertionError("unexpected log path")
                raise failure

        def update_job(operation_id: str, **changes: object) -> JobRecord:
            events.append("persist")
            self.assertEqual(operation_id, self.operation_id)
            self.assertEqual(changes["expected_updated_at"], "v1")
            self.assertEqual(changes["phase"], JobPhase.ACTIVE)
            self.assertIs(changes["ever_started"], True)
            self.assertEqual(changes["current_node"], "node01")
            self.assertEqual(changes["last_node"], "node01")
            return active

        state = SimpleNamespace(update_job=Mock(side_effect=update_job))
        follower = LogFollower(
            Client(),  # type: ignore[arg-type]
            source.ref,  # type: ignore[arg-type]
            ".hpc-alloc/allocation.log",
            tracker=JobMonitor.tracker(source),
            output=io.BytesIO(),
            retry_budget=_no_patience(),
        )

        def project(_ctx: object, _paths: object) -> bool:
            events.append("project")
            return True

        with (
            patch("hpc_alloc.commands._sync_ssh_projection", side_effect=project) as sync,
            self.assertRaises(SchedulerUnavailable) as raised,
        ):
            _follow_with_reconciliation(
                ctx=SimpleNamespace(state=state),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=follower.client,
                job=source,
                follower=follower,
            )

        self.assertIs(raised.exception, failure)
        self.assertEqual(events, ["observe", "persist", "project", "log-size"])
        sync.assert_called_once()
        self.assertEqual(follower.tracker.assessment.phase.value, "UNCERTAIN")

    def test_running_checkpoint_repairs_real_compute_alias_before_log_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.for_home(Path(directory))
            repository = StateRepository(
                paths.state_db,
                machine_id_factory=lambda: "deadbeef1234",
            ).initialize()
            owner = repository.get_or_create_machine_id("laptop")
            job_name = slurm_job_name("allocation", self.operation_id)
            comment = format_tag(
                owner,
                self.operation_id,
                "laptop",
                "allocation",
                "dev",
            )
            repository.reserve_submission(
                operation_id=self.operation_id,
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                owner_id=owner,
                slurm_job_name=job_name,
                slurm_comment=comment,
                resources={"partition": "day", "time": "1:00:00", "cpus": 2},
            )
            source = repository.acknowledge_submission(self.operation_id, "12345")
            paths.config_dir.mkdir(parents=True, exist_ok=True)
            paths.config_file.write_text(
                """\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
[cluster.grace]
host = "grace.example.edu"
"""
            )
            context = SimpleNamespace(
                state=repository,
                config=Config.load(paths.config_file),
            )
            failure = SchedulerUnavailable("identical log size failure")

            class Client:
                def observe(inner_self, ref: object) -> QueueRow:
                    if ref != source.ref:
                        raise AssertionError("unexpected job identity")
                    return QueueRow(
                        job_id="12345",
                        state="RUNNING",
                        node="node01",
                        reason="",
                        time_left="1:00:00",
                        partition="day",
                        name=job_name,
                        submitted_at="2026-07-10T12:00:00",
                        comment=comment,
                    )

                def log_size(inner_self, _path: str):
                    raise failure

            client = Client()
            follower = LogFollower(
                client,  # type: ignore[arg-type]
                source.ref,  # type: ignore[arg-type]
                ".hpc-alloc/allocation.log",
                tracker=JobMonitor.tracker(source),
                output=io.BytesIO(),
                retry_budget=_no_patience(),
            )

            with self.assertRaises(SchedulerUnavailable) as raised:
                _follow_with_reconciliation(
                    ctx=context,
                    paths=paths,
                    client=client,
                    job=source,
                    follower=follower,
                )

            self.assertIs(raised.exception, failure)
            durable = repository.get_job(self.operation_id)
            self.assertEqual(durable.phase, JobPhase.ACTIVE)
            self.assertTrue(durable.ever_started)
            self.assertEqual(durable.current_node, "node01")
            self.assertEqual(durable.last_node, "node01")
            projection = paths.managed_ssh_config.read_text()
            self.assertIn("Host hpc-grace.dev", projection)
            self.assertIn("    HostName node01", projection)

    def test_terminal_checkpoint_leases_real_alias_before_log_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.for_home(Path(directory))
            repository = StateRepository(
                paths.state_db,
                machine_id_factory=lambda: "deadbeef1234",
            ).initialize()
            owner = repository.get_or_create_machine_id("laptop")
            job_name = slurm_job_name("allocation", self.operation_id)
            comment = format_tag(
                owner,
                self.operation_id,
                "laptop",
                "allocation",
                "dev",
            )
            repository.reserve_submission(
                operation_id=self.operation_id,
                cluster="grace",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                owner_id=owner,
                slurm_job_name=job_name,
                slurm_comment=comment,
                resources={"partition": "day", "time": "1:00:00", "cpus": 2},
            )
            repository.acknowledge_submission(self.operation_id, "12345")
            source = repository.update_job(
                self.operation_id,
                phase=JobPhase.ACTIVE,
                ever_started=True,
                current_node="node01",
                last_node="node01",
            )
            paths.config_dir.mkdir(parents=True, exist_ok=True)
            paths.config_file.write_text(
                """\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
[cluster.grace]
host = "grace.example.edu"
"""
            )
            sync_managed_config(
                config_path=paths.config_file,
                repository=repository,
                managed_path=paths.managed_ssh_config,
                lock_path=paths.ssh_config_lock,
                known_hosts=paths.known_hosts,
            )
            prior_projection = paths.managed_ssh_config.read_bytes()
            failure = SchedulerUnavailable("identical post-candidate log failure")

            class Client:
                def observe(inner_self, ref: object) -> QueueRow:
                    if ref != source.ref:
                        raise AssertionError("unexpected job identity")
                    return QueueRow(
                        job_id="12345",
                        state="COMPLETED",
                        node="node01",
                        reason="",
                        time_left="0:00",
                        partition="day",
                        name=job_name,
                        submitted_at="2026-07-10T12:00:00",
                        comment=comment,
                    )

                def final(
                    inner_self,
                    ref: object,
                    *,
                    attempts: tuple[float, ...],
                ) -> None:
                    if ref != source.ref or attempts != (0,):
                        raise AssertionError("unexpected accounting lookup")
                    return None

                def log_size(inner_self, _path: str):
                    raise failure

            client = Client()
            follower = LogFollower(
                client,  # type: ignore[arg-type]
                source.ref,  # type: ignore[arg-type]
                ".hpc-alloc/allocation.log",
                tracker=JobMonitor.tracker(source),
                output=io.BytesIO(),
                retry_budget=_no_patience(),
            )

            with (
                patch("hpc_alloc.ssh.retire_compute_masters") as retire,
                self.assertRaises(SchedulerUnavailable) as raised,
            ):
                _follow_with_reconciliation(
                    ctx=SimpleNamespace(
                        state=repository,
                        config=Config.load(paths.config_file),
                    ),
                    paths=paths,
                    client=client,
                    job=source,
                    follower=follower,
                )

            self.assertIs(raised.exception, failure)
            durable = repository.get_job(self.operation_id)
            self.assertEqual(durable.phase, JobPhase.TERMINAL_CANDIDATE)
            self.assertTrue(durable.ever_started)
            self.assertIsNone(durable.current_node)
            self.assertEqual(durable.last_node, "node01")
            self.assertEqual(
                durable.evidence_provenance,
                EvidenceProvenance.QUEUE_TERMINAL,
            )
            self.assertEqual(
                paths.managed_ssh_config.read_bytes(),
                prior_projection,
            )
            retire.assert_not_called()

    def test_revision_retry_checkpoints_terminal_row_before_accounting_error(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        durable = replace(source, updated_at="v2")
        candidate = replace(
            durable,
            phase=JobPhase.TERMINAL_CANDIDATE,
            updated_at="v3",
            ever_started=True,
            last_node="node02",
            terminal_state="COMPLETED",
            evidence_provenance=EvidenceProvenance.QUEUE_TERMINAL,
        )
        failure = SchedulerUnavailable("accounting failed after fresh queue row")
        events: list[str] = []

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                events.append("observe")
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

            def final(self, *_args: object, **_kwargs: object) -> None:
                events.append("accounting")
                raise failure

        writes = 0

        def persist(
            _ctx: object,
            _paths: object,
            _job: JobRecord,
            assessment: object,
            *,
            force_projection: bool,
            skip_unchanged_projection: bool,
        ) -> JobRecord:
            nonlocal writes
            writes += 1
            events.append("persist")
            self.assertEqual(
                assessment.phase,
                JobMonitor.tracker(candidate).assessment.phase,
            )
            self.assertTrue(skip_unchanged_projection)
            self.assertEqual(force_projection, writes > 1)
            if writes == 1:
                raise LifecycleRevisionConflict("stale")
            return candidate

        client = Client()
        follower = LogFollower(
            client,  # type: ignore[arg-type]
            source.ref,  # type: ignore[arg-type]
            ".hpc-alloc/allocation.log",
            tracker=JobMonitor.tracker(source),
            output=io.BytesIO(),
            retry_budget=_no_patience(),
        )

        with (
            patch("hpc_alloc.commands._persist_and_render", side_effect=persist),
            self.assertRaises(SchedulerUnavailable) as raised,
        ):
            _follow_with_reconciliation(
                ctx=SimpleNamespace(
                    state=SimpleNamespace(get_job=Mock(return_value=durable))
                ),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=client,
                job=source,
                follower=follower,
            )

        self.assertIs(raised.exception, failure)
        self.assertEqual(
            events,
            ["observe", "persist", "observe", "persist", "accounting"],
        )
        self.assertEqual(writes, 2)

    def test_repeated_semantic_noop_checkpoints_do_not_resync_projection(self) -> None:
        source = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v1",
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )
        assessment = JobMonitor.tracker(source).assessment
        update = Mock(return_value=source)
        ctx = SimpleNamespace(state=SimpleNamespace(update_job=update))
        paths = AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile"))

        with patch("hpc_alloc.commands._sync_ssh_projection") as project:
            first, _assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=paths,
                client=object(),
                job=source,
                assessment=assessment,
                skip_unchanged_projection=True,
            )
            second, _assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=paths,
                client=object(),
                job=first,
                assessment=assessment,
                skip_unchanged_projection=True,
            )

        self.assertEqual(first, source)
        self.assertEqual(second, source)
        self.assertEqual(update.call_count, 2)
        project.assert_not_called()

    def test_uncertainty_is_not_persisted_but_default_projection_is_repaired(self) -> None:
        source = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v1",
            ever_started=True,
            current_node="node01",
            last_node="node01",
        )
        tracker = JobMonitor.tracker(source)
        uncertain = tracker.accept(EvidenceEvent.scheduler_error("slurm unavailable"))
        update = Mock()

        with patch("hpc_alloc.commands._sync_ssh_projection") as project:
            updated, assessment, replacement = _persist_reconciled_assessment(
                ctx=SimpleNamespace(state=SimpleNamespace(update_job=update)),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=object(),
                job=source,
                assessment=uncertain,
            )

        self.assertIs(updated, source)
        self.assertIs(assessment, uncertain)
        self.assertIsNone(replacement)
        update.assert_not_called()
        project.assert_called_once()

    def test_revision_retry_repairs_projection_even_when_state_is_unchanged(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        durable = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v2",
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                return QueueRow(
                    job_id="12345",
                    state="RUNNING",
                    node="node02",
                    reason="",
                    time_left="1:00:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

        update = Mock(
            side_effect=(LifecycleRevisionConflict("stale"), durable)
        )
        state = SimpleNamespace(
            update_job=update,
            get_job=Mock(return_value=durable),
        )

        with patch("hpc_alloc.commands._sync_ssh_projection") as project:
            updated, assessment, tracker = _persist_reconciled_assessment(
                ctx=SimpleNamespace(state=state),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=Client(),
                job=source,
                assessment=JobMonitor.tracker(source).assessment,
                skip_unchanged_projection=True,
            )

        self.assertIs(updated, durable)
        self.assertEqual(assessment.current_node, "node02")
        self.assertIsNotNone(tracker)
        self.assertEqual(update.call_count, 2)
        project.assert_called_once()

    def test_revision_retry_repairs_projection_before_returning_uncertainty(self) -> None:
        source = self.job(phase=JobPhase.QUEUED, updated_at="v1")
        durable = self.job(
            phase=JobPhase.ACTIVE,
            updated_at="v2",
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                return QueueRow(
                    job_id="12345",
                    state="NEW_UNKNOWN_STATE",
                    node=None,
                    reason="",
                    time_left="1:00:00",
                    partition="day",
                    name=source.slurm_job_name,
                    submitted_at="2026-07-10T12:00:00",
                    comment=source.slurm_comment,
                )

        with (
            patch(
                "hpc_alloc.commands._persist_and_render",
                side_effect=LifecycleRevisionConflict("stale"),
            ) as persist,
            patch("hpc_alloc.commands._sync_ssh_projection") as project,
        ):
            updated, assessment, tracker = _persist_reconciled_assessment(
                ctx=SimpleNamespace(
                    state=SimpleNamespace(get_job=Mock(return_value=durable))
                ),
                paths=AppPaths.for_home(Path("/tmp/hpc-alloc-reconcile")),
                client=Client(),
                job=source,
                assessment=self.accounting_assessment("FAILED", "1:0"),
                skip_unchanged_projection=True,
                checkpoint_reconciliation_observations=True,
            )

        self.assertEqual(updated, durable)
        self.assertTrue(assessment.uncertain)
        self.assertIsNotNone(tracker)
        persist.assert_called_once()
        project.assert_called_once()

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
                self.rebases: list[object] = []
                self.drains = 0

            def follow(self, *, publish_assessment) -> FollowOutcome:
                assessment, replacement = publish_assessment(stale_final)
                if replacement is not None:
                    self.rebase(replacement)
                if assessment.final:
                    raise AssertionError("stale final authority was not replaced")
                if assessment.phase.value != JobPhase.REQUEUEING.value:
                    raise AssertionError("expected reconciled requeue authority")
                if self.drains:
                    raise AssertionError("stale final was drained before reconciliation")
                self.assert_progress()

                assessment, replacement = publish_assessment(authoritative_final)
                if replacement is not None:
                    self.rebase(replacement)
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
        forced_projections: list[bool] = []

        def persist(
            _ctx: object,
            _paths: object,
            job: JobRecord,
            assessment: object,
            *,
            force_projection: bool,
            skip_unchanged_projection: bool,
        ):
            nonlocal persisted
            persisted += 1
            forced_projections.append(force_projection)
            self.assertTrue(skip_unchanged_projection)
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
        self.assertEqual(forced_projections, [False, True, False])
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
        forced_projections: list[bool] = []

        def persist(
            _ctx: object,
            _paths: object,
            _job: JobRecord,
            assessment: object,
            *,
            force_projection: bool,
            skip_unchanged_projection: bool,
        ):
            phases.append(assessment.phase)
            forced_projections.append(force_projection)
            self.assertFalse(skip_unchanged_projection)
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
        self.assertEqual(forced_projections, [False, True, True])
        self.assertEqual(client.observations, 2)
        self.assertEqual(assessment.current_node, "node04")
        self.assertIsNotNone(tracker)

    def test_fresh_final_after_revision_race_rebases_before_drain(self) -> None:
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

            def follow(self, *, publish_assessment) -> FollowOutcome:
                self.follow_calls += 1
                assessment, replacement = publish_assessment(stale)
                if replacement is not None:
                    self.rebase(replacement)
                if assessment.final:
                    self.drain()
                return FollowOutcome(
                    assessment,
                    assessment.terminal_state,
                    self.offset,
                    self.total_bytes_written,
                )

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
        forced_projections: list[bool] = []

        def persist(
            _ctx: object,
            _paths: object,
            _job: JobRecord,
            _assessment: object,
            *,
            force_projection: bool,
            skip_unchanged_projection: bool,
        ):
            nonlocal writes
            writes += 1
            forced_projections.append(force_projection)
            self.assertTrue(skip_unchanged_projection)
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
        self.assertEqual(forced_projections, [False, True])
        follower.assert_progress()


if __name__ == "__main__":
    unittest.main()
