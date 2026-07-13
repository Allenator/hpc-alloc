from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import cmd_logs, cmd_recover, cmd_status, cmd_sync, cmd_up
from hpc_alloc.errors import HpcAllocError, SchedulerUnavailable
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    OperationPhase,
)
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.slurm import AccountingRecord, QueueRow, RawQueueRow, RawQueueScan
from hpc_alloc.state import StateRepository


OPERATION_ID = "a" * 32
OWNER_ID = "deadbeef1234"


class ConfigStub:
    clusters = {"grace": SimpleNamespace()}

    @staticmethod
    def resolve_cluster(_cluster: str | None) -> str:
        return "grace"


class TransportStub:
    def __init__(self) -> None:
        self.bootstrap = Mock()
        self.require_node = Mock()


class SchedulerStub:
    def __init__(
        self,
        *,
        observations: tuple[QueueRow | BaseException, ...] = (),
        scan_rows: tuple[RawQueueRow, ...] = (),
        accounting: AccountingRecord | BaseException | None = None,
        recovered: AccountingRecord | None = None,
    ) -> None:
        self.observations = list(observations)
        self.scan_rows = scan_rows
        self.accounting = accounting
        self.recovered = recovered
        self.observe_calls = 0
        self.final_calls = 0
        self.tail_calls = 0
        self.verify_calls = 0

    def scan(self, *, auth: object) -> RawQueueScan:
        return RawQueueScan(self.scan_rows)

    def observe(self, _ref: object, *, auth: object) -> QueueRow:
        self.observe_calls += 1
        if not self.observations:
            raise AssertionError("unexpected exact scheduler observation")
        observation = self.observations.pop(0)
        if isinstance(observation, BaseException):
            raise observation
        return observation

    def final(
        self, _ref: object, *, attempts: tuple[float, ...], auth: object
    ) -> AccountingRecord | None:
        self.final_calls += 1
        if isinstance(self.accounting, BaseException):
            raise self.accounting
        return self.accounting

    def tail_log(self, _path: str, _lines: int) -> bytes:
        self.tail_calls += 1
        return b"unexpected stale log\n"

    def find_accounting_by_name(
        self, _job_name: str, *, auth: object
    ) -> AccountingRecord | None:
        return self.recovered

    def verify_accounting_identity(
        self, _ref: object, _job_name: str, _comment: str
    ) -> None:
        self.verify_calls += 1


class CommandLifecycleCasTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.paths = AppPaths.for_home(Path(directory.name))
        self.state = StateRepository(
            self.paths.state_db,
            machine_id_factory=lambda: OWNER_ID,
        ).initialize()
        self.owner = self.state.get_or_create_machine_id("laptop")
        self.context = SimpleNamespace(config=ConfigStub(), state=self.state)
        self.entrypoint = Path("/tmp/hpc-alloc")

    def reserve(
        self,
        *,
        acknowledged: bool = True,
        phase: JobPhase | None = None,
        ever_started: bool = False,
        current_node: str | None = None,
    ):
        job_name = slurm_job_name("allocation", OPERATION_ID)
        comment = format_tag(
            self.owner,
            OPERATION_ID,
            "laptop",
            "allocation",
            "dev",
        )
        self.state.reserve_submission(
            operation_id=OPERATION_ID,
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            owner_id=self.owner,
            slurm_job_name=job_name,
            slurm_comment=comment,
            resources={"partition": "day", "time": "1:00:00", "cpus": 2},
        )
        job = self.state.get_job(OPERATION_ID)
        if acknowledged:
            job = self.state.acknowledge_submission(OPERATION_ID, "12345")
        if phase is not None:
            job = self.state.update_job(
                OPERATION_ID,
                phase=phase,
                ever_started=True if ever_started else None,
                current_node=current_node,
                last_node=current_node,
            )
        return job

    def row(
        self,
        state: str,
        *,
        node: str | None = None,
        reason: str = "None",
    ) -> QueueRow:
        return QueueRow(
            job_id="12345",
            state=state,
            node=node,
            reason=reason,
            time_left="1:00:00",
            partition="day",
            name=slurm_job_name("allocation", OPERATION_ID),
            submitted_at="2026-07-12T11:00:00",
            comment=format_tag(
                self.owner,
                OPERATION_ID,
                "laptop",
                "allocation",
                "dev",
            ),
        )

    @staticmethod
    def raw(row: QueueRow) -> RawQueueRow:
        return RawQueueRow(
            job_id=row.job_id,
            state=row.state,
            node=row.node or "",
            reason=row.reason,
            time_left=row.time_left,
            partition=row.partition,
            name=row.name,
            submitted_at=row.submitted_at,
            comment=row.comment,
        )

    def accounting(self, state: str = "COMPLETED", exit_code: str = "0:0"):
        return AccountingRecord(
            job_id="12345",
            state=state,
            exit_code=exit_code,
            job_name=slurm_job_name("allocation", OPERATION_ID),
            comment=format_tag(
                self.owner,
                OPERATION_ID,
                "laptop",
                "allocation",
                "dev",
            ),
        )

    @contextmanager
    def revision_race(self, mutation):
        original_update = self.state.update_job
        injected = False

        def racing_update(operation_id: str, **changes: object):
            nonlocal injected
            if not injected and changes.get("expected_updated_at") is not None:
                injected = True
                mutation(original_update)
            return original_update(operation_id, **changes)

        with patch.object(self.state, "update_job", side_effect=racing_update):
            yield
        self.assertTrue(injected, "test did not reach the lifecycle CAS boundary")

    def finalize_race(
        self,
        update,
        *,
        state: str = "COMPLETED",
        exit_code: str = "0:0",
    ) -> None:
        update(
            OPERATION_ID,
            phase=JobPhase.FINAL,
            terminal_state=state,
            exit_code=exit_code,
            final_source=FinalSource.ACCOUNTING,
        )

    def activate_race(self, update) -> None:
        update(
            OPERATION_ID,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node02",
            last_node="node02",
        )

    def test_status_reassesses_after_revision_race_and_renders_fresh_payload(self) -> None:
        self.reserve()
        pending = self.row("PENDING", reason="Resources")
        fresh = self.row("RUNNING", node="node03")
        client = SchedulerStub(
            observations=(pending, fresh),
            scan_rows=(self.raw(pending),),
        )
        transport = TransportStub()
        stdout = io.StringIO()

        with (
            self.revision_race(self.activate_race),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch(
                "hpc_alloc.commands._sync_ssh_projection", return_value=True
            ) as project,
            redirect_stdout(stdout),
        ):
            result = cmd_status(
                SimpleNamespace(json=True),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["jobs"][0]["phase"], JobPhase.ACTIVE.value)
        self.assertEqual(payload["jobs"][0]["current_node"], "node03")
        self.assertEqual(self.state.get_job(OPERATION_ID).current_node, "node03")
        self.assertEqual(client.observe_calls, 2)
        project.assert_called_once_with(self.context, self.paths)

    def test_up_reports_fresh_ready_state_without_resubmitting(self) -> None:
        source = self.reserve()
        pending = self.row("PENDING", reason="Resources")
        fresh = self.row("RUNNING", node="node03")
        client = SchedulerStub(observations=(pending, fresh))
        transport = TransportStub()
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

        with (
            self.revision_race(self.activate_race),
            patch("hpc_alloc.commands._resource_values", return_value=resources),
            patch("hpc_alloc.commands._submit_job", return_value=source) as submit,
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch("hpc_alloc.commands.info") as report,
        ):
            result = cmd_up(
                SimpleNamespace(
                    cluster=None,
                    name="dev",
                    wait_timeout=30,
                    dry_run=False,
                    no_wait=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        submit.assert_called_once()
        self.assertEqual(client.observe_calls, 2)
        self.assertTrue(
            any("ready on node03" in str(call.args[0]) for call in report.call_args_list)
        )

    def test_up_reports_fresh_final_state_without_resubmitting(self) -> None:
        source = self.reserve()
        client = SchedulerStub(observations=(self.row("RUNNING", node="node01"),))
        transport = TransportStub()
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

        with (
            self.revision_race(self.finalize_race),
            patch("hpc_alloc.commands._resource_values", return_value=resources),
            patch("hpc_alloc.commands._submit_job", return_value=source) as submit,
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch("hpc_alloc.commands.info"),
            self.assertRaisesRegex(
                HpcAllocError, "ended before becoming active: COMPLETED"
            ),
        ):
            cmd_up(
                SimpleNamespace(
                    cluster=None,
                    name="dev",
                    wait_timeout=30,
                    dry_run=False,
                    no_wait=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        submit.assert_called_once()
        self.assertEqual(client.observe_calls, 1)

    def test_nonfollow_logs_tails_after_fresh_unknown_start_final(self) -> None:
        self.reserve()
        client = SchedulerStub(
            observations=(self.row("PENDING", reason="Resources"),)
        )
        transport = TransportStub()
        output = io.BytesIO()

        def boot_fail(update) -> None:
            self.finalize_race(update, state="BOOT_FAIL", exit_code="1:0")

        with (
            self.revision_race(boot_fail),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch(
                "hpc_alloc.commands.sys.stdout",
                SimpleNamespace(buffer=output),
            ),
        ):
            result = cmd_logs(
                SimpleNamespace(
                    target=f"@{OPERATION_ID}",
                    cluster=None,
                    lines=20,
                    follow=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        self.assertEqual(client.tail_calls, 1)
        self.assertEqual(output.getvalue(), b"unexpected stale log\n")

    def test_nonfollow_logs_still_rejects_pending_job_without_start_proof(self) -> None:
        self.reserve()
        client = SchedulerStub(
            observations=(self.row("PENDING", reason="Resources"),)
        )

        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(TransportStub(), client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            self.assertRaisesRegex(HpcAllocError, "has not started"),
        ):
            cmd_logs(
                SimpleNamespace(
                    target=f"@{OPERATION_ID}",
                    cluster=None,
                    lines=20,
                    follow=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(client.tail_calls, 0)

    def test_nonfollow_logs_uses_queue_final_during_accounting_outage(self) -> None:
        self.reserve()
        self.state.update_job(
            OPERATION_ID,
            phase=JobPhase.FINAL,
            evidence_provenance=EvidenceProvenance.ABSENT,
            evidence_detail="job was absent from two exact queue observations",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        client = SchedulerStub(
            accounting=SchedulerUnavailable("accounting is temporarily unavailable")
        )
        output = io.BytesIO()

        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(TransportStub(), client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch(
                "hpc_alloc.commands.sys.stdout",
                SimpleNamespace(buffer=output),
            ),
            patch("hpc_alloc.commands.info") as report,
        ):
            result = cmd_logs(
                SimpleNamespace(
                    target=f"@{OPERATION_ID}",
                    cluster=None,
                    lines=20,
                    follow=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        self.assertEqual(client.final_calls, 1)
        self.assertEqual(client.tail_calls, 1)
        self.assertEqual(output.getvalue(), b"unexpected stale log\n")
        report.assert_called_once_with(
            "scheduler unavailable; reading the operation-scoped log from "
            "durable final/start evidence"
        )

    def test_nonfollow_logs_uses_durable_start_after_race_then_fresh_outage(self) -> None:
        self.reserve()
        client = SchedulerStub(
            observations=(
                self.row("PENDING", reason="Resources"),
                SchedulerUnavailable("slurm controller unavailable"),
            )
        )
        transport = TransportStub()
        output = io.BytesIO()

        with (
            self.revision_race(self.activate_race),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch(
                "hpc_alloc.commands.sys.stdout",
                SimpleNamespace(buffer=output),
            ),
            patch("hpc_alloc.commands.info") as report,
        ):
            result = cmd_logs(
                SimpleNamespace(
                    target=f"@{OPERATION_ID}",
                    cluster=None,
                    lines=20,
                    follow=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        self.assertEqual(client.observe_calls, 2)
        self.assertEqual(client.tail_calls, 1)
        self.assertEqual(output.getvalue(), b"unexpected stale log\n")
        report.assert_called_once_with(
            "scheduler unavailable; reading the operation-scoped log from "
            "durable final/start evidence"
        )

    def test_active_allocation_does_not_reach_node_after_fresh_final(self) -> None:
        self.reserve()
        client = SchedulerStub(observations=(self.row("RUNNING", node="node01"),))
        transport = TransportStub()
        run = Mock(side_effect=AssertionError("rsync must not run"))

        with (
            self.revision_race(self.finalize_race),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch("hpc_alloc.commands._sync_ssh_projection", return_value=True),
            patch("hpc_alloc.commands.subprocess.run", run),
            self.assertRaisesRegex(HpcAllocError, " is FINAL;"),
        ):
            cmd_sync(
                SimpleNamespace(
                    target=f"@{OPERATION_ID}",
                    cluster=None,
                    src="./source",
                    dst="~/destination",
                    pull=False,
                    delete=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        transport.require_node.assert_not_called()
        run.assert_not_called()

    def test_accounting_recovery_keeps_adoption_and_reconciles_enrichment(self) -> None:
        self.reserve(acknowledged=False)
        self.state.mark_submission_ambiguous(OPERATION_ID, "sbatch reply lost")
        record = self.accounting()
        client = SchedulerStub(
            observations=(self.row("COMPLETED"),),
            accounting=record,
            recovered=record,
        )
        transport = TransportStub()

        with (
            self.revision_race(self.activate_race),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, client),
            ),
            patch(
                "hpc_alloc.commands._sync_ssh_projection", return_value=True
            ) as project,
            patch("hpc_alloc.commands.info"),
        ):
            result = cmd_recover(
                SimpleNamespace(
                    operation_id=OPERATION_ID,
                    cluster=None,
                    abandon=False,
                    yes=False,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        operation = self.state.get_operation(OPERATION_ID)
        recovered = self.state.get_job(OPERATION_ID)
        self.assertEqual(operation.phase, OperationPhase.ACKNOWLEDGED)
        self.assertEqual(operation.job_id, "12345")
        self.assertEqual(recovered.job_id, "12345")
        self.assertEqual(recovered.phase, JobPhase.FINAL)
        self.assertEqual(recovered.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(recovered.terminal_state, "COMPLETED")
        self.assertEqual(client.observe_calls, 1)
        self.assertEqual(client.final_calls, 1)
        self.assertEqual(client.verify_calls, 1)
        project.assert_called_once_with(self.context, self.paths)


if __name__ == "__main__":
    unittest.main()
