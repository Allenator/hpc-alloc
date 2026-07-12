from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.commands import cmd_status
from hpc_alloc.config import Config
from hpc_alloc.errors import HostKeyChanged, JobIdReused, TransportLost
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.models import FinalSource, JobKind, JobPhase
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.slurm import AccountingRecord, QueueRow, RawQueueRow, RawQueueScan
from hpc_alloc.state import StateRepository


OPERATION_ID = "a" * 32
OWNER_ID = "deadbeef1234"


class AvailableTransport:
    def bootstrap(self, _cluster: str, _auth: object) -> None:
        return None


class UnavailableTransport:
    def bootstrap(self, cluster: str, _auth: object) -> None:
        raise TransportLost(f"{cluster} is offline")


class HostKeyTransport:
    def __init__(self, error: HostKeyChanged) -> None:
        self.error = error

    def bootstrap(self, _cluster: str, _auth: object) -> None:
        raise self.error


class EmptyClient:
    def scan(self, *, auth: object) -> RawQueueScan:
        return RawQueueScan(())


class StatusClient:
    def __init__(
        self,
        rows: tuple[RawQueueRow, ...],
        *,
        observations: tuple[QueueRow | None, ...] = (),
        finals: tuple[AccountingRecord | None, ...] = (),
    ) -> None:
        self.rows = rows
        self.observations = list(observations)
        self.finals = list(finals)

    def scan(self, *, auth: object) -> RawQueueScan:
        return RawQueueScan(self.rows)

    def observe(self, _ref: object, *, auth: object) -> QueueRow | None:
        if not self.observations:
            raise AssertionError("unexpected targeted observe")
        result = self.observations.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def final(
        self, _ref: object, *, attempts: tuple[int, ...], auth: object
    ) -> AccountingRecord | None:
        if not self.finals:
            raise AssertionError("unexpected accounting lookup")
        return self.finals.pop(0)

    def assert_complete(self) -> None:
        if self.observations or self.finals:
            raise AssertionError(
                f"unconsumed status script: {len(self.observations)} observations, "
                f"{len(self.finals)} accounting results"
            )


class StatusMulticlusterTests(unittest.TestCase):
    def make_context(
        self, *, clusters: tuple[str, ...] = ("grace",), primary: str = "grace"
    ) -> tuple[AppPaths, object, StateRepository, str]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paths = AppPaths.for_home(Path(directory.name))
        paths.config_dir.mkdir(parents=True)
        cluster_text = "".join(
            f"[cluster.{cluster}]\nhost = \"{cluster}.example.edu\"\n"
            for cluster in clusters
        )
        paths.config_file.write_text(
            "[identity]\n"
            'netid = "ab1234"\n'
            "[defaults]\n"
            f'cluster = "{primary}"\n'
            + cluster_text
        )
        config = Config.load(paths.config_file)
        state = StateRepository(
            paths.state_db, machine_id_factory=lambda: OWNER_ID
        ).initialize()
        owner = state.get_or_create_machine_id("laptop")
        return paths, SimpleNamespace(config=config, state=state), state, owner

    @staticmethod
    def reserve(
        state: StateRepository,
        owner: str,
        operation_id: str,
        *,
        name: str = "dev",
        job_id: str | None = None,
    ) -> tuple[str, str]:
        job_name = slurm_job_name("allocation", operation_id)
        comment = format_tag(owner, operation_id, "laptop", "allocation", name)
        state.reserve_submission(
            cluster="grace",
            logical_name=name,
            kind=JobKind.ALLOCATION,
            owner_id=owner,
            slurm_job_name=job_name,
            slurm_comment=comment,
            operation_id=operation_id,
        )
        if job_id is not None:
            state.acknowledge_submission(operation_id, job_id)
        return job_name, comment

    @staticmethod
    def raw_row(
        operation_id: str,
        owner: str,
        job_id: str,
        *,
        name: str = "dev",
        state: str = "RUNNING",
        node: str = "node01",
    ) -> RawQueueRow:
        return RawQueueRow(
            job_id=job_id,
            state=state,
            node=node,
            reason="None",
            time_left="1:00:00",
            partition="day",
            name=slurm_job_name("allocation", operation_id),
            submitted_at="2026-07-12T11:00:00",
            comment=format_tag(
                owner, operation_id, "laptop", "allocation", name
            ),
        )

    @staticmethod
    def strict_row(raw: RawQueueRow) -> QueueRow:
        return QueueRow(
            job_id=raw.job_id,
            state=raw.state,
            node=raw.node,
            reason=raw.reason,
            time_left=raw.time_left,
            partition=raw.partition,
            name=raw.name,
            submitted_at=raw.submitted_at,
            comment=raw.comment,
        )

    def run_status(
        self, paths: AppPaths, context: object, client: object
    ) -> dict[str, object]:
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(AvailableTransport(), client),
            ),
            redirect_stdout(stdout),
        ):
            result = cmd_status(
                SimpleNamespace(json=True),
                ctx=context,
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
        self.assertEqual(result, 0)
        return json.loads(stdout.getvalue())

    def test_secondary_transport_loss_is_uncertainty_and_json_contract_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            paths = AppPaths.for_home(home)
            paths.config_dir.mkdir(parents=True)
            paths.config_file.write_text(
                """\
[identity]
netid = "ab1234"
[defaults]
cluster = "primary"
[cluster.primary]
host = "primary.example.edu"
[cluster.secondary]
host = "secondary.example.edu"
"""
            )
            config = Config.load(paths.config_file)
            state = StateRepository(
                paths.state_db, machine_id_factory=lambda: OWNER_ID
            ).initialize()
            owner = state.get_or_create_machine_id("laptop")
            state.reserve_submission(
                cluster="secondary",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                owner_id=owner,
                slurm_job_name=slurm_job_name("allocation", OPERATION_ID),
                slurm_comment=format_tag(
                    owner, OPERATION_ID, "laptop", "allocation", "dev"
                ),
                operation_id=OPERATION_ID,
            )
            state.acknowledge_submission(OPERATION_ID, "12345")
            context = SimpleNamespace(config=config, state=state)

            def services(_ctx: object, _paths: object, _entry: object, cluster: str):
                if cluster == "primary":
                    return AvailableTransport(), EmptyClient()
                return UnavailableTransport(), EmptyClient()

            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("hpc_alloc.commands._services", side_effect=services),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = cmd_status(
                    SimpleNamespace(json=True),
                    ctx=context,
                    paths=paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )

            self.assertEqual(result, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(set(payload), {"jobs", "discovered", "operations"})
            self.assertEqual(payload["jobs"][0]["phase"], "UNCERTAIN")
            self.assertEqual(payload["jobs"][0]["cluster"], "secondary")
            self.assertEqual(state.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
            self.assertIn("preserving its state", stderr.getvalue())

    def test_secondary_bootstrap_host_key_change_is_always_fatal(self) -> None:
        paths, context, _state, _owner = self.make_context(
            clusters=("grace", "secondary")
        )
        changed = HostKeyChanged(
            "SSH host-key verification failed for hpc-secondary.login"
        )

        def services(_ctx: object, _paths: object, _entry: object, cluster: str):
            if cluster == "grace":
                return AvailableTransport(), EmptyClient()
            return HostKeyTransport(changed), EmptyClient()

        stdout = io.StringIO()
        with (
            patch("hpc_alloc.commands._services", side_effect=services),
            redirect_stdout(stdout),
        ):
            with self.assertRaises(HostKeyChanged) as raised:
                cmd_status(
                    SimpleNamespace(json=True),
                    ctx=context,
                    paths=paths,
                    entrypoint=Path("/tmp/hpc-alloc"),
                )
        self.assertIs(raised.exception, changed)
        self.assertEqual(stdout.getvalue(), "")

    def test_secondary_targeted_host_key_change_is_always_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.for_home(Path(directory))
            paths.config_dir.mkdir(parents=True)
            paths.config_file.write_text(
                """\
[identity]
netid = "ab1234"
[defaults]
cluster = "primary"
[cluster.primary]
host = "primary.example.edu"
[cluster.secondary]
host = "secondary.example.edu"
"""
            )
            config = Config.load(paths.config_file)
            state = StateRepository(
                paths.state_db, machine_id_factory=lambda: OWNER_ID
            ).initialize()
            owner = state.get_or_create_machine_id("laptop")
            state.reserve_submission(
                cluster="secondary",
                logical_name="dev",
                kind=JobKind.ALLOCATION,
                owner_id=owner,
                slurm_job_name=slurm_job_name("allocation", OPERATION_ID),
                slurm_comment=format_tag(
                    owner, OPERATION_ID, "laptop", "allocation", "dev"
                ),
                operation_id=OPERATION_ID,
            )
            state.acknowledge_submission(OPERATION_ID, "12345")
            context = SimpleNamespace(config=config, state=state)
            changed = HostKeyChanged(
                "SSH host-key verification failed during targeted observation"
            )
            secondary = StatusClient((), observations=(changed,))

            def services(_ctx: object, _paths: object, _entry: object, cluster: str):
                if cluster == "primary":
                    return AvailableTransport(), EmptyClient()
                return AvailableTransport(), secondary

            stdout = io.StringIO()
            with (
                patch("hpc_alloc.commands._services", side_effect=services),
                redirect_stdout(stdout),
            ):
                with self.assertRaises(HostKeyChanged) as raised:
                    cmd_status(
                        SimpleNamespace(json=True),
                        ctx=context,
                        paths=paths,
                        entrypoint=Path("/tmp/hpc-alloc"),
                    )
            self.assertIs(raised.exception, changed)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(state.get_job(OPERATION_ID).phase, JobPhase.QUEUED)
            secondary.assert_complete()

    def test_unrelated_array_and_multinode_scan_rows_do_not_break_managed_status(self) -> None:
        paths, context, state, owner = self.make_context()
        job_name, comment = self.reserve(
            state, owner, OPERATION_ID, job_id="12345"
        )
        bound = self.raw_row(OPERATION_ID, owner, "12345")
        unrelated = RawQueueRow(
            job_id="98765_4",
            state="RUNNING",
            node="node[02-05]",
            reason="None",
            time_left="0:20:00",
            partition="day*",
            name="ordinary-array-job",
            submitted_at="2026-07-12T11:01:00",
            comment="not-an-hpc-alloc-tag",
        )
        client = StatusClient(
            (unrelated, bound), observations=(self.strict_row(bound),)
        )

        payload = self.run_status(paths, context, client)

        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["phase"], JobPhase.ACTIVE.value)
        self.assertEqual(
            payload["jobs"][0]["selector"], f"grace:@{OPERATION_ID}"
        )
        self.assertEqual(payload["discovered"], [])
        self.assertEqual(state.get_job(OPERATION_ID).slurm_job_name, job_name)
        self.assertEqual(state.get_job(OPERATION_ID).slurm_comment, comment)
        client.assert_complete()

    def test_job_finalized_during_pass_appears_once_not_as_local_final_conflict(self) -> None:
        paths, context, state, owner = self.make_context()
        job_name, comment = self.reserve(
            state, owner, OPERATION_ID, job_id="12345"
        )
        stale_scan_row = self.raw_row(OPERATION_ID, owner, "12345")
        accounting = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=job_name,
            comment=comment,
        )
        client = StatusClient(
            (stale_scan_row,), observations=(None,), finals=(accounting,)
        )

        payload = self.run_status(paths, context, client)

        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["phase"], JobPhase.FINAL.value)
        self.assertEqual(
            payload["jobs"][0]["selector"], f"grace:@{OPERATION_ID}"
        )
        self.assertEqual(payload["discovered"], [])
        self.assertEqual(
            state.get_job(OPERATION_ID).final_source, FinalSource.ACCOUNTING
        )
        client.assert_complete()

    def test_bound_row_and_extra_same_operation_emit_one_duplicate_operation(self) -> None:
        paths, context, state, owner = self.make_context()
        self.reserve(state, owner, OPERATION_ID, job_id="12345")
        bound = self.raw_row(OPERATION_ID, owner, "12345")
        duplicate = self.raw_row(OPERATION_ID, owner, "12346")
        client = StatusClient(
            (bound, duplicate), observations=(self.strict_row(bound),)
        )

        payload = self.run_status(paths, context, client)

        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(
            payload["jobs"][0]["selector"], f"grace:@{OPERATION_ID}"
        )
        self.assertEqual(len(payload["discovered"]), 1)
        discovered = payload["discovered"][0]
        self.assertEqual(discovered["jobid"], "12346")
        self.assertEqual(discovered["selector"], f"grace:@{OPERATION_ID}")
        self.assertEqual(discovered["job_kind"], JobKind.ALLOCATION.value)
        self.assertEqual(discovered["classification"], "duplicate-operation")
        self.assertNotIn("kind", discovered)
        client.assert_complete()

    def test_reused_numeric_id_for_different_operation_is_not_local_final_conflict(self) -> None:
        paths, context, state, owner = self.make_context()
        self.reserve(state, owner, OPERATION_ID, job_id="12345")
        state.update_job(
            OPERATION_ID,
            phase=JobPhase.FINAL,
            ever_started=True,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        replacement_operation = "b" * 32
        recycled = self.raw_row(
            replacement_operation, owner, "12345", name="replacement"
        )
        client = StatusClient((recycled,))

        payload = self.run_status(paths, context, client)

        self.assertEqual(payload["jobs"], [])
        self.assertEqual(len(payload["discovered"]), 1)
        discovered = payload["discovered"][0]
        self.assertEqual(discovered["selector"], f"grace:@{replacement_operation}")
        self.assertEqual(discovered["job_kind"], JobKind.ALLOCATION.value)
        self.assertEqual(discovered["classification"], "untracked-owned")
        self.assertNotEqual(discovered["classification"], "local-final-conflict")
        client.assert_complete()

    def test_nonfinal_old_operation_reconciles_recycled_id_and_reports_replacement(self) -> None:
        class ImmediateMonitor(JobMonitor):
            def __init__(self, client: object) -> None:
                super().__init__(client, confirmation_delay=0)

        paths, context, state, owner = self.make_context()
        self.reserve(state, owner, OPERATION_ID, job_id="12345")
        replacement_operation = "b" * 32
        recycled = self.raw_row(
            replacement_operation, owner, "12345", name="replacement"
        )
        mismatch = JobIdReused(
            "job grace:12345 now belongs to a different operation"
        )
        client = StatusClient(
            (recycled,),
            observations=(mismatch, mismatch),
            finals=(None, None),
        )

        with patch("hpc_alloc.monitor.JobMonitor", ImmediateMonitor):
            payload = self.run_status(paths, context, client)

        self.assertEqual(len(payload["jobs"]), 1)
        old = payload["jobs"][0]
        self.assertEqual(old["selector"], f"grace:@{OPERATION_ID}")
        self.assertEqual(old["phase"], JobPhase.FINAL.value)
        self.assertEqual(old["final_source"], FinalSource.CONFIRMED_QUEUE.value)
        self.assertIn("different operation", old["evidence_detail"])
        self.assertEqual(len(payload["discovered"]), 1)
        replacement = payload["discovered"][0]
        self.assertEqual(
            replacement["selector"], f"grace:@{replacement_operation}"
        )
        self.assertEqual(replacement["classification"], "untracked-owned")
        self.assertEqual(state.get_job(OPERATION_ID).phase, JobPhase.FINAL)
        client.assert_complete()


if __name__ == "__main__":
    unittest.main()
