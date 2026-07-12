from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import cmd_logs, cmd_why
from hpc_alloc.config import Config
from hpc_alloc.errors import (
    AuthRequired,
    HostKeyChanged,
    SchedulerUnavailable,
    StateConflict,
)
from hpc_alloc.models import FinalSource, JobKind, JobPhase, OperationPhase
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.slurm import LogSizeResult, QueueRow
from hpc_alloc.state import StateRepository


SUBMIT_FAILED_ID = "a" * 32
ABANDONED_ID = "b" * 32
UNRESOLVED_ID = "c" * 32
HISTORICAL_ID = "d" * 32
RECYCLED_ID = "e" * 32
OWNER_ID = "deadbeef1234"


class LocalFinalCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = AppPaths.for_home(Path(self.directory.name))
        self.paths.config_dir.mkdir(parents=True)
        self.paths.config_file.write_text(
            """\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
[cluster.grace]
host = "grace.example.edu"
"""
        )
        self.config = Config.load(self.paths.config_file)
        self.state = StateRepository(
            self.paths.state_db, machine_id_factory=lambda: OWNER_ID
        ).initialize()
        self.owner = self.state.get_or_create_machine_id("laptop")
        self.context = SimpleNamespace(config=self.config, state=self.state)
        self._reserve(SUBMIT_FAILED_ID, "failed")
        self.state.fail_submission(SUBMIT_FAILED_ID, "sbatch rejected the request")
        self._reserve(ABANDONED_ID, "abandoned")
        self.state.mark_submission_ambiguous(ABANDONED_ID, "sbatch reply was lost")
        self.state.abandon_operation(
            ABANDONED_ID, "operator chose to abandon local recovery"
        )
        self._reserve(UNRESOLVED_ID, "unresolved")
        self.state.mark_submission_ambiguous(
            UNRESOLVED_ID, "sbatch may have committed"
        )
        self._reserve(HISTORICAL_ID, "historical")
        self.state.acknowledge_submission(HISTORICAL_ID, "12345")
        self.state.update_job(
            HISTORICAL_ID,
            phase=JobPhase.ACTIVE,
            ever_started=True,
            current_node="node01",
        )
        self.state.update_job(
            HISTORICAL_ID,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            exit_code="0:0",
            final_source=FinalSource.ACCOUNTING,
        )
        self._reserve(RECYCLED_ID, "recycled")
        self.state.acknowledge_submission(RECYCLED_ID, "54321")

    def _reserve(self, operation_id: str, name: str) -> None:
        self.state.reserve_submission(
            cluster="grace",
            logical_name=name,
            kind=JobKind.ALLOCATION,
            owner_id=self.owner,
            slurm_job_name=slurm_job_name("allocation", operation_id),
            slurm_comment=format_tag(
                self.owner, operation_id, "laptop", "allocation", name
            ),
            operation_id=operation_id,
        )

    @staticmethod
    def _no_remote(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a job without a Slurm ID must not construct remote services")

    def invoke_why(self, operation_id: str, *, json_output: bool) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("hpc_alloc.commands._services", side_effect=self._no_remote),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = cmd_why(
                SimpleNamespace(
                    target=f"grace:@{operation_id}",
                    cluster=None,
                    json=json_output,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_why_json_reports_local_final_verdicts_without_recovery_or_remote_calls(self) -> None:
        cases = (
            (SUBMIT_FAILED_ID, "SUBMIT_FAILED", FinalSource.SUBMIT_FAILED.value),
            (ABANDONED_ID, "ABANDONED", FinalSource.ABANDONED.value),
        )
        for operation_id, terminal_state, final_source in cases:
            with self.subTest(final_source=final_source):
                result, stdout, stderr = self.invoke_why(
                    operation_id, json_output=True
                )
                self.assertEqual(result, 0)
                self.assertEqual(stderr, "")
                payload = json.loads(stdout)
                self.assertEqual(payload["selector"], f"grace:@{operation_id}")
                self.assertIsNone(payload["jobid"])
                self.assertEqual(payload["status"], JobPhase.FINAL.value)
                self.assertEqual(payload["terminal_state"], terminal_state)
                self.assertEqual(payload["final_source"], final_source)
                self.assertNotIn("hpc-alloc recover", payload["diagnosis"])

    def test_why_text_reports_local_final_verdicts_without_recovery_or_remote_calls(self) -> None:
        for operation_id, verdict_word in (
            (SUBMIT_FAILED_ID, "failed"),
            (ABANDONED_ID, "abandoned"),
        ):
            with self.subTest(verdict=verdict_word):
                result, stdout, stderr = self.invoke_why(
                    operation_id, json_output=False
                )
                self.assertEqual(result, 0)
                self.assertEqual(stdout, "")
                self.assertIn(operation_id, stderr)
                self.assertIn(JobPhase.FINAL.value, stderr)
                self.assertIn(verdict_word, stderr.lower())
                self.assertNotIn("hpc-alloc recover", stderr)

    def test_why_unresolved_submission_has_recovery_guidance_and_no_remote_calls(self) -> None:
        operation = self.state.get_operation(UNRESOLVED_ID)
        self.assertEqual(operation.phase, OperationPhase.AMBIGUOUS)

        result, stdout, stderr = self.invoke_why(UNRESOLVED_ID, json_output=True)

        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["selector"], f"grace:@{UNRESOLVED_ID}")
        self.assertIsNone(payload["jobid"])
        self.assertEqual(payload["status"], JobPhase.SUBMITTING.value)
        self.assertIsNone(payload.get("final_source"))
        self.assertIn(f"hpc-alloc recover {UNRESOLVED_ID}", payload["diagnosis"])

        result, stdout, stderr = self.invoke_why(UNRESOLVED_ID, json_output=False)
        self.assertEqual(result, 0)
        self.assertEqual(stdout, "")
        self.assertIn(UNRESOLVED_ID, stderr)
        self.assertIn(JobPhase.SUBMITTING.value, stderr)
        self.assertIn(f"hpc-alloc recover {UNRESOLVED_ID}", stderr)

    def test_logs_and_follow_reject_local_finals_without_remote_calls(self) -> None:
        for operation_id, verdict_word in (
            (SUBMIT_FAILED_ID, "failed"),
            (ABANDONED_ID, "abandoned"),
        ):
            for follow in (False, True):
                with self.subTest(verdict=verdict_word, follow=follow):
                    with (
                        patch(
                            "hpc_alloc.commands._services",
                            side_effect=self._no_remote,
                        ),
                        self.assertRaises(StateConflict) as raised,
                    ):
                        cmd_logs(
                            SimpleNamespace(
                                target=f"grace:@{operation_id}",
                                cluster=None,
                                lines=100,
                                follow=follow,
                            ),
                            ctx=self.context,
                            paths=self.paths,
                            entrypoint=Path("/tmp/hpc-alloc"),
                        )
                    message = str(raised.exception)
                    self.assertIn(operation_id, message)
                    self.assertIn(verdict_word, message.lower())
                    self.assertIn("no managed log", message.lower())
                    self.assertNotIn("hpc-alloc recover", message)

    def test_logs_and_follow_keep_genuinely_unresolved_submission_local(self) -> None:
        for follow in (False, True):
            with self.subTest(follow=follow):
                with (
                    patch(
                        "hpc_alloc.commands._services", side_effect=self._no_remote
                    ),
                    self.assertRaises(StateConflict) as raised,
                ):
                    cmd_logs(
                        SimpleNamespace(
                            target=f"grace:@{UNRESOLVED_ID}",
                            cluster=None,
                            lines=100,
                            follow=follow,
                        ),
                        ctx=self.context,
                        paths=self.paths,
                        entrypoint=Path("/tmp/hpc-alloc"),
                    )
                self.assertIn(
                    f"hpc-alloc recover {UNRESOLVED_ID}", str(raised.exception)
                )

    def test_historical_why_uses_persisted_verdict_after_numeric_id_reuse(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("historical why observed a recycled numeric ID")

            def final(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("authoritative persisted accounting was re-queried")

            def tail_log(self, path: str, lines: int) -> bytes:
                self.path = path
                self.lines = lines
                return b"historical output\n"

        client = Client()
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            redirect_stdout(stdout),
        ):
            result = cmd_why(
                SimpleNamespace(
                    target=f"grace:@{HISTORICAL_ID}",
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["terminal_state"], "COMPLETED")
        self.assertEqual(payload["final_source"], FinalSource.ACCOUNTING.value)
        self.assertIn(HISTORICAL_ID, client.path)

    def test_why_preserves_access_failures_from_optional_log_tail(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        for failure in (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        ):
            with self.subTest(failure=type(failure).__name__):
                client = SimpleNamespace(tail_log=Mock(side_effect=failure))
                with patch(
                    "hpc_alloc.commands._services",
                    return_value=(Transport(), client),
                ):
                    with self.assertRaises(type(failure)) as raised:
                        cmd_why(
                            SimpleNamespace(
                                target=f"grace:@{HISTORICAL_ID}",
                                cluster=None,
                                json=True,
                            ),
                            ctx=self.context,
                            paths=self.paths,
                            entrypoint=Path("/tmp/hpc-alloc"),
                        )

                self.assertIs(raised.exception, failure)

    def test_why_still_ignores_ordinary_optional_log_tail_failure(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        failure = SchedulerUnavailable("log tail is temporarily unavailable")
        client = SimpleNamespace(tail_log=Mock(side_effect=failure))
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            redirect_stdout(stdout),
        ):
            result = cmd_why(
                SimpleNamespace(
                    target=f"grace:@{HISTORICAL_ID}",
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["detail"], [])

    def test_historical_logs_never_observe_recycled_numeric_id(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.paths: list[str] = []

            def observe(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("historical logs observed a recycled numeric ID")

            def final(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("historical logs queried recycled accounting")

            def tail_log(self, path: str, _lines: int) -> bytes:
                self.paths.append(path)
                return b"historical output\n"

            def log_size(self, path: str) -> LogSizeResult:
                self.paths.append(path)
                return LogSizeResult.available(len(b"historical output\n"))

            def read_log_chunk(self, path: str, offset: int, *, limit: int) -> bytes:
                self.paths.append(path)
                return b"historical output\n"[offset : offset + limit]

        for follow in (False, True):
            with self.subTest(follow=follow):
                client = Client()
                output = io.BytesIO()
                with (
                    patch(
                        "hpc_alloc.commands._services",
                        return_value=(Transport(), client),
                    ),
                    patch(
                        "hpc_alloc.commands.sys.stdout",
                        SimpleNamespace(buffer=output),
                    ),
                ):
                    result = cmd_logs(
                        SimpleNamespace(
                            target=f"grace:@{HISTORICAL_ID}",
                            cluster=None,
                            lines=100,
                            follow=follow,
                        ),
                        ctx=self.context,
                        paths=self.paths,
                        entrypoint=Path("/tmp/hpc-alloc"),
                    )
                self.assertEqual(result, 0)
                self.assertEqual(output.getvalue(), b"historical output\n")
                self.assertTrue(client.paths)
                self.assertTrue(all(HISTORICAL_ID in path for path in client.paths))

    def test_why_reconciles_nonfinal_recycled_id_without_adopting_replacement(self) -> None:
        from hpc_alloc.errors import JobIdReused

        class ImmediateMonitor(JobMonitor):
            def __init__(self, client: object) -> None:
                super().__init__(client, confirmation_delay=0)

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.observations = 0
                self.accounting_checks = 0

            def observe(self, *_args: object, **_kwargs: object) -> object:
                self.observations += 1
                raise JobIdReused(
                    "job grace:54321 now belongs to a different operation"
                )

            def final(self, *_args: object, **_kwargs: object) -> None:
                self.accounting_checks += 1
                return None

        client = Client()
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            patch("hpc_alloc.monitor.JobMonitor", ImmediateMonitor),
            redirect_stdout(stdout),
        ):
            result = cmd_why(
                SimpleNamespace(
                    target=f"grace:@{RECYCLED_ID}",
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["selector"], f"grace:@{RECYCLED_ID}")
        self.assertEqual(payload["status"], JobPhase.FINAL.value)
        self.assertEqual(payload["final_source"], FinalSource.CONFIRMED_QUEUE.value)
        self.assertIn("different operation", payload["diagnosis"])
        self.assertEqual(client.observations, 2)
        self.assertGreaterEqual(client.accounting_checks, 2)

    def test_why_uses_identity_checked_pending_reason_without_second_observation(self) -> None:
        job = self.state.get_job(RECYCLED_ID)

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.observations = 0

            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                self.observations += 1
                if self.observations > 1:
                    raise AssertionError("why performed an untracked second observation")
                return QueueRow(
                    job_id=job.job_id or "",
                    state="PENDING",
                    node=None,
                    reason="Priority",
                    time_left="1:00:00",
                    partition="cpu",
                    name=job.slurm_job_name,
                    submitted_at="2026-07-12T12:00:00",
                    comment=job.slurm_comment,
                )

            def estimated_start(self, *_args: object, **_kwargs: object) -> str:
                return "N/A"

            def priority(self, *_args: object, **_kwargs: object) -> str:
                return ""

        client = Client()
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            redirect_stdout(stdout),
        ):
            result = cmd_why(
                SimpleNamespace(
                    target=f"grace:@{RECYCLED_ID}",
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertEqual(result, 0)
        self.assertEqual(client.observations, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], JobPhase.QUEUED.value)
        self.assertEqual(payload["diagnosis"], "queue contention (Priority)")


if __name__ == "__main__":
    unittest.main()
