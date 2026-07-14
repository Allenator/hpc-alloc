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
    ConfigInvalid,
    HostKeyChanged,
    ProtocolViolation,
    RemoteCommandFailed,
    SchedulerUnavailable,
    StateConflict,
)
from hpc_alloc.models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    OperationPhase,
)
from hpc_alloc.monitor import JobMonitor
from hpc_alloc.ownership import format_tag, slurm_job_name
from hpc_alloc.paths import AppPaths
from hpc_alloc.slurm import AccountingRecord, LogSizeResult, QueueRow
from hpc_alloc.ssh import AuthMode
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

    def test_historical_why_enriches_persisted_accounting_without_queue_observation(
        self,
    ) -> None:
        historical = self.state.get_job(HISTORICAL_ID)
        assert historical.ref is not None
        record = AccountingRecord(
            job_id="12345",
            state="COMPLETED",
            exit_code="0:0",
            job_name=slurm_job_name("allocation", HISTORICAL_ID),
            comment=format_tag(
                self.owner,
                HISTORICAL_ID,
                "laptop",
                "allocation",
                "historical",
            ),
            extra=("00:20:00", "01:00:00"),
        )

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.final_calls: list[tuple[object, dict[str, object]]] = []

            def observe(self, *_args: object, **_kwargs: object) -> object:
                raise AssertionError("historical why observed a recycled numeric ID")

            def final(self, ref: object, **kwargs: object) -> AccountingRecord:
                self.final_calls.append((ref, kwargs))
                return record

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
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                return_value=True,
            ) as projection,
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
        self.assertEqual(payload["elapsed"], "00:20:00")
        self.assertEqual(payload["timelimit"], "01:00:00")
        self.assertIn(HISTORICAL_ID, client.path)
        self.assertEqual(
            client.final_calls,
            [
                (
                    historical.ref,
                    {
                        "attempts": (0, 2, 2),
                        "auth": AuthMode.NONINTERACTIVE,
                        "extra_fields": ("Elapsed", "Timelimit"),
                    },
                )
            ],
        )
        self.assertEqual(projection.call_count, 1)

    def test_why_enriches_a_nonfinal_job_reconciled_directly_to_accounting(
        self,
    ) -> None:
        queued = self.state.get_job(RECYCLED_ID)
        assert queued.ref is not None
        base_record = AccountingRecord(
            job_id="54321",
            state="FAILED",
            exit_code="7:0",
            job_name=slurm_job_name("allocation", RECYCLED_ID),
            comment=format_tag(
                self.owner,
                RECYCLED_ID,
                "laptop",
                "allocation",
                "recycled",
            ),
        )
        timing_record = AccountingRecord(
            job_id=base_record.job_id,
            state=base_record.state,
            exit_code=base_record.exit_code,
            job_name=base_record.job_name,
            comment=base_record.comment,
            extra=("00:12:34", "01:00:00"),
        )

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.observations = 0
                self.final_calls: list[tuple[object, dict[str, object]]] = []

            def observe(self, *_args: object, **_kwargs: object) -> None:
                self.observations += 1
                return None

            def final(self, ref: object, **kwargs: object) -> AccountingRecord:
                self.final_calls.append((ref, kwargs))
                return timing_record if kwargs.get("extra_fields") else base_record

            def tail_log(self, _path: str, _lines: int) -> bytes:
                return b"failed output\n"

        client = Client()
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                return_value=True,
            ) as projection,
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
        self.assertEqual(payload["terminal_state"], "FAILED")
        self.assertEqual(payload["exit_code"], "7:0")
        self.assertEqual(payload["final_source"], FinalSource.ACCOUNTING.value)
        self.assertEqual(payload["elapsed"], "00:12:34")
        self.assertEqual(payload["timelimit"], "01:00:00")
        stored = self.state.get_job(RECYCLED_ID)
        self.assertEqual(stored.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(stored.terminal_state, "FAILED")
        self.assertEqual(stored.exit_code, "7:0")
        self.assertEqual(client.observations, 1)
        # One accounting query, not two.  `why` now asks for its display-only
        # columns on the assessment's own read, so the record that assessment
        # fetches is the record `why` needs.  It used to run the entire retry
        # ladder a second time -- the heaviest and slowest query the tool makes --
        # for a record that could not have changed in the twenty lines between.
        self.assertEqual(len(client.final_calls), 1)
        self.assertEqual(client.final_calls[0][0], queued.ref)
        self.assertEqual(
            client.final_calls[0][1]["extra_fields"], ("Elapsed", "Timelimit")
        )
        self.assertEqual(client.final_calls[0][1]["auth"], AuthMode.NONINTERACTIVE)
        self.assertEqual(projection.call_count, 1)

    def test_why_omits_stale_timing_that_disagrees_with_accounting_verdict(
        self,
    ) -> None:
        record = AccountingRecord(
            job_id="12345",
            state="FAILED",
            exit_code="7:0",
            job_name=slurm_job_name("allocation", HISTORICAL_ID),
            comment=format_tag(
                self.owner,
                HISTORICAL_ID,
                "laptop",
                "allocation",
                "historical",
            ),
            extra=("00:12:34", "01:00:00"),
        )

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        client = SimpleNamespace(
            final=Mock(return_value=record),
            tail_log=Mock(return_value=b"historical output\n"),
        )
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
        self.assertEqual(payload["exit_code"], "0:0")
        self.assertEqual(payload["final_source"], FinalSource.ACCOUNTING.value)
        self.assertNotIn("elapsed", payload)
        self.assertNotIn("timelimit", payload)
        stored = self.state.get_job(HISTORICAL_ID)
        self.assertEqual(stored.terminal_state, "COMPLETED")
        self.assertEqual(stored.exit_code, "0:0")

    def test_why_preserves_access_failures_from_accounting_enrichment(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        for failure in (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        ):
            with self.subTest(failure=type(failure).__name__):
                client = SimpleNamespace(
                    final=Mock(side_effect=failure),
                    tail_log=Mock(),
                )
                with (
                    patch(
                        "hpc_alloc.commands._services",
                        return_value=(Transport(), client),
                    ),
                    self.assertRaises(type(failure)) as raised,
                ):
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
                client.tail_log.assert_not_called()

    def test_why_ignores_nonaccess_accounting_enrichment_failures(self) -> None:
        self.state.update_job(
            RECYCLED_ID,
            phase=JobPhase.FINAL,
            terminal_state="CANCELLED",
            exit_code="0:15",
            final_source=FinalSource.ACCOUNTING,
        )
        self.assertFalse(self.state.get_job(RECYCLED_ID).ever_started)

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        for failure in (
            SchedulerUnavailable("accounting is temporarily unavailable"),
            ProtocolViolation("accounting returned malformed output"),
        ):
            with self.subTest(failure=type(failure).__name__):
                client = SimpleNamespace(
                    final=Mock(side_effect=failure),
                    tail_log=Mock(return_value=b"historical output\n"),
                )
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
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["terminal_state"], "CANCELLED")
                self.assertEqual(payload["exit_code"], "0:15")
                self.assertFalse(payload["ever_started"])
                self.assertEqual(
                    payload["final_source"], FinalSource.ACCOUNTING.value
                )
                self.assertNotIn("elapsed", payload)
                self.assertNotIn("timelimit", payload)
                self.assertEqual(
                    payload["detail"],
                    ["--- log tail ---", "historical output"],
                )
                client.tail_log.assert_called_once()

    def test_queue_final_enrichment_failure_still_propagates(self) -> None:
        self.state.update_job(
            RECYCLED_ID,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            evidence_provenance=EvidenceProvenance.ABSENT,
            evidence_detail="job was absent from two exact queue observations",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        failure = SchedulerUnavailable("accounting is temporarily unavailable")

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.final_calls = 0

            def final(self, *_args: object, **_kwargs: object) -> None:
                self.final_calls += 1
                if self.final_calls == 1:
                    return None
                raise failure

        client = Client()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            self.assertRaises(SchedulerUnavailable) as raised,
        ):
            cmd_why(
                SimpleNamespace(
                    target=f"grace:@{RECYCLED_ID}",
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )

        self.assertIs(raised.exception, failure)
        self.assertEqual(client.final_calls, 2)
        self.assertEqual(
            self.state.get_job(RECYCLED_ID).final_source,
            FinalSource.CONFIRMED_QUEUE,
        )

    def test_why_persists_delayed_accounting_before_rendering_json(self) -> None:
        self.state.update_job(
            RECYCLED_ID,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            evidence_provenance=EvidenceProvenance.ABSENT,
            evidence_detail="job was absent from two exact queue observations",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )
        record = AccountingRecord(
            job_id="54321",
            state="FAILED",
            exit_code="7:0",
            job_name=slurm_job_name("allocation", RECYCLED_ID),
            comment=format_tag(
                self.owner, RECYCLED_ID, "laptop", "allocation", "recycled"
            ),
            extra=("00:12:34", "01:00:00"),
        )

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def __init__(self) -> None:
                self.final_calls: list[dict[str, object]] = []

            def final(self, *_args: object, **kwargs: object):
                self.final_calls.append(kwargs)
                # Delayed accounting: slurmdbd has not caught up on the first
                # read, and the record appears on the retry.  Keyed on *when* the
                # call happens, not on which columns it asks for -- `why` now
                # requests its display columns on the assessment's own read, so
                # keying on extra_fields would have handed the record over
                # immediately and stopped exercising the retry at all.
                return record if len(self.final_calls) > 1 else None

            def tail_log(self, _path: str, _lines: int) -> bytes:
                return b"failed output\n"

        client = Client()
        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), client),
            ),
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                return_value=True,
            ) as projection,
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
        self.assertEqual(payload["status"], JobPhase.FINAL.value)
        self.assertEqual(payload["phase"], JobPhase.FINAL.value)
        self.assertEqual(payload["terminal_state"], "FAILED")
        self.assertEqual(payload["exit_code"], "7:0")
        self.assertEqual(payload["final_source"], FinalSource.ACCOUNTING.value)
        self.assertEqual(payload["diagnosis"], "final state FAILED")
        self.assertEqual(payload["elapsed"], "00:12:34")
        self.assertEqual(payload["timelimit"], "01:00:00")
        stored = self.state.get_job(RECYCLED_ID)
        self.assertEqual(stored.final_source, FinalSource.ACCOUNTING)
        self.assertEqual(stored.terminal_state, "FAILED")
        self.assertEqual(stored.exit_code, "7:0")
        self.assertGreaterEqual(projection.call_count, 2)

    def test_why_no_result_retry_keeps_queue_final_without_timing_fields(self) -> None:
        self.state.update_job(
            RECYCLED_ID,
            phase=JobPhase.FINAL,
            terminal_state="COMPLETED",
            evidence_provenance=EvidenceProvenance.ABSENT,
            evidence_detail="job was absent from two exact queue observations",
            final_source=FinalSource.CONFIRMED_QUEUE,
        )

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        client = SimpleNamespace(
            final=Mock(return_value=None),
            tail_log=Mock(return_value=b""),
        )
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
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            payload["final_source"], FinalSource.CONFIRMED_QUEUE.value
        )
        self.assertNotIn("elapsed", payload)
        self.assertNotIn("timelimit", payload)
        self.assertEqual(client.final.call_count, 2)
        self.assertEqual(
            self.state.get_job(RECYCLED_ID).final_source,
            FinalSource.CONFIRMED_QUEUE,
        )

    def test_why_preserves_access_failures_from_optional_log_tail(self) -> None:
        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        for failure in (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        ):
            with self.subTest(failure=type(failure).__name__):
                client = SimpleNamespace(
                    final=Mock(return_value=None),
                    tail_log=Mock(side_effect=failure),
                )
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
        client = SimpleNamespace(
            final=Mock(return_value=None),
            tail_log=Mock(side_effect=failure),
        )
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

            def tail_log(self, _path: str, _lines: int) -> bytes:
                return b""

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

    def _pending_why_payload(
        self,
        *,
        reason: str,
        estimate: str | BaseException = "N/A",
        priority: str | BaseException = "",
        reservations: str | BaseException = "",
    ) -> dict[str, object]:
        job = self.state.get_job(RECYCLED_ID)

        def value(result: str | BaseException) -> str:
            if isinstance(result, BaseException):
                raise result
            return result

        class Transport:
            def bootstrap(self, *_args: object, **_kwargs: object) -> None:
                return None

        class Client:
            def observe(self, *_args: object, **_kwargs: object) -> QueueRow:
                return QueueRow(
                    job_id=job.job_id or "",
                    state="PENDING",
                    node=None,
                    reason=reason,
                    time_left="1:00:00",
                    partition="cpu",
                    name=job.slurm_job_name,
                    submitted_at="2026-07-12T12:00:00",
                    comment=job.slurm_comment,
                )

            def estimated_start(self, *_args: object, **_kwargs: object) -> str:
                return value(estimate)

            def priority(self, *_args: object, **_kwargs: object) -> str:
                return value(priority)

            def reservations(self, *_args: object, **_kwargs: object) -> str:
                return value(reservations)

        stdout = io.StringIO()
        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(Transport(), Client()),
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
        return json.loads(stdout.getvalue())

    def test_why_omits_only_failed_optional_pending_enrichment(self) -> None:
        payload = self._pending_why_payload(
            reason="Priority",
            estimate=RemoteCommandFailed("squeue --start lost the job"),
            priority="priority detail",
        )
        self.assertEqual(payload["diagnosis"], "queue contention (Priority)")
        self.assertEqual(payload["detail"], ["priority detail"])

        payload = self._pending_why_payload(
            reason="Priority",
            estimate="2026-07-14T09:00:00",
            priority=SchedulerUnavailable("sprio unavailable"),
        )
        self.assertEqual(payload["diagnosis"], "queue contention (Priority)")
        self.assertEqual(
            payload["detail"],
            ["estimated start: 2026-07-14T09:00:00"],
        )

        payload = self._pending_why_payload(
            reason="Reservation",
            reservations=RemoteCommandFailed("reservation query failed"),
        )
        self.assertIn("nodes are reserved", str(payload["diagnosis"]))
        self.assertEqual(payload["detail"], [])

    def test_why_optional_pending_enrichment_preserves_access_failures(self) -> None:
        failures = (
            (
                "Priority",
                {"estimate": AuthRequired("Duo required")},
            ),
            (
                "Reservation",
                {"reservations": HostKeyChanged("host key changed")},
            ),
        )
        for reason, kwargs in failures:
            failure = next(iter(kwargs.values()))
            assert isinstance(failure, BaseException)
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)) as raised:
                    self._pending_why_payload(reason=reason, **kwargs)
                self.assertIs(raised.exception, failure)

    def test_qualified_local_history_survives_cluster_removal(self) -> None:
        self.paths.config_file.write_text(
            """\
[identity]
netid = "ab1234"
[defaults]
cluster = "beta"
[cluster.beta]
host = "beta.example.edu"
"""
        )
        self.config = Config.load(self.paths.config_file)
        self.context = SimpleNamespace(config=self.config, state=self.state)

        for operation_id in (SUBMIT_FAILED_ID, ABANDONED_ID):
            with self.subTest(operation_id=operation_id):
                result, stdout, _stderr = self.invoke_why(
                    operation_id,
                    json_output=True,
                )
                self.assertEqual(result, 0)
                payload = json.loads(stdout)
                self.assertEqual(
                    payload["selector"],
                    f"grace:@{operation_id}",
                )

                with (
                    patch(
                        "hpc_alloc.commands._services",
                        side_effect=self._no_remote,
                    ),
                    self.assertRaisesRegex(StateConflict, "no managed log"),
                ):
                    cmd_logs(
                        SimpleNamespace(
                            target=f"grace:@{operation_id}",
                            cluster=None,
                            lines=10,
                            follow=False,
                        ),
                        ctx=self.context,
                        paths=self.paths,
                        entrypoint=Path("/tmp/hpc-alloc"),
                    )

    def test_removed_cluster_history_never_falls_back_to_current_cluster(self) -> None:
        self.paths.config_file.write_text(
            """\
[identity]
netid = "ab1234"
[defaults]
cluster = "beta"
[cluster.beta]
host = "beta.example.edu"
"""
        )
        self.config = Config.load(self.paths.config_file)
        self.context = SimpleNamespace(config=self.config, state=self.state)
        clusters: list[str] = []

        def services(
            _ctx: object,
            _paths: object,
            _entrypoint: object,
            cluster: str,
        ) -> object:
            clusters.append(cluster)
            raise ConfigInvalid(f"cluster {cluster!r} is not configured")

        with (
            patch("hpc_alloc.commands._services", side_effect=services),
            self.assertRaisesRegex(ConfigInvalid, "grace"),
        ):
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

        self.assertEqual(clusters, ["grace"])

        with self.assertRaisesRegex(ConfigInvalid, "grace"):
            cmd_why(
                SimpleNamespace(
                    target="grace:@" + "f" * 32,
                    cluster=None,
                    json=True,
                ),
                ctx=self.context,
                paths=self.paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )


if __name__ == "__main__":
    unittest.main()
