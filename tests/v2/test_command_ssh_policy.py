from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

from hpc_alloc.commands import (
    _services,
    _sync_ssh_projection,
    cmd_cancel,
    cmd_connect,
    cmd_down,
    cmd_ssh,
    cmd_sync,
)
from hpc_alloc.config import Config
from hpc_alloc.errors import AuthRequired, ConfigInvalid, HostKeyChanged, TransportLost
from hpc_alloc.models import JobKind
from hpc_alloc.paths import AppPaths


class ServiceProjectionTests(unittest.TestCase):
    def context(self) -> tuple[AppPaths, object]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paths = AppPaths.for_home(Path(directory.name))
        paths.config_dir.mkdir(parents=True)
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
        active = SimpleNamespace(
            cluster="grace",
            logical_name="dev",
            kind=JobKind.ALLOCATION,
            current_node="node01",
        )
        state = SimpleNamespace(
            list_jobs=Mock(return_value=[active]),
        )
        return paths, SimpleNamespace(
            config=Config.load(paths.config_file),
            state=state,
            primary_cluster="grace",
        )

    def test_services_projects_new_cluster_before_transport_construction(
        self,
    ) -> None:
        paths, context = self.context()
        _sync_ssh_projection(context, paths)
        self.assertNotIn("hpc-bouchet.login", paths.managed_ssh_config.read_text())
        with paths.config_file.open("a", encoding="utf-8") as handle:
            handle.write('[cluster.bouchet]\nhost = "bouchet.example.edu"\n')
        context.config = Config.load(paths.config_file)
        transport = object()
        client = object()

        def construct_transport(*_args: object, **_kwargs: object) -> object:
            projection = paths.managed_ssh_config.read_text()
            self.assertIn("Host hpc-grace.login", projection)
            self.assertIn("Host hpc-bouchet.login", projection)
            self.assertIn("    HostName bouchet.example.edu", projection)
            self.assertIn("Host hpc-grace.dev", projection)
            self.assertIn("    HostName node01", projection)
            return transport

        with (
            patch(
                "hpc_alloc.ssh.SshTransport",
                side_effect=construct_transport,
            ) as transport_constructor,
            patch(
                "hpc_alloc.slurm.SlurmClient",
                return_value=client,
            ) as client_constructor,
        ):
            result = _services(
                context,
                paths,
                Path("/tmp/hpc-alloc"),
                "bouchet",
            )

        self.assertEqual(result, (transport, client))
        transport_constructor.assert_called_once()
        client_constructor.assert_called_once_with(transport, "bouchet")

    def test_projection_failure_prevents_service_construction(self) -> None:
        paths, context = self.context()
        failure = ConfigInvalid("managed SSH projection failed")

        with (
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                side_effect=failure,
            ) as project,
            patch("hpc_alloc.ssh.SshTransport") as transport_constructor,
            patch("hpc_alloc.slurm.SlurmClient") as client_constructor,
        ):
            with self.assertRaises(ConfigInvalid) as raised:
                _services(
                    context,
                    paths,
                    Path("/tmp/hpc-alloc"),
                    "grace",
                )

        self.assertIs(raised.exception, failure)
        project.assert_called_once_with(context, paths)
        transport_constructor.assert_not_called()
        client_constructor.assert_not_called()

    def test_retirement_failure_warns_but_does_not_block_projection(self) -> None:
        paths, context = self.context()
        _sync_ssh_projection(context, paths)
        active = context.state.list_jobs.return_value[0]
        active.current_node = None
        stderr = io.StringIO()

        with (
            patch(
                "hpc_alloc.ssh.retire_compute_masters",
                side_effect=RuntimeError("local cleanup failed"),
            ),
            redirect_stderr(stderr),
        ):
            changed = _sync_ssh_projection(context, paths)

        self.assertTrue(changed)
        self.assertNotIn("Host hpc-grace.dev", paths.managed_ssh_config.read_text())
        self.assertIn("warning", stderr.getvalue())
        self.assertIn("local cleanup failed", stderr.getvalue())

    def test_invalid_cluster_is_rejected_before_projection(self) -> None:
        paths, context = self.context()

        with (
            patch("hpc_alloc.commands._sync_ssh_projection") as project,
            patch("hpc_alloc.ssh.SshTransport") as transport_constructor,
            patch("hpc_alloc.slurm.SlurmClient") as client_constructor,
        ):
            with self.assertRaisesRegex(ConfigInvalid, "is not configured"):
                _services(
                    context,
                    paths,
                    Path("/tmp/hpc-alloc"),
                    "missing",
                )

        project.assert_not_called()
        transport_constructor.assert_not_called()
        client_constructor.assert_not_called()


class CommandSshPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = AppPaths.for_home(Path("/tmp/hpc-alloc-command-ssh-policy"))
        self.entrypoint = Path("/tmp/hpc-alloc")
        self.job = SimpleNamespace(cluster="grace", logical_name="dev")

    def test_cmd_ssh_propagates_compute_host_key_failure_without_exec(self) -> None:
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace.dev")
        transport = SimpleNamespace(require_node=Mock(side_effect=failure))
        execvp = Mock(side_effect=AssertionError("ssh must not execute"))

        with (
            patch(
                "hpc_alloc.commands._active_allocation",
                return_value=(self.job, transport),
            ) as active,
            patch("hpc_alloc.commands.os.execvp", execvp),
        ):
            with self.assertRaises(HostKeyChanged) as raised:
                cmd_ssh(
                    SimpleNamespace(args=["dev"], cluster=None),
                    ctx=object(),
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, failure)
        active.assert_called_once_with(
            ANY,
            self.paths,
            self.entrypoint,
            "dev",
            None,
        )
        transport.require_node.assert_called_once_with("hpc-grace.dev")
        execvp.assert_not_called()

    def test_cmd_sync_propagates_compute_host_key_failure_without_rsync(self) -> None:
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace.dev")
        transport = SimpleNamespace(require_node=Mock(side_effect=failure))
        run = Mock(side_effect=AssertionError("rsync must not execute"))
        args = SimpleNamespace(
            target="dev",
            cluster=None,
            src="./source",
            dst="~/destination",
            pull=False,
            delete=False,
        )

        with (
            patch(
                "hpc_alloc.commands._active_allocation",
                return_value=(self.job, transport),
            ) as active,
            patch("hpc_alloc.commands.subprocess.run", run),
        ):
            with self.assertRaises(HostKeyChanged) as raised:
                cmd_sync(
                    args,
                    ctx=object(),
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, failure)
        active.assert_called_once_with(
            ANY,
            self.paths,
            self.entrypoint,
            "dev",
            None,
        )
        transport.require_node.assert_called_once_with("hpc-grace.dev")
        run.assert_not_called()

    def test_cmd_connect_reports_every_node_then_raises_first_host_key_failure(self) -> None:
        failure = HostKeyChanged("SSH host-key verification failed for hpc-grace.bad")

        def require_node(alias: str) -> None:
            if alias == "hpc-grace.bad":
                raise failure
            if alias == "hpc-grace.offline":
                raise TransportLost("network unavailable")

        transport = SimpleNamespace(
            bootstrap=Mock(),
            run=Mock(
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout_text="login01\n",
                    stderr="",
                )
            ),
            require_node=Mock(side_effect=require_node),
        )
        jobs = [
            SimpleNamespace(
                cluster="grace",
                kind=JobKind.ALLOCATION,
                logical_name="bad",
                current_node="node01",
            ),
            SimpleNamespace(
                cluster="grace",
                kind=JobKind.ALLOCATION,
                logical_name="good",
                current_node="node02",
            ),
            SimpleNamespace(
                cluster="grace",
                kind=JobKind.ALLOCATION,
                logical_name="offline",
                current_node="node03",
            ),
        ]
        context = SimpleNamespace(
            config=SimpleNamespace(resolve_cluster=lambda _cluster: "grace"),
            state=SimpleNamespace(list_jobs=Mock(return_value=jobs)),
        )

        with (
            patch("hpc_alloc.commands._services", return_value=(transport, object())),
            patch("hpc_alloc.commands.info") as report,
        ):
            with self.assertRaises(HostKeyChanged) as raised:
                cmd_connect(
                    SimpleNamespace(cluster=None, reset=False, push=False),
                    ctx=context,
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, failure)
        self.assertEqual(
            transport.require_node.call_args_list,
            [
                call("hpc-grace.bad"),
                call("hpc-grace.good"),
                call("hpc-grace.offline"),
            ],
        )
        report.assert_any_call("node node01 ('bad'): host-key")
        report.assert_any_call("node node02 ('good'): ok")
        report.assert_any_call("node node03 ('offline'): network")
        context.state.list_jobs.assert_called_once_with(include_final=False)

    def test_cmd_down_all_stops_on_access_failure_after_projecting_prior_changes(self) -> None:
        for failure in (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        ):
            with self.subTest(failure=type(failure).__name__):
                jobs = [
                    SimpleNamespace(
                        cluster="grace",
                        kind=JobKind.ALLOCATION,
                        logical_name=name,
                        job_id=job_id,
                    )
                    for name, job_id in (
                        ("released", "101"),
                        ("blocked", "102"),
                        ("untouched", "103"),
                    )
                ]
                context = SimpleNamespace(
                    state=SimpleNamespace(list_jobs=Mock(return_value=jobs))
                )
                transport = SimpleNamespace(bootstrap=Mock())
                client = object()
                events: list[str] = []

                def cancel(
                    _ctx: object,
                    _paths: object,
                    _client: object,
                    job: object,
                ) -> object:
                    name = str(getattr(job, "logical_name"))
                    events.append(f"cancel:{name}")
                    if name == "released":
                        return SimpleNamespace(
                            status=SimpleNamespace(value="CANCELLED")
                        )
                    if name == "blocked":
                        raise failure
                    raise AssertionError("later allocation must not be attempted")

                def sync(_ctx: object, _paths: object) -> bool:
                    events.append("sync")
                    # False means the projection was already current, not that
                    # synchronization failed.
                    return False

                def report(message: str) -> None:
                    events.append(f"report:{message}")

                with (
                    patch(
                        "hpc_alloc.commands._services",
                        return_value=(transport, client),
                    ) as services,
                    patch("hpc_alloc.commands._cancel_record", side_effect=cancel),
                    patch(
                        "hpc_alloc.commands._sync_ssh_projection",
                        side_effect=sync,
                    ) as project,
                    patch("hpc_alloc.commands.info", side_effect=report),
                ):
                    with self.assertRaises(type(failure)) as raised:
                        cmd_down(
                            SimpleNamespace(all=True, target=None, cluster=None),
                            ctx=context,
                            paths=self.paths,
                            entrypoint=self.entrypoint,
                        )

                self.assertIs(raised.exception, failure)
                self.assertEqual(
                    events,
                    [
                        "cancel:released",
                        "cancel:blocked",
                        f"report:could not release grace:blocked: {failure}",
                        "sync",
                        "report:cancelled allocation grace:released (101)",
                    ],
                )
                self.assertEqual(services.call_count, 2)
                self.assertEqual(transport.bootstrap.call_count, 2)
                project.assert_called_once_with(context, self.paths)

    def test_cmd_down_all_does_not_report_prior_success_when_access_unwind_projection_fails(
        self,
    ) -> None:
        for failure in (
            HostKeyChanged("SSH host-key verification failed for hpc-grace"),
            AuthRequired("SSH authentication failed for hpc-grace"),
        ):
            with self.subTest(failure=type(failure).__name__):
                jobs = [
                    SimpleNamespace(
                        cluster="grace",
                        kind=JobKind.ALLOCATION,
                        logical_name=name,
                        job_id=job_id,
                    )
                    for name, job_id in (
                        ("released", "101"),
                        ("blocked", "102"),
                        ("untouched", "103"),
                    )
                ]
                context = SimpleNamespace(
                    state=SimpleNamespace(list_jobs=Mock(return_value=jobs))
                )
                transport = SimpleNamespace(bootstrap=Mock())
                client = object()
                projection_failure = ConfigInvalid("managed SSH projection failed")
                events: list[str] = []

                def cancel(
                    _ctx: object,
                    _paths: object,
                    _client: object,
                    job: object,
                ) -> object:
                    name = str(getattr(job, "logical_name"))
                    events.append(f"cancel:{name}")
                    if name == "released":
                        return SimpleNamespace(
                            status=SimpleNamespace(value="CANCELLED")
                        )
                    if name == "blocked":
                        raise failure
                    raise AssertionError("later allocation must not be attempted")

                def sync(_ctx: object, _paths: object) -> bool:
                    events.append("sync")
                    raise projection_failure

                def report(message: str) -> None:
                    events.append(f"report:{message}")

                with (
                    patch(
                        "hpc_alloc.commands._services",
                        return_value=(transport, client),
                    ) as services,
                    patch("hpc_alloc.commands._cancel_record", side_effect=cancel),
                    patch(
                        "hpc_alloc.commands._sync_ssh_projection",
                        side_effect=sync,
                    ) as project,
                    patch("hpc_alloc.commands.info", side_effect=report),
                ):
                    with self.assertRaises(type(failure)) as raised:
                        cmd_down(
                            SimpleNamespace(all=True, target=None, cluster=None),
                            ctx=context,
                            paths=self.paths,
                            entrypoint=self.entrypoint,
                        )

                self.assertIs(raised.exception, failure)
                self.assertEqual(
                    events,
                    [
                        "cancel:released",
                        "cancel:blocked",
                        f"report:could not release grace:blocked: {failure}",
                        "sync",
                        "report:warning: could not synchronize managed SSH config "
                        "while recovering from another error "
                        f"({projection_failure})",
                    ],
                )
                self.assertFalse(
                    any("cancelled allocation" in event for event in events)
                )
                self.assertEqual(services.call_count, 2)
                self.assertEqual(transport.bootstrap.call_count, 2)
                project.assert_called_once_with(context, self.paths)

    def test_cmd_cancel_projects_once_without_masking_cancellation_failure(self) -> None:
        failure = TransportLost("guarded cancellation reply was lost")
        projection_failure = HostKeyChanged("managed SSH projection failed")
        job = SimpleNamespace(
            cluster="grace",
            logical_name="dev",
            job_id="101",
        )
        context = object()
        transport = SimpleNamespace(bootstrap=Mock())

        with (
            patch("hpc_alloc.commands._resolve_managed_job", return_value=job),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, object()),
            ),
            patch("hpc_alloc.commands._cancel_record", side_effect=failure),
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                side_effect=projection_failure,
            ) as project,
            patch("hpc_alloc.commands.info") as report,
        ):
            with self.assertRaises(TransportLost) as raised:
                cmd_cancel(
                    SimpleNamespace(target="101", cluster=None),
                    ctx=context,
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, failure)
        transport.bootstrap.assert_called_once_with("grace")
        project.assert_called_once_with(context, self.paths)
        self.assertIn("managed SSH projection failed", report.call_args.args[0])

    def test_single_target_down_projects_once_without_masking_failure(self) -> None:
        failure = TransportLost("guarded cancellation reply was lost")
        projection_failure = HostKeyChanged("managed SSH projection failed")
        job = SimpleNamespace(
            cluster="grace",
            kind=JobKind.ALLOCATION,
            logical_name="dev",
            job_id="101",
        )
        context = object()
        transport = SimpleNamespace(bootstrap=Mock())

        with (
            patch("hpc_alloc.commands._resolve_managed_job", return_value=job),
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, object()),
            ),
            patch("hpc_alloc.commands._cancel_record", side_effect=failure),
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                side_effect=projection_failure,
            ) as project,
            patch("hpc_alloc.commands.info") as report,
        ):
            with self.assertRaises(TransportLost) as raised:
                cmd_down(
                    SimpleNamespace(all=False, target="dev", cluster=None),
                    ctx=context,
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, failure)
        transport.bootstrap.assert_called_once_with("grace")
        project.assert_called_once_with(context, self.paths)
        self.assertIn("managed SSH projection failed", report.call_args.args[0])

    def test_down_does_not_announce_success_before_projection_succeeds(self) -> None:
        job = SimpleNamespace(
            cluster="grace",
            kind=JobKind.ALLOCATION,
            logical_name="dev",
            job_id="101",
        )
        context = object()
        transport = SimpleNamespace(bootstrap=Mock())
        projection_failure = ConfigInvalid("managed SSH projection failed")

        with (
            patch("hpc_alloc.commands._resolve_managed_job", return_value=job),
            patch("hpc_alloc.commands._services", return_value=(transport, object())),
            patch(
                "hpc_alloc.commands._cancel_record",
                return_value=SimpleNamespace(status=SimpleNamespace(value="CANCELLED")),
            ),
            patch(
                "hpc_alloc.commands._sync_ssh_projection",
                side_effect=projection_failure,
            ),
            patch("hpc_alloc.commands.info") as report,
        ):
            with self.assertRaises(ConfigInvalid) as raised:
                cmd_down(
                    SimpleNamespace(all=False, target="dev", cluster=None),
                    ctx=context,
                    paths=self.paths,
                    entrypoint=self.entrypoint,
                )

        self.assertIs(raised.exception, projection_failure)
        self.assertFalse(
            any("cancelled allocation" in str(call_.args[0]).lower() for call_ in report.call_args_list)
        )

    def test_cmd_down_all_routes_each_allocation_to_its_own_cluster(self) -> None:
        jobs = [
            SimpleNamespace(
                cluster=cluster,
                kind=JobKind.ALLOCATION,
                logical_name=name,
                job_id=job_id,
            )
            for cluster, name, job_id in (
                ("grace", "dev", "101"),
                ("bouchet", "gpu", "202"),
            )
        ]
        context = SimpleNamespace(
            state=SimpleNamespace(list_jobs=Mock(return_value=jobs))
        )
        transports = {
            cluster: SimpleNamespace(bootstrap=Mock())
            for cluster in ("grace", "bouchet")
        }

        def services(
            _ctx: object,
            _paths: object,
            _entrypoint: object,
            cluster: str,
        ) -> tuple[object, object]:
            return transports[cluster], SimpleNamespace(cluster=cluster)

        with (
            patch("hpc_alloc.commands._services", side_effect=services) as service_factory,
            patch(
                "hpc_alloc.commands._cancel_record",
                return_value=SimpleNamespace(
                    status=SimpleNamespace(value="CANCELLED")
                ),
            ) as cancel,
            patch("hpc_alloc.commands._sync_ssh_projection") as project,
        ):
            result = cmd_down(
                SimpleNamespace(all=True, target=None, cluster=None),
                ctx=context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            [call.args[3] for call in service_factory.call_args_list],
            ["grace", "bouchet"],
        )
        transports["grace"].bootstrap.assert_called_once_with("grace")
        transports["bouchet"].bootstrap.assert_called_once_with("bouchet")
        self.assertEqual([call.args[3] for call in cancel.call_args_list], jobs)
        project.assert_called_once_with(context, self.paths)

    def test_cmd_down_all_explicit_cluster_restricts_the_job_set(self) -> None:
        jobs = [
            SimpleNamespace(
                cluster=cluster,
                kind=JobKind.ALLOCATION,
                logical_name=name,
                job_id=job_id,
            )
            for cluster, name, job_id in (
                ("grace", "dev", "101"),
                ("bouchet", "gpu", "202"),
            )
        ]
        context = SimpleNamespace(
            state=SimpleNamespace(list_jobs=Mock(return_value=jobs))
        )
        transport = SimpleNamespace(bootstrap=Mock())

        with (
            patch(
                "hpc_alloc.commands._services",
                return_value=(transport, SimpleNamespace()),
            ) as service_factory,
            patch(
                "hpc_alloc.commands._cancel_record",
                return_value=SimpleNamespace(
                    status=SimpleNamespace(value="CANCELLED")
                ),
            ) as cancel,
            patch("hpc_alloc.commands._sync_ssh_projection"),
        ):
            result = cmd_down(
                SimpleNamespace(all=True, target=None, cluster="grace"),
                ctx=context,
                paths=self.paths,
                entrypoint=self.entrypoint,
            )

        self.assertEqual(result, 0)
        service_factory.assert_called_once_with(
            context,
            self.paths,
            self.entrypoint,
            "grace",
        )
        transport.bootstrap.assert_called_once_with("grace")
        cancel.assert_called_once_with(
            context,
            self.paths,
            service_factory.return_value[1],
            jobs[0],
        )


if __name__ == "__main__":
    unittest.main()
