from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, call, patch

from hpc_alloc.commands import cmd_connect, cmd_down, cmd_ssh, cmd_sync
from hpc_alloc.errors import AuthRequired, HostKeyChanged, TransportLost
from hpc_alloc.models import JobKind
from hpc_alloc.paths import AppPaths


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

                def cancel(_ctx: object, _client: object, job: object) -> object:
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
                    return True

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
                    ["cancel:released", "cancel:blocked", "sync"],
                )
                self.assertEqual(services.call_count, 2)
                self.assertEqual(transport.bootstrap.call_count, 2)
                project.assert_called_once_with(context, self.paths)

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
        self.assertEqual([call.args[2] for call in cancel.call_args_list], jobs)
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
        cancel.assert_called_once_with(context, service_factory.return_value[1], jobs[0])


if __name__ == "__main__":
    unittest.main()
