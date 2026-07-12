from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hpc_alloc.commands import cmd_run
from hpc_alloc.errors import TransportLost
from hpc_alloc.models import JobKind
from hpc_alloc.paths import AppPaths


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


if __name__ == "__main__":
    unittest.main()
