"""Child-process driver for command-level broken-pipe regression tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hpc_alloc.commands import cmd_logs, cmd_run
from hpc_alloc.models import JobKind, JobPhase, JobRecord
from hpc_alloc.paths import AppPaths


OPERATION_ID = "a" * 32


class PipeFollower:
    """Write until the parent closes the pipe and the kernel reports EPIPE."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def follow(self) -> object:
        chunk = b"broken-pipe-regression\n" * 1024
        while True:
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()


def job() -> JobRecord:
    return JobRecord(
        operation_id=OPERATION_ID,
        cluster="grace",
        logical_name="run",
        kind=JobKind.RUN,
        owner_id="deadbeef1234",
        slurm_job_name=f"hpcalloc-v2-run-{OPERATION_ID}",
        slurm_comment=(
            f"hpc-alloc:v2:deadbeef1234:{OPERATION_ID}:laptop:run:-"
        ),
        phase=JobPhase.QUEUED,
        job_id="12345",
        updated_at="2026-07-12T00:00:00+00:00",
    )


def record_cancellation(marker: Path, *_args: object, **_kwargs: object) -> None:
    with marker.open("a", encoding="utf-8") as handle:
        handle.write("cancel\n")


def run_child(mode: str, marker: Path) -> int:
    paths = AppPaths.for_home(marker.parent)
    transport = SimpleNamespace(bootstrap=lambda *_args, **_kwargs: None)
    client = object()
    current_job = job()
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            resolve_cluster=lambda _cluster: "grace",
            resolve_option=lambda _key, _cluster, fallback=None: fallback,
        ),
        state=object(),
    )

    with (
        patch("hpc_alloc.commands._services", return_value=(transport, client)),
        patch("hpc_alloc.commands._remote_home", return_value="/home/me"),
        patch("hpc_alloc.commands._submit_job", return_value=current_job),
        patch("hpc_alloc.commands._resolve_managed_job", return_value=current_job),
        patch(
            "hpc_alloc.commands._cancel_record",
            side_effect=lambda *args, **kwargs: record_cancellation(
                marker, *args, **kwargs
            ),
        ),
        patch("hpc_alloc.streaming.LogFollower", PipeFollower),
    ):
        if os.environ.get("HPC_ALLOC_BROKEN_PIPE_GATE") == "1":
            if sys.stdin.buffer.read(1) != b"1":
                raise RuntimeError("broken-pipe child did not receive its start signal")
        if mode.startswith("run"):
            return cmd_run(
                SimpleNamespace(
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
                    detach=mode == "run-detach",
                ),
                ctx=ctx,
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
        if mode.startswith("logs"):
            return cmd_logs(
                SimpleNamespace(
                    target=f"grace:@{OPERATION_ID}",
                    cluster=None,
                    lines=100,
                    follow=True,
                ),
                ctx=ctx,
                paths=paths,
                entrypoint=Path("/tmp/hpc-alloc"),
            )
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(run_child(sys.argv[1], Path(sys.argv[2])))
