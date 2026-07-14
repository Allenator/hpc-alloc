"""Command-line entry point for hpc-alloc v2."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Sequence

from .errors import HpcAllocError
from .output import neutralize_stderr, neutralize_stdout


def add_cluster_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster", help="configured cluster name")


def add_resource_flags(parser: argparse.ArgumentParser) -> None:
    add_cluster_flag(parser)
    parser.add_argument("-p", "--partition", help="Slurm partition")
    parser.add_argument("-t", "--time", help="walltime in Slurm format")
    parser.add_argument("-c", "--cpus", type=int, help="CPUs per task")
    parser.add_argument("--mem", help="memory per node, e.g. 64G")
    parser.add_argument("-G", "--gpus", help="GPU request, e.g. h200:1")
    parser.add_argument("-C", "--constraint", help="Slurm node constraint")
    parser.add_argument(
        "--dry-run", action="store_true", help="print scheduler submission command only"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpc-alloc",
        description="Allocate and manage YCRC compute nodes safely over SSH.",
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    setup = sub.add_parser("setup", help="write v2 configuration and initialize state")
    setup.add_argument("--netid", help="Yale NetID")
    setup.add_argument("--cluster", default="bouchet", help="cluster name")
    setup.add_argument("--host", help="login host (default: <cluster>.ycrc.yale.edu)")
    setup.add_argument("--force", action="store_true", help="replace the v2 config")
    setup.add_argument(
        "--identity-file",
        help="SSH private key to authenticate with (default: keep the configured one)",
    )

    config = sub.add_parser("config", help="show validated effective configuration")
    add_cluster_flag(config)
    config.add_argument("--json", action="store_true")

    connect = sub.add_parser("connect", help="establish and health-check SSH connections")
    add_cluster_flag(connect)
    connect.add_argument("--reset", action="store_true", help="tear down masters first")
    connect.add_argument("--push", action="store_true", help="authenticate with one Duo push")

    up = sub.add_parser("up", help="submit a persistent sleeper allocation")
    up.add_argument("--name", default="dev", help="logical allocation name")
    add_resource_flags(up)
    up.add_argument("--idle-timeout", type=int, metavar="MIN")
    up.add_argument("--no-wait", action="store_true")
    up.add_argument("--wait-timeout", type=int, default=1800)

    run = sub.add_parser("run", help="submit a short-lived batch command")
    add_resource_flags(run)
    run.add_argument("--chdir", help="remote working directory")
    run.add_argument("--detach", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD")

    status = sub.add_parser("status", help="show managed and discovered jobs")
    status.add_argument("--json", action="store_true")

    why = sub.add_parser("why", help="diagnose a job")
    why.add_argument("target", nargs="?", help="name, job ID, @operation, or qualified selector")
    add_cluster_flag(why)
    why.add_argument("--json", action="store_true")

    logs = sub.add_parser("logs", help="show or follow a job log")
    logs.add_argument("target", help="name, job ID, @operation, or qualified selector")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("-n", "--lines", type=int, default=100)
    add_cluster_flag(logs)

    cancel = sub.add_parser("cancel", help="cancel an exact v2-owned job")
    cancel.add_argument("target", help="job ID, @operation, or cluster-qualified form")
    add_cluster_flag(cancel)

    down = sub.add_parser("down", help="cancel one or all managed allocations")
    down.add_argument(
        "target",
        nargs="?",
        help="name, job ID, @operation, or cluster-qualified selector",
    )
    down.add_argument("--all", action="store_true")
    add_cluster_flag(down)

    ssh = sub.add_parser("ssh", help="open a shell or run a command on an allocation")
    ssh.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="[NAME|JOBID|@OPERATION] [-- CMD...]",
    )
    add_cluster_flag(ssh)

    sync = sub.add_parser("sync", help="rsync files to or from an allocation")
    sync.add_argument(
        "target", help="name, job ID, @operation, or cluster-qualified selector"
    )
    sync.add_argument("src")
    sync.add_argument("dst")
    sync.add_argument("--pull", action="store_true")
    sync.add_argument("--delete", action="store_true")
    add_cluster_flag(sync)

    avail = sub.add_parser("avail", help="show free CPUs and GPUs")
    add_cluster_flag(avail)
    avail.add_argument("-p", "--partition")
    avail.add_argument("--json", action="store_true")

    partitions = sub.add_parser("partitions", help="list Slurm partitions")
    add_cluster_flag(partitions)
    partitions.add_argument("--json", action="store_true")

    recover = sub.add_parser("recover", help="reconcile an unresolved remote mutation")
    recover.add_argument("operation_id", nargs="?")
    recover.add_argument("--abandon", action="store_true")
    recover.add_argument("--yes", action="store_true")
    add_cluster_flag(recover)

    return parser


def _signal_as_interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def _install_interrupt_signals() -> None:
    """Route every terminating signal through the KeyboardInterrupt path.

    SIGHUP is what a closed terminal delivers.  Left at its default disposition
    it kills the process outright -- no ``except``, no ``finally`` -- so a
    foreground ``run`` would skip its cancellation path and leak the detached
    Slurm job for its whole walltime, even though Ctrl-C and SIGTERM both
    release it.
    """

    signal.signal(signal.SIGTERM, _signal_as_interrupt)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_as_interrupt)


def main(argv: Sequence[str] | None = None, *, entrypoint: Path | None = None) -> int:
    """Parse and dispatch one CLI invocation, returning its command status.

    Argparse usage failures raise SystemExit before dispatch and the application
    error boundary.
    """

    from .commands import dispatch

    args = build_parser().parse_args(argv)
    _install_interrupt_signals()
    try:
        return int(dispatch(args, entrypoint=entrypoint or Path(sys.argv[0]).resolve()) or 0)
    except KeyboardInterrupt:
        try:
            print(file=sys.stderr)
        except BrokenPipeError:
            neutralize_stderr()
        return 130
    except BrokenPipeError:
        neutralize_stdout()
        neutralize_stderr()
        # Context-aware streaming paths handle this themselves. This is only a
        # final no-traceback fallback for non-job payload commands.
        return 141
    except HpcAllocError as exc:
        try:
            print(f"hpc-alloc: error: {exc}", file=sys.stderr, flush=True)
        except BrokenPipeError:
            neutralize_stdout()
            neutralize_stderr()
        return exc.exit_code
