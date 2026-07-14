"""Command orchestration for hpc-alloc v2.

Services in this module may format user output, but transport, Slurm parsing,
lifecycle policy, and durable transactions remain in their dedicated modules.
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from .config import Config
from .errors import (
    EXIT_SUBMITTED_NOT_READY,
    AmbiguousSubmission,
    AuthRequired,
    ConfigInvalid,
    HpcAllocError,
    HostKeyChanged,
    IdentityMismatch,
    LifecycleRevisionConflict,
    LocalToolUnavailable,
    OperationBusy,
    RecordNotFound,
    StateConflict,
    TransportLost,
)
from .locking import (
    SETUP_LOCK_TIMEOUT_SECONDS,
    configuration_scope_lock,
    operation_scope_lock,
)
from .models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    OperationKind,
    OperationPhase,
)
from .ownership import normalize_host_label, parse_tag, slurm_job_name
from .output import neutralize_stderr, neutralize_stdout
from .paths import AppPaths
from .retry import PollBackoff, RetryBudget
from .selectors import SelectorKind, canonical_job_selector, parse_selector, unique_job
from .ssh_config import (
    allocation_alias,
    atomic_write_600,
    ensure_include,
    sync_managed_config,
)


DEFAULT_PARTITION = "day"
DEFAULT_GPU_PARTITION = "gpu"
DEFAULT_TIME = "4:00:00"
DEFAULT_CPUS = 2
DEFAULT_GPU_IDLE_MINUTES = 30
REMOTE_LOG_DIR = ".hpc-alloc"
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}\Z")
# Characters a remote POSIX shell cannot expand, split on, or execute.  A leading
# `~` is deliberately allowed: the documented `hpc-alloc sync … '~/project'` form
# relies on the remote shell expanding it.
_REMOTE_SYNC_PATH = re.compile(r"[A-Za-z0-9_@%+=:,./~-]+")
GPU_RE = re.compile(r"(?:[A-Za-z0-9][A-Za-z0-9_.-]*:)?[1-9][0-9]*\Z")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _is_job_id(value: str) -> bool:
    return re.fullmatch(r"[0-9]+", value) is not None


def info(message: str) -> None:
    print(f"hpc-alloc: {message}", file=sys.stderr, flush=True)


def _pipe_aware_info(message: str) -> None:
    """Emit a diagnostic and neutralize fd 2 if that write proves it broken."""

    try:
        info(message)
    except BrokenPipeError:
        neutralize_stderr()
        raise


def _best_effort_info(message: str) -> None:
    """Emit a diagnostic without letting a broken output pipe replace policy."""

    try:
        _pipe_aware_info(message)
    except BrokenPipeError:
        pass


def _secondary_info(message: str) -> None:
    """Report secondary context without replacing an in-flight primary error."""

    try:
        _best_effort_info(message)
    except BaseException:
        pass


def _durable_job_phase(ctx: Any, job: Any) -> JobPhase | None:
    """Best-effort phase lookup used only to qualify failure guidance."""

    try:
        return ctx.state.get_job(job.operation_id).phase
    except BaseException:
        phase = getattr(job, "phase", None)
        try:
            return JobPhase(phase)
        except (TypeError, ValueError):
            return None


def _report_up_interrupt(ctx: Any, job: Any) -> None:
    selector = canonical_job_selector(job)
    if _durable_job_phase(ctx, job) is JobPhase.FINAL:
        _secondary_info(
            f"allocation wait interrupted after {selector} reached a durable final "
            f"state; inspect it with `hpc-alloc status` or `hpc-alloc why {selector}`"
        )
        return
    _secondary_info(
        f"allocation wait interrupted; {selector} was not cancelled and may remain "
        f"queued or running; check `hpc-alloc status` and release it with "
        f"`hpc-alloc down {selector}`"
    )


def _report_run_follow_failure(ctx: Any, job: Any) -> None:
    selector = canonical_job_selector(job)
    if _durable_job_phase(ctx, job) is JobPhase.FINAL:
        _secondary_info(
            f"foreground follow stopped after {selector} reached a durable final state; "
            f"inspect `hpc-alloc logs {selector}` or `hpc-alloc why {selector}`"
        )
        return
    _secondary_info(
        f"foreground follow stopped; {selector} was not cancelled and may continue; "
        f"reattach with `hpc-alloc logs {selector} -f` or cancel it with "
        f"`hpc-alloc cancel {selector}`"
    )


def machine_host() -> str:
    return normalize_host_label(platform.node())


def _sync_ssh_projection_repository(repository: Any, paths: AppPaths) -> bool:
    """Project aliases and best-effort retire masters from the old projection."""

    from .ssh import retire_compute_masters

    warnings: list[str] = []

    def retire(retirement: Any) -> None:
        try:
            warnings.extend(
                retire_compute_masters(retirement, ssh_dir=paths.ssh_dir)
            )
        except Exception as exc:
            # Local cleanup must not prevent authoritative lifecycle state
            # from removing or retargeting an obsolete alias.  Asynchronous
            # interrupts still propagate so the old projection remains
            # available for the next invocation to repair.
            warnings.append(f"could not retire obsolete SSH masters: {exc}")

    changed = sync_managed_config(
        config_path=paths.config_file,
        repository=repository,
        managed_path=paths.managed_ssh_config,
        lock_path=paths.ssh_config_lock,
        known_hosts=paths.known_hosts,
        before_replace=retire,
    )
    for warning in warnings:
        _best_effort_info(f"warning: {warning}")
    return changed


def _sync_ssh_projection(ctx: Any, paths: AppPaths) -> bool:
    return _sync_ssh_projection_repository(ctx.state, paths)


@dataclass
class _SshProjectionResult:
    synchronized: bool = False


@contextmanager
def _ssh_projection_scope(
    ctx: Any, paths: AppPaths
) -> Iterator[_SshProjectionResult]:
    """Synchronize derived SSH aliases after success or while unwinding.

    A projection error is authoritative on the successful path.  During
    exception unwinding it is only a secondary local-repair failure, so report
    it without replacing the exception that interrupted reconciliation.
    """

    result = _SshProjectionResult()
    try:
        yield result
    except BaseException:
        try:
            _sync_ssh_projection(ctx, paths)
            result.synchronized = True
        except BaseException as projection_error:
            try:
                info(
                    "warning: could not synchronize managed SSH config while "
                    f"recovering from another error ({projection_error})"
                )
            except BaseException:
                pass
        raise
    else:
        _sync_ssh_projection(ctx, paths)
        result.synchronized = True


def _submission_recovery_guidance(
    operation_id: str, *, job_id: str | None = None
) -> str:
    known_job = f" (trusted Slurm job ID {job_id})" if job_id is not None else ""
    return (
        f"submission {operation_id} may have committed{known_job}; do not resubmit; "
        f"run `hpc-alloc recover {operation_id}`"
    )


def _cancellation_recovery_guidance(
    operation_id: str, *, ambiguous: bool = True
) -> str:
    state = "is ambiguous" if ambiguous else "may remain unresolved after interruption"
    return (
        f"cancellation {operation_id} {state}; run "
        f"`hpc-alloc recover {operation_id}`"
    )


def _best_effort_cancel_interrupt_reconciliation(
    ctx: Any,
    *,
    operation_id: str,
    target_job_operation_id: str,
    known_intent: bool,
) -> None:
    """Close an undispatched interrupt or retain an actionable recovery ID.

    This runs while the cancellation's exclusive operation lock is still held.
    Durable phase is authoritative because ``mark_cancel_dispatching`` commits
    AMBIGUOUS immediately before the guarded one-shot scheduler mutation.
    Every failure here is secondary to the original asynchronous interrupt.
    """

    def report(*, ambiguous: bool) -> None:
        try:
            _best_effort_info(
                _cancellation_recovery_guidance(
                    operation_id,
                    ambiguous=ambiguous,
                )
            )
        except BaseException:
            pass

    def matching_cancel(operation: Any) -> bool:
        return (
            operation.kind is OperationKind.CANCEL
            and operation.target_job_operation_id == target_job_operation_id
        )

    try:
        operation = ctx.state.get_operation(operation_id)
    except RecordNotFound:
        # An interrupt inside begin_cancel's transaction rolled the reservation
        # back.  Absence while its lock is held proves there is no intent to
        # recover and guarded dispatch was never entered.
        return
    except BaseException:
        if known_intent:
            report(ambiguous=False)
        return

    # Never mutate or advertise recovery for an unexpected durable identity.
    if not matching_cancel(operation):
        return

    if operation.phase is OperationPhase.CANCEL_PENDING:
        try:
            ctx.state.fail_cancel_operation(
                operation_id,
                "cancellation was interrupted before guarded dispatch",
            )
            return
        except BaseException:
            # The close may have committed before its caller was interrupted.
            # Re-read before telling the user that recovery is still required.
            try:
                operation = ctx.state.get_operation(operation_id)
            except BaseException:
                report(ambiguous=False)
                return
            if not matching_cancel(operation):
                return
            if operation.phase is OperationPhase.CANCEL_PENDING:
                report(ambiguous=False)
            elif operation.phase is OperationPhase.AMBIGUOUS:
                report(ambiguous=True)
            return

    if operation.phase is OperationPhase.AMBIGUOUS:
        report(ambiguous=True)


def _best_effort_mark_submission_ambiguous(
    ctx: Any, operation_id: str, detail: str
) -> None:
    """Retain recovery intent without ever replacing the primary failure."""

    try:
        ctx.state.mark_submission_ambiguous(operation_id, detail)
    except BaseException:
        # The operation remains recoverable from PREPARED if this write could
        # not commit.  It may also already be ACKNOWLEDGED if an interruption
        # arrived after the acknowledgement transaction committed.
        pass


def _toml_string(value: str) -> str:
    # TOML basic strings use JSON-compatible escaping for this validated subset.
    return json.dumps(value, ensure_ascii=False)


def _render_initial_config(netid: str, cluster: str, host: str, identity_file: str | None) -> str:
    identity_line = (
        f"identity_file = {_toml_string(identity_file)}\n" if identity_file else ""
    )
    return (
        "# hpc-alloc v2 configuration (Python 3.11+).\n\n"
        "[identity]\n"
        f"netid = {_toml_string(netid)}\n\n"
        "[ssh]\n"
        f"{identity_line}\n"
        "[defaults]\n"
        f"cluster = {_toml_string(cluster)}\n"
        "# partition = \"day\"\n"
        "# gpu_partition = \"gpu\"\n"
        "# time = \"4:00:00\"\n"
        "# cpus = 2\n"
        "# mem = \"16G\"\n"
        "# idle_timeout = 30\n\n"
        f"[cluster.{cluster}]\n"
        f"host = {_toml_string(host)}\n"
    )


def _validated_initial_config(
    netid: str,
    cluster: str,
    host: str,
    identity_file: str | None,
) -> tuple[str, Config]:
    """Render and parse a setup candidate without touching application state."""

    text = _render_initial_config(netid, cluster, host, identity_file)
    with tempfile.TemporaryDirectory(prefix="hpc-alloc-setup-") as directory:
        staged = Path(directory) / "config.toml"
        atomic_write_600(staged, text)
        return text, Config.load(staged)


def _setup_blocker_ids(jobs: list[Any], operations: list[Any]) -> str:
    identities = sorted(
        {job.operation_id for job in jobs}
        | {operation.operation_id for operation in operations}
    )
    return ", ".join(identities)


def _setup_scope_conflict(
    reason: str,
    jobs: list[Any],
    operations: list[Any],
) -> StateConflict:
    recovery = ""
    if operations:
        commands = ", ".join(
            f"`hpc-alloc recover {operation.operation_id}`"
            for operation in operations
        )
        recovery = f" Resolve unresolved operations with {commands}."
    return StateConflict(
        f"forced setup cannot {reason} while durable work remains; "
        f"blocking operation IDs: {_setup_blocker_ids(jobs, operations)}. "
        "Run `hpc-alloc status`, finish or cancel non-final jobs, and retry."
        f"{recovery}"
    )


def _setup_host_identity(host: str) -> tuple[str, str]:
    """Canonicalize one already-validated host for setup-scope comparison."""

    try:
        # Config parsing has already removed matching brackets.  Parse before
        # DNS normalization so an IPv6 zone identifier remains opaque and
        # case-sensitive, including a trailing dot that belongs to the zone.
        return ("ip", str(ipaddress.ip_address(host)))
    except ValueError:
        return ("dns", host.removesuffix(".").lower())


def _validate_setup_scope(
    prior: Config | None,
    candidate: Config,
    jobs: list[Any],
    operations: list[Any],
) -> None:
    """Reject an unprovable or changed scope while durable work remains."""

    if not jobs and not operations:
        return
    if prior is None:
        raise _setup_scope_conflict(
            "replace a missing or invalid prior configuration",
            jobs,
            operations,
        )
    if prior.identity.netid != candidate.identity.netid:
        raise _setup_scope_conflict("change the configured NetID", jobs, operations)

    referenced_clusters = sorted(
        {job.cluster for job in jobs} | {operation.cluster for operation in operations}
    )
    for cluster in referenced_clusters:
        old = prior.clusters.get(cluster)
        new = candidate.clusters.get(cluster)
        if old is None:
            raise _setup_scope_conflict(
                f"replace cluster {cluster!r} whose prior host cannot be proven",
                jobs,
                operations,
            )
        if new is None:
            raise _setup_scope_conflict(
                f"remove blocker-referenced cluster {cluster!r}",
                jobs,
                operations,
            )
        if _setup_host_identity(old.host) != _setup_host_identity(new.host):
            raise _setup_scope_conflict(
                f"change the host for blocker-referenced cluster {cluster!r}",
                jobs,
                operations,
            )


def _identity_key_pair(paths: AppPaths, identity_file: str) -> tuple[Path, Path]:
    """Map a configured ``identity_file`` to its (private, public) key paths."""

    if identity_file.startswith("~/"):
        private = paths.home / identity_file[2:]
    else:
        private = Path(identity_file)
    return private, Path(f"{private}.pub")


def _pinned_ssh_key(paths: AppPaths, identity_file: str) -> tuple[Path, str]:
    """Resolve an explicitly chosen key, refusing to silently substitute another.

    ``IdentitiesOnly yes`` means the configured key is the *only* one OpenSSH
    will offer, so quietly falling back to a different one does not degrade
    gracefully -- it produces an authentication failure the user cannot diagnose,
    against a cluster where the original key is the one actually registered.
    """

    private, public = _identity_key_pair(paths, identity_file)
    if private.is_file() and public.is_file():
        return public, identity_file
    raise ConfigInvalid(
        f"SSH key {identity_file} is missing its "
        f"{'private' if public.is_file() else 'public'} half "
        f"(expected {private} and {public}); restore it, or choose another key "
        "with `hpc-alloc setup --force --identity-file PATH`"
    )


def _existing_ssh_key(
    paths: AppPaths, preferred: str | None = None
) -> tuple[Path, str] | None:
    """Choose the key setup should authenticate with.

    An explicit ``--identity-file``, or the key already recorded in the config,
    always wins over the standard-name probe below.  The probe only knows four
    hard-coded filenames, so a key named anything else -- ``~/.ssh/id_yale``,
    say, which is the one actually registered with the cluster -- was invisible
    to it: `setup --force`, the documented way to fix a NetID typo, silently
    repointed the config at whichever standard-named key happened to exist, and
    because ``IdentitiesOnly yes`` offers only that key, every later command
    failed to authenticate with no way to see why.
    """

    if preferred is not None:
        return _pinned_ssh_key(paths, preferred)
    for filename in (
        "id_ed25519_hpc_alloc.pub",
        "id_ed25519.pub",
        "id_rsa.pub",
        "id_ecdsa.pub",
    ):
        public = paths.ssh_dir / filename
        private = public.with_suffix("")
        if public.is_file() and private.is_file():
            return public, "~/.ssh/" + filename.removesuffix(".pub")
    return None


def _find_or_create_ssh_key(
    paths: AppPaths, preferred: str | None = None
) -> tuple[Path, str]:
    existing = _existing_ssh_key(paths, preferred)
    if existing is not None:
        return existing
    try:
        paths.ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        paths.ssh_dir.chmod(0o700)
    except OSError as exc:
        raise ConfigInvalid(f"cannot prepare {paths.ssh_dir}: {exc}") from exc
    private = paths.ssh_dir / "id_ed25519_hpc_alloc"
    if private.exists() or private.with_suffix(".pub").exists():
        raise ConfigInvalid(
            f"incomplete SSH key pair at {private}; repair or remove both key files"
        )
    info("no SSH key found — generating an ed25519 key")
    command = ["ssh-keygen", "-t", "ed25519", "-f", str(private)]
    if not sys.stdin.isatty():
        command += ["-N", ""]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise ConfigInvalid("ssh-keygen is required for setup but was not found") from exc
    except subprocess.CalledProcessError as exc:
        raise ConfigInvalid(f"ssh-keygen failed with exit status {exc.returncode}") from exc
    except OSError as exc:
        raise ConfigInvalid(f"cannot execute ssh-keygen: {exc}") from exc
    return private.with_suffix(".pub"), "~/.ssh/id_ed25519_hpc_alloc"


def cmd_setup(args: Any, *, paths: AppPaths, entrypoint: Path) -> int:
    from .state import StateRepository
    from .ssh_config import resolve_user_ssh_config

    netid = args.netid
    if not netid and sys.stdin.isatty():
        netid = input("Yale NetID: ").strip()
    if not netid:
        raise ConfigInvalid("NetID required: hpc-alloc setup --netid YOUR_NETID")
    cluster = args.cluster
    if not NAME_RE.fullmatch(cluster):
        raise ConfigInvalid(f"invalid cluster name {cluster!r}")
    host = args.host or f"{cluster}.ycrc.yale.edu"

    # Candidate and dotfile-target validation intentionally precede the lock.
    # Invalid setup input must not create an application lock, key, or journal.
    _candidate_text, candidate = _validated_initial_config(
        netid,
        cluster,
        host,
        "~/.ssh/id_ed25519_hpc_alloc",
    )
    resolve_user_ssh_config(paths.user_ssh_config)

    # Bounded: every other command holds this lock shared for its entire
    # lifetime -- hours, for a `run` or `logs -f` -- so a blocking acquire left
    # `setup` hanging silently and indefinitely, and flock's lack of writer
    # preference let a stream of short commands starve it outright.  Fail fast
    # and say why instead.
    with configuration_scope_lock(
        paths.config_scope_lock,
        exclusive=True,
        timeout=SETUP_LOCK_TIMEOUT_SECONDS,
    ):
        # This check is authoritative: another setup may have completed while
        # this invocation was validating its candidate.
        config_exists = paths.config_file.exists()
        if config_exists and not args.force:
            raise ConfigInvalid(
                "v2 configuration already exists; pass setup --force to replace it",
                path=paths.config_file,
            )

        repository = StateRepository(paths.state_db)
        repository.initialize()
        jobs, operations = repository.snapshot_setup_scope_blockers()

        prior: Config | None = None
        if config_exists:
            try:
                prior = Config.load(paths.config_file)
            except ConfigInvalid:
                # Invalid prior input is replaceable only when no durable work
                # depends on proving that input's identity and cluster hosts.
                prior = None
        _validate_setup_scope(prior, candidate, jobs, operations)

        # Key choice is authoritative only under the same exclusive lock that
        # serializes other setup invocations.
        #
        # `--force` exists to repair a NetID or host mistake.  Re-keying is a far
        # more consequential act -- the key has to be registered with the cluster
        # before it works -- so it must be asked for explicitly, never arrive as a
        # side effect of fixing a typo.  An existing configured key therefore
        # wins over the standard-name probe.
        preferred_identity = getattr(args, "identity_file", None) or (
            prior.ssh.identity_file if prior is not None else None
        )
        existing_key = _existing_ssh_key(paths, preferred_identity)
        planned_identity = (
            existing_key[1] if existing_key else "~/.ssh/id_ed25519_hpc_alloc"
        )
        text, committed_candidate = _validated_initial_config(
            netid,
            cluster,
            host,
            planned_identity,
        )
        _validate_setup_scope(prior, committed_candidate, jobs, operations)

        public_key, identity_file = _find_or_create_ssh_key(paths, preferred_identity)
        if identity_file != planned_identity:
            raise StateConflict("SSH key selection changed during setup; rerun setup")
        try:
            encoded_key = public_key.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ConfigInvalid(f"cannot read SSH public key {public_key}: {exc}") from exc
        if not encoded_key:
            raise ConfigInvalid(f"SSH public key {public_key} is empty")

        repository.get_or_create_machine_id(machine_host())
        atomic_write_600(paths.config_file, text)
        _sync_ssh_projection_repository(repository, paths)
        ensure_include(paths.user_ssh_config)
    info(f"configured cluster {cluster!r} ({host}) for NetID {netid!r}")
    print("Your SSH public key:")
    print("  " + encoded_key)
    print("\nNext: upload it at https://sshkeys.ycrc.yale.edu/, then run hpc-alloc connect")
    return 0


def _load_context(args: Any, paths: AppPaths) -> Any:
    from .context import RuntimeContext

    context_command = args.command_name
    if context_command in {"up", "run"} and getattr(args, "dry_run", False):
        # A dry run needs authoritative config, but must not initialize or
        # mutate the journal merely to render a command.
        context_command = "dry-run"
    return RuntimeContext.load(
        command=context_command,
        explicit_cluster=getattr(args, "cluster", None),
        paths=paths,
    )


def cmd_config(args: Any, *, ctx: Any, **_kwargs: Any) -> int:
    if ctx.config is None:
        assert ctx.config_error is not None
        print(
            json.dumps({"config_file": str(ctx.paths.config_file), "error": str(ctx.config_error)}, indent=2)
            if args.json
            else f"invalid configuration: {ctx.config_error}"
        )
        return 1
    cluster = ctx.config.resolve_cluster(args.cluster)
    payload = {
        "config_file": str(ctx.config.path),
        "state_file": str(ctx.paths.state_db),
        "primary_cluster": cluster,
        "config": ctx.config.as_dict(),
        "effective": {
            key: ctx.config.resolve_option(key, cluster, fallback=fallback)
            for key, fallback in {
                "partition": DEFAULT_PARTITION,
                "gpu_partition": DEFAULT_GPU_PARTITION,
                "time": DEFAULT_TIME,
                "cpus": DEFAULT_CPUS,
                "mem": None,
                "idle_timeout": DEFAULT_GPU_IDLE_MINUTES,
            }.items()
        },
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    info(f"validated config: {ctx.config.path}; effective for {cluster!r}")
    print(f"  netid            {ctx.config.identity.netid}")
    print(f"  identity_file    {ctx.config.ssh.identity_file or '(ssh defaults)'}")
    for key, value in payload["effective"].items():
        print(f"  {key:<16} {value if value is not None else '(Slurm default)'}")
    return 0


def _services(ctx: Any, paths: AppPaths, entrypoint: Path, cluster: str | None = None) -> tuple[Any, Any]:
    from .slurm import SlurmClient
    from .ssh import SshTransport
    selected = ctx.config.resolve_cluster(cluster or getattr(ctx, "primary_cluster", None))
    # A prior projection write may have failed after authoritative config or
    # lifecycle state committed.  Repair that derived file before OpenSSH can
    # resolve either a login or allocation alias.
    _sync_ssh_projection(ctx, paths)
    # The Include is exactly as derived, and exactly as repairable, as the file
    # it points at -- and without it OpenSSH sees no managed Host block at all,
    # so every alias becomes an unresolvable hostname and the command dies at
    # exit 3 ("check the VPN").  The remedy the docs then prescribe, `hpc-alloc
    # connect`, routes through here too and would fail identically forever.  The
    # call is idempotent and does not rewrite an Include that is already present.
    ensure_include(paths.user_ssh_config)
    # The transport reads its compute aliases straight from the projection it
    # just synchronized above.  That file is what OpenSSH resolves and the only
    # place the node leases are applied, so it is the single source of truth for
    # which allocations own an alias and a ControlMaster.  Re-deriving the set
    # here from JobRecord.current_node -- which the repository nulls for every
    # non-ACTIVE phase -- produced a smaller, disagreeing set.
    transport = SshTransport(
        ctx.config,
        paths,
        entrypoint=entrypoint,
        info=info,
    )
    return transport, SlurmClient(transport, selected)


def _validate_positive(value: int | None, name: str, default: int) -> int:
    result = default if value is None else value
    if result <= 0:
        raise ConfigInvalid(f"{name} must be positive")
    return result


def _resource_values(args: Any, config: Config, cluster: str) -> dict[str, Any]:
    gpus = getattr(args, "gpus", None)
    if gpus is not None and GPU_RE.fullmatch(gpus) is None:
        raise ConfigInvalid("--gpus must be NUMBER or TYPE:NUMBER with a positive count")
    partition = getattr(args, "partition", None)
    if partition is None:
        partition = config.resolve_option(
            "gpu_partition" if gpus else "partition",
            cluster,
            fallback=DEFAULT_GPU_PARTITION if gpus else DEFAULT_PARTITION,
        )
        if gpus:
            info(f"--gpus given without --partition; using {partition!r}")
    walltime = getattr(args, "time", None)
    if walltime is None:
        walltime = config.resolve_option("time", cluster, fallback=DEFAULT_TIME)
    cpus = _validate_positive(
        getattr(args, "cpus", None),
        "--cpus",
        int(config.resolve_option("cpus", cluster, fallback=DEFAULT_CPUS)),
    )
    memory = getattr(args, "mem", None)
    if memory is None:
        memory = config.resolve_option("mem", cluster, fallback=None)
    idle = getattr(args, "idle_timeout", None)
    if idle is not None and gpus is None:
        raise ConfigInvalid("--idle-timeout applies only when --gpus is requested")
    if idle is None:
        idle = int(
            config.resolve_option(
                "idle_timeout", cluster, fallback=DEFAULT_GPU_IDLE_MINUTES
            )
        )
    if idle < 0:
        raise ConfigInvalid("--idle-timeout must be non-negative")
    resources = {
        "partition": partition,
        "time": walltime,
        "cpus": cpus,
        "mem": memory,
        "gpus": gpus,
        "constraint": getattr(args, "constraint", None),
        "idle_timeout": idle if gpus and idle else None,
        "chdir": getattr(args, "chdir", None),
    }
    for key in ("partition", "time", "cpus", "mem"):
        if resources[key] is not None:
            Config.validate_resource_override(key, resources[key])
    constraint = resources["constraint"]
    if constraint is not None and (not constraint or CONTROL_RE.search(constraint)):
        raise ConfigInvalid("--constraint must be non-empty and contain no control characters")
    chdir = resources["chdir"]
    if chdir is not None and (not chdir or CONTROL_RE.search(chdir)):
        raise ConfigInvalid("--chdir must be non-empty and contain no control characters")
    return resources


def _sleeper_command(idle_minutes: int | None) -> str:
    if not idle_minutes:
        return "sleep infinity"
    return (
        f'echo "hpc-alloc: self-releases after {idle_minutes} min of GPU idleness"; '
        "idle=0; while true; do sleep 60; "
        "u=$(nvidia-smi --query-gpu=utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null | sort -rn | head -n1); "
        'case "$u" in ""|*[!0-9]*) u=100;; esac; '
        'if [ "$u" -gt 5 ]; then idle=0; else idle=$((idle+1)); fi; '
        f'if [ "$idle" -ge {idle_minutes} ]; then '
        'echo "hpc-alloc: GPU idle timeout, releasing"; exit 0; fi; done'
    )


def _remote_command(tokens: list[str]) -> str:
    """Preserve one explicit shell string or quote a multi-token argv."""

    if not tokens:
        raise ConfigInvalid("remote command cannot be empty")
    return tokens[0] if len(tokens) == 1 else shlex.join(tokens)


def cmd_connect(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .errors import AuthRequired, HostKeyChanged
    from .ssh import AuthMode, ProbeStatus, RetryPolicy
    cluster = ctx.config.resolve_cluster(args.cluster)
    transport, _client = _services(ctx, paths, entrypoint, cluster)
    if args.reset:
        transport.heal(cluster)
        info(f"closed SSH masters for {cluster} and swept dead sockets")
    if args.push:
        status = transport.probe(cluster)
        if status != ProbeStatus.OK and transport.master_alive(cluster):
            transport.heal(cluster)
            status = transport.probe(cluster)
        if status == ProbeStatus.AUTH:
            transport.push_login(cluster)
    transport.bootstrap(cluster, AuthMode.INTERACTIVE_BOOTSTRAP)
    result = transport.run(
        cluster,
        "hostname",
        auth=AuthMode.NONINTERACTIVE,
        retry=RetryPolicy.SAFE_READ,
    )
    if result.returncode != 0:
        raise HpcAllocError(result.stderr.strip() or f"hostname failed on {cluster}")
    info(f"login OK: {result.stdout_text.strip()}")
    host_key_failure: HostKeyChanged | None = None
    # Health-check exactly the aliases the projection published -- the same set
    # `heal` retires.  Deriving it from current_node instead silently skipped
    # every seat whose node is held under a lease (a suspended allocation, a
    # scheduler terminal candidate), which are precisely the ones a user runs
    # `connect` to ask about.
    for alias, node in sorted(transport.compute_endpoints(cluster).items()):
        name = alias.rpartition(".")[2]
        try:
            transport.require_node(alias)
        except HostKeyChanged as exc:
            info(f"node {node} ({name!r}): host-key")
            host_key_failure = host_key_failure or exc
        except AuthRequired:
            info(f"node {node} ({name!r}): auth")
        except TransportLost:
            info(f"node {node} ({name!r}): network")
        else:
            info(f"node {node} ({name!r}): ok")
    if host_key_failure is not None:
        raise host_key_failure
    return 0


def cmd_partitions(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    cluster = ctx.config.resolve_cluster(args.cluster)
    transport, client = _services(ctx, paths, entrypoint, cluster)
    transport.bootstrap(cluster)
    text = client.partitions()
    parsed = [line.split("|") for line in text.strip().splitlines() if line.strip()]
    keys = ["partition", "avail", "timelimit", "nodes", "cpus", "memory", "gres", "features"]
    rows = [dict(zip(keys, row)) for row in parsed[1:] if len(row) == len(keys)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        info("no partitions returned")
        return 0
    widths = {key: max(len(key), *(len(str(row.get(key, ""))) for row in rows)) for key in keys}
    print("  ".join(key.upper().ljust(widths[key]) for key in keys))
    for row in rows:
        print("  ".join(str(row.get(key, "")).ljust(widths[key]) for key in keys))
    return 0


def cmd_avail(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    cluster = ctx.config.resolve_cluster(args.cluster)
    transport, client = _services(ctx, paths, entrypoint, cluster)
    transport.bootstrap(cluster)
    text = client.availability()
    payload: dict[str, Any] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        name, node_state, _node, gres, gres_used, cpus = fields[:6]
        name = name.rstrip("*")
        if args.partition and name != args.partition:
            continue
        part = payload.setdefault(
            name,
            {
                "nodes": {"idle": 0, "mix": 0, "alloc": 0, "other": 0},
                "cpus_idle": 0,
                "cpus_total": 0,
                "gpus": {},
            },
        )
        normalized = node_state.rstrip("*~#%!$@+^-").lower()
        if normalized.startswith("idle"):
            category = "idle"
        elif normalized.startswith("mix"):
            category = "mix"
        elif normalized.startswith(("alloc", "comp")):
            category = "alloc"
        else:
            category = "other"
        part["nodes"][category] += 1
        if category == "other":
            continue
        try:
            _allocated, idle, _other, total = (int(value) for value in cpus.split("/"))
        except (ValueError, TypeError):
            raise HpcAllocError(f"scheduler returned invalid CPU accounting: {cpus!r}")
        part["cpus_idle"] += idle
        part["cpus_total"] += total
        for source, key in ((gres, "total"), (gres_used, "used")):
            for gpu_type, count in re.findall(r"gpu:(?:([^:,()\s]+):)?(\d+)", source or ""):
                label = gpu_type or "gpu"
                entry = part["gpus"].setdefault(label, {"total": 0, "used": 0})
                entry[key] += int(count)
    for part in payload.values():
        for gpu in part["gpus"].values():
            gpu["free"] = gpu["total"] - gpu["used"]
    if args.json:
        print(json.dumps({"partitions": payload}, indent=2))
        return 0
    if not payload:
        info("no matching partitions")
        return 0
    fmt = "{:<14} {:<22} {:<16} {}"
    print(fmt.format("PARTITION", "NODES idle/mix/alloc/off", "CPUS free/total", "GPUS free/total"))
    for name, part in sorted(payload.items()):
        nodes = part["nodes"]
        gpus = "  ".join(
            f"{kind} {data['free']}/{data['total']}" for kind, data in sorted(part["gpus"].items())
        ) or "-"
        print(
            fmt.format(
                name,
                f"{nodes['idle']}/{nodes['mix']}/{nodes['alloc']}/{nodes['other']}",
                f"{part['cpus_idle']}/{part['cpus_total']}",
                gpus,
            )
        )
    return 0


def _assessment_payload(job: Any, assessment: Any) -> dict[str, Any]:
    return {
        "operation_id": job.operation_id,
        "selector": canonical_job_selector(job),
        "jobid": job.job_id,
        "cluster": job.cluster,
        "name": job.logical_name,
        "kind": job.kind.value,
        "phase": assessment.phase.value,
        "scheduler_state": assessment.scheduler_state,
        "evidence_detail": assessment.detail or None,
        "ever_started": job.ever_started,
        "current_node": job.current_node,
        "last_node": job.last_node,
        "terminal_state": job.terminal_state,
        "exit_code": job.exit_code,
        "final_source": job.final_source.value if job.final_source else None,
        "partition": job.resources.get("partition"),
        "time": job.resources.get("time"),
        "gpus": job.resources.get("gpus"),
        "alias": (
            allocation_alias(job.cluster, job.logical_name)
            if job.kind == JobKind.ALLOCATION and assessment.current_node
            else None
        ),
    }


def _reconcile_status(
    *, ctx: Any, paths: AppPaths, entrypoint: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from .errors import AuthRequired, HostKeyChanged, SchedulerUnavailable
    from .monitor import JobMonitor
    from .ssh import AuthMode

    # Correlate the entire pass against one local identity graph.  In
    # particular, remember jobs that were live at the start even if this pass
    # finalizes them, so their already-captured remote rows are never emitted a
    # second time as local-final conflicts.
    local_at_start = ctx.state.list_jobs(include_final=True)
    jobs = [job for job in local_at_start if job.phase != JobPhase.FINAL]
    local_by_identity = {
        (job.cluster, job.operation_id): job for job in local_at_start
    }
    rows: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    primary = ctx.config.resolve_cluster(None)
    scans: dict[str, Any] = {}
    clients: dict[str, Any] = {}
    # Connectivity, authentication, and scheduler outages on a secondary are
    # soft availability failures.  A changed host key is an integrity failure
    # on every cluster and must never be hidden behind successful status JSON.
    soft_availability_errors = (TransportLost, AuthRequired, SchedulerUnavailable)
    for cluster in ctx.config.clusters:
        transport, client = _services(ctx, paths, entrypoint, cluster)
        clients[cluster] = client
        auth = AuthMode.INTERACTIVE_BOOTSTRAP if cluster == primary else AuthMode.NONINTERACTIVE
        try:
            transport.bootstrap(cluster, auth)
            scans[cluster] = client.scan(auth=AuthMode.NONINTERACTIVE)
        except HostKeyChanged:
            raise
        except soft_availability_errors as exc:
            if cluster == primary:
                raise
            info(f"note: cluster {cluster!r} unavailable ({exc}); preserving its state")
            scans[cluster] = None

    for job in jobs:
        if job.job_id is None:
            assessment = JobMonitor.tracker(job).assessment
            payload = _assessment_payload(job, assessment)
            payload["phase"] = job.phase.value
            rows.append(payload)
            continue
        if scans.get(job.cluster) is None:
            payload = _assessment_payload(job, JobMonitor.tracker(job).assessment)
            payload["phase"] = "UNCERTAIN"
            rows.append(payload)
            continue
        client = clients[job.cluster]
        try:
            assessment = JobMonitor(client).assess(
                job, auth=AuthMode.NONINTERACTIVE
            ).assessment
            updated, assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=paths,
                client=client,
                job=job,
                assessment=assessment,
                monotonic_evidence=assessment,
                synchronize_projection=False,
            )
        except HostKeyChanged:
            raise
        except soft_availability_errors as exc:
            if job.cluster == primary:
                raise
            info(
                f"note: cluster {job.cluster!r} unavailable during managed observation "
                f"({exc}); preserving its state"
            )
            scans[job.cluster] = None
            durable = ctx.state.get_job(job.operation_id)
            payload = _assessment_payload(
                durable, JobMonitor.tracker(durable).assessment
            )
            payload["phase"] = "UNCERTAIN"
            rows.append(payload)
            continue
        rows.append(_assessment_payload(updated, assessment))

    machine = ctx.state.get_machine()
    owner_id = machine.machine_id if machine else ""
    remote_groups: dict[tuple[str, str], list[tuple[Any, Any, bool]]] = {}
    for cluster, scan in scans.items():
        if scan is None:
            continue
        for row in scan.rows:
            tag = parse_tag(row.comment)
            if tag is None:
                continue
            exact_derived_name = row.name == slurm_job_name(tag.kind, tag.operation_id)
            remote_groups.setdefault((cluster, tag.operation_id), []).append(
                (row, tag, exact_derived_name)
            )

    for identity, group in sorted(remote_groups.items()):
        cluster, operation_id = identity
        local = local_by_identity.get(identity)
        exact_remote_count = sum(
            1
            for row, _tag, exact_name in group
            if exact_name and row.job_id.isascii() and row.job_id.isdigit()
        )
        for row, tag, exact_name in sorted(group, key=lambda item: item[0].job_id):
            scalar_job_id = row.job_id.isascii() and row.job_id.isdigit()
            complete_local_match = bool(
                local is not None
                and exact_name
                and row.name == local.slurm_job_name
                and row.comment == local.slurm_comment
            )
            if local is not None and complete_local_match:
                if local.phase != JobPhase.FINAL and local.job_id == row.job_id:
                    # This row is represented by the managed jobs entry even if
                    # reconciliation finalized it during the current pass.
                    continue
                if local.phase == JobPhase.FINAL and exact_remote_count == 1:
                    classification = "local-final-conflict"
                elif local.job_id is None and exact_remote_count == 1:
                    classification = "unresolved-operation-match"
                else:
                    classification = "duplicate-operation"
            elif local is not None:
                classification = "operation-identity-conflict"
            elif not exact_name or not scalar_job_id:
                classification = "operation-identity-conflict"
            elif exact_remote_count > 1:
                classification = "duplicate-operation"
            else:
                classification = (
                    "other-machine" if tag.owner_id != owner_id else "untracked-owned"
                )
            node = row.node.strip()
            if not node or node.lower() in {"(null)", "none", "n/a"}:
                node = None
            discovered.append(
                {
                    "cluster": cluster,
                    "jobid": row.job_id,
                    "operation_id": operation_id,
                    "selector": f"{cluster}:@{operation_id}",
                    "job_kind": tag.kind,
                    "classification": classification,
                    "state": row.state.strip().upper(),
                    "node": node,
                    "partition": row.partition.rstrip("*"),
                    "time_left": row.time_left,
                    "owner": tag.host,
                    "name": tag.logical_name,
                }
            )

    operations = [
        {
            "operation_id": op.operation_id,
            "selector": f"{op.cluster}:@{op.target_job_operation_id}",
            "kind": op.kind.value,
            "phase": op.phase.value,
            "cluster": op.cluster,
            "target": op.logical_name,
            "jobid": op.job_id,
            "detail": op.detail,
        }
        for op in ctx.state.list_unresolved_operations()
    ]
    return rows, discovered, operations


def cmd_status(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    with _ssh_projection_scope(ctx, paths):
        rows, discovered, operations = _reconcile_status(
            ctx=ctx,
            paths=paths,
            entrypoint=entrypoint,
        )
    if args.json:
        print(
            json.dumps(
                {"jobs": rows, "discovered": discovered, "operations": operations},
                indent=2,
            )
        )
        return 0
    if not rows and not discovered and not operations:
        info("no managed or discovered jobs")
        return 0
    if rows:
        fmt = "{:<14} {:<9} {:<20} {:<14} {:<14} {}"
        print(fmt.format("NAME", "JOB", "PHASE", "CLUSTER", "NODE", "SSH ALIAS"))
        for row in rows:
            print(
                fmt.format(
                    row["name"],
                    row["jobid"] or "-",
                    row["phase"],
                    row["cluster"],
                    row["current_node"] or "-",
                    row["alias"] or "-",
                )
            )
    if discovered:
        print("\nDISCOVERED V2 JOBS")
        for row in discovered:
            print(
                f"  {row['cluster']}:{row['jobid']} {row['state']} "
                f"{row['job_kind']}/{row['classification']} "
                f"operation={row['operation_id']}"
            )
    if operations:
        print("\nUNRESOLVED OPERATIONS")
        for operation in operations:
            print(
                f"  {operation['operation_id']} {operation['kind']} {operation['phase']} "
                f"({operation['cluster']}:{operation['target']}) — hpc-alloc recover "
                f"{operation['operation_id']}"
            )
    return 0


def _reconcile_name_holder(
    ctx: Any, client: Any, cluster: str, kind: JobKind, logical_name: str
) -> None:
    """Finalize a departed allocation that still reserves this logical name.

    A successful `down` deliberately leaves its job TERMINAL_CANDIDATE: an
    acknowledged cancellation proves the request was *accepted*, not that the
    job reached a terminal state, and this codebase never fabricates finality
    from a mutation acknowledgement.  But the live-allocation unique index
    reserves the name for every non-FINAL row, so `up --name dev` immediately
    after `down dev` failed with a StateConflict until some unrelated later
    command happened to finalize the row.

    Reconciling here pays the cost exactly where it is needed -- only when a
    same-named allocation still holds the name -- and finalizes it only on
    genuine two-observation or accounting evidence.  An allocation that is still
    alive (or still draining in COMPLETING, and so still holding its node)
    simply does not finalize, and reserve_submission raises its usual, accurate
    conflict.
    """

    from .monitor import JobMonitor, persist_assessment
    from .ssh import AuthMode

    if kind != JobKind.ALLOCATION:
        return
    try:
        holders = [
            job
            for job in ctx.state.list_jobs(include_final=False)
            if job.kind == JobKind.ALLOCATION
            and job.cluster == cluster
            and job.logical_name == logical_name
            and job.ref is not None
        ]
        for holder in holders:
            assessment = JobMonitor(client).assess(
                holder,
                auth=AuthMode.NONINTERACTIVE,
                confirm=True,
            ).assessment
            if assessment.final:
                persist_assessment(ctx.state, holder, assessment)
    except HpcAllocError:
        # Never mask the reservation's authoritative conflict below: if the name
        # really is still held, that is the error the user must see.
        return


def _submit_job(
    *,
    ctx: Any,
    paths: AppPaths,
    entrypoint: Path,
    cluster: str,
    kind: JobKind,
    logical_name: str,
    resources: dict[str, Any],
    wrap: str,
    logfile_template: str,
    dry_run: bool,
) -> Any:
    from .slurm import SubmissionSpec
    from .ssh import AuthMode

    if not NAME_RE.fullmatch(logical_name):
        raise ConfigInvalid(f"invalid logical name {logical_name!r}")
    operation_id = uuid.uuid4().hex
    logfile = logfile_template.format(operation_id=operation_id)
    host = machine_host()
    owner_id = (
        f"dryrun-{operation_id[:12]}"
        if dry_run
        else ctx.state.get_or_create_machine_id(host)
    )
    spec = SubmissionSpec(
        operation_id=operation_id,
        owner_id=owner_id,
        owner_host=host,
        kind=kind,
        logical_name=logical_name,
        partition=resources["partition"],
        walltime=resources["time"],
        cpus=resources["cpus"],
        mem=resources.get("mem"),
        gpus=resources.get("gpus"),
        constraint=resources.get("constraint"),
        chdir=resources.get("chdir"),
        wrap=wrap,
        logfile=logfile,
        log_directory=REMOTE_LOG_DIR,
    )
    if dry_run:
        print(spec.command())
        return None
    transport, client = _services(ctx, paths, entrypoint, cluster)
    transport.bootstrap(cluster, AuthMode.INTERACTIVE_BOOTSTRAP)
    _reconcile_name_holder(ctx, client, cluster, kind, logical_name)
    # Retry-safe filesystem preparation is deliberately outside the durable
    # scheduler mutation boundary.  It cannot submit a job, so any failure
    # here leaves no local intent and no possible remote scheduler side effect.
    client.prepare_submission(spec, auth=AuthMode.NONINTERACTIVE)
    result: Any | None = None
    dispatch_may_have_started = False
    with operation_scope_lock(
        paths.operation_locks_dir,
        operation_id,
        blocking=True,
    ):
        try:
            ctx.state.reserve_submission(
                operation_id=operation_id,
                cluster=cluster,
                logical_name=logical_name,
                kind=kind,
                owner_id=owner_id,
                slurm_job_name=spec.job_name,
                slurm_comment=spec.comment,
                resources=resources,
            )
            # From this point onward an asynchronous interrupt is
            # conservatively treated as possibly overlapping the one-shot
            # scheduler mutation.
            dispatch_may_have_started = True
            try:
                result = client.submit(spec, auth=AuthMode.NONINTERACTIVE)
            except (AuthRequired, HostKeyChanged) as exc:
                # These typed SSH failures prove the mutation was never
                # dispatched.  Clear the flag before the failure write so an
                # interrupt there cannot create false ambiguity.
                dispatch_may_have_started = False
                ctx.state.fail_submission(operation_id, str(exc))
                raise
            except AmbiguousSubmission as exc:
                guidance = _submission_recovery_guidance(operation_id)
                try:
                    ctx.state.mark_submission_ambiguous(operation_id, str(exc))
                except Exception:
                    # PREPARED remains unresolved and recoverable if this
                    # secondary journal update cannot commit.
                    pass
                raise AmbiguousSubmission(guidance) from exc

            job_id = result.job_id
            guidance = _submission_recovery_guidance(operation_id, job_id=job_id)
            try:
                return ctx.state.acknowledge_submission(operation_id, job_id)
            except Exception as exc:
                _best_effort_mark_submission_ambiguous(
                    ctx,
                    operation_id,
                    f"Slurm acknowledged job {job_id}, but local acknowledgement failed: {exc}",
                )
                raise AmbiguousSubmission(guidance) from exc
        except KeyboardInterrupt:
            if not dispatch_may_have_started:
                # The reservation may have committed immediately before the
                # interrupt reached Python, but remote dispatch was not entered.
                try:
                    ctx.state.fail_submission(
                        operation_id,
                        "submission was interrupted before remote dispatch",
                    )
                except RecordNotFound:
                    # An interrupt inside the reservation transaction rolls
                    # both rows back.  With the operation guard still held,
                    # absence proves there is no local recovery intent and no
                    # remote dispatch to reconcile.
                    pass
                except BaseException:
                    try:
                        _best_effort_info(_submission_recovery_guidance(operation_id))
                    except BaseException:
                        # Recovery guidance is secondary to the original
                        # interrupt even if the diagnostic stream itself is
                        # unexpectedly unusable.
                        pass
                raise
            # Keep the operation guard until conservative ambiguity is durable.
            job_id = None
            if result is not None:
                try:
                    job_id = result.job_id
                except BaseException:
                    pass
            guidance = _submission_recovery_guidance(operation_id, job_id=job_id)
            detail = "submission was interrupted after remote dispatch may have started"
            if job_id is not None:
                detail = (
                    f"Slurm acknowledged job {job_id}, but local acknowledgement "
                    "was interrupted"
                )
            _best_effort_mark_submission_ambiguous(ctx, operation_id, detail)
            _best_effort_info(guidance)
            raise


def _persist_and_render(
    ctx: Any,
    paths: AppPaths,
    job: Any,
    assessment: Any,
    *,
    force_projection: bool = False,
    skip_unchanged_projection: bool = False,
) -> Any:
    from .monitor import persist_assessment

    updated = persist_assessment(ctx.state, job, assessment)
    # StateRepository normalizes and CAS-checks every candidate before
    # returning the original durable values for a semantic no-op.  Stream
    # checkpoints may suppress that redundant projection rewrite, while the
    # CAS still detects a concurrently advanced lifecycle revision.  Other
    # command paths retain their repair-on-success behavior by default.
    if (
        force_projection
        or not skip_unchanged_projection
        or updated != job
    ):
        _sync_ssh_projection(ctx, paths)
    return updated


def _persist_reconciled_assessment(
    *,
    ctx: Any,
    paths: AppPaths | None,
    client: Any,
    job: Any,
    assessment: Any,
    monotonic_evidence: Any | None = None,
    synchronize_projection: bool = True,
    skip_unchanged_projection: bool = False,
    checkpoint_reconciliation_observations: bool = False,
) -> tuple[Any, Any, Any | None]:
    """Persist lifecycle evidence, replacing stale policy after revision races.

    The first candidate may have been collected by a long-lived follower.  If
    its source revision changed, that candidate is never retried.  Each later
    candidate comes from a fresh exact scheduler assessment seeded only with
    durable authority and the follower's monotonic start/last-node evidence.
    """

    from .monitor import JobMonitor, accept_observation, persist_assessment
    from .ssh import AuthMode

    source = job
    candidate = assessment
    fresh_tracker = None
    evidence = monotonic_evidence or assessment
    monotonic_started = bool(evidence.ever_started)
    monotonic_last_node = evidence.last_node
    while True:
        if candidate.uncertain:
            # Operational uncertainty is process-local.  Any successful queue
            # evidence preceding this boundary was already checkpointed by
            # the follower's publication hook.  Non-stream commands retain
            # their normal projection repair, and a streaming CAS race must
            # repair the projection potentially missed by its competing
            # writer, but no uncertainty is written to the lifecycle row.
            if synchronize_projection and (
                fresh_tracker is not None or not skip_unchanged_projection
            ):
                if paths is None:
                    raise ValueError(
                        "paths are required when synchronizing the SSH projection"
                    )
                _sync_ssh_projection(ctx, paths)
            return source, candidate, fresh_tracker
        try:
            if synchronize_projection:
                if paths is None:
                    raise ValueError(
                        "paths are required when synchronizing the SSH projection"
                    )
                updated = _persist_and_render(
                    ctx,
                    paths,
                    source,
                    candidate,
                    # A competing writer may have committed lifecycle state
                    # and then failed its derived projection.  Once a CAS
                    # race has supplied fresh authority, repair that
                    # projection even when the retry is a semantic DB no-op.
                    force_projection=fresh_tracker is not None,
                    skip_unchanged_projection=skip_unchanged_projection,
                )
            else:
                updated = persist_assessment(ctx.state, source, candidate)
        except LifecycleRevisionConflict:
            durable = ctx.state.get_job(job.operation_id)
            if (
                checkpoint_reconciliation_observations
                and durable.phase is JobPhase.FINAL
                and durable.final_source
                in {FinalSource.SUBMIT_FAILED, FinalSource.ABANDONED}
            ):
                # Explicit local finality cannot be rewritten through the
                # scheduler-evidence API.  It is already durable authority;
                # only its possibly missed derived projection needs repair.
                authoritative_tracker = JobMonitor.tracker(durable)
                if synchronize_projection:
                    if paths is None:
                        raise ValueError(
                            "paths are required when synchronizing the SSH projection"
                        )
                    _sync_ssh_projection(ctx, paths)
                return durable, authoritative_tracker.assessment, authoritative_tracker
            seeded = replace(
                durable,
                ever_started=(durable.ever_started or monotonic_started),
                last_node=(durable.last_node or monotonic_last_node),
            )
            fresh_tracker = JobMonitor.tracker(seeded)
            if checkpoint_reconciliation_observations:
                if seeded.phase is JobPhase.FINAL or seeded.ref is None:
                    candidate = fresh_tracker.assessment
                else:
                    _row, candidate = accept_observation(
                        fresh_tracker,
                        lambda: client.observe(
                            seeded.ref,
                            auth=AuthMode.NONINTERACTIVE,
                        ),
                    )
            else:
                candidate = JobMonitor(client).assess(
                    seeded,
                    auth=AuthMode.NONINTERACTIVE,
                    tracker=fresh_tracker,
                ).assessment
            monotonic_started = monotonic_started or candidate.ever_started
            if candidate.last_node:
                monotonic_last_node = candidate.last_node
            source = seeded
            continue
        if updated.phase is JobPhase.FINAL:
            authoritative_tracker = JobMonitor.tracker(updated)
            return (
                updated,
                authoritative_tracker.assessment,
                authoritative_tracker if fresh_tracker is not None else None,
            )
        return updated, candidate, fresh_tracker


def _follow_with_reconciliation(
    *,
    ctx: Any,
    paths: AppPaths,
    client: Any,
    job: Any,
    follower: Any,
) -> tuple[Any, Any]:
    """Publish each successful observation before following it as policy."""

    source = job

    def publish(assessment: Any) -> tuple[Any, Any | None]:
        nonlocal source
        updated, assessment, fresh_tracker = _persist_reconciled_assessment(
            ctx=ctx,
            paths=paths,
            client=client,
            job=source,
            assessment=assessment,
            monotonic_evidence=assessment,
            skip_unchanged_projection=True,
            checkpoint_reconciliation_observations=True,
        )
        source = updated
        return assessment, fresh_tracker

    # LogFollower invokes ``publish`` immediately after successful scheduler
    # evidence and adopts any replacement tracker before notes, log access,
    # draining, or sleeping.  Consequently a stale final CAS can reconcile
    # back to live without one byte of stale-policy drain.
    outcome = follower.follow(publish_assessment=publish)
    if not outcome.log_complete:
        # The job's own exit status is deliberately preserved.  Flipping it here
        # would tell an agent the *job* failed when in fact only our read of its
        # log did -- inviting a duplicate GPU submission, which is the worse
        # outcome.  But the shortfall must be loud: the output above is knowingly
        # incomplete, and silence would present a truncated result as a complete
        # one.  The remote log still exists, so re-reading it is cheap.
        _best_effort_info(
            f"warning: could not read the log of {source.cluster}:{source.job_id} to the end; "
            f"the output above may be truncated — re-read it with "
            f"`hpc-alloc logs {source.cluster}:{source.job_id}`"
        )
    return source, outcome.assessment


def cmd_up(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .lifecycle import AssessmentPhase
    from .monitor import JobMonitor
    from .ssh import AuthMode
    cluster = ctx.config.resolve_cluster(args.cluster)
    if (
        not NAME_RE.fullmatch(args.name)
        or args.name.isdigit()
        or args.name in {"login", "run"}
    ):
        raise ConfigInvalid(f"invalid or reserved allocation name {args.name!r}")
    if args.wait_timeout < 0:
        raise ConfigInvalid("--wait-timeout must be non-negative")
    resources = _resource_values(args, ctx.config, cluster)
    job = _submit_job(
        ctx=ctx,
        paths=paths,
        entrypoint=entrypoint,
        cluster=cluster,
        kind=JobKind.ALLOCATION,
        logical_name=args.name,
        resources=resources,
        wrap=_sleeper_command(resources["idle_timeout"]),
        logfile_template=f"{REMOTE_LOG_DIR}/alloc-{{operation_id}}.log",
        dry_run=args.dry_run,
    )
    if job is None:
        return 0
    try:
        info(
            f"submitted {cluster}:{job.job_id} for allocation {cluster}:{args.name} "
            f"({resources['partition']}, {resources['time']}; "
            f"{canonical_job_selector(job)})"
        )
        if args.no_wait:
            return 0
        _transport, client = _services(ctx, paths, entrypoint, cluster)
        monitor = JobMonitor(client)
        started = time.monotonic()
        budget = RetryBudget(info=info)
        # A fixed 5s poll meant a job that queued for the default 30 minutes made
        # 360 controller queries, every one of them learning the same thing.  Back
        # off while nothing changes; any movement drops straight back to the floor.
        backoff = PollBackoff()
        while True:
            try:
                result = monitor.assess(job, auth=AuthMode.NONINTERACTIVE)
            except HpcAllocError as exc:
                # One slurmctld restart or VPN blip used to abort the wait
                # outright, leaving a submitted GPU job queued and unwatched.
                # absorb() re-raises anything untimely or untreatable by waiting.
                budget.absorb(exc)
                continue
            budget.reset()
            assessment = result.assessment
            job, assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=paths,
                client=client,
                job=job,
                assessment=assessment,
                monotonic_evidence=assessment,
            )
            if assessment.phase == AssessmentPhase.ACTIVE and assessment.current_node:
                alias = allocation_alias(cluster, args.name)
                info(
                    f"allocation {cluster}:{args.name} ready on {assessment.current_node} "
                    f"(job {job.job_id}; ssh {alias})"
                )
                return 0
            if assessment.final:
                raise HpcAllocError(
                    f"job {job.job_id} ended before becoming active: "
                    f"{assessment.terminal_state or assessment.scheduler_state or 'unknown'}"
                )
            elapsed = int(time.monotonic() - started)
            info(
                f"job {job.job_id}: {assessment.scheduler_state or assessment.phase.value} "
                f"(waited {elapsed}s)"
            )
            if elapsed >= args.wait_timeout:
                info(
                    f"job {job.job_id} is still queued after {elapsed}s; it remains "
                    f"submitted and tracked — do not resubmit. Wait for it with "
                    f"`hpc-alloc status` or `hpc-alloc logs {cluster}:{job.job_id} -f`, "
                    f"or release it with `hpc-alloc down {args.name}`"
                )
                return EXIT_SUBMITTED_NOT_READY
            time.sleep(
                backoff.interval(
                    (
                        assessment.phase,
                        assessment.scheduler_state,
                        assessment.current_node,
                    )
                )
            )
    except KeyboardInterrupt:
        _report_up_interrupt(ctx, job)
        raise


def _remote_home(ctx: Any, client: Any, cluster: str) -> str:
    netid = ctx.config.identity.netid
    host = ctx.config.clusters[cluster].host
    cached = ctx.state.get_cluster_cache(cluster, "remote_home")
    if (
        isinstance(cached, dict)
        and set(cached) == {"path", "netid", "host"}
        and isinstance(cached["path"], str)
        and cached["path"].startswith("/")
        and "\x00" not in cached["path"]
        and "\n" not in cached["path"]
        and cached["netid"] == netid
        and cached["host"] == host
    ):
        return cached["path"]
    home = client.remote_home()
    ctx.state.set_cluster_cache(
        cluster,
        "remote_home",
        {"path": home, "netid": netid, "host": host},
    )
    return home


def _cancellation_lifecycle_evidence(
    assessment: Any,
    *,
    cancellation_detail: str | None = None,
) -> dict[str, Any]:
    """Translate one canonical assessment into repository merge arguments."""

    provenance = assessment.evidence_provenance
    evidence_detail = (assessment.detail or None) if provenance is not None else None
    if cancellation_detail is not None and provenance is None:
        provenance = EvidenceProvenance.CANCELLATION
        evidence_detail = cancellation_detail
    evidence = {
        # False is lack of evidence, not evidence that a durable True is wrong.
        "ever_started": True if assessment.ever_started else None,
        "last_node": assessment.last_node,
        "observation_epoch": assessment.observation_epoch,
    }
    if provenance is not None:
        evidence["evidence_provenance"] = provenance
        evidence["evidence_detail"] = evidence_detail
    return evidence


def _cancel_record(ctx: Any, paths: AppPaths, client: Any, job: Any) -> Any:
    if job.ref is None:
        raise StateConflict("cannot cancel a submission without a confirmed Slurm job ID")
    operation_id = uuid.uuid4().hex
    operation: Any | None = None
    with operation_scope_lock(
        paths.operation_locks_dir,
        operation_id,
        blocking=True,
    ):
        try:
            operation = ctx.state.begin_cancel(
                job.operation_id,
                operation_id=operation_id,
            )
            return _cancel_record_owned(ctx, client, job, operation)
        except KeyboardInterrupt:
            _best_effort_cancel_interrupt_reconciliation(
                ctx,
                operation_id=operation_id,
                target_job_operation_id=job.operation_id,
                known_intent=operation is not None,
            )
            raise


def _cancel_record_owned(
    ctx: Any,
    client: Any,
    job: Any,
    operation: Any,
) -> Any:
    from .errors import ProtocolViolation, SchedulerUnavailable
    from .lifecycle import EvidenceEvent
    from .monitor import JobMonitor
    from .slurm import CancellationInspectionStatus, CancellationStatus

    current = job
    while True:
        assert current.ref is not None
        try:
            inspection = client.inspect_cancel(current.ref)
        except HpcAllocError as exc:
            ctx.state.fail_cancel_operation(operation.operation_id, str(exc))
            raise

        tracker = JobMonitor.tracker(current)
        assessment = tracker.assessment
        try:
            if inspection.status == CancellationInspectionStatus.ALREADY_FINAL:
                record = inspection.final_record
                if record is None:
                    detail = "cancellation inspection omitted its final accounting record"
                    ctx.state.fail_cancel_operation(operation.operation_id, detail)
                    raise ProtocolViolation(detail)
                tracker.begin_observation_epoch()
                assessment = tracker.accept(EvidenceEvent.final(record))
                ctx.state.resolve_operation(
                    operation.operation_id,
                    final_source=assessment.final_source or FinalSource.ACCOUNTING,
                    expected_target_updated_at=current.updated_at,
                    detail=inspection.detail,
                    terminal_state=assessment.terminal_state,
                    exit_code=assessment.exit_code,
                    **_cancellation_lifecycle_evidence(assessment),
                )
                return inspection
            if inspection.status == CancellationInspectionStatus.CONFIRMED_ABSENT:
                ctx.state.resolve_cancel_departed(
                    operation.operation_id,
                    detail=(
                        inspection.detail
                        or "job absence was confirmed before cancellation"
                    ),
                    expected_target_updated_at=current.updated_at,
                )
                return inspection
            if inspection.status == CancellationInspectionStatus.IDENTITY_MISMATCH:
                ctx.state.fail_cancel_operation(
                    operation.operation_id,
                    inspection.detail or "job identity changed before cancellation",
                )
                raise IdentityMismatch(inspection.detail)
            if inspection.status != CancellationInspectionStatus.READY:
                detail = f"unsupported cancellation inspection status {inspection.status}"
                ctx.state.fail_cancel_operation(operation.operation_id, detail)
                raise ProtocolViolation(detail)

            # The real Slurm adapter always returns the exact row that
            # authorized this cancellation.  The optional field keeps
            # compatibility for callers constructing the public result type.
            if inspection.queue_row is not None:
                tracker.begin_observation_epoch()
                assessment = tracker.accept(EvidenceEvent.queue(inspection.queue_row))

            # Commit the may-have-run state immediately before the mutation.
            # A revision race rolls this transaction back to CANCEL_PENDING,
            # so it is safe to discard the inspection and repeat read-only
            # preflight.  Once this commits, execute_cancel must run once.
            dispatch_evidence = _cancellation_lifecycle_evidence(assessment)
            if not assessment.uncertain:
                dispatch_evidence["phase"] = JobPhase(assessment.phase.value)
                dispatch_evidence["current_node"] = assessment.current_node
            ctx.state.mark_cancel_dispatching(
                operation.operation_id,
                expected_target_updated_at=current.updated_at,
                terminal_state=assessment.terminal_state,
                exit_code=assessment.exit_code,
                **dispatch_evidence,
            )
        except LifecycleRevisionConflict:
            current = ctx.state.get_job(job.operation_id)
            continue
        dispatch_ref = current.ref
        break

    assert dispatch_ref is not None
    try:
        outcome = client.execute_cancel(dispatch_ref)
    except (AuthRequired, HostKeyChanged) as exc:
        # The adapter preserves these only when guarded cancellation never
        # crossed the remote dispatch boundary.
        ctx.state.fail_cancel_operation(operation.operation_id, str(exc))
        raise
    except HpcAllocError as exc:
        ctx.state.mark_cancel_ambiguous(operation.operation_id, str(exc))
        raise TransportLost(
            _cancellation_recovery_guidance(operation.operation_id)
        ) from exc

    if outcome.status == CancellationStatus.CANCELLED:
        ctx.state.resolve_cancel_departed(
            operation.operation_id,
            detail="cancellation request acknowledged",
        )
        return outcome
    if outcome.status == CancellationStatus.LEFT_QUEUE:
        ctx.state.resolve_cancel_departed(
            operation.operation_id,
            outcome.detail,
        )
        return outcome
    if outcome.status == CancellationStatus.MUTATION_AMBIGUOUS:
        ctx.state.mark_cancel_ambiguous(operation.operation_id, outcome.detail)
        raise TransportLost(_cancellation_recovery_guidance(operation.operation_id))
    if outcome.status == CancellationStatus.IDENTITY_MISMATCH:
        ctx.state.fail_cancel_operation(
            operation.operation_id,
            outcome.detail or "job identity changed before guarded cancellation",
        )
        raise IdentityMismatch(outcome.detail)
    if outcome.status == CancellationStatus.GUARD_FAILED:
        ctx.state.fail_cancel_operation(
            operation.operation_id,
            outcome.detail or "cancellation guard rejected the mutation",
        )
        raise SchedulerUnavailable(
            outcome.detail or "cancellation guard could not verify the exact job"
        )

    # An unknown post-dispatch result can never be downgraded to a safe retry.
    detail = outcome.detail or f"unrecognized cancellation result {outcome.status}"
    ctx.state.mark_cancel_ambiguous(operation.operation_id, detail)
    raise TransportLost(_cancellation_recovery_guidance(operation.operation_id))


def cmd_run(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .lifecycle import EvidenceTracker, state_code
    from .ssh import AuthMode
    from .streaming import LogFollower

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ConfigInvalid("no command given — usage: hpc-alloc run [options] -- CMD ...")
    cluster = ctx.config.resolve_cluster(args.cluster)
    resources = _resource_values(args, ctx.config, cluster)
    if resources.get("chdir") and resources["chdir"].startswith("~"):
        raw_chdir = resources["chdir"]
        if not re.fullmatch(r"~(?:/.*)?", raw_chdir):
            raise ConfigInvalid("~user paths are unsupported; use ~/... or an absolute path")
    if not args.dry_run:
        transport, client = _services(ctx, paths, entrypoint, cluster)
        transport.bootstrap(cluster, AuthMode.INTERACTIVE_BOOTSTRAP)
        home = _remote_home(ctx, client, cluster)
        if resources.get("chdir") and resources["chdir"].startswith("~"):
            resources["chdir"] = home + (resources["chdir"][1:] or "")
        logfile = f"{home}/{REMOTE_LOG_DIR}/run-{{operation_id}}.log"
    else:
        logfile = f"{REMOTE_LOG_DIR}/run-{{operation_id}}.log"
    wrap = _remote_command(command)
    job = _submit_job(
        ctx=ctx,
        paths=paths,
        entrypoint=entrypoint,
        cluster=cluster,
        kind=JobKind.RUN,
        logical_name="run",
        resources=resources,
        wrap=wrap,
        logfile_template=logfile,
        dry_run=args.dry_run,
    )
    if job is None:
        return 0
    try:
        log_path = _job_log_path(job)
        _pipe_aware_info(
            f"submitted run {cluster}:{job.job_id} "
            f"({resources['partition']}, {resources['time']}; "
            f"{canonical_job_selector(job)})"
        )
        if args.detach:
            _pipe_aware_info(f"follow with `hpc-alloc logs {cluster}:{job.job_id} -f`")
            return 0
        follower = LogFollower(
            client,
            job.ref,
            log_path,
            tracker=EvidenceTracker(
                ever_started=job.ever_started,
                current_node=job.current_node,
                last_node=job.last_node,
            ),
            info=_pipe_aware_info,
        )
        updated, assessment = _follow_with_reconciliation(
            ctx=ctx,
            paths=paths,
            client=client,
            job=job,
            follower=follower,
        )
    except BrokenPipeError:
        neutralize_stdout()
        if args.detach:
            return 141
        try:
            _cancel_record(ctx, paths, client, job)
        except HpcAllocError as exc:
            _best_effort_info(
                f"output pipe closed — could not confirm cancellation of foreground "
                f"job {cluster}:{job.job_id}: {exc}"
            )
        else:
            _best_effort_info(
                f"output pipe closed — cancelled foreground job {cluster}:{job.job_id}"
            )
        return 141
    except KeyboardInterrupt:
        if args.detach:
            raise
        try:
            _cancel_record(ctx, paths, client, job)
        except HpcAllocError as exc:
            _best_effort_info(f"could not confirm cancellation: {exc}")
        else:
            _best_effort_info(f"cancelled foreground job {cluster}:{job.job_id}")
        raise
    except HpcAllocError:
        _report_run_follow_failure(ctx, job)
        raise
    state = state_code(assessment.terminal_state or "")
    if not state:
        raise HpcAllocError(
            f"job {updated.job_id} left the queue without a final state; run `hpc-alloc why "
            f"{cluster}:{updated.job_id}`"
        )
    if assessment.exit_code and ":" in assessment.exit_code:
        code = int(assessment.exit_code.split(":", 1)[0])
    else:
        code = 0 if state == "COMPLETED" else 1
    if state != "COMPLETED" and code == 0:
        code = 1
    info(f"job {updated.job_id} finished: {state} (exit {code})")
    return code


def _resolve_managed_job(
    ctx: Any,
    target: str,
    *,
    explicit_cluster: str | None,
    kind: JobKind | None = None,
    include_final: bool = False,
) -> Any:
    selector = parse_selector(target, explicit_cluster)
    jobs = ctx.state.list_jobs(include_final=include_final)
    if kind is not None:
        jobs = [job for job in jobs if job.kind == kind]
    try:
        job = unique_job(jobs, selector)
    except IdentityMismatch:
        # Retain the established configuration error for an unknown qualifier
        # when no durable historical identity can authorize it.
        if selector.cluster:
            ctx.config.resolve_cluster(selector.cluster)
        raise
    if selector.cluster and not (
        include_final and job.phase is JobPhase.FINAL
    ):
        ctx.config.resolve_cluster(selector.cluster)
    return job


def cmd_cancel(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    selector = parse_selector(args.target, args.cluster)
    if selector.kind not in {SelectorKind.JOB_ID, SelectorKind.OPERATION_ID}:
        raise IdentityMismatch("cancel requires a job ID or @operation selector")
    job = _resolve_managed_job(
        ctx,
        args.target,
        explicit_cluster=args.cluster,
    )
    with _ssh_projection_scope(ctx, paths):
        transport, client = _services(ctx, paths, entrypoint, job.cluster)
        transport.bootstrap(job.cluster)
        outcome = _cancel_record(ctx, paths, client, job)
    info(f"{outcome.status.value.lower().replace('_', ' ')}: {job.cluster}:{job.job_id}")
    return 0


def _default_allocation(ctx: Any, explicit_cluster: str | None = None) -> Any:
    jobs = [
        job
        for job in ctx.state.list_jobs(include_final=False)
        if job.kind == JobKind.ALLOCATION
        and (explicit_cluster is None or job.cluster == explicit_cluster)
    ]
    if not jobs:
        raise IdentityMismatch("no active managed allocations")
    if len(jobs) == 1:
        return jobs[0]
    dev = [job for job in jobs if job.logical_name == "dev"]
    if len(dev) == 1:
        return dev[0]
    raise IdentityMismatch("multiple allocations exist; use cluster:name")


def cmd_down(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    if args.all and args.target:
        raise ConfigInvalid("down accepts either a target or --all, not both")
    if args.all:
        jobs = [
            job
            for job in ctx.state.list_jobs(include_final=False)
            if job.kind == JobKind.ALLOCATION
            and (args.cluster is None or job.cluster == args.cluster)
        ]
    elif args.target:
        jobs = [
            _resolve_managed_job(
                ctx,
                args.target,
                explicit_cluster=args.cluster,
                kind=JobKind.ALLOCATION,
            )
        ]
    else:
        # `down` is irreversible.  It used to fall back to an implicit target --
        # the sole allocation, or whichever one happened to be named `dev` when
        # several existed -- so an omitted selector silently cancelled a
        # different, still-in-use seat and exited 0 with no prompt.  Naming the
        # target (or --all) costs one word.
        candidates = sorted(
            f"{job.cluster}:{job.logical_name}"
            for job in ctx.state.list_jobs(include_final=False)
            if job.kind == JobKind.ALLOCATION
            and (args.cluster is None or job.cluster == args.cluster)
        )
        listed = ", ".join(candidates) if candidates else "none are active"
        raise ConfigInvalid(
            f"down requires an allocation target or --all (active: {listed})"
        )
    if not jobs:
        info("no matching active allocations")
        return 0
    failed = 0
    successes: list[str] = []
    projection = _SshProjectionResult()
    try:
        with _ssh_projection_scope(ctx, paths) as projection:
            for job in jobs:
                try:
                    transport, client = _services(ctx, paths, entrypoint, job.cluster)
                    transport.bootstrap(job.cluster)
                    outcome = _cancel_record(ctx, paths, client, job)
                    successes.append(
                        f"{outcome.status.value.lower().replace('_', ' ')} allocation "
                        f"{job.cluster}:{job.logical_name} ({job.job_id})"
                    )
                except (AuthRequired, HostKeyChanged, TransportLost) as exc:
                    info(f"could not release {job.cluster}:{job.logical_name}: {exc}")
                    # Security/authentication failures are not best-effort
                    # scheduler errors.  Stop before touching another allocation;
                    # the projection scope repairs state changed by earlier jobs.
                    #
                    # TransportLost belongs here for a sharper reason: it is how
                    # _cancel_record_owned reports a *possibly dispatched*
                    # cancellation.  Folding it into the best-effort `failed`
                    # counter returned exit 1 -- "validation/scheduler" -- for
                    # the one outcome an agent must never blindly retry, and the
                    # docs promise exit 3 for it.  A blind retry then hits the
                    # pending-cancellation index and fails forever.
                    raise
                except HpcAllocError as exc:
                    failed += 1
                    info(f"could not release {job.cluster}:{job.logical_name}: {exc}")
                    if not args.all:
                        raise
    except (AuthRequired, HostKeyChanged, TransportLost):
        if args.all and projection.synchronized:
            for message in successes:
                info(message)
        raise
    for message in successes:
        info(message)
    return 1 if failed else 0


def _job_log_path(job: Any) -> str:
    prefix = "run" if job.kind == JobKind.RUN else "alloc"
    return f"{REMOTE_LOG_DIR}/{prefix}-{job.operation_id}.log"


def _local_no_id_diagnosis(job: Any) -> str | None:
    """Describe a durable local final verdict that has no remote job ID."""

    if job.phase is not JobPhase.FINAL or job.ref is not None:
        return None
    if job.final_source is FinalSource.SUBMIT_FAILED:
        return "submission failed before a Slurm job ID was acknowledged"
    if job.final_source is FinalSource.ABANDONED:
        return (
            "submission was explicitly abandoned without an acknowledged Slurm job ID; "
            "an untracked remote orphan may remain"
        )
    return None


def _optional_scheduler_diagnostic(fetch: Callable[[], str]) -> str | None:
    """Return output-only scheduler detail without discarding core evidence."""

    try:
        return fetch()
    except (AuthRequired, HostKeyChanged):
        raise
    except HpcAllocError:
        return None


def cmd_logs(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .errors import SchedulerUnavailable
    from .monitor import JobMonitor
    from .ssh import AuthMode
    from .streaming import LogFollower

    if args.lines < 0:
        raise ConfigInvalid("--lines must be non-negative")
    job = _resolve_managed_job(
        ctx,
        args.target,
        explicit_cluster=args.cluster,
        include_final=True,
    )
    if job.ref is None:
        local_diagnosis = _local_no_id_diagnosis(job)
        if local_diagnosis is not None:
            raise StateConflict(
                f"{local_diagnosis} for operation {job.operation_id}; "
                "no managed log is available"
            )
        raise StateConflict(
            f"submission {job.operation_id} is unresolved; run `hpc-alloc recover {job.operation_id}`"
        )
    transport, client = _services(ctx, paths, entrypoint, job.cluster)
    # Establish Duo once before entering the strictly noninteractive follower.
    transport.bootstrap(job.cluster, AuthMode.INTERACTIVE_BOOTSTRAP)
    log_path = _job_log_path(job)
    if not args.follow:
        assessment = None
        try:
            assessment = JobMonitor(client).assess(
                job, auth=AuthMode.NONINTERACTIVE
            ).assessment
            job, assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=paths,
                client=client,
                job=job,
                assessment=assessment,
                monotonic_evidence=assessment,
            )
        except SchedulerUnavailable:
            job = ctx.state.get_job(job.operation_id)
            if not JobMonitor.tracker(job).assessment.log_eligible:
                raise
            assessment = None
            info(
                "scheduler unavailable; reading the operation-scoped log from "
                "durable final/start evidence"
            )
        if assessment is not None and not assessment.log_eligible:
            raise HpcAllocError(
                f"job {job.cluster}:{job.job_id} has not started; use logs -f to wait safely"
            )
        sys.stdout.buffer.write(client.tail_log(log_path, args.lines))
        sys.stdout.buffer.flush()
        return 0
    follower = LogFollower(
        client,
        job.ref,
        log_path,
        tracker=JobMonitor.tracker(job),
        info=_pipe_aware_info,
    )
    try:
        _updated, assessment = _follow_with_reconciliation(
            ctx=ctx,
            paths=paths,
            client=client,
            job=job,
            follower=follower,
        )
    except BrokenPipeError:
        neutralize_stdout()
        _best_effort_info(
            f"output pipe closed — detached; job {job.cluster}:{job.job_id} continues "
            f"(reattach: hpc-alloc logs {job.cluster}:{job.job_id} -f)"
        )
        return 141
    except KeyboardInterrupt:
        _best_effort_info(
            f"detached; job {job.cluster}:{job.job_id} continues "
            f"(reattach: hpc-alloc logs {job.cluster}:{job.job_id} -f)"
        )
        raise
    if assessment.terminal_state:
        info(f"job {job.job_id} ended: {assessment.terminal_state}")
    return 0


def cmd_why(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .lifecycle import AssessmentPhase, EvidenceEvent
    from .monitor import JobMonitor
    from .ssh import AuthMode

    if args.target:
        job = _resolve_managed_job(
            ctx,
            args.target,
            explicit_cluster=args.cluster,
            include_final=True,
        )
    else:
        job = _default_allocation(ctx, args.cluster)
    if job.ref is None:
        local_diagnosis = _local_no_id_diagnosis(job)
        if local_diagnosis is not None:
            assessment = JobMonitor.tracker(job).assessment
            result = _assessment_payload(job, assessment)
            result["status"] = assessment.phase.value
            result["diagnosis"] = local_diagnosis
            result["detail"] = []
        else:
            assessment = JobMonitor.tracker(job).assessment
            result = _assessment_payload(job, assessment)
            # SUBMITTING is a durable local phase, whereas the lifecycle
            # tracker treats it as queue-like while awaiting a Slurm ID.
            result["phase"] = job.phase.value
            result["status"] = job.phase.value
            result["diagnosis"] = (
                "submission has no acknowledged Slurm job ID; "
                f"run hpc-alloc recover {job.operation_id}"
            )
            result["detail"] = []
    else:
        transport, client = _services(ctx, paths, entrypoint, job.cluster)
        transport.bootstrap(job.cluster, AuthMode.INTERACTIVE_BOOTSTRAP)
        assessment = JobMonitor(client).assess(job, auth=AuthMode.NONINTERACTIVE).assessment
        job, assessment, _tracker = _persist_reconciled_assessment(
            ctx=ctx,
            paths=paths,
            client=client,
            job=job,
            assessment=assessment,
            monotonic_evidence=assessment,
        )

        # A queue-derived final may predate slurmdbd, while an accounting final
        # can be durable without the output-only timing fields requested by
        # why.  Retry exact accounting for either source, routing only delayed
        # queue upgrades through the lifecycle tracker before rendering.
        accounting_record = None
        if assessment.final and job.final_source in {
            FinalSource.CONFIRMED_QUEUE,
            FinalSource.ACCOUNTING,
        }:
            try:
                record = client.final(
                    job.ref,
                    attempts=(0, 2, 2),
                    auth=AuthMode.NONINTERACTIVE,
                    extra_fields=("Elapsed", "Timelimit"),
                )
            except (AuthRequired, HostKeyChanged):
                raise
            except HpcAllocError:
                if job.final_source is not FinalSource.ACCOUNTING:
                    raise
                # The durable accounting verdict remains authoritative when
                # this output-only timing lookup is temporarily unavailable.
                record = None
            if record is not None:
                if job.final_source is FinalSource.CONFIRMED_QUEUE:
                    tracker = JobMonitor.tracker(job)
                    enriched = tracker.accept(EvidenceEvent.final(record))
                    job, assessment, _tracker = _persist_reconciled_assessment(
                        ctx=ctx,
                        paths=paths,
                        client=client,
                        job=job,
                        assessment=enriched,
                        monotonic_evidence=enriched,
                    )
                if (
                    assessment.final
                    and job.final_source is FinalSource.ACCOUNTING
                    and assessment.terminal_state == record.state
                    and assessment.exit_code == record.exit_code
                ):
                    accounting_record = record

        # Build every durable field only after reconciliation so phase,
        # terminal state, source, exit code, and diagnosis share one authority.
        result = _assessment_payload(job, assessment)
        result["status"] = assessment.phase.value
        detail: list[str] = []
        state = assessment.scheduler_state or ""
        if assessment.phase == AssessmentPhase.QUEUED:
            if state == "PENDING":
                # The lifecycle assessment already contains the reason from
                # the identity-checked observation.  A second raw observation
                # would open a race in which a recycled numeric ID escapes the
                # monitor's evidence handling.
                reason = assessment.detail or "unknown"
                if reason.startswith(("QOSMax", "QOSGrp", "AssocGrp", "MaxTRESPer", "QOSUsage")):
                    diagnosis = f"per-user/group resource cap ({reason})"
                elif "ReqNodeNotAvail" in reason or "Reserv" in reason:
                    diagnosis = "nodes are reserved, commonly for maintenance; a shorter walltime may fit"
                    reservations = _optional_scheduler_diagnostic(client.reservations)
                    if reservations is not None:
                        detail.extend(reservations.strip().splitlines()[:5])
                else:
                    diagnosis = f"queue contention ({reason})"
                    estimate = _optional_scheduler_diagnostic(
                        lambda: client.estimated_start(job.job_id)
                    )
                    if estimate and estimate != "N/A":
                        detail.append(f"estimated start: {estimate}")
                    priority = _optional_scheduler_diagnostic(
                        lambda: client.priority(job.job_id)
                    )
                    if priority is not None:
                        detail.extend(priority.strip().splitlines()[:3])
            else:
                diagnosis = f"queued ({state or assessment.phase.value})"
        elif assessment.phase == AssessmentPhase.ACTIVE:
            diagnosis = f"running on {assessment.current_node}"
        elif assessment.phase == AssessmentPhase.STARTED_INACTIVE:
            diagnosis = (
                f"{state or assessment.phase.value}; execution has started but is not "
                "currently active"
            )
        elif assessment.phase == AssessmentPhase.REQUEUEING:
            diagnosis = f"{state or assessment.phase.value}; previously started and may run again"
        elif assessment.final:
            final_state = assessment.terminal_state
            diagnosis = (
                f"final state {final_state}"
                if final_state
                else assessment.detail or "final state unknown"
            )
            if accounting_record is not None and len(accounting_record.extra) >= 2:
                result["elapsed"] = accounting_record.extra[0]
                result["timelimit"] = accounting_record.extra[1]
            if assessment.log_eligible:
                try:
                    tail = client.tail_log(_job_log_path(job), 15).decode(errors="replace").splitlines()
                    detail += ["--- log tail ---", *tail[-15:]]
                except (AuthRequired, HostKeyChanged):
                    raise
                except HpcAllocError:
                    pass
        else:
            diagnosis = assessment.detail or "scheduler evidence is uncertain"
        result["diagnosis"] = diagnosis
        result["detail"] = detail
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    info(
        f"{result.get('cluster')}:{result.get('jobid') or result.get('operation_id')}: "
        f"{result['status']} — {result['diagnosis']}"
    )
    for line in result.get("detail", []):
        print(f"  {line}")
    return 0


def _active_allocation(
    ctx: Any,
    paths: AppPaths,
    entrypoint: Path,
    target: str | None,
    explicit_cluster: str | None,
) -> tuple[Any, Any]:
    from .lifecycle import AssessmentPhase
    from .monitor import JobMonitor

    job = (
        _resolve_managed_job(
            ctx,
            target,
            explicit_cluster=explicit_cluster,
            kind=JobKind.ALLOCATION,
        )
        if target
        else _default_allocation(ctx, explicit_cluster)
    )
    if job.ref is None:
        raise StateConflict(f"allocation submission {job.operation_id} is unresolved")
    transport, client = _services(ctx, paths, entrypoint, job.cluster)
    transport.bootstrap(job.cluster)
    assessment = JobMonitor(client).assess(job).assessment
    job, assessment, _tracker = _persist_reconciled_assessment(
        ctx=ctx,
        paths=paths,
        client=client,
        job=job,
        assessment=assessment,
        monotonic_evidence=assessment,
    )
    if assessment.phase != AssessmentPhase.ACTIVE or not assessment.current_node:
        raise HpcAllocError(
            f"allocation {job.cluster}:{job.logical_name} is {assessment.phase.value}; "
            "run hpc-alloc status"
        )
    return job, transport


def cmd_ssh(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    tokens = list(args.args)
    target: str | None = None
    if tokens and tokens[0] == "--":
        tokens = tokens[1:]
    elif tokens:
        target = tokens.pop(0)
        if tokens and tokens[0] == "--":
            tokens = tokens[1:]
    job, transport = _active_allocation(ctx, paths, entrypoint, target, args.cluster)
    alias = allocation_alias(job.cluster, job.logical_name)
    transport.require_node(alias)
    from .ssh import ssh_argv

    argv = ssh_argv(
        alias,
        _remote_command(tokens) if tokens else None,
        batch=bool(tokens),
    )
    try:
        os.execvp(argv[0], argv)
    except OSError as exc:
        raise LocalToolUnavailable(f"cannot execute ssh: {exc}") from exc


def _safe_remote_sync_path(path: str) -> str:
    """Reject a remote rsync path the remote login shell would re-interpret.

    rsync passes its remote path through the remote shell, and neither macOS's
    openrsync nor the `--protect-args` flag it lacks can prevent that.  So a path
    containing a space split into several arguments, and `*`, `$VAR`, `` `cmd` ``
    and `$(cmd)` were expanded or executed on the cluster.  With `--delete` on a
    push, a mis-split destination silently deleted files under the *wrong* remote
    directory and still exited 0.

    The tool depends on that shell pass for one thing only -- the documented
    leading `~` -- so rather than quote the path (which would break `~`), restrict
    it to characters the shell cannot do anything with.
    """

    if not path:
        raise ConfigInvalid("remote sync path must not be empty")
    if _REMOTE_SYNC_PATH.fullmatch(path) is None:
        raise ConfigInvalid(
            f"unsafe remote sync path {path!r}: the remote shell re-parses this "
            "path, so it may contain only letters, digits and _@%+=:,./~- "
            "(no spaces, quotes, globs, or $(...) substitutions)"
        )
    return path


def cmd_sync(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    from .ssh import ssh_transport_argv

    job, transport = _active_allocation(ctx, paths, entrypoint, args.target, args.cluster)
    alias = allocation_alias(job.cluster, job.logical_name)
    transport.require_node(alias)
    source, destination = args.src, args.dst
    if args.pull:
        source = f"{alias}:{_safe_remote_sync_path(source)}"
    else:
        destination = f"{alias}:{_safe_remote_sync_path(destination)}"
    # rsync hands the remote spec to the remote *login shell*, which re-splits
    # and expands it.  The documented `~/project` form depends on exactly that,
    # so the path is deliberately left unquoted -- and _safe_remote_sync_path is
    # what makes the shell's re-parse a no-op for everything else.  `-e` gets the
    # same hardened transport as every other SSH call site rather than a bare
    # `ssh`, so a dead master cannot drop the transfer to an interactive password
    # prompt or hang without a connect timeout.
    command = ["rsync", "-az", "-e", shlex.join(ssh_transport_argv())]
    if args.delete:
        command.append("--delete")
    command += ["--", source, destination]
    info(shlex.join(command))
    try:
        completed = subprocess.run(command)
    except OSError as exc:
        raise LocalToolUnavailable(f"cannot execute rsync: {exc}") from exc
    status = completed.returncode
    if status < 0:
        # A signal-killed child reports -N.  Returned verbatim it reaches the
        # shell as 256-N (241 for SIGTERM, 247 for SIGKILL) -- neither rsync's
        # status, which this command contracts to pass through, nor any
        # documented hpc-alloc exit code.  Report the conventional 128+N.
        status = 128 - status
    return status


def _recover_submission(ctx: Any, client: Any, operation: Any, job: Any) -> bool:
    from .ssh import AuthMode

    scan = client.scan(auth=AuthMode.NONINTERACTIVE)
    candidates = [
        row
        for row in scan.rows
        if row.comment == job.slurm_comment and row.name == job.slurm_job_name
    ]
    unsupported = [row.job_id for row in candidates if not (row.job_id.isascii() and row.job_id.isdigit())]
    if unsupported:
        raise AmbiguousSubmission(
            f"operation {operation.operation_id} matches unsupported non-scalar live jobs: "
            + ", ".join(unsupported)
        )
    matches = [row.job_id for row in candidates]
    if len(matches) > 1:
        raise AmbiguousSubmission(
            f"operation {operation.operation_id} matches multiple live jobs: {', '.join(matches)}"
        )
    if len(matches) == 1:
        ctx.state.acknowledge_submission(operation.operation_id, matches[0])
        info(f"adopted {operation.cluster}:{matches[0]} for operation {operation.operation_id}")
        return True
    finder = getattr(client, "find_accounting_by_name", None)
    record = finder(job.slurm_job_name, auth=AuthMode.NONINTERACTIVE) if finder else None
    if record is not None:
        from .models import JobRef

        recovered_ref = JobRef(
            cluster=operation.cluster,
            job_id=record.job_id,
            owner_id=job.owner_id,
            operation_id=operation.operation_id,
            slurm_job_name=job.slurm_job_name,
            slurm_comment=job.slurm_comment,
        )
        client.verify_accounting_identity(recovered_ref, record.job_name, record.comment)
        adopted = ctx.state.acknowledge_submission(operation.operation_id, record.job_id)
        if record.final:
            # Recovery must not maintain a second scheduler-state taxonomy.
            # Feed authoritative accounting through the same lifecycle engine
            # used by normal monitoring so start history and log eligibility
            # stay identical for states such as BOOT_FAIL and REVOKED.
            from .lifecycle import EvidenceEvent
            from .monitor import JobMonitor

            tracker = JobMonitor.tracker(adopted)
            assessment = tracker.accept(EvidenceEvent.final(record))
            adopted, assessment, _tracker = _persist_reconciled_assessment(
                ctx=ctx,
                paths=None,
                client=client,
                job=adopted,
                assessment=assessment,
                monotonic_evidence=assessment,
                synchronize_projection=False,
            )
        info(f"recovered accounting job {operation.cluster}:{adopted.job_id}")
        return True
    return False


def _has_durable_scheduler_final(job: Any) -> bool:
    return (
        job.phase is JobPhase.FINAL
        and job.final_source in {FinalSource.ACCOUNTING, FinalSource.CONFIRMED_QUEUE}
    )


def _can_recover_cancel_locally(operation: Any, job: Any) -> bool:
    return operation.kind is OperationKind.CANCEL and (
        operation.phase is OperationPhase.CANCEL_PENDING
        or _has_durable_scheduler_final(job)
    )


def _prioritize_local_cancel_recoveries(ctx: Any, operations: list[Any]) -> list[Any]:
    """Stable-partition bulk recovery so local cancellation work cannot be stranded."""

    local: list[Any] = []
    remote: list[Any] = []
    for operation in operations:
        locally_recoverable = (
            operation.kind is OperationKind.CANCEL
            and operation.phase is OperationPhase.CANCEL_PENDING
        )
        if operation.kind is OperationKind.CANCEL and not locally_recoverable:
            job = ctx.state.get_job(operation.target_job_operation_id)
            locally_recoverable = _can_recover_cancel_locally(operation, job)
        (local if locally_recoverable else remote).append(operation)
    return local + remote


def _best_effort_recover_local_cancellations(
    ctx: Any,
    paths: AppPaths,
    operations: list[Any],
) -> None:
    """Close newly local work after a bulk remote failure without masking it."""

    for selected_operation in operations:
        try:
            with operation_scope_lock(
                paths.operation_locks_dir,
                selected_operation.operation_id,
                blocking=False,
            ):
                operation = ctx.state.get_operation(selected_operation.operation_id)
                if not operation.unresolved:
                    continue
                job = ctx.state.get_job(operation.target_job_operation_id)
                if _can_recover_cancel_locally(operation, job):
                    _recover_cancel(ctx, None, operation, job)
        except Exception:
            # The remote failure remains primary.  In particular, an active
            # operation lock must not turn this bounded local sweep into a wait
            # or replace the exact access/transport error being propagated.
            #
            # Catch Exception, never BaseException: this loop performs durable
            # cancellation writes, so swallowing KeyboardInterrupt/SystemExit
            # would let a Ctrl-C (or the SIGTERM/SIGHUP the CLI converts into
            # one) keep mutating the remaining operations instead of stopping.
            continue


def _resolve_cancel_from_durable_final(
    ctx: Any,
    operation: Any,
    job: Any,
) -> bool:
    """Atomically close ambiguity from scheduler-final evidence already stored."""

    current = job
    while _has_durable_scheduler_final(current):
        source = current.final_source
        assert source is not None
        lifecycle_evidence: dict[str, Any] = {
            "ever_started": True if current.ever_started else None,
            "last_node": current.last_node,
            "observation_epoch": current.observation_epoch,
        }
        if source is FinalSource.CONFIRMED_QUEUE:
            lifecycle_evidence["evidence_provenance"] = current.evidence_provenance
            lifecycle_evidence["evidence_detail"] = current.evidence_detail
        try:
            ctx.state.resolve_operation(
                operation.operation_id,
                final_source=source,
                expected_target_updated_at=current.updated_at,
                detail="local cancellation recovery adopted durable final scheduler evidence",
                terminal_state=current.terminal_state,
                exit_code=current.exit_code,
                **lifecycle_evidence,
            )
        except LifecycleRevisionConflict:
            # A concurrent observer may upgrade confirmed queue evidence to
            # accounting.  Rebase on that exact durable row; finality itself
            # is monotonic and no scheduler access is needed.
            current = ctx.state.get_job(operation.target_job_operation_id)
            continue
        info(f"recovered final cancellation {operation.cluster}:{operation.job_id}")
        return True
    return False


def _recover_cancel(ctx: Any, client: Any | None, operation: Any, job: Any) -> bool:
    """Reconcile a cancellation using local phase and read-only evidence only."""

    from .errors import ProtocolViolation
    from .monitor import JobMonitor
    from .ssh import AuthMode

    if operation.phase is OperationPhase.CANCEL_PENDING:
        # mark_cancel_dispatching is committed before execute_cancel.  A
        # surviving CANCEL_PENDING row therefore proves no mutation crossed
        # this process boundary and can be closed without contacting Slurm.
        ctx.state.fail_cancel_operation(
            operation.operation_id,
            "recovery closed an undispatched cancellation; no remote mutation was issued",
        )
        info(f"closed undispatched cancellation {operation.operation_id}")
        return True
    if operation.phase is not OperationPhase.AMBIGUOUS:
        raise StateConflict(
            f"cancellation {operation.operation_id} is not recoverable from {operation.phase.value}"
        )
    if _has_durable_scheduler_final(job):
        return _resolve_cancel_from_durable_final(ctx, operation, job)
    if job.ref is None:
        return False
    if client is None:
        raise StateConflict("ambiguous cancellation recovery requires a read-only scheduler client")

    # JobMonitor performs only exact queue/accounting reads.  If another
    # observer advances the target before the atomic resolution, discard the
    # stale verdict and assess the fresh durable revision.  Recovery remains
    # read-only throughout and never replays the cancellation mutation.
    current = job
    while True:
        assessment = JobMonitor(client).assess(
            current,
            auth=AuthMode.NONINTERACTIVE,
            confirm=True,
        ).assessment
        observed = assessment.scheduler_state or assessment.phase.value
        if assessment.uncertain:
            # No usable observation at all.  The cancellation must stay guarded:
            # releasing it on uncertainty could let a retry dispatch a second
            # mutation while the first may still have been delivered.
            ctx.state.mark_cancel_ambiguous(
                operation.operation_id,
                f"read-only recovery remains inconclusive ({observed})",
            )
            return False
        if not assessment.final:
            # The job was positively observed still alive, which proves the
            # ambiguous cancellation never took effect -- cancelling terminates a
            # job, it does not requeue one.  Resolving the operation as failed
            # releases the one-pending-cancel index so an explicit retry can
            # dispatch a fresh, freshly-guarded cancellation.  Unlike a
            # submission, a repeated cancellation is idempotent, so the extreme
            # conservatism the submit path needs is not warranted here.
            #
            # Re-marking it AMBIGUOUS (the old behaviour) held that index
            # forever: every later `down`/`cancel` hit StateConflict("job
            # already has a pending cancellation") and every later `recover`
            # landed back here, so a live GPU allocation could never be released
            # through the CLI at all -- the user had to wait out its walltime.
            ctx.state.fail_cancel_operation(
                operation.operation_id,
                f"read-only recovery observed the job still {observed}; "
                "the cancellation did not take effect",
            )
            info(
                f"cancellation {operation.operation_id} did not take effect — "
                f"{operation.cluster}:{operation.job_id} is still {observed}; "
                f"release it with `hpc-alloc cancel {operation.cluster}:{operation.job_id}`"
            )
            return True
        if assessment.final_source is None:
            raise ProtocolViolation("final cancellation recovery evidence lacks provenance")
        try:
            ctx.state.resolve_operation(
                operation.operation_id,
                final_source=assessment.final_source,
                expected_target_updated_at=current.updated_at,
                detail="read-only cancellation recovery confirmed the job is final",
                terminal_state=assessment.terminal_state,
                exit_code=assessment.exit_code,
                **_cancellation_lifecycle_evidence(assessment),
            )
        except LifecycleRevisionConflict:
            current = ctx.state.get_job(job.operation_id)
            continue
        break
    info(f"recovered final cancellation {operation.cluster}:{operation.job_id}")
    return True


def cmd_recover(
    args: Any, *, ctx: Any, paths: AppPaths, entrypoint: Path, **_kwargs: Any
) -> int:
    if args.operation_id:
        operation = ctx.state.get_operation(args.operation_id)
        if args.cluster:
            requested_cluster = ctx.config.resolve_cluster(args.cluster)
            if operation.cluster != requested_cluster:
                raise IdentityMismatch(
                    f"operation {operation.operation_id} belongs to cluster "
                    f"{operation.cluster!r}, not requested cluster {requested_cluster!r}"
                )
        operations = [operation]
    else:
        operations = ctx.state.list_unresolved_operations()
        if args.cluster:
            requested_cluster = ctx.config.resolve_cluster(args.cluster)
            operations = [
                operation
                for operation in operations
                if operation.cluster == requested_cluster
            ]
        operations = _prioritize_local_cancel_recoveries(ctx, operations)

    with _ssh_projection_scope(ctx, paths):
        if args.operation_id and not operations[0].unresolved:
            operation = operations[0]
            if args.abandon:
                raise StateConflict(
                    f"operation {operation.operation_id} is not unresolved "
                    f"(durable phase {operation.phase.value})"
                )
            info(
                f"operation {operation.operation_id} is already resolved "
                f"with durable phase {operation.phase.value}"
            )
            return 0
        if not operations:
            info("no unresolved operations")
            return 0
        if args.abandon:
            if args.operation_id is None or len(operations) != 1:
                raise StateConflict("--abandon requires one explicit operation ID")
            operation = operations[0]
            if not args.yes:
                if not sys.stdin.isatty():
                    raise StateConflict("--abandon requires --yes without a terminal")
                answer = input(
                    f"Abandon {operation.operation_id}? A remote orphan may remain [y/N]: "
                ).strip().lower()
                if answer not in {"y", "yes"}:
                    info("not abandoned")
                    return 1
            with operation_scope_lock(
                paths.operation_locks_dir,
                operation.operation_id,
                blocking=False,
            ):
                operation = ctx.state.get_operation(operation.operation_id)
                if not operation.unresolved:
                    raise StateConflict(
                        f"operation {operation.operation_id} is not unresolved "
                        f"(durable phase {operation.phase.value})"
                    )
                ctx.state.abandon_operation(
                    operation.operation_id,
                    "explicitly abandoned; remote side effect was not changed",
                )
            info(f"abandoned local operation {operation.operation_id}")
            return 0
        unresolved = 0
        for index, selected_operation in enumerate(operations):
            remote_attempted = False
            try:
                with operation_scope_lock(
                    paths.operation_locks_dir,
                    selected_operation.operation_id,
                    blocking=False,
                ):
                    operation = ctx.state.get_operation(selected_operation.operation_id)
                    if not operation.unresolved:
                        if args.operation_id:
                            info(
                                f"operation {operation.operation_id} is already resolved "
                                f"with durable phase {operation.phase.value}"
                            )
                        continue
                    job = ctx.state.get_job(operation.target_job_operation_id)
                    if _can_recover_cancel_locally(operation, job):
                        resolved = _recover_cancel(ctx, None, operation, job)
                    else:
                        remote_attempted = True
                        transport, client = _services(
                            ctx,
                            paths,
                            entrypoint,
                            operation.cluster,
                        )
                        transport.bootstrap(operation.cluster)
                        if operation.kind == OperationKind.SUBMIT:
                            resolved = _recover_submission(ctx, client, operation, job)
                        else:
                            resolved = _recover_cancel(ctx, client, operation, job)
                    if not resolved:
                        unresolved += 1
                        info(
                            f"operation {operation.operation_id} remains unresolved; "
                            "no conclusive remote outcome found"
                        )
            except OperationBusy as exc:
                if args.operation_id is not None:
                    raise
                # A bulk sweep must not die because one operation is in use by a
                # live process.  A concurrent `down` holds its cancel
                # operation's lock for the whole guarded cancel, and
                # _prioritize_local_cancel_recoveries deterministically sorts
                # exactly that operation to the front -- and because the lock
                # conflict is raised by the `with` itself, before
                # remote_attempted is set, the handler below skipped even the
                # local fallback and aborted the sweep on item 0, stranding
                # every genuinely orphaned operation behind it.  The busy
                # operation is owned by the process holding it; leave it there.
                unresolved += 1
                info(f"skipped {selected_operation.operation_id}: {exc}")
                continue
            except HpcAllocError:
                if args.operation_id is None and remote_attempted:
                    _best_effort_recover_local_cancellations(
                        ctx,
                        paths,
                        operations[index:],
                    )
                raise
        return 1 if unresolved else 0


HANDLERS: dict[str, Callable[..., int]] = {
    "setup": cmd_setup,
    "config": cmd_config,
    "connect": cmd_connect,
    "up": cmd_up,
    "run": cmd_run,
    "status": cmd_status,
    "why": cmd_why,
    "logs": cmd_logs,
    "cancel": cmd_cancel,
    "down": cmd_down,
    "ssh": cmd_ssh,
    "sync": cmd_sync,
    "avail": cmd_avail,
    "partitions": cmd_partitions,
    "recover": cmd_recover,
}


def dispatch(args: Any, *, entrypoint: Path) -> int:
    paths = AppPaths.for_home()
    handler = HANDLERS[args.command_name]
    if args.command_name == "setup":
        return handler(args, paths=paths, entrypoint=entrypoint)
    lock_free = args.command_name == "config" or (
        args.command_name in {"up", "run"}
        and getattr(args, "dry_run", False)
    )
    if lock_free:
        ctx = _load_context(args, paths)
        return handler(args, ctx=ctx, paths=paths, entrypoint=entrypoint)
    with configuration_scope_lock(paths.config_scope_lock, exclusive=False):
        ctx = _load_context(args, paths)
        return handler(args, ctx=ctx, paths=paths, entrypoint=entrypoint)
