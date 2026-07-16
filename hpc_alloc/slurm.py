"""Typed Slurm protocol over :mod:`hpc_alloc.ssh`.

All scheduler command construction and parsing lives here.  Control output is
wrapped in a random, length-checked envelope: login banners are discarded and
arbitrary command or log bytes cannot be mistaken for scheduler metadata.
"""

from __future__ import annotations

import math
import re
import secrets
import shlex
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Callable, Iterable, Mapping, Sequence

from .errors import (
    AmbiguousSubmission,
    AuthRequired,
    HpcAllocError,
    HostKeyChanged,
    IdentityMismatch,
    JobIdReused,
    ProtocolViolation,
    RemoteCommandFailed,
    SchedulerUnavailable,
)
from .models import JobKind, JobRef
from .ownership import COMPUTE_NODE_RE, format_tag, slurm_job_name
from .ssh import AuthMode, RetryPolicy, SshTransport


FINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

# The final states a job can reach WITHOUT ever having run: the node never booted
# (BOOT_FAIL), or the job was cancelled / hit its start deadline / had a
# federation reservation revoked.  Every other final state implies execution
# began.  This is the conservative split -- ever_started gates log streaming, so
# a state where a start is uncertain is treated as "did not start" rather than
# reaching for a log that may not exist.  Kept beside FINAL_STATES, and the two
# are the sole source of the lifecycle layer's "proves started" taxonomy, so they
# cannot drift: a newly added final state lands on exactly one side, and a
# partition test in the lifecycle suite fails until it is classified.
TERMINAL_WITHOUT_START = frozenset({"BOOT_FAIL", "CANCELLED", "DEADLINE", "REVOKED"})

# The terminal states Slurm actually requeues a job out of.  With the default
# JobRequeue=1, a NODE_FAIL job is restarted under the same job ID, and a
# PreemptMode=REQUEUE preemption does the same -- so observing one of these
# states, in the queue *or* in accounting, does not prove the job is dead.
# Consumers must therefore require a second, independent observation before
# accepting one as final; a single observation and the accounting read taken in
# the same poll cycle both describe the same instant inside the requeue window.
REQUEUE_ELIGIBLE_FINAL = frozenset({"NODE_FAIL", "PREEMPTED"})

_SQUEUE_INVALID_SINGLETON = "slurm_load_jobs error: Invalid job id specified"
_SACCT_BASE_FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "JobName%255",
    "Comment%255",
)
MAX_LOG_CHUNK_BYTES = 1024 * 1024
_LOG_CHUNK_SIGPIPE_STATUS = 85
_MAX_QUEUE_PAYLOAD_BYTES = 32 * 1024 * 1024
_MAX_QUEUE_FIELD_BYTES = 64 * 1024
_DRY_RUN_HOME = '"${HOME:?}"'


def _dry_run_path(path: str) -> str:
    """Quote a remote path for a paste-ready dry-run shell command.

    SSH commands start in the remote user's home directory.  Keep that home
    symbolic so the displayed command can be pasted into the target login
    shell, while quoting the user-controlled suffix independently from the
    expansion itself.
    """

    if PurePosixPath(path).is_absolute():
        return shlex.quote(path)
    if path == "~":
        suffix = ""
    elif path.startswith("~/"):
        suffix = path[1:]
    else:
        suffix = f"/{path}"
    return f"{_DRY_RUN_HOME}{shlex.quote(suffix)}"


@dataclass(frozen=True, slots=True)
class QueueRow:
    job_id: str
    state: str
    node: str | None
    reason: str
    time_left: str
    partition: str
    name: str
    submitted_at: str
    comment: str


@dataclass(frozen=True, slots=True)
class RawQueueRow:
    """One broad-scan row before managed-job semantic validation."""

    job_id: str
    state: str
    node: str
    reason: str
    time_left: str
    partition: str
    name: str
    submitted_at: str
    comment: str


@dataclass(frozen=True, slots=True)
class RawQueueScan:
    """Semantically tolerant queue discovery result.

    Array IDs, multi-node expressions, and other rows outside hpc-alloc's
    managed singleton contract remain available to discovery callers.  The
    transport grammar itself is still strict: malformed, oversized, or
    control-bearing rows fail the entire scan.  Callers must never use this
    broad result as negative lifecycle evidence; :meth:`SlurmClient.observe`
    provides the strict targeted contract.
    """

    rows: tuple[RawQueueRow, ...]
    remote_time: str | None = None

    def items(self):
        return ((row.job_id, row) for row in self.rows)


@dataclass(frozen=True, slots=True)
class AccountingRecord:
    job_id: str
    state: str
    exit_code: str
    job_name: str
    comment: str
    extra: tuple[str, ...] = ()

    @property
    def state_code(self) -> str:
        words = self.state.split()
        return words[0].rstrip("+") if words else ""

    @property
    def final(self) -> bool:
        return self.state_code in FINAL_STATES


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    job_id: str
    raw_output: str


@dataclass(frozen=True, slots=True)
class SubmissionSpec:
    """Semantic batch submission; command construction remains in this module."""

    operation_id: str
    owner_id: str
    owner_host: str
    kind: JobKind
    logical_name: str | None
    partition: str
    walltime: str
    cpus: int
    logfile: str
    wrap: str
    mem: str | None = None
    gpus: str | None = None
    constraint: str | None = None
    chdir: str | None = None
    log_directory: str = ".hpc-alloc"

    @property
    def job_name(self) -> str:
        return slurm_job_name(self.kind.value, self.operation_id)

    @property
    def comment(self) -> str:
        return format_tag(
            self.owner_id,
            self.operation_id,
            self.owner_host,
            self.kind.value,
            self.logical_name,
        )

    def _preparation(self, quote: Callable[[str], str]) -> str:
        # Single source for the retention window and predicate so the dry-run
        # rendering and the real setup can never diverge; only the quoting
        # differs between them.
        log_directory = quote(self.log_directory)
        return (
            f"mkdir -p {log_directory} && "
            f"(find {log_directory} -name '*.log' -mtime +30 -delete "
            "2>/dev/null || true)"
        )

    def preparation_command(self) -> str:
        """Return idempotent remote setup that never invokes ``sbatch``."""

        return self._preparation(shlex.quote)

    def _sbatch_argv(self) -> list[str]:
        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={self.job_name}",
            f"--comment={self.comment}",
            f"--time={self.walltime}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={self.cpus}",
            "--output",
            self.logfile,
            f"--partition={self.partition}",
        ]
        if self.mem:
            argv.append(f"--mem={self.mem}")
        if self.gpus:
            argv.append(f"--gpus={self.gpus}")
        if self.constraint:
            argv.append(f"--constraint={self.constraint}")
        if self.chdir:
            argv.append(f"--chdir={self.chdir}")
        argv += ["--wrap", self.wrap]
        return argv

    def sbatch_command(self) -> str:
        """Return only the one-shot scheduler mutation."""

        argv = self._sbatch_argv()
        return shlex.join(argv)

    def command(self) -> str:
        """Render a paste-ready operation for the target login shell."""

        preparation = self._preparation(_dry_run_path)

        argv = self._sbatch_argv()
        rendered = [shlex.quote(argument) for argument in argv]
        rendered[argv.index("--output") + 1] = _dry_run_path(self.logfile)
        if self.chdir:
            chdir_index = argv.index(f"--chdir={self.chdir}")
            rendered[chdir_index] = f"--chdir={_dry_run_path(self.chdir)}"
        return f"{preparation} && {' '.join(rendered)}"


class CancellationInspectionStatus(StrEnum):
    READY = "READY"
    ALREADY_FINAL = "ALREADY_FINAL"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"


@dataclass(frozen=True, slots=True)
class CancellationInspection:
    status: CancellationInspectionStatus
    detail: str = ""
    final_record: AccountingRecord | None = None
    queue_row: QueueRow | None = None


class CancellationStatus(StrEnum):
    CANCELLED = "CANCELLED"
    IDENTITY_MISMATCH = "IDENTITY_MISMATCH"
    LEFT_QUEUE = "LEFT_QUEUE"
    GUARD_FAILED = "GUARD_FAILED"
    MUTATION_AMBIGUOUS = "MUTATION_AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class CancellationResult:
    status: CancellationStatus
    detail: str = ""


class LogSizeStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    UNREADABLE = "UNREADABLE"


@dataclass(frozen=True, slots=True)
class LogSizeResult:
    status: LogSizeStatus
    size: int | None = None
    detail: str = ""

    @classmethod
    def available(cls, size: int) -> "LogSizeResult":
        return cls(LogSizeStatus.AVAILABLE, size)


@dataclass(frozen=True, slots=True)
class _FramedResult:
    returncode: int
    payload: bytes
    stderr: str


def _job_id(value: JobRef | str | int) -> str:
    raw = str(value.job_id if isinstance(value, JobRef) else value)
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ProtocolViolation(f"invalid Slurm job ID {raw!r}")
    return raw


def _queue_node(raw: str) -> str | None:
    value = raw.strip()
    if not value or value.lower() in {"(null)", "none", "n/a"}:
        return None
    if COMPUTE_NODE_RE.fullmatch(value) is None:
        raise ProtocolViolation(f"squeue returned an invalid compute-node name {value!r}")
    return value


def _valid_exit_code(value: str) -> bool:
    match = re.fullmatch(r"([0-9]{1,3}):([0-9]{1,3})", value)
    return bool(match and int(match.group(1)) <= 255 and int(match.group(2)) <= 255)


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_invalid_singleton(result: _FramedResult, *, returncode: int = 1) -> bool:
    return (
        result.returncode == returncode
        and result.payload == b""
        and result.stderr
        in {
            _SQUEUE_INVALID_SINGLETON,
            _SQUEUE_INVALID_SINGLETON + "\n",
        }
    )


def _strict_queue_row(raw: RawQueueRow) -> QueueRow:
    if re.fullmatch(r"[0-9]+", raw.job_id) is None:
        raise ProtocolViolation(f"squeue returned an invalid job ID {raw.job_id!r}")
    state = raw.state.strip().upper()
    if re.fullmatch(r"[A-Z_]+", state) is None:
        raise ProtocolViolation(f"squeue returned an invalid job state {raw.state!r}")
    if (
        not raw.partition
        or re.fullmatch(r"[A-Za-z0-9_.-]+\*?", raw.partition) is None
        or _has_control(raw.name)
        or _has_control(raw.comment)
    ):
        raise ProtocolViolation("squeue returned unsafe identity or partition fields")
    return QueueRow(
        job_id=raw.job_id,
        state=state,
        node=_queue_node(raw.node),
        reason=raw.reason,
        time_left=raw.time_left,
        partition=raw.partition.rstrip("*"),
        name=raw.name,
        submitted_at=raw.submitted_at,
        comment=raw.comment,
    )


class SlurmClient:
    """One cluster's scheduler, accounting, diagnostic, and log operations."""

    def __init__(
        self,
        transport: SshTransport,
        cluster: str,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        token_factory: Callable[[int], str] = secrets.token_hex,
    ) -> None:
        self.transport = transport
        self.cluster = cluster
        self._sleep = sleeper
        self._token = token_factory

    def _framed(
        self,
        command: str,
        *,
        retry: RetryPolicy,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        timeout: float = 60,
        max_payload_bytes: int | None = None,
    ) -> _FramedResult:
        """Run *command* and decode a random byte-length envelope.

        The marker is emitted after the command has completed and its stdout
        and stderr are captured.  Therefore anything printed before the remote
        command (for example an SSH login banner) is outside the envelope and
        ignored.
        """

        nonce = self._token(16)
        if max_payload_bytes is not None and (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes < 0
        ):
            raise ValueError("frame payload limit must be a non-negative integer")
        marker_text = f"HPC_ALLOC_V2_{nonce}"
        marker = b"\x1e" + marker_text.encode("ascii") + b" "
        temp_template = "${TMPDIR:-/tmp}/hpc-alloc.XXXXXX"
        emit_streams = (
            'if [ "$n" -ge 0 ] 2>/dev/null && [ "$e" -ge 0 ] 2>/dev/null; '
            'then cat "$tmp"; cat "$err"; fi; '
            if max_payload_bytes is None
            else f'if [ "$n" -ge 0 ] 2>/dev/null && [ "$e" -ge 0 ] 2>/dev/null '
                 f'&& [ "$n" -le {max_payload_bytes} ]; '
                 'then cat "$tmp"; cat "$err"; fi; '
        )
        wrapped = (
            f'tmp=$(mktemp "{temp_template}") || {{ '
            f"printf '\\036{marker_text} 125 0 0\\n'; exit 0; }}; "
            f'err=$(mktemp "{temp_template}") || {{ rm -f "$tmp"; '
            f"printf '\\036{marker_text} 125 0 0\\n'; exit 0; }}; "
            "trap 'rm -f \"$tmp\" \"$err\"' EXIT; "
            "trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM; "
            f"( {command} ) >\"$tmp\" 2>\"$err\"; rc=$?; "
            "n=$(wc -c <\"$tmp\") || n=-1; "
            "e=$(wc -c <\"$err\") || e=-1; "
            f"printf '\\036{marker_text} %s %s %s\\n' \"$rc\" \"$n\" \"$e\"; "
            + emit_streams
            + "exit 0"
        )
        result = self.transport.run(
            self.cluster,
            wrapped,
            auth=auth,
            retry=retry,
            timeout=timeout,
            binary=True,
        )
        raw = result.stdout_bytes
        # The wrapper marker precedes the payload.  Use the first marker so an
        # arbitrary binary payload containing the same byte sequence cannot be
        # reinterpreted as a second control header.
        start = raw.find(marker)
        if start < 0:
            detail = result.stderr.strip()
            if result.returncode != 0:
                raise RemoteCommandFailed(
                    f"remote wrapper failed on {self.cluster}: {detail or result.returncode}"
                )
            message = "remote reply did not contain its random frame marker"
            if detail:
                message = f"{message}: {detail}"
            raise ProtocolViolation(message)
        header_start = start + len(marker)
        header_end = raw.find(b"\n", header_start)
        if header_end < 0:
            raise ProtocolViolation("remote reply contained a truncated frame header")
        fields = raw[header_start:header_end].split()
        if len(fields) != 3:
            raise ProtocolViolation("remote reply contained a malformed frame header")
        try:
            returncode, length, stderr_length = (int(field) for field in fields)
        except ValueError as exc:
            raise ProtocolViolation("remote reply frame status was not numeric") from exc
        if (
            returncode < 0
            or returncode > 255
            or length < 0
            or stderr_length < 0
        ):
            raise ProtocolViolation("remote reply frame status was outside its valid range")
        if max_payload_bytes is not None and length > max_payload_bytes:
            raise ProtocolViolation(
                f"remote reply exceeded payload limit ({length} > {max_payload_bytes})"
            )
        streams = raw[header_end + 1 :]
        expected_length = length + stderr_length
        if len(streams) != expected_length:
            raise ProtocolViolation(
                "remote reply length mismatch "
                f"(declared {length} stdout and {stderr_length} stderr, "
                f"received {len(streams)})"
            )
        payload = streams[:length]
        command_stderr = streams[length:].decode("utf-8", errors="replace")
        return _FramedResult(returncode, payload, command_stderr)

    @staticmethod
    def _require_success(result: _FramedResult, operation: str) -> bytes:
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit {result.returncode}"
            raise SchedulerUnavailable(f"{operation} failed on the cluster: {detail}")
        return result.payload

    def _raw_queue(
        self,
        selected_ids: Sequence[str] | None,
        *,
        auth: AuthMode,
    ) -> RawQueueScan:
        delimiter = f"__HPC_{self._token(12)}__"
        meta = f"__HPC_TIME_{self._token(12)}__"
        queue_format = delimiter.join(["%i", "%T", "%N", "%r", "%L", "%P", "%j", "%V", "%k"])
        selected = ""
        if selected_ids is not None:
            selected = f" -j {shlex.quote(','.join(selected_ids))}"
        command = (
            f"TZ=UTC squeue --me -h{selected} -o {shlex.quote(queue_format)} && "
            f"printf '\\n%s' {shlex.quote(meta)} && date -u +%Y-%m-%dT%H:%M:%S"
        )
        framed = self._framed(
            command,
            retry=RetryPolicy.SAFE_READ,
            auth=auth,
            max_payload_bytes=_MAX_QUEUE_PAYLOAD_BYTES,
        )
        # Bouchet's squeue reports a job that has just left the queue as this
        # specific rc=1 error for a singleton `-j` lookup.  That is meaningful
        # absence evidence only under the exact shape below.  Broad snapshots,
        # multi-ID queries, output-bearing replies, and every other scheduler
        # failure remain errors rather than being normalized to an empty queue.
        if (
            selected_ids is not None
            and len(selected_ids) == 1
            and _is_invalid_singleton(framed)
        ):
            return RawQueueScan(())
        try:
            payload = self._require_success(framed, "squeue").decode(
                "utf-8", errors="strict"
            )
        except UnicodeDecodeError as exc:
            raise ProtocolViolation("squeue returned non-UTF-8 output") from exc
        queue_text, found, remote_time = payload.rpartition(meta)
        if not found:
            raise ProtocolViolation("squeue reply omitted its clock record")
        remote_time = remote_time.strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", remote_time):
            raise ProtocolViolation(f"squeue returned an invalid remote clock {remote_time!r}")
        rows: list[RawQueueRow] = []
        for raw_line in queue_text.split("\n"):
            line = raw_line.rstrip("\r")
            if not line:
                continue
            fields = line.split(delimiter)
            if len(fields) != 9:
                raise ProtocolViolation("squeue returned a row with the wrong field count")
            if any(_has_control(field) for field in fields):
                raise ProtocolViolation("squeue returned a row containing control characters")
            if any(
                len(field.encode("utf-8")) > _MAX_QUEUE_FIELD_BYTES
                for field in fields
            ):
                raise ProtocolViolation("squeue returned an oversized field")
            rows.append(RawQueueRow(
                job_id=fields[0],
                state=fields[1],
                node=fields[2],
                reason=fields[3],
                time_left=fields[4],
                partition=fields[5],
                name=fields[6],
                submitted_at=fields[7],
                comment=fields[8],
            ))
        return RawQueueScan(tuple(rows), remote_time)

    def scan(
        self,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> RawQueueScan:
        """Return a tolerant unfiltered scan for discovery and recovery only."""

        return self._raw_queue(None, auth=auth)

    def observe(
        self,
        ref: JobRef | str | int,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> QueueRow | None:
        job_id = _job_id(ref)
        raw = self._raw_queue((job_id,), auth=auth)
        unexpected = [candidate.job_id for candidate in raw.rows if candidate.job_id != job_id]
        if unexpected:
            raise ProtocolViolation(
                f"targeted squeue for job {job_id} returned unexpected job IDs"
            )
        candidates = [candidate for candidate in raw.rows if candidate.job_id == job_id]
        if len(candidates) > 1:
            raise ProtocolViolation(f"targeted squeue returned duplicate job ID {job_id}")
        row = _strict_queue_row(candidates[0]) if candidates else None
        if row is not None and isinstance(ref, JobRef):
            # Always.  This was behind a `verify_live` switch that nothing ever
            # set, and whose only possible effect was to disable the recycled-ID
            # check -- so the sole reason to reach for it would be to silence an
            # IdentityMismatch that is telling the truth.
            self.verify_live_identity(ref, row.name, row.comment)
        return row

    @staticmethod
    def verify_live_identity(ref: JobRef, job_name: str, comment: str) -> None:
        """Verify the complete identity exposed by a live queue row.

        Live observations and mutation guards must never use the weaker
        accounting rule: ``squeue`` preserves the submitted comment, so an
        omission there is itself an identity mismatch.
        """

        name_mismatch = job_name != ref.slurm_job_name
        comment_mismatch = comment != ref.slurm_comment
        if name_mismatch and comment_mismatch:
            raise JobIdReused(
                f"job {ref.cluster}:{ref.job_id} now belongs to another operation"
            )
        if name_mismatch or comment_mismatch:
            raise IdentityMismatch(
                f"job {ref.cluster}:{ref.job_id} no longer matches operation {ref.operation_id}"
            )

    @staticmethod
    def verify_accounting_identity(ref: JobRef, job_name: str, comment: str) -> None:
        """Verify a parent accounting record without trusting a missing field.

        Some Slurm accounting deployments retain ``JobName`` but return an
        empty ``Comment`` even though the live queue row contained it.  The
        persisted v2 job name embeds the full random operation UUID, so that
        exact name is sufficient only when accounting omitted Comment.  A
        nonempty comment remains authoritative and must match byte-for-byte.
        """

        try:
            operation_names = {
                slurm_job_name(JobKind.ALLOCATION.value, ref.operation_id),
                slurm_job_name(JobKind.RUN.value, ref.operation_id),
            }
        except ValueError:
            operation_names = set()
        if (
            ref.slurm_job_name not in operation_names
            or job_name != ref.slurm_job_name
            or (comment != "" and comment != ref.slurm_comment)
        ):
            raise IdentityMismatch(
                f"accounting record {ref.cluster}:{ref.job_id} does not match "
                f"operation {ref.operation_id}"
            )

    def accounting(
        self,
        ref: JobRef | str | int,
        *,
        extra_fields: Sequence[str] = (),
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> AccountingRecord | None:
        job_id = _job_id(ref)
        allowed_extra = {"Elapsed", "Timelimit", "Reason", "Partition", "NodeList"}
        if any(field not in allowed_extra for field in extra_fields):
            raise ProtocolViolation("unsupported sacct field requested")
        fields = (*_SACCT_BASE_FIELDS, *extra_fields)
        command = f"sacct -j {job_id} -X -n -P -o {shlex.quote(','.join(fields))}"
        framed = self._framed(command, retry=RetryPolicy.SAFE_READ, auth=auth)
        payload = self._require_success(framed, "sacct").decode("utf-8", errors="replace")
        records: list[AccountingRecord] = []
        malformed = False
        malformed_exact = False
        for line in payload.split("\n"):
            if not line.strip():
                continue
            values = line.rstrip("\r").split("|")
            if len(values) != len(fields):
                malformed = True
                malformed_exact = malformed_exact or values[0] == job_id
                continue
            if (
                re.fullmatch(r"[0-9]+", values[0]) is None
                or not _valid_exit_code(values[2])
                or not values[1].strip()
                or re.fullmatch(r"[A-Z_]+", AccountingRecord(
                    values[0], values[1], values[2], values[3], values[4]
                ).state_code) is None
                or _has_control(values[1])
                or _has_control(values[3])
                or _has_control(values[4])
            ):
                malformed = True
                malformed_exact = malformed_exact or values[0] == job_id
                continue
            records.append(
                AccountingRecord(
                    job_id=values[0],
                    state=values[1],
                    exit_code=values[2],
                    job_name=values[3],
                    comment=values[4],
                    extra=tuple(values[5:]),
                )
            )
        exact_id = [record for record in records if record.job_id == job_id]
        if malformed_exact:
            raise ProtocolViolation(f"sacct returned a malformed record for job {job_id}")
        if isinstance(ref, JobRef):
            # A recycled ID normally differs in both operation-derived
            # identity fields.  If accounting instead preserves our exact
            # comment under a foreign name, the record is internally
            # inconsistent with the persisted identity and must fail closed
            # rather than being dismissed as unrelated history.
            if any(
                record.job_name != ref.slurm_job_name
                and record.comment == ref.slurm_comment
                for record in exact_id
            ):
                raise IdentityMismatch(
                    f"accounting record {ref.cluster}:{ref.job_id} does not match "
                    f"operation {ref.operation_id}"
                )
            # Slurm job IDs are recycled.  sacct can retain an older parent
            # record with the same numeric ID, so select the persisted
            # operation-derived name before deciding whether the reply is a
            # duplicate.  Comment verification remains strict below.
            exact = [record for record in exact_id if record.job_name == ref.slurm_job_name]
        else:
            exact = exact_id
        if len(exact) > 1:
            raise ProtocolViolation(f"sacct returned duplicate parent records for job {job_id}")
        if not exact:
            if (
                isinstance(ref, JobRef)
                and exact_id
                and len(exact_id) == len(records)
                and not malformed
            ):
                # Every trustworthy record belongs to another operation that
                # previously used this recycled numeric ID.  Those rows say
                # nothing about the exact persisted job and are equivalent to
                # accounting lag/missing data.  A matching JobName continues
                # through the strict Comment check below.
                return None
            if malformed or records:
                raise ProtocolViolation(f"sacct returned no trustworthy exact record for job {job_id}")
            return None
        record = exact[0]
        if isinstance(ref, JobRef):
            self.verify_accounting_identity(ref, record.job_name, record.comment)
        return record

    def final(
        self,
        ref: JobRef | str | int,
        *,
        attempts: Sequence[float] = (0,),
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        extra_fields: Sequence[str] = (),
    ) -> AccountingRecord | None:
        for delay in attempts:
            if delay:
                self._sleep(delay)
            record = self.accounting(ref, extra_fields=extra_fields, auth=auth)
            if record is not None and record.final:
                return record
        return None

    def find_accounting_by_name(
        self,
        job_name: str,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> AccountingRecord | None:
        """Find one recent parent record by exact v2 Slurm job name.

        This is the recovery path for an ``sbatch`` that may have committed
        before its reply was lost.  Duplicate exact records are deliberately a
        protocol error: recovery must never guess which remote mutation to
        adopt.
        """

        if (
            not job_name
            or any(ord(character) < 32 or ord(character) == 127 for character in job_name)
            or len(job_name) > 255
        ):
            raise ProtocolViolation(f"invalid accounting job name {job_name!r}")
        fields = _SACCT_BASE_FIELDS
        command = (
            "sacct -X -n -P -S now-30days "
            f"--name {shlex.quote(job_name)} -o {shlex.quote(','.join(fields))}"
        )
        framed = self._framed(command, retry=RetryPolicy.SAFE_READ, auth=auth)
        payload = self._require_success(framed, "sacct recovery").decode(
            "utf-8", errors="replace"
        )
        matches: list[AccountingRecord] = []
        malformed = False
        for line in payload.split("\n"):
            if not line.strip():
                continue
            values = line.rstrip("\r").split("|")
            if (
                len(values) != len(fields)
                or re.fullmatch(r"[0-9]+", values[0]) is None
                or not _valid_exit_code(values[2])
                or not values[1].strip()
                or re.fullmatch(r"[A-Z_]+", AccountingRecord(
                    values[0], values[1], values[2], values[3], values[4]
                ).state_code) is None
                or _has_control(values[1])
                or _has_control(values[3])
                or _has_control(values[4])
            ):
                malformed = True
                continue
            if values[3] != job_name:
                continue
            matches.append(
                AccountingRecord(
                    job_id=values[0],
                    state=values[1],
                    exit_code=values[2],
                    job_name=values[3],
                    comment=values[4],
                )
            )
        if malformed:
            raise ProtocolViolation("sacct recovery returned a malformed record")
        if len(matches) > 1:
            raise ProtocolViolation(
                f"sacct returned multiple records named {job_name!r}; refusing ambiguous recovery"
            )
        return matches[0] if matches else None

    @staticmethod
    def _submission_command(
        command_or_argv: SubmissionSpec | str | Sequence[str],
    ) -> str:
        if isinstance(command_or_argv, SubmissionSpec):
            return command_or_argv.sbatch_command()
        if isinstance(command_or_argv, str):
            return command_or_argv
        argv = [str(item) for item in command_or_argv]
        if not argv or argv[0] != "sbatch":
            argv.insert(0, "sbatch")
        if "--parsable" not in argv:
            argv.insert(1, "--parsable")
        return shlex.join(argv)

    def prepare_submission(
        self,
        spec: SubmissionSpec,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        timeout: float = 60,
    ) -> None:
        """Perform retry-safe remote setup before the submission intent.

        The generated command cannot invoke ``sbatch``.  Callers may therefore
        treat every failure here as definitively pre-submission and must call
        this method before crossing their durable mutation boundary.
        """

        framed = self._framed(
            spec.preparation_command(),
            retry=RetryPolicy.SAFE_READ,
            auth=auth,
            timeout=timeout,
        )
        if framed.returncode != 0:
            detail = framed.stderr.strip() or f"exit {framed.returncode}"
            raise RemoteCommandFailed(f"submission preparation failed: {detail}")
        if framed.payload != b"":
            raise ProtocolViolation("submission preparation returned unexpected output")

    def submit(
        self,
        command_or_argv: SubmissionSpec | str | Sequence[str],
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        timeout: float = 60,
    ) -> SubmissionResult:
        command = self._submission_command(command_or_argv)
        try:
            framed = self._framed(
                command,
                retry=RetryPolicy.NEVER,
                auth=auth,
                timeout=timeout,
            )
        except (AuthRequired, HostKeyChanged):
            # SshTransport raises these only when authentication/host-key
            # verification rejects the connection before the remote command
            # can be dispatched.  Preserve that definitive no-mutation fact.
            raise
        except HpcAllocError as exc:
            raise AmbiguousSubmission(
                "sbatch may have committed, but its exact reply was lost; reconcile by operation ID"
            ) from exc
        if framed.returncode != 0:
            detail = framed.stderr.strip()
            if not detail and framed.payload:
                detail = framed.payload.decode("utf-8", errors="replace").strip()
            raise AmbiguousSubmission(
                "sbatch returned a nonzero status after dispatch and may have committed"
                + (f": {detail}" if detail else f" (exit {framed.returncode})")
            )
        output = framed.payload.decode("utf-8", errors="replace").strip()
        lines = output.splitlines()
        match = re.fullmatch(r"([0-9]+)(?:;[^\s;]+)?", lines[0] if len(lines) == 1 else "")
        if match is None:
            raise AmbiguousSubmission(
                f"sbatch returned an untrustworthy reply after possible commit: {output!r}"
            )
        return SubmissionResult(match.group(1), output)

    def inspect_cancel(
        self,
        ref: JobRef,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        confirmation_delay: float = 3,
    ) -> CancellationInspection:
        """Read scheduler state before a cancellation is eligible to dispatch.

        This method never builds or invokes ``scancel``.  Any exception it
        raises is therefore a definitive pre-mutation failure that can safely
        close a locally prepared cancellation intent.  Queue absence is
        accepted only twice consecutively, with exact final accounting allowed
        to short-circuit after either observation.
        """

        if (
            isinstance(confirmation_delay, bool)
            or not isinstance(confirmation_delay, (int, float))
            or not math.isfinite(confirmation_delay)
            or confirmation_delay < 0
        ):
            raise ValueError("cancellation confirmation delay must be finite and non-negative")
        for observation in range(2):
            try:
                row = self.observe(ref, auth=auth)
            except IdentityMismatch as exc:
                return CancellationInspection(
                    CancellationInspectionStatus.IDENTITY_MISMATCH,
                    str(exc),
                )
            if row is not None:
                return CancellationInspection(
                    CancellationInspectionStatus.READY,
                    queue_row=row,
                )
            try:
                record = self.final(ref, auth=auth)
            except IdentityMismatch as exc:
                return CancellationInspection(
                    CancellationInspectionStatus.IDENTITY_MISMATCH,
                    str(exc),
                )
            if record is not None:
                # A NODE_FAIL or PREEMPTED record on the FIRST look may be the
                # reaped failed attempt of a job Slurm is requeueing under the
                # same ID.  Accepting it here resolves the cancellation as
                # already-final and never issues the mutation, so the requeued
                # instance runs on untracked -- the exact single-observation reap
                # the queue and streaming paths defer via
                # awaits_requeue_confirmation.  This loop is that same
                # two-observation rule, so require the second observation for a
                # requeue-eligible state: it either sees the requeued instance
                # back in the queue (-> READY, and it gets cancelled) or confirms
                # the death.  A genuinely terminal state (COMPLETED, CANCELLED,
                # ...) still resolves on the first look.
                if observation == 0 and record.state_code in REQUEUE_ELIGIBLE_FINAL:
                    self._sleep(confirmation_delay)
                    continue
                return CancellationInspection(
                    CancellationInspectionStatus.ALREADY_FINAL,
                    f"job ended as {record.state}",
                    record,
                )
            if observation == 0:
                self._sleep(confirmation_delay)
        return CancellationInspection(
            CancellationInspectionStatus.CONFIRMED_ABSENT,
            "job was absent from two exact queue observations with no final accounting",
        )

    def execute_cancel(
        self,
        ref: JobRef,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        timeout: float = 60,
    ) -> CancellationResult:
        """Execute one guarded cancellation with no diagnostic follow-up.

        The caller must durably mark its intent ambiguous immediately before
        entering this method.  Once the framed remote command is dispatched,
        every missing or untrusted acknowledgement is conservatively reported
        as :attr:`CancellationStatus.MUTATION_AMBIGUOUS`; this method never
        retries and never performs accounting reads that could blur that
        mutation boundary.
        """

        try:
            job_id = _job_id(ref)
        except ProtocolViolation as exc:
            return CancellationResult(CancellationStatus.GUARD_FAILED, str(exc))
        # Re-check the exact, already-validated identity in the same remote
        # script that performs the mutation.  This closes the gap after
        # inspect_cancel: if the row disappears or changes, scancel is never
        # reached.
        expected = f"{job_id}|{ref.slurm_job_name}|{ref.slurm_comment}"
        guarded_cancel = (
            f"row=$(squeue --me -h -j {job_id} -o '%i|%j|%k'); qrc=$?; "
            "if [ \"$qrc\" -ne 0 ]; then "
            "[ -z \"$row\" ] || printf '%s' \"$row\"; exit 46; fi; "
            "[ -n \"$row\" ] || exit 44; "
            f"[ \"$row\" = {shlex.quote(expected)} ] || exit 45; "
            f"scancel -- {job_id} || exit 47"
        )
        try:
            framed = self._framed(
                guarded_cancel,
                retry=RetryPolicy.NEVER,
                auth=auth,
                timeout=timeout,
            )
        except (AuthRequired, HostKeyChanged):
            # These terminal SSH checks fail before the guarded remote script
            # starts, so the caller may safely close its dispatch intent.
            raise
        except HpcAllocError as exc:
            return CancellationResult(
                CancellationStatus.MUTATION_AMBIGUOUS,
                str(exc),
            )
        if framed.returncode == 0 and framed.payload == b"":
            return CancellationResult(CancellationStatus.CANCELLED)
        if framed.returncode == 45 and framed.payload == b"":
            return CancellationResult(
                CancellationStatus.IDENTITY_MISMATCH,
                f"job {ref.cluster}:{job_id} changed identity before cancellation",
            )
        guard_absent = (
            framed.returncode == 44
            and framed.payload == b""
            and framed.stderr == ""
        ) or _is_invalid_singleton(framed, returncode=46)
        if guard_absent:
            return CancellationResult(
                CancellationStatus.LEFT_QUEUE,
                f"job {ref.cluster}:{job_id} left the queue before cancellation; "
                "scancel was not issued",
            )
        if framed.returncode in {44, 45, 46}:
            return CancellationResult(
                CancellationStatus.GUARD_FAILED,
                f"could not safely verify job {ref.cluster}:{job_id} immediately "
                f"before cancellation (guard exit {framed.returncode})",
            )
        detail = framed.stderr.strip()
        if not detail and framed.payload:
            detail = framed.payload.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = f"guarded cancellation returned exit {framed.returncode}"
        return CancellationResult(
            CancellationStatus.MUTATION_AMBIGUOUS,
            detail,
        )

    def log_size(self, path: str, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> LogSizeResult:
        quoted = shlex.quote(path)
        command = (
            f"if [ ! -e {quoted} ]; then printf 'MISSING'; "
            f"elif [ ! -r {quoted} ]; then printf 'UNREADABLE'; "
            f"else n=$(wc -c < {quoted}) && printf 'AVAILABLE:%s' \"$n\" || printf 'UNREADABLE'; fi"
        )
        framed = self._framed(command, retry=RetryPolicy.SAFE_READ, auth=auth)
        payload = self._require_success(framed, "log size").decode("ascii", errors="replace").strip()
        if payload == "MISSING":
            return LogSizeResult(LogSizeStatus.MISSING)
        if payload == "UNREADABLE":
            return LogSizeResult(LogSizeStatus.UNREADABLE)
        # BSD wc (including macOS) pads its numeric output; GNU wc generally
        # does not.  Whitespace around the otherwise-strict integer is valid.
        match = re.fullmatch(r"AVAILABLE:\s*([0-9]+)\s*", payload)
        if match is None:
            raise ProtocolViolation(f"invalid log-size result {payload!r}")
        return LogSizeResult.available(int(match.group(1)))

    def read_log_chunk(
        self,
        path: str,
        offset: int,
        *,
        limit: int = MAX_LOG_CHUNK_BYTES,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> bytes:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("log offset cannot be negative")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_LOG_CHUNK_BYTES
        ):
            raise ValueError(
                f"log chunk limit must be between 1 and {MAX_LOG_CHUNK_BYTES} bytes"
            )
        quoted = shlex.quote(path)
        status_template = "${TMPDIR:-/tmp}/hpc-alloc-log-chunk.XXXXXX"
        command = (
            f"[ -r {quoted} ] || exit 1; "
            f'status_file=$(mktemp "{status_template}") || exit 125; '
            "trap 'rm -f \"$status_file\"' 0; "
            "trap 'exit 129' 1; trap 'exit 130' 2; trap 'exit 143' 15; "
            "{ "
            f"tail -c +{offset + 1} -- {quoted}; source_rc=$?; "
            "printf '%s\\n' \"$source_rc\" >\"$status_file\"; "
            f"}} | head -c {limit}; sink_rc=$?; "
            '[ "$sink_rc" -eq 0 ] || { '
            "printf 'log-chunk sink failed with exit %s\\n' \"$sink_rc\" >&2; "
            "exit 125; }; "
            'IFS= read -r source_rc <"$status_file" || { '
            "printf '%s\\n' 'log-chunk source status was unavailable' >&2; "
            "exit 125; }; "
            'case "$source_rc" in \'\'|*[!0-9]*) '
            "printf '%s\\n' 'log-chunk source status was invalid' >&2; "
            "exit 125;; esac; "
            '[ "$source_rc" -le 255 ] 2>/dev/null || { '
            "printf '%s\\n' 'log-chunk source status was invalid' >&2; "
            "exit 125; }; "
            '[ "$source_rc" -eq 0 ] && exit 0; '
            'if [ "$source_rc" -gt 128 ]; then '
            'signal_name=$(kill -l "$source_rc" 2>/dev/null) || signal_name=; '
            f'case "$signal_name" in PIPE|SIGPIPE) exit {_LOG_CHUNK_SIGPIPE_STATUS};; esac; '
            "fi; "
            "printf 'log-chunk source failed with exit %s\\n' \"$source_rc\" >&2; "
            "exit 125"
        )
        framed = self._framed(
            command,
            retry=RetryPolicy.SAFE_READ,
            auth=auth,
            max_payload_bytes=limit,
        )
        if len(framed.payload) > limit:
            raise ProtocolViolation(
                f"log chunk exceeded requested limit ({len(framed.payload)} > {limit})"
            )
        if framed.returncode == _LOG_CHUNK_SIGPIPE_STATUS:
            if len(framed.payload) == limit:
                return framed.payload
            detail = framed.stderr.strip() or (
                "log source received SIGPIPE before the requested byte limit"
            )
            raise RemoteCommandFailed(f"cannot read log {path!r}: {detail}")
        if framed.returncode != 0:
            detail = framed.stderr.strip() or f"exit {framed.returncode}"
            raise RemoteCommandFailed(f"cannot read log {path!r}: {detail}")
        return framed.payload

    def tail_log(
        self,
        path: str,
        lines: int,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
    ) -> bytes:
        if lines < 0:
            raise ValueError("line count cannot be negative")
        quoted = shlex.quote(path)
        status_template = "${TMPDIR:-/tmp}/hpc-alloc-tail.XXXXXX"
        command = (
            f"[ -r {quoted} ] || exit 1; "
            f"[ {lines} -eq 0 ] && exit 0; "
            f'status=$(mktemp "{status_template}") || exit 125; '
            "trap 'rm -f \"$status\"' 0; "
            "trap 'exit 129' 1; trap 'exit 130' 2; trap 'exit 143' 15; "
            "{ "
            f"tail -c {MAX_LOG_CHUNK_BYTES} -- {quoted}; source_rc=$?; "
            "printf '%s\\n' \"$source_rc\" >\"$status\"; "
            f"}} | tail -n {lines}; sink_rc=$?; "
            '[ "$sink_rc" -eq 0 ] || exit "$sink_rc"; '
            'IFS= read -r source_rc <"$status" || { '
            "printf '%s\\n' 'log-tail source status was unavailable' >&2; exit 125; }; "
            'case "$source_rc" in \'\'|*[!0-9]*) '
            "printf '%s\\n' 'log-tail source status was invalid' >&2; exit 125;; esac; "
            '[ "$source_rc" -le 255 ] 2>/dev/null || { '
            "printf '%s\\n' 'log-tail source status was invalid' >&2; exit 125; }; "
            '[ "$source_rc" -eq 0 ] || exit "$source_rc"; '
            "exit 0"
        )
        framed = self._framed(
            command,
            retry=RetryPolicy.SAFE_READ,
            auth=auth,
            max_payload_bytes=MAX_LOG_CHUNK_BYTES,
        )
        if framed.returncode != 0:
            detail = framed.stderr.strip() or f"exit {framed.returncode}"
            raise RemoteCommandFailed(f"cannot read log {path!r}: {detail}")
        if len(framed.payload) > MAX_LOG_CHUNK_BYTES:
            raise ProtocolViolation("bounded log tail exceeded its byte limit")
        return framed.payload

    def remote_home(self, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        framed = self._framed("printf '%s' \"$HOME\"", retry=RetryPolicy.SAFE_READ, auth=auth)
        try:
            home = self._require_success(framed, "remote home").decode(
                "utf-8", errors="strict"
            )
        except UnicodeDecodeError as exc:
            raise ProtocolViolation("remote home directory was not valid UTF-8") from exc
        if not home.startswith("/") or "\x00" in home or "\n" in home:
            raise ProtocolViolation(f"remote shell returned an invalid home directory {home!r}")
        return home

    def _text_read(self, command: str, operation: str, *, auth: AuthMode) -> str:
        framed = self._framed(command, retry=RetryPolicy.SAFE_READ, auth=auth)
        return self._require_success(framed, operation).decode("utf-8", errors="replace")

    def partitions(self, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        return self._text_read(
            "sinfo -o '%P|%a|%l|%D|%c|%m|%G|%f'",
            "sinfo partitions",
            auth=auth,
        )

    def availability(self, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        fields = "Partition:30,StateCompact:12,NodeHost:30,Gres:60,GresUsed:60,CPUsState:30"
        return self._text_read(
            f"sinfo -h -N -O {shlex.quote(fields)}",
            "sinfo availability",
            auth=auth,
        )

    def estimated_start(self, job_id: str | int, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        job = _job_id(job_id)
        return self._text_read(
            f"squeue -j {job} --start -h -o '%S'",
            "squeue estimated start",
            auth=auth,
        ).strip()

    def priority(self, job_id: str | int, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        job = _job_id(job_id)
        return self._text_read(f"sprio -l -j {job}", "sprio", auth=auth)

    def reservations(self, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        return self._text_read("scontrol -o show reservation", "scontrol reservations", auth=auth)

    def user_access(self, netid: str, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        """Raw unix groups and scheduler associations, for local eligibility gating.

        Best-effort by design: the pipeline always exits 0, so a cluster that is
        missing the accounting tool simply yields no association rows and the
        caller falls open rather than blocking a submit.
        """

        netid_q = shlex.quote(netid)
        command = (
            "printf 'GROUPS '; id -Gn 2>/dev/null; printf 'ASSOC\\n'; "
            f"sacctmgr -n -P show assoc user={netid_q} format=Account,Partition,QOS "
            "2>/dev/null; true"
        )
        return self._text_read(command, "user access", auth=auth)

    def partition_access(self, *, auth: AuthMode = AuthMode.NONINTERACTIVE) -> str:
        """Raw one-line-per-partition access rules, for local eligibility gating."""

        return self._text_read(
            "scontrol -o show partition 2>/dev/null || true",
            "partition access",
            auth=auth,
        )


__all__ = [
    "AccountingRecord",
    "CancellationInspection",
    "CancellationInspectionStatus",
    "CancellationResult",
    "CancellationStatus",
    "FINAL_STATES",
    "MAX_LOG_CHUNK_BYTES",
    "LogSizeResult",
    "LogSizeStatus",
    "QueueRow",
    "RawQueueRow",
    "RawQueueScan",
    "SlurmClient",
    "SubmissionSpec",
    "SubmissionResult",
]
