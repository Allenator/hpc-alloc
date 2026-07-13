"""SQLite-backed durable job and mutation state.

The repository's transactions contain only local database work.  Callers must
leave a transaction before invoking SSH or any other subprocess, then record
the result in a new transaction.  This short-lock rule is central to safe CLI
concurrency.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
import stat
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import (
    LifecycleRevisionConflict,
    RecordNotFound,
    StateConflict,
    StateInvalid,
)
from .models import (
    EvidenceProvenance,
    FinalSource,
    JobKind,
    JobPhase,
    JobRecord,
    MachineRecord,
    OperationKind,
    OperationPhase,
    OperationRecord,
    UNRESOLVED_OPERATION_PHASES,
)
from .ownership import (
    IDENTIFIER_RE,
    OPERATION_RE,
    parse_tag,
    slurm_job_name as build_slurm_job_name,
)


SCHEMA_VERSION = 5
_SCHEMA_TABLES = frozenset({"metadata", "machine", "jobs", "operations", "cluster_cache"})
_SCHEMA_COLUMNS = {
    "metadata": {"schema_version"},
    "machine": {"singleton", "machine_id", "hostname", "created_at", "updated_at"},
    "jobs": {
        "operation_id",
        "cluster",
        "logical_name",
        "kind",
        "owner_id",
        "slurm_job_name",
        "slurm_comment",
        "job_id",
        "phase",
        "resources_json",
        "ever_started",
        "current_node",
        "last_node",
        "terminal_state",
        "exit_code",
        "observation_epoch",
        "evidence_provenance",
        "evidence_detail",
        "final_source",
        "created_at",
        "updated_at",
        "finalized_at",
    },
    "operations": {
        "operation_id",
        "kind",
        "phase",
        "cluster",
        "logical_name",
        "target_job_operation_id",
        "job_id",
        "detail",
        "created_at",
        "updated_at",
        "resolved_at",
    },
    "cluster_cache": {"cluster", "cache_key", "value_json", "updated_at", "expires_at"},
}
_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z")
_UNSET = object()
_LOCAL_FINAL_SOURCES = frozenset({FinalSource.SUBMIT_FAILED, FinalSource.ABANDONED})
_FINAL_AUTHORITY = {
    FinalSource.CONFIRMED_QUEUE: 1,
    FinalSource.ACCOUNTING: 2,
}


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


class StateRepository:
    """Own the current state database and its concurrency invariants."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        clock: Callable[[], datetime] = _default_clock,
        machine_id_factory: Callable[[], str] = lambda: secrets.token_hex(12),
    ) -> None:
        self.path = Path(path)
        self.busy_timeout_ms = busy_timeout_ms
        self._clock = clock
        self._machine_id_factory = machine_id_factory
        self._initialized = False

    def _now(self) -> str:
        return _timestamp(self._clock())

    def _prepare_path(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError as exc:
            raise StateInvalid(f"cannot secure state directory: {exc}", path=self.path) from exc
        if self.path.is_symlink():
            raise StateInvalid("state database must not be a symbolic link", path=self.path)
        if self.path.exists() and not stat.S_ISREG(self.path.stat().st_mode):
            raise StateInvalid("state database is not a regular file", path=self.path)

    def _connect_raw(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=self.busy_timeout_ms / 1_000,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {int(self.busy_timeout_ms)}")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise StateInvalid(f"cannot open state database: {exc}", path=self.path) from exc

    def initialize(self) -> "StateRepository":
        """Create or validate the current schema and secure its containing directory."""

        self._prepare_path()
        previous_umask = os.umask(0o077)
        try:
            connection = self._connect_raw()
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if "metadata" in tables:
                    try:
                        rows = connection.execute("SELECT schema_version FROM metadata").fetchall()
                    except sqlite3.Error as exc:
                        raise StateInvalid("state schema metadata is corrupt", path=self.path) from exc
                    if len(rows) != 1 or rows[0][0] != SCHEMA_VERSION:
                        version = "missing" if not rows else rows[0][0]
                        raise StateInvalid(
                            f"unsupported state schema {version!r}; expected {SCHEMA_VERSION}",
                            path=self.path,
                        )
                if tables and tables != _SCHEMA_TABLES:
                    missing = sorted(_SCHEMA_TABLES - tables)
                    extra = sorted(tables - _SCHEMA_TABLES)
                    detail = []
                    if missing:
                        detail.append(f"missing {', '.join(missing)}")
                    if extra:
                        detail.append(f"unexpected {', '.join(extra)}")
                    raise StateInvalid(
                        "database is not an hpc-alloc state database with an intact schema"
                        + (f" ({'; '.join(detail)})" if detail else ""),
                        path=self.path,
                    )
                connection.execute("PRAGMA journal_mode = WAL")
                self._create_schema(connection)
                for table, expected_columns in _SCHEMA_COLUMNS.items():
                    actual_columns = {
                        row[1]
                        for row in connection.execute(
                            f"PRAGMA table_info({table})"
                        ).fetchall()
                    }
                    if actual_columns != expected_columns:
                        raise StateInvalid(
                            f"state table {table!r} has an unsupported schema",
                            path=self.path,
                        )
                check = connection.execute("PRAGMA quick_check(1)").fetchone()
                if check is None or check[0] != "ok":
                    raise StateInvalid("state database integrity check failed", path=self.path)
            finally:
                connection.close()
            try:
                self.path.chmod(0o600)
            except OSError as exc:
                raise StateInvalid(f"cannot secure state database: {exc}", path=self.path) from exc
            self._initialized = True
            return self
        except sqlite3.Error as exc:
            raise StateInvalid(f"cannot initialize state database: {exc}", path=self.path) from exc
        finally:
            os.umask(previous_umask)

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        job_kinds = ",".join(f"'{value.value}'" for value in JobKind)
        job_phases = ",".join(f"'{value.value}'" for value in JobPhase)
        evidence_provenance = ",".join(
            f"'{value.value}'" for value in EvidenceProvenance
        )
        final_sources = ",".join(f"'{value.value}'" for value in FinalSource)
        operation_kinds = ",".join(f"'{value.value}'" for value in OperationKind)
        operation_phases = ",".join(f"'{value.value}'" for value in OperationPhase)
        connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS metadata (
                schema_version INTEGER NOT NULL
            );
            INSERT INTO metadata(schema_version)
                SELECT {SCHEMA_VERSION} WHERE NOT EXISTS (SELECT 1 FROM metadata);

            CREATE TABLE IF NOT EXISTS machine (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                machine_id TEXT NOT NULL UNIQUE,
                hostname TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                operation_id TEXT PRIMARY KEY,
                cluster TEXT NOT NULL,
                logical_name TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ({job_kinds})),
                owner_id TEXT NOT NULL,
                slurm_job_name TEXT NOT NULL,
                slurm_comment TEXT NOT NULL,
                job_id TEXT,
                phase TEXT NOT NULL CHECK (phase IN ({job_phases})),
                resources_json TEXT NOT NULL,
                ever_started INTEGER NOT NULL DEFAULT 0 CHECK (ever_started IN (0, 1)),
                current_node TEXT,
                last_node TEXT,
                terminal_state TEXT,
                exit_code TEXT,
                observation_epoch INTEGER NOT NULL DEFAULT 0
                    CHECK (observation_epoch >= 0),
                evidence_provenance TEXT CHECK (
                    evidence_provenance IS NULL
                    OR evidence_provenance IN ({evidence_provenance})
                ),
                evidence_detail TEXT,
                final_source TEXT CHECK (
                    final_source IS NULL OR final_source IN ({final_sources})
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finalized_at TEXT,
                CHECK (
                    (phase = 'FINAL' AND final_source IS NOT NULL AND finalized_at IS NOT NULL)
                    OR
                    (phase <> 'FINAL' AND final_source IS NULL AND finalized_at IS NULL)
                ),
                CHECK (
                    (phase = 'TERMINAL_CANDIDATE' AND evidence_provenance IS NOT NULL)
                    OR
                    (final_source = 'confirmed-queue' AND evidence_provenance IS NOT NULL)
                    OR
                    (
                        phase <> 'TERMINAL_CANDIDATE'
                        AND COALESCE(final_source, '') <> 'confirmed-queue'
                        AND evidence_provenance IS NULL
                    )
                ),
                CHECK (evidence_detail IS NULL OR evidence_provenance IS NOT NULL)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_live_allocation
                ON jobs(cluster, logical_name)
                WHERE kind = 'allocation' AND phase <> 'FINAL';
            CREATE INDEX IF NOT EXISTS jobs_by_remote_id ON jobs(cluster, job_id);
            CREATE INDEX IF NOT EXISTS jobs_by_name ON jobs(logical_name, cluster);

            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ({operation_kinds})),
                phase TEXT NOT NULL CHECK (phase IN ({operation_phases})),
                cluster TEXT NOT NULL,
                logical_name TEXT NOT NULL,
                target_job_operation_id TEXT NOT NULL
                    REFERENCES jobs(operation_id) ON DELETE CASCADE,
                job_id TEXT,
                detail TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending_cancel
                ON operations(target_job_operation_id)
                WHERE kind = 'cancel'
                  AND phase IN ('CANCEL_PENDING', 'AMBIGUOUS');
            CREATE INDEX IF NOT EXISTS operations_by_phase ON operations(phase, updated_at);

            CREATE TABLE IF NOT EXISTS cluster_cache (
                cluster TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                PRIMARY KEY (cluster, cache_key)
            );
            COMMIT;
            """
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Open a short, immediate local transaction.

        This intentionally yields only a database connection.  Transport and
        subprocess services must be called after leaving the context.
        """

        self._ensure_initialized()
        connection = self._connect_raw()
        try:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                    raise StateConflict("state database is busy; retry the command") from exc
                raise StateInvalid(f"cannot begin state transaction: {exc}", path=self.path) from exc
            yield connection
            try:
                connection.execute("COMMIT")
            except sqlite3.Error as exc:
                raise StateInvalid(f"cannot commit state transaction: {exc}", path=self.path) from exc
        except sqlite3.IntegrityError:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        except sqlite3.Error as exc:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise StateInvalid(f"state database operation failed: {exc}", path=self.path) from exc
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        """Translate every read-side SQLite failure at the repository boundary."""

        self._ensure_initialized()
        connection = self._connect_raw()
        try:
            yield connection
        except sqlite3.Error as exc:
            raise StateInvalid(f"state database read failed: {exc}", path=self.path) from exc
        finally:
            connection.close()

    def get_machine(self) -> MachineRecord | None:
        with self._read_connection() as connection:
            row = connection.execute("SELECT * FROM machine WHERE singleton = 1").fetchone()
        return self._machine_from_row(row) if row is not None else None

    def get_or_create_machine_id(self, hostname: str) -> str:
        if not hostname or any(not char.isprintable() for char in hostname):
            raise StateConflict("hostname must be non-empty printable text")
        now = self._now()
        candidate = self._machine_id_factory()
        if not isinstance(candidate, str) or IDENTIFIER_RE.fullmatch(candidate) is None:
            raise StateInvalid("machine ID generator returned an invalid value", path=self.path)
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO machine(singleton, machine_id, hostname, created_at, updated_at)
                   VALUES(1, ?, ?, ?, ?) ON CONFLICT(singleton) DO NOTHING""",
                (candidate, hostname, now, now),
            )
            row = connection.execute("SELECT * FROM machine WHERE singleton = 1").fetchone()
            assert row is not None
            machine = self._machine_from_row(row)
            if machine.hostname != hostname:
                connection.execute(
                    "UPDATE machine SET hostname = ?, updated_at = ? WHERE singleton = 1",
                    (hostname, now),
                )
        return machine.machine_id

    def reserve_submission(
        self,
        *,
        cluster: str,
        logical_name: str,
        kind: JobKind | str,
        owner_id: str,
        slurm_job_name: str,
        slurm_comment: str,
        resources: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> OperationRecord:
        """Durably reserve a name before the one permitted remote submission."""

        kind = JobKind(kind)
        operation_id = operation_id or uuid.uuid4().hex
        self._validate_text_fields(
            cluster=cluster,
            logical_name=logical_name,
            owner_id=owner_id,
            slurm_job_name=slurm_job_name,
            slurm_comment=slurm_comment,
            operation_id=operation_id,
        )
        if OPERATION_RE.fullmatch(operation_id) is None:
            raise StateConflict("operation_id must be 32 lowercase hexadecimal characters")
        if IDENTIFIER_RE.fullmatch(cluster) is None:
            raise StateConflict("cluster is not a valid identifier")
        if kind is JobKind.ALLOCATION and (
            logical_name.isdigit() or logical_name in {"login", "run"}
        ):
            raise StateConflict(f"allocation name {logical_name!r} is reserved or ambiguous")
        if kind is JobKind.RUN and logical_name != "run":
            raise StateConflict("run jobs must use the logical name 'run'")
        tag = parse_tag(slurm_comment)
        if (
            tag is None
            or tag.owner_id != owner_id
            or tag.operation_id != operation_id
            or tag.kind != kind.value
            or tag.logical_name != logical_name
            or slurm_job_name != build_slurm_job_name(kind.value, operation_id)
        ):
            raise StateConflict("submission identity metadata is inconsistent")
        resources_json = self._encode_object(resources or {}, "resources")
        now = self._now()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """INSERT INTO jobs(
                           operation_id, cluster, logical_name, kind, owner_id,
                           slurm_job_name, slurm_comment, phase, resources_json,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        operation_id,
                        cluster,
                        logical_name,
                        kind.value,
                        owner_id,
                        slurm_job_name,
                        slurm_comment,
                        JobPhase.SUBMITTING.value,
                        resources_json,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    """INSERT INTO operations(
                           operation_id, kind, phase, cluster, logical_name,
                           target_job_operation_id, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        operation_id,
                        OperationKind.SUBMIT.value,
                        OperationPhase.PREPARED.value,
                        cluster,
                        logical_name,
                        operation_id,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "jobs.cluster, jobs.logical_name" in str(exc):
                raise StateConflict(
                    f"allocation {cluster}:{logical_name} already has a non-final job or submission"
                ) from exc
            raise StateConflict(f"operation {operation_id} already exists") from exc
        return self.get_operation(operation_id)

    def acknowledge_submission(self, operation_id: str, job_id: str) -> JobRecord:
        self._validate_text_fields(operation_id=operation_id, job_id=job_id)
        if not job_id.isascii() or not job_id.isdigit():
            raise StateConflict(f"Slurm job ID must be numeric, got {job_id!r}")
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if OperationKind(operation["kind"]) is not OperationKind.SUBMIT:
                raise StateConflict(f"operation {operation_id} is not a submission")
            if OperationPhase(operation["phase"]) not in {
                OperationPhase.PREPARED,
                OperationPhase.AMBIGUOUS,
                OperationPhase.ACKNOWLEDGED,
            }:
                raise StateConflict(f"submission {operation_id} cannot be acknowledged from {operation['phase']}")
            operation_phase = OperationPhase(operation["phase"])
            existing = operation["job_id"]
            if existing is not None and existing != job_id:
                raise StateConflict(
                    f"submission {operation_id} is already bound to a different Slurm job"
                )
            if operation_phase is not OperationPhase.ACKNOWLEDGED:
                job = self._require_job_row(connection, operation_id)
                job_now = self._next_job_timestamp(now, job["updated_at"])
                connection.execute(
                    """UPDATE jobs SET job_id = ?, phase = ?, updated_at = ?
                       WHERE operation_id = ?""",
                    (job_id, JobPhase.QUEUED.value, job_now, operation_id),
                )
                connection.execute(
                    """UPDATE operations SET job_id = ?, phase = ?, detail = NULL,
                       updated_at = ?, resolved_at = ? WHERE operation_id = ?""",
                    (job_id, OperationPhase.ACKNOWLEDGED.value, now, now, operation_id),
                )
        return self.get_job(operation_id)

    def mark_submission_ambiguous(self, operation_id: str, detail: str) -> OperationRecord:
        self._validate_text_fields(operation_id=operation_id)
        detail = self._sanitize_detail(detail)
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.SUBMIT.value:
                raise StateConflict(f"operation {operation_id} is not a submission")
            if OperationPhase(operation["phase"]) not in {
                OperationPhase.PREPARED,
                OperationPhase.AMBIGUOUS,
            }:
                raise StateConflict(f"submission {operation_id} is no longer unresolved")
            connection.execute(
                "UPDATE operations SET phase = ?, detail = ?, updated_at = ? WHERE operation_id = ?",
                (OperationPhase.AMBIGUOUS.value, detail, now, operation_id),
            )
        return self.get_operation(operation_id)

    def begin_cancel(
        self, target_job_operation_id: str, *, operation_id: str | None = None
    ) -> OperationRecord:
        operation_id = operation_id or uuid.uuid4().hex
        self._validate_text_fields(
            target_job_operation_id=target_job_operation_id, operation_id=operation_id
        )
        if OPERATION_RE.fullmatch(operation_id) is None:
            raise StateConflict("operation_id must be 32 lowercase hexadecimal characters")
        now = self._now()
        try:
            with self.transaction() as connection:
                job = self._require_job_row(connection, target_job_operation_id)
                if job["job_id"] is None:
                    raise StateConflict("cannot cancel a submission without a confirmed Slurm job ID")
                if job["phase"] == JobPhase.FINAL.value:
                    raise StateConflict("cannot cancel a final job")
                connection.execute(
                    """INSERT INTO operations(
                           operation_id, kind, phase, cluster, logical_name,
                           target_job_operation_id, job_id, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        operation_id,
                        OperationKind.CANCEL.value,
                        OperationPhase.CANCEL_PENDING.value,
                        job["cluster"],
                        job["logical_name"],
                        target_job_operation_id,
                        job["job_id"],
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "operations.target_job_operation_id" in str(exc):
                raise StateConflict("job already has a pending cancellation") from exc
            raise StateConflict(f"operation {operation_id} already exists") from exc
        return self.get_operation(operation_id)

    def mark_cancel_dispatching(
        self,
        operation_id: str,
        detail: str = "guarded cancellation dispatch started; outcome is not yet known",
        *,
        expected_target_updated_at: str | None = None,
        phase: JobPhase | str | None = None,
        ever_started: bool | None = None,
        current_node: str | None | object = _UNSET,
        last_node: str | None | object = _UNSET,
        terminal_state: str | None | object = _UNSET,
        exit_code: str | None | object = _UNSET,
        observation_epoch: int | None = None,
        evidence_provenance: EvidenceProvenance | str | None | object = _UNSET,
        evidence_detail: str | None | object = _UNSET,
    ) -> OperationRecord:
        """Make a cancel conservatively ambiguous before remote dispatch.

        Callers must commit this transition immediately before invoking the
        one-shot guarded mutation.  A crash after this transaction can produce
        a false ambiguity, but can never make a possibly-issued remote
        cancellation look safe to retry.
        """

        self._validate_text_fields(operation_id=operation_id)
        detail = self._sanitize_detail(detail)
        self._validate_job_update_values(
            current_node=current_node,
            last_node=last_node,
            terminal_state=terminal_state,
            exit_code=exit_code,
        )
        provenance, evidence_detail = self._prepare_lifecycle_evidence(
            last_node=last_node,
            observation_epoch=observation_epoch,
            evidence_provenance=evidence_provenance,
            evidence_detail=evidence_detail,
        )
        target_phase = phase
        if target_phase is None and provenance is not _UNSET and provenance is not None:
            target_phase = JobPhase.TERMINAL_CANDIDATE
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.CANCEL.value:
                raise StateConflict(f"operation {operation_id} is not a cancellation")
            if operation["phase"] != OperationPhase.CANCEL_PENDING.value:
                raise StateConflict(f"cancellation {operation_id} is not pending dispatch")
            connection.execute(
                "UPDATE operations SET phase = ?, detail = ?, updated_at = ? "
                "WHERE operation_id = ?",
                (
                    OperationPhase.AMBIGUOUS.value,
                    detail,
                    now,
                    operation_id,
                ),
            )
            if (
                expected_target_updated_at is not None
                or target_phase is not None
                or ever_started is not None
                or current_node is not _UNSET
                or last_node is not _UNSET
                or terminal_state is not _UNSET
                or exit_code is not _UNSET
                or observation_epoch is not None
                or provenance is not _UNSET
                or evidence_detail is not _UNSET
            ):
                self._update_job_row(
                    connection,
                    operation["target_job_operation_id"],
                    now=now,
                    expected_updated_at=expected_target_updated_at,
                    phase=target_phase,
                    ever_started=ever_started,
                    current_node=current_node,
                    last_node=last_node,
                    terminal_state=terminal_state,
                    exit_code=exit_code,
                    observation_epoch=observation_epoch,
                    evidence_provenance=provenance,
                    evidence_detail=evidence_detail,
                )
        return self.get_operation(operation_id)

    def mark_cancel_ambiguous(self, operation_id: str, detail: str) -> OperationRecord:
        """Update diagnostic detail for an already-ambiguous cancellation."""

        self._validate_text_fields(operation_id=operation_id)
        detail = self._sanitize_detail(detail)
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.CANCEL.value:
                raise StateConflict(f"operation {operation_id} is not a cancellation")
            if operation["phase"] != OperationPhase.AMBIGUOUS.value:
                raise StateConflict(f"cancellation {operation_id} was not dispatched ambiguously")
            connection.execute(
                "UPDATE operations SET detail = ?, updated_at = ? WHERE operation_id = ?",
                (detail, now, operation_id),
            )
        return self.get_operation(operation_id)

    def resolve_operation(
        self,
        operation_id: str,
        *,
        final_source: FinalSource | str,
        expected_target_updated_at: str | None = None,
        detail: str | None = None,
        terminal_state: str | None = None,
        exit_code: str | None = None,
        ever_started: bool | None = None,
        last_node: str | None | object = _UNSET,
        observation_epoch: int | None = None,
        evidence_provenance: EvidenceProvenance | str | None | object = _UNSET,
        evidence_detail: str | None | object = _UNSET,
    ) -> OperationRecord:
        """Resolve a cancellation with durable final scheduler evidence.

        A cancellation acknowledgement without final evidence must use
        :meth:`resolve_cancel_departed`; keeping the policies separate prevents
        a successful cancellation request from being misrecorded as a final
        scheduler verdict.
        """

        if detail is not None:
            detail = self._sanitize_detail(detail)
        source = self._final_source(final_source)
        if source in {FinalSource.SUBMIT_FAILED, FinalSource.ABANDONED}:
            raise StateConflict("local final verdicts require their atomic operation API")
        self._validate_job_update_values(
            last_node=last_node,
            terminal_state=terminal_state,
            exit_code=exit_code,
        )
        provenance, evidence_detail = self._prepare_lifecycle_evidence(
            last_node=last_node,
            observation_epoch=observation_epoch,
            evidence_provenance=evidence_provenance,
            evidence_detail=evidence_detail,
        )
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.CANCEL.value:
                raise StateConflict(
                    "submission operations are closed by acknowledgement, failure, or abandonment"
                )
            if OperationPhase(operation["phase"]) not in {
                OperationPhase.CANCEL_PENDING,
                OperationPhase.AMBIGUOUS,
            }:
                raise StateConflict(f"cancellation {operation_id} is not unresolved")
            connection.execute(
                """UPDATE operations SET phase = ?, detail = ?, updated_at = ?, resolved_at = ?
                   WHERE operation_id = ?""",
                (OperationPhase.RESOLVED.value, detail, now, now, operation_id),
            )
            self._update_job_row(
                connection,
                operation["target_job_operation_id"],
                now=now,
                expected_updated_at=expected_target_updated_at,
                phase=JobPhase.FINAL,
                ever_started=ever_started,
                last_node=last_node,
                terminal_state=terminal_state,
                exit_code=exit_code,
                observation_epoch=observation_epoch,
                evidence_provenance=(
                    EvidenceProvenance.CANCELLATION
                    if source is FinalSource.CONFIRMED_QUEUE
                    and provenance is _UNSET
                    else provenance
                    if source is FinalSource.CONFIRMED_QUEUE
                    else None
                ),
                evidence_detail=(
                    detail
                    if source is FinalSource.CONFIRMED_QUEUE
                    and evidence_detail is _UNSET
                    else evidence_detail
                    if source is FinalSource.CONFIRMED_QUEUE
                    else None
                ),
                final_source=source,
            )
            result = self._require_operation_row(connection, operation_id)
        return self._operation_from_row(result)

    def resolve_cancel_departed(
        self,
        operation_id: str,
        detail: str | None = None,
        *,
        expected_target_updated_at: str | None = None,
        terminal_state: str | None = None,
        exit_code: str | None = None,
        final_source: FinalSource | str | None = None,
        ever_started: bool | None = None,
        last_node: str | None | object = _UNSET,
        observation_epoch: int | None = None,
        evidence_provenance: EvidenceProvenance | str | None | object = _UNSET,
        evidence_detail: str | None | object = _UNSET,
    ) -> OperationRecord:
        """Atomically close a cancellation and record target departure.

        Without ``final_source`` the job becomes ``TERMINAL_CANDIDATE``: a
        successful cancellation mutation proves only that the request was
        accepted.  When the caller also has final scheduler evidence, the same
        transaction performs the authority-aware final merge.
        """

        if detail is not None:
            detail = self._sanitize_detail(detail)
        source = self._final_source(final_source) if final_source is not None else None
        if source in {FinalSource.SUBMIT_FAILED, FinalSource.ABANDONED}:
            raise StateConflict("local final verdicts require their atomic operation API")
        self._validate_job_update_values(
            last_node=last_node,
            terminal_state=terminal_state,
            exit_code=exit_code,
        )
        provenance, evidence_detail = self._prepare_lifecycle_evidence(
            last_node=last_node,
            observation_epoch=observation_epoch,
            evidence_provenance=evidence_provenance,
            evidence_detail=evidence_detail,
        )
        has_explicit_lifecycle_evidence = (
            ever_started is not None
            or last_node is not _UNSET
            or observation_epoch is not None
            or provenance is not _UNSET
            or evidence_detail is not _UNSET
            or terminal_state is not None
            or exit_code is not None
        )
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.CANCEL.value:
                raise StateConflict(f"operation {operation_id} is not a cancellation")
            if OperationPhase(operation["phase"]) not in {
                OperationPhase.CANCEL_PENDING,
                OperationPhase.AMBIGUOUS,
            }:
                raise StateConflict(f"cancellation {operation_id} is not unresolved")
            connection.execute(
                """UPDATE operations SET phase = ?, detail = ?, updated_at = ?, resolved_at = ?
                   WHERE operation_id = ?""",
                (OperationPhase.RESOLVED.value, detail, now, now, operation_id),
            )
            target_id = operation["target_job_operation_id"]
            target = self._require_job_row(connection, target_id)
            if source is not None:
                self._update_job_row(
                    connection,
                    target_id,
                    now=now,
                    expected_updated_at=expected_target_updated_at,
                    phase=JobPhase.FINAL,
                    ever_started=ever_started,
                    last_node=last_node,
                    terminal_state=terminal_state,
                    exit_code=exit_code,
                    observation_epoch=observation_epoch,
                    evidence_provenance=(
                        EvidenceProvenance.CANCELLATION
                        if source is FinalSource.CONFIRMED_QUEUE
                        and provenance is _UNSET
                        else provenance
                        if source is FinalSource.CONFIRMED_QUEUE
                        else None
                    ),
                    evidence_detail=(
                        detail
                        if source is FinalSource.CONFIRMED_QUEUE
                        and evidence_detail is _UNSET
                        else evidence_detail
                        if source is FinalSource.CONFIRMED_QUEUE
                        else None
                    ),
                    final_source=source,
                )
            elif target["phase"] != JobPhase.FINAL.value:
                retain_preflight_evidence = (
                    target["phase"] == JobPhase.TERMINAL_CANDIDATE.value
                    and not has_explicit_lifecycle_evidence
                )
                self._update_job_row(
                    connection,
                    target_id,
                    now=now,
                    expected_updated_at=expected_target_updated_at,
                    phase=JobPhase.TERMINAL_CANDIDATE,
                    ever_started=ever_started,
                    last_node=last_node,
                    terminal_state=terminal_state,
                    exit_code=exit_code,
                    observation_epoch=observation_epoch,
                    evidence_provenance=(
                        _UNSET
                        if retain_preflight_evidence
                        else EvidenceProvenance.CANCELLATION
                        if provenance is _UNSET
                        else provenance
                    ),
                    evidence_detail=(
                        _UNSET
                        if retain_preflight_evidence
                        else detail
                        if evidence_detail is _UNSET
                        else evidence_detail
                    ),
                )
            else:
                # The target may have finalized after cancellation dispatch.
                # With no explicit fresh evidence, preserve it byte-for-byte;
                # pre-dispatch evidence must never be replayed here.
                self._update_job_row(
                    connection,
                    target_id,
                    now=now,
                    expected_updated_at=expected_target_updated_at,
                    phase=JobPhase.FINAL,
                    ever_started=ever_started,
                    last_node=last_node,
                    observation_epoch=observation_epoch,
                )
            result = self._require_operation_row(connection, operation_id)
        return self._operation_from_row(result)

    def fail_submission(self, operation_id: str, detail: str) -> OperationRecord:
        """Atomically fail a known-uncommitted submission and release its name."""

        detail = self._sanitize_detail(detail)
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.SUBMIT.value:
                raise StateConflict(f"operation {operation_id} is not a submission")
            if operation["phase"] != OperationPhase.PREPARED.value:
                raise StateConflict(
                    f"submission {operation_id} cannot fail from {operation['phase']}"
                )
            connection.execute(
                """UPDATE operations SET phase = ?, detail = ?, updated_at = ?, resolved_at = ?
                   WHERE operation_id = ?""",
                (OperationPhase.FAILED.value, detail, now, now, operation_id),
            )
            self._update_job_row(
                connection,
                operation["target_job_operation_id"],
                now=now,
                phase=JobPhase.FINAL,
                terminal_state="SUBMIT_FAILED",
                final_source=FinalSource.SUBMIT_FAILED,
                allow_local_final=True,
            )
            result = self._require_operation_row(connection, operation_id)
        return self._operation_from_row(result)

    def fail_cancel_operation(self, operation_id: str, detail: str) -> OperationRecord:
        """Close a definitively failed cancellation without changing its job."""

        detail = self._sanitize_detail(detail)
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if operation["kind"] != OperationKind.CANCEL.value:
                raise StateConflict(f"operation {operation_id} is not a cancellation")
            if OperationPhase(operation["phase"]) not in {
                OperationPhase.CANCEL_PENDING,
                OperationPhase.AMBIGUOUS,
            }:
                raise StateConflict(f"cancellation {operation_id} is not unresolved")
            connection.execute(
                """UPDATE operations SET phase = ?, detail = ?, updated_at = ?, resolved_at = ?
                   WHERE operation_id = ?""",
                (OperationPhase.FAILED.value, detail, now, now, operation_id),
            )
            result = self._require_operation_row(connection, operation_id)
        return self._operation_from_row(result)

    def abandon_operation(self, operation_id: str, detail: str | None = None) -> OperationRecord:
        if detail is not None:
            detail = self._sanitize_detail(detail)
        now = self._now()
        with self.transaction() as connection:
            operation = self._require_operation_row(connection, operation_id)
            if OperationPhase(operation["phase"]) not in UNRESOLVED_OPERATION_PHASES:
                raise StateConflict(f"operation {operation_id} is not unresolved")
            connection.execute(
                """UPDATE operations SET phase = ?, detail = ?, updated_at = ?, resolved_at = ?
                   WHERE operation_id = ?""",
                (OperationPhase.ABANDONED.value, detail, now, now, operation_id),
            )
            if operation["kind"] == OperationKind.SUBMIT.value:
                self._update_job_row(
                    connection,
                    operation["target_job_operation_id"],
                    now=now,
                    phase=JobPhase.FINAL,
                    terminal_state="ABANDONED",
                    final_source=FinalSource.ABANDONED,
                    allow_local_final=True,
                )
            result = self._require_operation_row(connection, operation_id)
        return self._operation_from_row(result)

    def get_job(self, operation_id: str) -> JobRecord:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFound(f"job operation {operation_id} does not exist")
        return self._job_from_row(row)

    def find_jobs(
        self,
        *,
        cluster: str | None = None,
        logical_name: str | None = None,
        job_id: str | None = None,
        kind: JobKind | str | None = None,
        include_final: bool = True,
    ) -> list[JobRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("cluster", cluster),
            ("logical_name", logical_name),
            ("job_id", job_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(JobKind(kind).value)
        if not include_final:
            clauses.append("phase <> ?")
            parameters.append(JobPhase.FINAL.value)
        query = "SELECT * FROM jobs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, operation_id"
        with self._read_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._job_from_row(row) for row in rows]

    def list_jobs(self, *, include_final: bool = True) -> list[JobRecord]:
        return self.find_jobs(include_final=include_final)

    def snapshot_setup_scope_blockers(
        self,
    ) -> tuple[list[JobRecord], list[OperationRecord]]:
        """Read every setup-scope blocker from one SQLite snapshot.

        Forced setup uses both non-final jobs and unresolved operation intents
        to decide which configured cluster identities remain authoritative.
        The explicit read transaction prevents a concurrent lifecycle update
        from being observed between the two queries.
        """

        unresolved = tuple(
            phase.value
            for phase in sorted(UNRESOLVED_OPERATION_PHASES, key=lambda item: item.value)
        )
        placeholders = ",".join("?" for _ in unresolved)
        with self._read_connection() as connection:
            connection.execute("BEGIN")
            try:
                job_rows = connection.execute(
                    """SELECT * FROM jobs WHERE phase <> ?
                       ORDER BY created_at, operation_id""",
                    (JobPhase.FINAL.value,),
                ).fetchall()
                operation_rows = connection.execute(
                    f"""SELECT * FROM operations WHERE phase IN ({placeholders})
                        ORDER BY created_at, operation_id""",
                    unresolved,
                ).fetchall()
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        return (
            [self._job_from_row(row) for row in job_rows],
            [self._operation_from_row(row) for row in operation_rows],
        )

    def update_job(
        self,
        operation_id: str,
        *,
        expected_updated_at: str | None = None,
        phase: JobPhase | str | None = None,
        ever_started: bool | None = None,
        current_node: str | None | object = _UNSET,
        last_node: str | None | object = _UNSET,
        terminal_state: str | None | object = _UNSET,
        exit_code: str | None | object = _UNSET,
        observation_epoch: int | None = None,
        evidence_provenance: EvidenceProvenance | str | None | object = _UNSET,
        evidence_detail: str | None | object = _UNSET,
        final_source: FinalSource | str | None | object = _UNSET,
    ) -> JobRecord:
        """Merge lifecycle evidence against the row read inside the write transaction."""

        self._validate_job_update_values(
            current_node=current_node,
            last_node=last_node,
            terminal_state=terminal_state,
            exit_code=exit_code,
        )
        if (
            observation_epoch is not None
            and (
                isinstance(observation_epoch, bool)
                or not isinstance(observation_epoch, int)
                or observation_epoch < 0
            )
        ):
            raise StateConflict("observation_epoch must be a non-negative integer")
        provenance: EvidenceProvenance | None | object
        if evidence_provenance is _UNSET:
            provenance = _UNSET
        elif evidence_provenance is None:
            provenance = None
        else:
            provenance = self._evidence_provenance(evidence_provenance)
        if evidence_detail is not _UNSET and evidence_detail is not None:
            evidence_detail = self._sanitize_detail(evidence_detail)
        source: FinalSource | None | object
        if final_source is _UNSET:
            source = _UNSET
        elif final_source is None:
            source = None
        else:
            source = self._final_source(final_source)
            if source in _LOCAL_FINAL_SOURCES:
                raise StateConflict("local final verdicts require their atomic operation API")
        now = self._now()
        with self.transaction() as connection:
            result = self._update_job_row(
                connection,
                operation_id,
                now=now,
                expected_updated_at=expected_updated_at,
                phase=phase,
                ever_started=ever_started,
                current_node=current_node,
                last_node=last_node,
                terminal_state=terminal_state,
                exit_code=exit_code,
                observation_epoch=observation_epoch,
                evidence_provenance=provenance,
                evidence_detail=evidence_detail,
                final_source=source,
            )
        return self._job_from_row(result)

    def _update_job_row(
        self,
        connection: sqlite3.Connection,
        operation_id: str,
        *,
        now: str,
        expected_updated_at: str | None = None,
        phase: JobPhase | str | None = None,
        ever_started: bool | None = None,
        current_node: str | None | object = _UNSET,
        last_node: str | None | object = _UNSET,
        terminal_state: str | None | object = _UNSET,
        exit_code: str | None | object = _UNSET,
        observation_epoch: int | None = None,
        evidence_provenance: EvidenceProvenance | None | object = _UNSET,
        evidence_detail: str | None | object = _UNSET,
        final_source: FinalSource | None | object = _UNSET,
        allow_local_final: bool = False,
    ) -> sqlite3.Row:
        """Apply one lifecycle merge using a fresh row in ``connection``."""

        # This read is intentionally inside every caller's BEGIN IMMEDIATE
        # boundary.  Callers must not derive merge policy from a JobRecord read
        # before the transaction.
        row = self._require_job_row(connection, operation_id)
        if expected_updated_at is not None and row["updated_at"] != expected_updated_at:
            raise LifecycleRevisionConflict(
                f"job {operation_id} changed while scheduler evidence was collected; "
                "rerun the command"
            )
        new_phase = JobPhase(phase) if phase is not None else JobPhase(row["phase"])
        old_phase = JobPhase(row["phase"])
        if old_phase is JobPhase.FINAL and new_phase is not JobPhase.FINAL:
            raise StateConflict("a final job cannot transition back to a live phase")

        merged_epoch = max(
            int(row["observation_epoch"]),
            observation_epoch if observation_epoch is not None else 0,
        )

        started = bool(row["ever_started"])
        if ever_started is False and started:
            raise StateConflict("ever_started is monotonic and cannot be cleared")
        started = started or bool(ever_started)

        node = row["current_node"] if current_node is _UNSET else current_node
        prior_last_node = (
            row["last_node"] if last_node is _UNSET or last_node is None else last_node
        )
        if new_phase is not JobPhase.ACTIVE:
            node = None
        elif node and (last_node is _UNSET or last_node is None):
            prior_last_node = node

        parsed_source = final_source
        if parsed_source is not _UNSET and parsed_source is not None:
            parsed_source = FinalSource(parsed_source)
            if parsed_source in _LOCAL_FINAL_SOURCES and not allow_local_final:
                raise StateConflict("local final verdicts require their atomic operation API")

        parsed_provenance = evidence_provenance
        if parsed_provenance is not _UNSET and parsed_provenance is not None:
            parsed_provenance = EvidenceProvenance(parsed_provenance)
        cleaned_detail = evidence_detail

        if new_phase is JobPhase.FINAL:
            if old_phase is JobPhase.FINAL:
                existing_source = FinalSource(row["final_source"])
                finalized_at = row["finalized_at"]
                if existing_source in _LOCAL_FINAL_SOURCES:
                    # Explicit local outcomes are never reinterpreted as a
                    # scheduler verdict, even if an abandoned remote job later
                    # becomes visible.  Return the row untouched so even its
                    # durable timestamps remain immutable.
                    return row
                else:
                    incoming_source = (
                        existing_source
                        if parsed_source is _UNSET or parsed_source is None
                        else parsed_source
                    )
                    if incoming_source in _LOCAL_FINAL_SOURCES:
                        raise StateConflict("a scheduler final cannot become a local verdict")
                    existing_rank = _FINAL_AUTHORITY[existing_source]
                    incoming_rank = _FINAL_AUTHORITY[incoming_source]
                    if incoming_rank < existing_rank:
                        merged_source = existing_source
                        merged_terminal = row["terminal_state"]
                        merged_exit = row["exit_code"]
                        merged_provenance = row["evidence_provenance"]
                        merged_detail = row["evidence_detail"]
                    elif incoming_rank > existing_rank:
                        merged_source = incoming_source
                        merged_terminal = (
                            row["terminal_state"]
                            if terminal_state is _UNSET or terminal_state is None
                            else terminal_state
                        )
                        merged_exit = (
                            row["exit_code"]
                            if exit_code is _UNSET or exit_code is None
                            else exit_code
                        )
                        if incoming_source is FinalSource.CONFIRMED_QUEUE:
                            merged_provenance = parsed_provenance
                            merged_detail = cleaned_detail
                        else:
                            merged_provenance = None
                            merged_detail = None
                    else:
                        merged_source = existing_source
                        merged_terminal = self._merge_equal_final_value(
                            "terminal_state", row["terminal_state"], terminal_state
                        )
                        merged_exit = self._merge_equal_final_value(
                            "exit_code", row["exit_code"], exit_code
                        )
                        merged_provenance = row["evidence_provenance"]
                        merged_detail = row["evidence_detail"]
            else:
                if parsed_source is _UNSET or parsed_source is None:
                    raise StateConflict("final job evidence requires final_source provenance")
                merged_source = parsed_source
                merged_terminal = (
                    row["terminal_state"]
                    if terminal_state is _UNSET or terminal_state is None
                    else terminal_state
                )
                merged_exit = (
                    row["exit_code"]
                    if exit_code is _UNSET or exit_code is None
                    else exit_code
                )
                if merged_source is FinalSource.SUBMIT_FAILED and merged_terminal != "SUBMIT_FAILED":
                    raise StateConflict("submit-failed final must retain SUBMIT_FAILED state")
                if merged_source is FinalSource.ABANDONED and merged_terminal != "ABANDONED":
                    raise StateConflict("abandoned final must retain ABANDONED state")
                if merged_source is FinalSource.CONFIRMED_QUEUE:
                    merged_provenance = (
                        row["evidence_provenance"]
                        if parsed_provenance is _UNSET
                        and old_phase is JobPhase.TERMINAL_CANDIDATE
                        else None
                        if parsed_provenance is _UNSET
                        else parsed_provenance
                    )
                    merged_detail = (
                        row["evidence_detail"]
                        if cleaned_detail is _UNSET
                        and old_phase is JobPhase.TERMINAL_CANDIDATE
                        else None
                        if cleaned_detail is _UNSET
                        else cleaned_detail
                    )
                else:
                    merged_provenance = None
                    merged_detail = None
                finalized_at = now
        else:
            if parsed_source is not _UNSET and parsed_source is not None:
                raise StateConflict("only a final job may have final_source provenance")
            merged_source = None
            finalized_at = None
            if new_phase is JobPhase.TERMINAL_CANDIDATE:
                merged_terminal = (
                    row["terminal_state"]
                    if terminal_state is _UNSET or terminal_state is None
                    else terminal_state
                )
                merged_exit = (
                    row["exit_code"]
                    if exit_code is _UNSET or exit_code is None
                    else exit_code
                )
                merged_provenance = (
                    row["evidence_provenance"]
                    if parsed_provenance is _UNSET
                    and old_phase is JobPhase.TERMINAL_CANDIDATE
                    else None
                    if parsed_provenance is _UNSET
                    else parsed_provenance
                )
                merged_detail = (
                    row["evidence_detail"]
                    if cleaned_detail is _UNSET
                    and old_phase is JobPhase.TERMINAL_CANDIDATE
                    else None
                    if cleaned_detail is _UNSET
                    else cleaned_detail
                )
            else:
                # A successful live observation invalidates stale terminal
                # candidate metadata as well as its phase.
                merged_terminal = None
                merged_exit = None
                merged_provenance = None
                merged_detail = None

        if merged_provenance is _UNSET:
            merged_provenance = None
        if merged_detail is _UNSET:
            merged_detail = None
        if new_phase is JobPhase.TERMINAL_CANDIDATE and merged_provenance is None:
            raise StateConflict("terminal candidate requires evidence provenance")
        if (
            new_phase is JobPhase.FINAL
            and merged_source is FinalSource.CONFIRMED_QUEUE
            and merged_provenance is None
        ):
            raise StateConflict("confirmed-queue final requires evidence provenance")
        if merged_detail is not None and merged_provenance is None:
            raise StateConflict("evidence detail requires evidence provenance")

        merged_provenance_value = (
            merged_provenance.value
            if isinstance(merged_provenance, EvidenceProvenance)
            else merged_provenance
        )
        merged_source_value = merged_source.value if merged_source is not None else None
        merged_values = (
            new_phase.value,
            int(started),
            node,
            prior_last_node,
            merged_terminal,
            merged_exit,
            merged_epoch,
            merged_provenance_value,
            merged_detail,
            merged_source_value,
            finalized_at,
        )
        current_values = tuple(
            row[column]
            for column in (
                "phase",
                "ever_started",
                "current_node",
                "last_node",
                "terminal_state",
                "exit_code",
                "observation_epoch",
                "evidence_provenance",
                "evidence_detail",
                "final_source",
                "finalized_at",
            )
        )
        if merged_values == current_values:
            return row

        now = self._next_job_timestamp(now, row["updated_at"])
        if old_phase is not JobPhase.FINAL and new_phase is JobPhase.FINAL:
            finalized_at = now

        connection.execute(
            """UPDATE jobs SET phase = ?, ever_started = ?, current_node = ?,
               last_node = ?, terminal_state = ?, exit_code = ?, observation_epoch = ?,
               evidence_provenance = ?, evidence_detail = ?, final_source = ?,
               updated_at = ?, finalized_at = ? WHERE operation_id = ?""",
            (
                new_phase.value,
                int(started),
                node,
                prior_last_node,
                merged_terminal,
                merged_exit,
                merged_epoch,
                merged_provenance_value,
                merged_detail,
                merged_source_value,
                now,
                finalized_at,
                operation_id,
            ),
        )
        return self._require_job_row(connection, operation_id)

    def _next_job_timestamp(self, candidate: str, previous: str) -> str:
        """Return a job version timestamp strictly newer than ``previous``."""

        try:
            candidate_time = datetime.fromisoformat(candidate)
            previous_time = datetime.fromisoformat(previous)
            if candidate_time.tzinfo is None:
                candidate_time = candidate_time.replace(tzinfo=timezone.utc)
            if previous_time.tzinfo is None:
                previous_time = previous_time.replace(tzinfo=timezone.utc)
            candidate_time = candidate_time.astimezone(timezone.utc)
            previous_time = previous_time.astimezone(timezone.utc)
            if candidate_time <= previous_time:
                candidate_time = previous_time + timedelta(microseconds=1)
        except (OverflowError, TypeError, ValueError) as exc:
            raise StateInvalid("job revision timestamp is malformed", path=self.path) from exc
        return _timestamp(candidate_time)

    @staticmethod
    def _merge_equal_final_value(
        field_name: str,
        existing: str | None,
        incoming: str | None | object,
    ) -> str | None:
        if incoming is _UNSET or incoming is None:
            return existing
        if existing is None:
            return incoming
        if existing != incoming:
            raise StateConflict(
                f"conflicting equal-authority final {field_name}: {existing!r} != {incoming!r}"
            )
        return existing

    def delete_job(self, operation_id: str) -> None:
        with self.transaction() as connection:
            row = self._require_job_row(connection, operation_id)
            if row["phase"] != JobPhase.FINAL.value:
                raise StateConflict("only final jobs may be deleted")
            connection.execute("DELETE FROM jobs WHERE operation_id = ?", (operation_id,))

    def get_operation(self, operation_id: str) -> OperationRecord:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            raise RecordNotFound(f"operation {operation_id} does not exist")
        return self._operation_from_row(row)

    def list_operations(self, *, unresolved_only: bool = False) -> list[OperationRecord]:
        query = "SELECT * FROM operations"
        parameters: list[str] = []
        if unresolved_only:
            placeholders = ",".join("?" for _ in UNRESOLVED_OPERATION_PHASES)
            query += f" WHERE phase IN ({placeholders})"
            parameters = [phase.value for phase in UNRESOLVED_OPERATION_PHASES]
        query += " ORDER BY created_at, operation_id"
        with self._read_connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._operation_from_row(row) for row in rows]

    def list_unresolved_operations(self) -> list[OperationRecord]:
        return self.list_operations(unresolved_only=True)

    def set_cluster_cache(
        self,
        cluster: str,
        key: str,
        value: Any,
        *,
        expires_at: datetime | None = None,
    ) -> None:
        self._validate_text_fields(cluster=cluster, key=key)
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise StateConflict("cluster cache value must contain only JSON values") from exc
        now = self._now()
        expiry = _timestamp(expires_at) if expires_at is not None else None
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO cluster_cache(cluster, cache_key, value_json, updated_at, expires_at)
                   VALUES (?, ?, ?, ?, ?) ON CONFLICT(cluster, cache_key) DO UPDATE SET
                   value_json = excluded.value_json, updated_at = excluded.updated_at,
                   expires_at = excluded.expires_at""",
                (cluster, key, encoded, now, expiry),
            )

    def get_cluster_cache(self, cluster: str, key: str) -> Any | None:
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT value_json, expires_at FROM cluster_cache WHERE cluster = ? AND cache_key = ?",
                (cluster, key),
            ).fetchone()
        if row is None:
            return None
        if not isinstance(row["value_json"], str) or (
            row["expires_at"] is not None and not isinstance(row["expires_at"], str)
        ):
            raise StateInvalid("cluster cache record is malformed", path=self.path)
        if row["expires_at"] is not None and row["expires_at"] <= self._now():
            self.delete_cluster_cache(cluster, key)
            return None
        try:
            return json.loads(row["value_json"], parse_constant=_reject_json_constant)
        except (TypeError, ValueError) as exc:
            raise StateInvalid("cluster cache contains invalid JSON", path=self.path) from exc

    def delete_cluster_cache(self, cluster: str, key: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM cluster_cache WHERE cluster = ? AND cache_key = ?", (cluster, key)
            )

    @staticmethod
    def _validate_text_fields(**values: str) -> None:
        for name, value in values.items():
            if not isinstance(value, str) or not value or any(
                not char.isprintable() for char in value
            ):
                raise StateConflict(f"{name} must be non-empty printable text")

    @staticmethod
    def _validate_job_update_values(
        *,
        current_node: str | None | object = _UNSET,
        last_node: str | None | object = _UNSET,
        terminal_state: str | None | object = _UNSET,
        exit_code: str | None | object = _UNSET,
    ) -> None:
        for field_name, value in (
            ("current_node", current_node),
            ("last_node", last_node),
            ("terminal_state", terminal_state),
            ("exit_code", exit_code),
        ):
            if value is _UNSET or value is None:
                continue
            if not isinstance(value, str) or not value or any(
                not character.isprintable() for character in value
            ):
                raise StateConflict(f"{field_name} must be printable text")
        for field_name, value in (("current_node", current_node), ("last_node", last_node)):
            if value is not _UNSET and value is not None and _NODE_NAME.fullmatch(value) is None:
                raise StateConflict(f"{field_name} is not a safe compute-node name")

    def _prepare_lifecycle_evidence(
        self,
        *,
        last_node: str | None | object,
        observation_epoch: int | None,
        evidence_provenance: EvidenceProvenance | str | None | object,
        evidence_detail: str | None | object,
    ) -> tuple[EvidenceProvenance | None | object, str | None | object]:
        """Validate lifecycle fields shared by atomic cancellation updates."""

        self._validate_job_update_values(last_node=last_node)
        if observation_epoch is not None and (
            isinstance(observation_epoch, bool)
            or not isinstance(observation_epoch, int)
            or observation_epoch < 0
        ):
            raise StateConflict("observation_epoch must be a non-negative integer")
        if evidence_provenance is _UNSET or evidence_provenance is None:
            provenance = evidence_provenance
        else:
            provenance = self._evidence_provenance(evidence_provenance)
        if evidence_detail is not _UNSET and evidence_detail is not None:
            evidence_detail = self._sanitize_detail(evidence_detail)
        return provenance, evidence_detail

    @staticmethod
    def _final_source(value: FinalSource | str) -> FinalSource:
        try:
            return FinalSource(value)
        except (TypeError, ValueError) as exc:
            raise StateConflict(f"unsupported final_source {value!r}") from exc

    @staticmethod
    def _evidence_provenance(
        value: EvidenceProvenance | str,
    ) -> EvidenceProvenance:
        try:
            return EvidenceProvenance(value)
        except (TypeError, ValueError) as exc:
            raise StateConflict(f"unsupported evidence_provenance {value!r}") from exc

    @staticmethod
    def _sanitize_detail(value: str) -> str:
        if not isinstance(value, str) or not value:
            raise StateConflict("operation detail must be a non-empty string")
        cleaned = " ".join(
            "".join(character if character.isprintable() else " " for character in value).split()
        )
        return (cleaned or "unspecified remote failure")[:4096]

    @staticmethod
    def _encode_object(value: Mapping[str, Any], name: str) -> str:
        if not isinstance(value, Mapping):
            raise StateConflict(f"{name} must be a mapping")
        try:
            return json.dumps(
                dict(value),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise StateConflict(f"{name} must contain only JSON values") from exc

    @staticmethod
    def _require_job_row(connection: sqlite3.Connection, operation_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM jobs WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFound(f"job operation {operation_id} does not exist")
        return row

    @staticmethod
    def _require_operation_row(connection: sqlite3.Connection, operation_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFound(f"operation {operation_id} does not exist")
        return row

    def _machine_from_row(self, row: sqlite3.Row) -> MachineRecord:
        try:
            values = {
                key: row[key]
                for key in ("machine_id", "hostname", "created_at", "updated_at")
            }
            if any(not isinstance(value, str) or not value for value in values.values()):
                raise TypeError
            if IDENTIFIER_RE.fullmatch(values["machine_id"]) is None or any(
                not character.isprintable() for character in values["hostname"]
            ):
                raise TypeError
            return MachineRecord(**values)
        except (IndexError, TypeError, ValueError) as exc:
            raise StateInvalid("machine record is malformed", path=self.path) from exc

    def _job_from_row(self, row: sqlite3.Row) -> JobRecord:
        try:
            resources = json.loads(
                row["resources_json"], parse_constant=_reject_json_constant
            )
            if not isinstance(resources, dict):
                raise TypeError
        except (IndexError, TypeError, ValueError) as exc:
            raise StateInvalid("job resources contain invalid JSON", path=self.path) from exc
        try:
            required = (
                "operation_id",
                "cluster",
                "logical_name",
                "owner_id",
                "slurm_job_name",
                "slurm_comment",
                "created_at",
                "updated_at",
            )
            if any(not isinstance(row[key], str) or not row[key] for key in required):
                raise TypeError
            if row["ever_started"] not in (0, 1):
                raise TypeError
            if (
                isinstance(row["observation_epoch"], bool)
                or not isinstance(row["observation_epoch"], int)
                or row["observation_epoch"] < 0
            ):
                raise TypeError
            job_id = row["job_id"]
            if job_id is not None and (
                not isinstance(job_id, str)
                or not job_id.isascii()
                or not job_id.isdigit()
            ):
                raise TypeError
            optional = (
                "current_node",
                "last_node",
                "terminal_state",
                "exit_code",
                "evidence_provenance",
                "evidence_detail",
                "final_source",
                "finalized_at",
            )
            if any(row[key] is not None and not isinstance(row[key], str) for key in optional):
                raise TypeError
            if any(
                row[key] is not None
                and any(not character.isprintable() for character in row[key])
                for key in optional
            ):
                raise TypeError
            if any(
                row[key] is not None and _NODE_NAME.fullmatch(row[key]) is None
                for key in ("current_node", "last_node")
            ):
                raise TypeError
            kind = JobKind(row["kind"])
            phase = JobPhase(row["phase"])
            final_source = (
                FinalSource(row["final_source"])
                if row["final_source"] is not None
                else None
            )
            evidence_provenance = (
                EvidenceProvenance(row["evidence_provenance"])
                if row["evidence_provenance"] is not None
                else None
            )
            operation_id = row["operation_id"]
            tag = parse_tag(row["slurm_comment"])
            if (
                OPERATION_RE.fullmatch(operation_id) is None
                or tag is None
                or tag.owner_id != row["owner_id"]
                or tag.operation_id != operation_id
                or tag.kind != kind.value
                or tag.logical_name != row["logical_name"]
                or row["slurm_job_name"]
                != build_slurm_job_name(kind.value, operation_id)
                or IDENTIFIER_RE.fullmatch(row["cluster"]) is None
                or (phase is JobPhase.FINAL) != (final_source is not None)
                or (phase is JobPhase.FINAL) != (row["finalized_at"] is not None)
                or (
                    phase is JobPhase.TERMINAL_CANDIDATE
                    and evidence_provenance is None
                )
                or (
                    final_source is FinalSource.CONFIRMED_QUEUE
                    and evidence_provenance is None
                )
                or (
                    phase is not JobPhase.TERMINAL_CANDIDATE
                    and final_source is not FinalSource.CONFIRMED_QUEUE
                    and evidence_provenance is not None
                )
                or (
                    row["evidence_detail"] is not None
                    and evidence_provenance is None
                )
                or (
                    final_source is FinalSource.SUBMIT_FAILED
                    and row["terminal_state"] != "SUBMIT_FAILED"
                )
                or (
                    final_source is FinalSource.ABANDONED
                    and row["terminal_state"] != "ABANDONED"
                )
                or (
                    kind is JobKind.ALLOCATION
                    and (
                        row["logical_name"].isdigit()
                        or row["logical_name"] in {"login", "run"}
                    )
                )
                or (kind is JobKind.RUN and row["logical_name"] != "run")
            ):
                raise TypeError
            return JobRecord(
                operation_id=operation_id,
                cluster=row["cluster"],
                logical_name=row["logical_name"],
                kind=kind,
                owner_id=row["owner_id"],
                slurm_job_name=row["slurm_job_name"],
                slurm_comment=row["slurm_comment"],
                phase=phase,
                resources=resources,
                job_id=job_id,
                ever_started=bool(row["ever_started"]),
                current_node=row["current_node"],
                last_node=row["last_node"],
                terminal_state=row["terminal_state"],
                exit_code=row["exit_code"],
                observation_epoch=row["observation_epoch"],
                evidence_provenance=evidence_provenance,
                evidence_detail=row["evidence_detail"],
                final_source=final_source,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                finalized_at=row["finalized_at"],
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise StateInvalid("job record is malformed", path=self.path) from exc

    def _operation_from_row(self, row: sqlite3.Row) -> OperationRecord:
        try:
            required = (
                "operation_id",
                "cluster",
                "logical_name",
                "target_job_operation_id",
                "created_at",
                "updated_at",
            )
            if any(not isinstance(row[key], str) or not row[key] for key in required):
                raise TypeError
            for key in ("job_id", "detail", "resolved_at"):
                if row[key] is not None and not isinstance(row[key], str):
                    raise TypeError
            if row["detail"] is not None and any(
                not character.isprintable() for character in row["detail"]
            ):
                raise TypeError
            if row["job_id"] is not None and (
                not row["job_id"].isascii() or not row["job_id"].isdigit()
            ):
                raise TypeError
            if (
                OPERATION_RE.fullmatch(row["operation_id"]) is None
                or OPERATION_RE.fullmatch(row["target_job_operation_id"]) is None
            ):
                raise TypeError
            return OperationRecord(
                operation_id=row["operation_id"],
                kind=OperationKind(row["kind"]),
                phase=OperationPhase(row["phase"]),
                cluster=row["cluster"],
                logical_name=row["logical_name"],
                target_job_operation_id=row["target_job_operation_id"],
                job_id=row["job_id"],
                detail=row["detail"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                resolved_at=row["resolved_at"],
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise StateInvalid("operation record is malformed", path=self.path) from exc
