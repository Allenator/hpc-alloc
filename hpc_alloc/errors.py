"""Typed failures shared by hpc-alloc's internal service boundaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path


# `up --wait` timed out with the job still queued.  This is neither success nor
# failure and must not be reported as either: exit 0 told an agent the seat was
# ready, so it would `ssh` into a queued allocation or re-submit; exit 1 says
# "application failure", so it would go hunting for a fault that is not there.
# The job is submitted, durable and healthy -- it is simply still waiting, which
# on a busy GPU partition is a routine outcome, not an edge case.  The only
# correct next action is to keep waiting or poll `status`, and that deserves a
# code an agent can branch on without parsing prose.
EXIT_SUBMITTED_NOT_READY = 4


class HpcAllocError(Exception):
    """Base class for an expected, user-actionable hpc-alloc failure."""

    exit_code = 1


class ConfigInvalid(HpcAllocError):
    """The authoritative configuration is absent, malformed, or inconsistent."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        prefix = f"{path}: " if path is not None else ""
        super().__init__(prefix + message)


class StateInvalid(HpcAllocError):
    """The state database is inaccessible, corrupt, or has the wrong schema."""

    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        prefix = f"{path}: " if path is not None else ""
        super().__init__(prefix + message)


class StateConflict(HpcAllocError):
    """A requested mutation conflicts with durable state or an active intent."""


class LifecycleRevisionConflict(StateConflict):
    """Scheduler evidence was collected against an outdated job revision."""


class StateIntegrityViolation(StateConflict, sqlite3.IntegrityError):
    """A database constraint rejected a write.

    Deliberately both.  It is a ``sqlite3.IntegrityError`` so that the two
    writers which inspect a unique-index violation to produce a friendlier
    message -- reserving a submission, beginning a cancellation -- keep working
    unchanged.  It is an :class:`HpcAllocError` so that a constraint violation
    *no writer anticipated* still reaches the CLI's error boundary as a typed
    failure with an exit code, instead of escaping as a raw traceback and
    breaking the no-traceback contract the whole boundary exists to uphold.
    """


class OperationBusy(StateConflict):
    """Another live hpc-alloc process holds this operation's scope lock.

    A StateConflict subclass so existing handlers keep their behaviour, but a
    distinct type because it says nothing about the operation itself: a bulk
    sweep must skip a busy operation and carry on rather than abort, whereas a
    conflict raised *by* an operation is a real failure.
    """


class RecordNotFound(HpcAllocError):
    """A requested durable job or operation does not exist."""


class AuthRequired(HpcAllocError):
    """Interactive authentication is required before work can continue."""

    exit_code = 3


class HostKeyChanged(HpcAllocError):
    """SSH rejected a host because its stored key no longer matches."""

    exit_code = 3


class TransportLost(HpcAllocError):
    """The remote transport failed or timed out."""

    exit_code = 3


class SchedulerUnavailable(HpcAllocError):
    """The transport is healthy but Slurm is unavailable."""


class RemoteCommandFailed(HpcAllocError):
    """A remote command ran and returned a non-zero status."""


class LocalToolUnavailable(HpcAllocError):
    """A required local executable could not be launched."""


class ProtocolViolation(HpcAllocError):
    """Remote output did not satisfy the expected typed protocol."""


class AmbiguousSubmission(HpcAllocError):
    """Submission may have committed, but no trustworthy reply was received."""


class IdentityMismatch(HpcAllocError):
    """A live Slurm job does not match the exact durable job identity."""


class JobIdReused(IdentityMismatch):
    """A persisted numeric Slurm job ID now belongs to another operation."""
