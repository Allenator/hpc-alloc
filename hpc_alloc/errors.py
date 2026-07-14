"""Typed failures shared by hpc-alloc's internal service boundaries."""

from __future__ import annotations

from pathlib import Path


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
