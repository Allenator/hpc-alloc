"""Stable, secure process locks for configuration and remote mutations."""

from __future__ import annotations

import errno
import fcntl
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import ConfigInvalid, OperationBusy, StateConflict, StateInvalid


_OPERATION_ID = re.compile(r"[0-9a-f]{32}\Z")
# How often a bounded acquire re-tries.  Short enough that `setup` starts
# promptly once a long-running command exits, cheap enough to poll.
_LOCK_POLL_SECONDS = 0.25
# `setup` is the only exclusive taker.  Long enough to ride out an ordinary
# short command, short enough that a multi-hour `run` fails fast and explains.
SETUP_LOCK_TIMEOUT_SECONDS = 30.0


@contextmanager
def configuration_scope_lock(
    path: Path, *, exclusive: bool, timeout: float | None = None
) -> Iterator[None]:
    """Hold a shared or exclusive lock on a stable configuration sibling.

    The lock is deliberately separate from every atomically replaced data
    file.  Opening with ``O_NOFOLLOW`` and validating the descriptor prevents
    a symlink or special file from becoming a lock target.

    ``timeout`` bounds an otherwise unbounded wait.  Every command holds this
    lock *shared* for its whole lifetime -- hours, for a `run` or `logs -f` --
    and `setup` needs it exclusively, so a plain blocking acquire left `setup`
    hanging silently and indefinitely with no output and no explanation.  Worse,
    flock grants no writer preference, so a steady trickle of short shared
    commands (exactly what an agent polling `status` produces) could starve the
    exclusive waiter forever.  A bounded wait turns both into a fast, explicit
    failure that names the cause.
    """

    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
    except OSError as exc:
        raise ConfigInvalid(
            f"cannot prepare configuration-lock directory {path.parent}: {exc}"
        ) from exc

    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ConfigInvalid(f"cannot open configuration-scope lock {path}: {exc}") from exc

    locked = False
    try:
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ConfigInvalid(
                    f"configuration-scope lock is not a regular file: {path}"
                )
            if metadata.st_uid != os.geteuid():
                raise ConfigInvalid(
                    f"configuration-scope lock is not owned by the current user: {path}"
                )
            if metadata.st_nlink != 1:
                raise ConfigInvalid(
                    f"configuration-scope lock must have exactly one hard link: {path}"
                )
            os.fchmod(descriptor, 0o600)
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            if timeout is None:
                fcntl.flock(descriptor, mode)
            else:
                deadline = time.monotonic() + timeout
                while True:
                    try:
                        fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise StateConflict(
                                "another hpc-alloc command is holding the "
                                f"configuration lock ({path}); it is still "
                                "running — stop it, or retry once it exits"
                            ) from exc
                        time.sleep(_LOCK_POLL_SECONDS)
            locked = True
        except OSError as exc:
            kind = "exclusive" if exclusive else "shared"
            raise ConfigInvalid(
                f"cannot acquire {kind} configuration-scope lock {path}: {exc}"
            ) from exc
        yield
    finally:
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(descriptor)
        except OSError:
            pass


@contextmanager
def operation_scope_lock(
    directory: Path,
    operation_id: str,
    *,
    blocking: bool,
) -> Iterator[None]:
    """Own one operation while it may still issue a remote mutation.

    Lock files are stable and intentionally retained.  Unlinking after use
    could let a waiter and a new opener lock different inodes for the same
    operation.  The containing directory and file are opened without
    following their final symlink components and validated through their open
    descriptors before the advisory lock is acquired.
    """

    if (
        not isinstance(operation_id, str)
        or _OPERATION_ID.fullmatch(operation_id) is None
    ):
        raise StateConflict("operation ID must be 32 lowercase hexadecimal characters")
    if not isinstance(blocking, bool):
        raise TypeError("blocking must be a boolean")

    lock_path = directory / f"{operation_id}.lock"
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_descriptor = os.open(directory, directory_flags)
    except OSError as exc:
        raise StateInvalid(
            f"cannot prepare operation-lock directory {directory}: {exc}",
            path=directory,
        ) from exc

    descriptor: int | None = None
    locked = False
    try:
        try:
            directory_metadata = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(directory_metadata.st_mode):
                raise StateInvalid(
                    f"operation-lock directory is not a directory: {directory}",
                    path=directory,
                )
            if directory_metadata.st_uid != os.geteuid():
                raise StateInvalid(
                    f"operation-lock directory is not owned by the current user: {directory}",
                    path=directory,
                )
            os.fchmod(directory_descriptor, 0o700)

            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                f"{operation_id}.lock",
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise StateInvalid(
                    f"operation lock is not a regular file: {lock_path}",
                    path=lock_path,
                )
            if metadata.st_uid != os.geteuid():
                raise StateInvalid(
                    f"operation lock is not owned by the current user: {lock_path}",
                    path=lock_path,
                )
            if metadata.st_nlink != 1:
                raise StateInvalid(
                    f"operation lock must have exactly one hard link: {lock_path}",
                    path=lock_path,
                )
            os.fchmod(descriptor, 0o600)
            lock_flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(descriptor, lock_flags)
            except OSError as exc:
                if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise OperationBusy(
                        f"operation {operation_id} is active in another hpc-alloc "
                        "process; retry after it exits"
                    ) from exc
                raise
            locked = True
        except StateConflict:
            raise
        except StateInvalid:
            raise
        except OSError as exc:
            raise StateInvalid(
                f"cannot acquire operation lock {lock_path}: {exc}",
                path=lock_path,
            ) from exc
        yield
    finally:
        if locked and descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.close(directory_descriptor)
        except OSError:
            pass
