"""Stable, secure process locks for authoritative local configuration scope."""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .errors import ConfigInvalid


@contextmanager
def configuration_scope_lock(
    path: Path, *, exclusive: bool
) -> Iterator[None]:
    """Hold a shared or exclusive lock on a stable configuration sibling.

    The lock is deliberately separate from every atomically replaced data
    file.  Opening with ``O_NOFOLLOW`` and validating the descriptor prevents
    a symlink or special file from becoming a lock target.
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
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
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
        os.close(descriptor)
