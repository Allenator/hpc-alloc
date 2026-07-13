"""Process-output helpers used by command and CLI error paths."""

from __future__ import annotations

import os
import sys


def _stream_fileno(stream: object, expected: int) -> int | None:
    try:
        descriptor = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return descriptor if descriptor == expected else None


def _descriptors_alias(first: int, second: int) -> bool:
    """Best-effort test whether two descriptors name the same open target."""

    try:
        return os.path.samestat(os.fstat(first), os.fstat(second))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _redirect_to_devnull(*descriptors: int) -> None:
    targets = set(descriptors)
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except (OSError, TypeError, ValueError):
        return

    try:
        for descriptor in descriptors:
            if devnull_fd == descriptor:
                continue
            try:
                os.dup2(devnull_fd, descriptor)
            except (OSError, TypeError, ValueError):
                pass
    finally:
        if devnull_fd not in targets:
            try:
                os.close(devnull_fd)
            except OSError:
                pass


def neutralize_stdout() -> None:
    """Best-effort redirect stdout to ``/dev/null`` after a broken pipe.

    Replacing the descriptor prevents Python's interpreter-shutdown flush from
    writing to the broken pipe again.  This helper must not mask the original
    error, including when stdout is an in-memory object without a descriptor.
    """

    stdout_fd = _stream_fileno(sys.stdout, 1)
    if stdout_fd is None:
        return

    stderr_fd = _stream_fileno(sys.stderr, 2)
    stderr_alias = stderr_fd is not None and _descriptors_alias(stdout_fd, stderr_fd)
    targets = (stdout_fd, stderr_fd) if stderr_alias else (stdout_fd,)
    _redirect_to_devnull(*targets)


def neutralize_stderr() -> None:
    """Best-effort redirect fd 2 after a write proves stderr is broken."""

    stderr_fd = _stream_fileno(sys.stderr, 2)
    if stderr_fd is not None:
        _redirect_to_devnull(stderr_fd)
