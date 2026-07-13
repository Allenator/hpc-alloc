"""Process-output helpers used by command and CLI error paths."""

from __future__ import annotations

import os
import sys


def neutralize_stdout() -> None:
    """Best-effort redirect stdout to ``/dev/null`` after a broken pipe.

    Replacing the descriptor prevents Python's interpreter-shutdown flush from
    writing to the broken pipe again.  This helper must not mask the original
    error, including when stdout is an in-memory object without a descriptor.
    """

    try:
        stdout_fd = sys.stdout.fileno()
    except (AttributeError, OSError, TypeError, ValueError):
        return
    if stdout_fd != 1:
        return

    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except (OSError, TypeError, ValueError):
        return

    try:
        if devnull_fd != 1:
            os.dup2(devnull_fd, 1)
    except (OSError, TypeError, ValueError):
        pass
    finally:
        if devnull_fd != 1:
            try:
                os.close(devnull_fd)
            except OSError:
                pass
