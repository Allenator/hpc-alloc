"""Bounded patience for transient scheduler and transport failures.

v1 wrapped both of its polling loops in bounded retries: roughly two minutes of
patience for scheduler errors, and a ten-minute reconnect window for transport
drops.  The v2 rewrite kept only ``SshTransport.run``'s single heal-and-retry,
which fires solely on an ssh rc-255 or timeout -- so it never retries
``SchedulerUnavailable`` at all, because a scheduler query that runs and exits
nonzero leaves ssh itself exiting 0.

The result was that one scheduler restart, a twenty-second laptop sleep, or a
VPN blip aborted ``up``'s wait and every ``run`` / ``logs -f`` stream outright,
while the GPU job it was watching kept running and burning its allocation.  This
module restores the missing budgets in one place so both loops share them.
"""

from __future__ import annotations

import time
from typing import Callable

from .errors import HpcAllocError, SchedulerUnavailable, TransportLost


# A scheduler hiccup (a controller restart) clears in seconds; a transport drop
# (VPN renegotiation, a closed laptop lid) can take minutes.
SCHEDULER_PATIENCE_SECONDS = 120.0
TRANSPORT_PATIENCE_SECONDS = 600.0
RETRY_INTERVAL_SECONDS = 15.0


class RetryBudget:
    """Absorb transient failures inside a polling loop, within a time budget.

    An *episode* begins at the first failure and ends at the next success.  The
    budget is measured from the start of the episode, not from the last failure,
    so a flapping connection cannot extend it indefinitely; and it is reset on
    every success, so a blip an hour ago cannot combine with a blip now.

    Authentication and host-key failures are never absorbed.  They are not
    transient -- they need the user, and retrying them silently would either
    hammer a Duo prompt or spin until the budget expired on a fault that time
    cannot heal.
    """

    def __init__(
        self,
        *,
        scheduler_patience: float = SCHEDULER_PATIENCE_SECONDS,
        transport_patience: float = TRANSPORT_PATIENCE_SECONDS,
        interval: float = RETRY_INTERVAL_SECONDS,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        info: Callable[[str], None] | None = None,
    ) -> None:
        self._scheduler_patience = scheduler_patience
        self._transport_patience = transport_patience
        self._interval = interval
        self._sleep = sleeper
        self._clock = clock
        self._info = info
        self._started: float | None = None
        self._patience = 0.0

    def reset(self) -> None:
        """End the current failure episode after a successful observation."""

        self._started = None
        self._patience = 0.0

    def absorb(self, error: HpcAllocError) -> None:
        """Wait out a transient failure, or re-raise it.

        Returns normally when the caller should retry.  Re-raises the original
        error, with its exit code and message intact, when the failure is not
        retryable or the episode's budget is spent.
        """

        if isinstance(error, SchedulerUnavailable):
            patience = self._scheduler_patience
        elif isinstance(error, TransportLost):
            patience = self._transport_patience
        else:
            # Includes AuthRequired and HostKeyChanged, which time cannot heal,
            # and any typed failure that is a real answer rather than a blip.
            raise error

        now = self._clock()
        if self._started is None:
            self._started = now
            self._patience = patience
        else:
            # A mixed episode (a scheduler error, then a dropped transport) is
            # given the more generous of the two budgets, but still measured
            # from the moment the trouble started.
            self._patience = max(self._patience, patience)

        remaining = self._patience - (now - self._started)
        if remaining <= 0:
            raise error
        if self._info is not None:
            self._info(f"{error} — retrying for up to {int(remaining)}s")
        self._sleep(min(self._interval, remaining))


__all__ = [
    "RETRY_INTERVAL_SECONDS",
    "SCHEDULER_PATIENCE_SECONDS",
    "TRANSPORT_PATIENCE_SECONDS",
    "RetryBudget",
]
