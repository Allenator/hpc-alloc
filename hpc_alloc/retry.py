"""Pacing and patience for the polling loops.

Two separate concerns live here.  :class:`RetryBudget` decides how long to keep
waiting when an observation *fails*.  :class:`PollBackoff` decides how long to
wait between observations that *succeed*.  They are deliberately distinct: a
loop can be patient with errors while still being a good citizen on the wire.

Bounded patience for transient scheduler and transport failures.

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
from typing import Any, Callable

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


# Polling the scheduler is this tool's dominant load on a shared controller.
# The floor keeps a fast-moving job responsive; the ceiling keeps a long-lived
# one from making thousands of queries that all learn the same thing.
POLL_MIN_INTERVAL_SECONDS = 5.0
POLL_MAX_INTERVAL_SECONDS = 30.0

_UNOBSERVED = object()


def observation_signature(assessment: Any) -> tuple[object, ...]:
    """What counts, for :class:`PollBackoff`, as the job's situation changing.

    Any difference collapses the poll interval back to its floor, so a loop polls
    hardest exactly when the job is moving -- getting a node, starting,
    requeueing, dying -- and coasts when it is simply running.  Both polling
    loops (the `run`/`logs -f` follower and `up`'s wait) share this one
    definition; they used to hand-roll it separately and had already drifted --
    one omitted ``terminal_evidence``, so it missed a job entering a death
    candidate.
    """

    return (
        assessment.phase,
        assessment.scheduler_state,
        assessment.current_node,
        assessment.terminal_evidence,
    )


class PollBackoff:
    """Widen the gap between scheduler observations while nothing changes.

    Polling the scheduler every few seconds was the tool's dominant load on a
    shared Slurm controller: a four-hour run made thousands of observations, and
    almost every one of them learned nothing at all -- the job was still running,
    exactly as it had been on the previous poll.

    Exponential backoff is self-correcting in the right direction here.  A job
    that starts, or ends, quickly is still noticed quickly, because the interval
    has had no time to grow.  Only a job that has been sitting in one state for a
    long while is polled slowly -- and there the added latency is proportionally
    negligible: thirty seconds on top of a twenty-minute queue wait.

    Any observed change drops the interval straight back to the floor, so the
    loop is at its most responsive exactly when something is happening.
    """

    def __init__(
        self,
        *,
        minimum: float = POLL_MIN_INTERVAL_SECONDS,
        maximum: float = POLL_MAX_INTERVAL_SECONDS,
    ) -> None:
        if minimum <= 0 or maximum < minimum:
            raise ValueError("poll backoff needs 0 < minimum <= maximum")
        self._minimum = minimum
        self._maximum = maximum
        self._interval = minimum
        self._signature: object = _UNOBSERVED

    def interval(self, signature: object) -> float:
        """How long to wait before observing again.

        ``signature`` summarizes what the observation just saw.  When it differs
        from the previous one -- the job moved, got a node, changed state -- the
        interval collapses back to the floor.
        """

        if signature != self._signature:
            self._signature = signature
            self._interval = self._minimum
            return self._interval
        self._interval = min(self._interval * 2, self._maximum)
        return self._interval

    def reset(self) -> None:
        self._interval = self._minimum
        self._signature = _UNOBSERVED


__all__ = [
    "POLL_MAX_INTERVAL_SECONDS",
    "POLL_MIN_INTERVAL_SECONDS",
    "RETRY_INTERVAL_SECONDS",
    "SCHEDULER_PATIENCE_SECONDS",
    "TRANSPORT_PATIENCE_SECONDS",
    "PollBackoff",
    "observation_signature",
    "RetryBudget",
]
