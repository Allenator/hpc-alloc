"""The bounded patience that keeps a transient blip from killing a stream."""

from __future__ import annotations

import unittest

from hpc_alloc.errors import (
    AuthRequired,
    HostKeyChanged,
    SchedulerUnavailable,
    TransportLost,
)
from hpc_alloc.retry import PollBackoff, RetryBudget

from .fakes import VirtualClock


class PollBackoffTests(unittest.TestCase):
    def test_a_steady_job_is_polled_less_and_less_often(self) -> None:
        """Almost every observation of a long-running job learns nothing.

        A fixed few-second poll made a four-hour run issue thousands of
        scheduler queries, all of which reported the same thing.
        """

        backoff = PollBackoff(minimum=5, maximum=30)
        steady = ("ACTIVE", "RUNNING", "node01")

        intervals = [backoff.interval(steady) for _ in range(6)]

        self.assertEqual(intervals, [5, 10, 20, 30, 30, 30])

    def test_any_change_drops_straight_back_to_the_floor(self) -> None:
        """The loop must be most responsive exactly when the job is moving."""

        backoff = PollBackoff(minimum=5, maximum=30)
        pending = ("QUEUED", "PENDING", None)

        for _ in range(4):
            backoff.interval(pending)
        self.assertEqual(backoff.interval(pending), 30)

        # The job got a node: poll hard again.
        self.assertEqual(backoff.interval(("ACTIVE", "RUNNING", "node01")), 5)

    def test_a_fast_start_is_still_noticed_fast(self) -> None:
        """Backoff is self-correcting: it has no time to grow on a quick job.

        This is what makes a ceiling affordable at all -- the latency it adds
        lands only on jobs that have already been waiting a long time, where it
        is proportionally negligible.
        """

        backoff = PollBackoff(minimum=5, maximum=30)
        self.assertEqual(backoff.interval(("QUEUED", "PENDING", None)), 5)
        self.assertEqual(backoff.interval(("ACTIVE", "RUNNING", "node01")), 5)

    def test_the_floor_and_ceiling_must_be_coherent(self) -> None:
        with self.assertRaises(ValueError):
            PollBackoff(minimum=0, maximum=30)
        with self.assertRaises(ValueError):
            PollBackoff(minimum=30, maximum=5)


class RetryBudgetTests(unittest.TestCase):
    def budget(self, clock: VirtualClock, **overrides: float) -> RetryBudget:
        return RetryBudget(sleeper=clock.sleep, clock=clock.monotonic, **overrides)

    def test_a_transient_scheduler_failure_is_ridden_out(self) -> None:
        """A controller restart used to abort the whole stream.

        SshTransport.run's single heal-and-retry fires only on an ssh rc-255, so
        a scheduler query that ran and exited nonzero was never retried at all:
        `up`'s wait and every `run` / `logs -f` stream died outright while the
        GPU job they were watching kept running.
        """

        clock = VirtualClock()
        budget = self.budget(clock, scheduler_patience=120, interval=15)

        for _ in range(4):
            budget.absorb(SchedulerUnavailable("controller is restarting"))

        self.assertEqual(clock.sleeps, [15, 15, 15, 15])

    def test_patience_is_bounded_and_the_original_error_still_surfaces(self) -> None:
        clock = VirtualClock()
        budget = self.budget(clock, scheduler_patience=30, interval=15)
        failure = SchedulerUnavailable("the scheduler is down for good")

        budget.absorb(failure)
        budget.absorb(failure)
        with self.assertRaises(SchedulerUnavailable) as raised:
            budget.absorb(failure)

        # The caller's typed error, exit code and message all survive intact.
        self.assertIs(raised.exception, failure)
        self.assertEqual(clock.sleeps, [15, 15])

    def test_a_success_ends_the_episode(self) -> None:
        """A blip an hour ago must not combine with a blip now.

        The budget is measured from the start of a failure episode and reset on
        every success, so a long-running stream that survives an outage does not
        carry a nearly-spent budget into the next one.
        """

        clock = VirtualClock()
        budget = self.budget(clock, scheduler_patience=30, interval=15)

        budget.absorb(SchedulerUnavailable("blip"))
        budget.absorb(SchedulerUnavailable("blip"))
        budget.reset()

        # A fresh episode gets the whole budget again rather than inheriting an
        # exhausted one.
        budget.absorb(SchedulerUnavailable("blip"))
        budget.absorb(SchedulerUnavailable("blip"))

        self.assertEqual(clock.sleeps, [15, 15, 15, 15])

    def test_a_flapping_connection_cannot_extend_the_budget_forever(self) -> None:
        """Patience runs from the start of the trouble, not the last failure."""

        clock = VirtualClock()
        budget = self.budget(clock, transport_patience=60, interval=15)
        failure = TransportLost("VPN keeps dropping")

        for _ in range(4):
            budget.absorb(failure)
        with self.assertRaises(TransportLost):
            budget.absorb(failure)

        self.assertEqual(clock.now, 60)

    def test_a_transport_drop_gets_the_longer_reconnect_window(self) -> None:
        """A VPN renegotiation or a closed laptop lid takes minutes, not seconds."""

        clock = VirtualClock()
        budget = self.budget(
            clock, scheduler_patience=30, transport_patience=600, interval=15
        )

        for _ in range(10):
            budget.absorb(TransportLost("transport dropped"))

        self.assertEqual(clock.now, 150)

    def test_authentication_and_host_key_failures_are_never_retried(self) -> None:
        """Time cannot heal them.

        Waiting out an expired credential or a changed host key would only spin
        until the budget expired -- or, worse, hammer a Duo prompt -- while
        hiding the one thing the user has to act on.
        """

        clock = VirtualClock()
        budget = self.budget(clock)

        for failure in (AuthRequired("Duo push denied"), HostKeyChanged("key changed")):
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)) as raised:
                    budget.absorb(failure)
                self.assertIs(raised.exception, failure)

        self.assertEqual(clock.sleeps, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
