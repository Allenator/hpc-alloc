from __future__ import annotations

import unittest

from .fakes import ExpectedCall, StrictProxy, StrictScript, VirtualClock


class StrictScriptTests(unittest.TestCase):
    def test_exact_script_and_ledger(self) -> None:
        script = StrictScript(
            [ExpectedCall("observe", result="alive", args=("job",))]
        )
        fake = StrictProxy(script)
        self.assertEqual(fake.observe("job"), "alive")
        self.assertEqual(script.count("observe"), 1)
        script.assert_complete()

    def test_unknown_and_missing_calls_fail(self) -> None:
        script = StrictScript([ExpectedCall("cancel")])
        fake = StrictProxy(script)
        with self.assertRaisesRegex(AssertionError, "expected cancel"):
            fake.submit()
        with self.assertRaisesRegex(AssertionError, "unconsumed"):
            script.assert_complete()

    def test_virtual_clock(self) -> None:
        clock = VirtualClock()
        clock.sleep(3.5)
        self.assertEqual(clock.monotonic(), 3.5)
        self.assertEqual(clock.sleeps, [3.5])


if __name__ == "__main__":
    unittest.main()
