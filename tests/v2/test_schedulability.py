from __future__ import annotations

import unittest

from hpc_alloc.schedulability import parse_probe, rank_probes


# Exact dry-run output captured live from Bouchet (note the stray " a ").
B200 = (
    "sbatch: Job 18555211 to start at 2026-07-16T18:18:13 a using 2 processors "
    "on nodes a1116u09n01 in partition gpu_b200"
)
H200 = (
    "sbatch: Job 18555214 to start at 2026-07-16T16:05:16 a using 2 processors "
    "on nodes a1122u11n01 in partition gpu_h200"
)
DAY = (
    "sbatch: Job 18555213 to start at 2026-07-16T20:25:38 a using 2 processors "
    "on nodes a1132u20n01 in partition day"
)
ERR = "sbatch: error: Batch job submission failed: Requested node configuration is not available"


class ParseProbeTests(unittest.TestCase):
    def test_schedulable_captures_the_start_timestamp(self) -> None:
        result = parse_probe("gpu_b200", B200)
        self.assertTrue(result.schedulable)
        self.assertEqual(result.start, "2026-07-16T18:18:13")
        self.assertEqual(result.detail, "")

    def test_error_is_not_schedulable_and_carries_the_reason(self) -> None:
        result = parse_probe("q", ERR)
        self.assertFalse(result.schedulable)
        self.assertIsNone(result.start)
        self.assertIn("node configuration", result.detail)

    def test_unrecognized_output_is_not_schedulable(self) -> None:
        result = parse_probe("q", "something unexpected\n")
        self.assertFalse(result.schedulable)
        self.assertIsNone(result.start)


class RankProbesTests(unittest.TestCase):
    def test_soonest_start_first_then_unschedulable_last(self) -> None:
        results = [
            parse_probe(name, text)
            for name, text in (
                ("gpu_b200", B200),
                ("gpu_h200", H200),
                ("day", DAY),
                ("full", ERR),
            )
        ]
        ranked = rank_probes(results)
        self.assertEqual(
            [r.partition for r in ranked], ["gpu_h200", "gpu_b200", "day", "full"]
        )
        self.assertFalse(ranked[-1].schedulable)


if __name__ == "__main__":
    unittest.main()
