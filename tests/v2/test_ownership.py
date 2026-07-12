from __future__ import annotations

import unittest

from hpc_alloc.ownership import (
    IDENTIFIER_RE,
    format_tag,
    normalize_host_label,
    parse_tag,
    slurm_job_name,
)


class OwnershipTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        op = "a" * 32
        tag = format_tag("deadbeef1234", op, "laptop", "allocation", "dev")
        parsed = parse_tag(tag)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.operation_id, op)
        self.assertEqual(parsed.logical_name, "dev")
        self.assertEqual(slurm_job_name("allocation", op), f"hpcalloc-v2-alloc-{op}")

    def test_old_and_malformed_tags_are_not_owned(self) -> None:
        self.assertIsNone(parse_tag("hpc-alloc:oldid:host"))
        self.assertIsNone(parse_tag("hpc-alloc:v2:bad:uuid:host:run:-"))

    def test_pathological_hostnames_normalize_deterministically(self) -> None:
        cases = (
            "_laptop.example",
            "-laptop.example",
            ".hidden",
            "💻.example",
            "",
            "---",
            "x" * 200,
        )
        for raw in cases:
            with self.subTest(raw=raw):
                first = normalize_host_label(raw)
                self.assertEqual(first, normalize_host_label(raw))
                self.assertIsNotNone(IDENTIFIER_RE.fullmatch(first))
                self.assertLessEqual(len(first), 63)
                tag = format_tag("deadbeef1234", "a" * 32, first, "run", "run")
                self.assertEqual(parse_tag(tag).host, first)  # type: ignore[union-attr]

    def test_lossy_host_labels_carry_collision_resistant_suffix(self) -> None:
        self.assertNotEqual(normalize_host_label("_same"), normalize_host_label("-same"))
        self.assertEqual(normalize_host_label("valid-host.example"), "valid-host")


if __name__ == "__main__":
    unittest.main()
