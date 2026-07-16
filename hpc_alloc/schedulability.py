"""Parse a scheduler dry-run probe into a start estimate per partition.

The probe asks the scheduler when a hypothetical request would start, without
submitting anything.  It turns idle-but-reserved capacity -- which the raw
availability counts cannot tell apart from capacity you can actually claim --
into a concrete "starts at T" answer, so a caller can rank partitions by when
the request would really run.  Estimates are advisory: the queue shifts, so a
reported start can move earlier or later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeResult:
    partition: str
    schedulable: bool
    start: str | None
    detail: str


_START_RE = re.compile(r"to start at (\S+)")
_ERROR_RE = re.compile(r"error:\s*(.*)", re.IGNORECASE)


def parse_probe(partition: str, text: str) -> ProbeResult:
    """Turn one dry-run probe's raw output into a typed result.

    A "to start at <time>" line means the request is schedulable (now or later);
    anything else is treated as not schedulable, carrying the scheduler's own
    error text when present.
    """

    start = _START_RE.search(text)
    if start is not None:
        return ProbeResult(partition, True, start.group(1), "")
    error = _ERROR_RE.search(text)
    detail = (error.group(1) if error is not None else text).strip()
    return ProbeResult(partition, False, None, detail[:200])


def rank_probes(results: list[ProbeResult]) -> list[ProbeResult]:
    """Schedulable first and soonest start earliest; not-schedulable last.

    ISO timestamps sort chronologically as plain strings, so the earliest start
    ranks first among the schedulable results.
    """

    return sorted(results, key=lambda result: (not result.schedulable, result.start or ""))
