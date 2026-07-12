"""Cluster-qualified selector parsing and ambiguity checks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from .errors import IdentityMismatch
from .ownership import OPERATION_RE


class SelectorKind(StrEnum):
    NAME = "name"
    JOB_ID = "job-id"
    OPERATION_ID = "operation-id"


@dataclass(frozen=True)
class Selector:
    cluster: str | None
    value: str

    @property
    def kind(self) -> SelectorKind:
        if self.value.startswith("@"):
            return SelectorKind.OPERATION_ID
        if self.value.isascii() and self.value.isdigit():
            return SelectorKind.JOB_ID
        return SelectorKind.NAME


def parse_selector(text: str, explicit_cluster: str | None = None) -> Selector:
    if ":" in text:
        cluster, value = text.split(":", 1)
        if not cluster or not value:
            raise IdentityMismatch(f"invalid qualified selector {text!r}")
        if explicit_cluster and explicit_cluster != cluster:
            raise IdentityMismatch(
                f"selector names cluster {cluster!r}, conflicting with --cluster {explicit_cluster!r}"
            )
        selector = Selector(cluster, value)
    else:
        selector = Selector(explicit_cluster, text)
    if selector.kind is SelectorKind.OPERATION_ID and OPERATION_RE.fullmatch(
        selector.value[1:]
    ) is None:
        raise IdentityMismatch(
            "operation selectors must be @ followed by 32 lowercase hexadecimal characters"
        )
    return selector


def canonical_job_selector(job: object) -> str:
    cluster = str(getattr(job, "cluster"))
    operation_id = str(getattr(job, "operation_id"))
    return f"{cluster}:@{operation_id}"


def _is_final(job: object) -> bool:
    phase = getattr(job, "phase", None)
    return str(getattr(phase, "value", phase)) == "FINAL"


def unique_job(
    jobs: Iterable[object],
    selector: Selector,
) -> object:
    matches = []
    field = {
        SelectorKind.NAME: "logical_name",
        SelectorKind.JOB_ID: "job_id",
        SelectorKind.OPERATION_ID: "operation_id",
    }[selector.kind]
    expected = selector.value[1:] if selector.kind is SelectorKind.OPERATION_ID else selector.value
    for job in jobs:
        if selector.cluster and getattr(job, "cluster") != selector.cluster:
            continue
        if str(getattr(job, field, "") or "") == expected:
            matches.append(job)
    # Numeric IDs and logical names are convenience locators.  A current job
    # takes precedence over retained history, while an explicit operation
    # selector always addresses exactly the requested durable record.
    if selector.kind is not SelectorKind.OPERATION_ID:
        live = [job for job in matches if not _is_final(job)]
        if live:
            matches = live
    if not matches:
        raise IdentityMismatch(f"no managed job matches {format_selector(selector)!r}")
    if len(matches) > 1:
        choices = ", ".join(sorted(canonical_job_selector(job) for job in matches))
        raise IdentityMismatch(f"{format_selector(selector)!r} matches multiple jobs; use {choices}")
    return matches[0]


def format_selector(selector: Selector) -> str:
    return f"{selector.cluster}:{selector.value}" if selector.cluster else selector.value


__all__ = [
    "Selector",
    "SelectorKind",
    "canonical_job_selector",
    "format_selector",
    "parse_selector",
    "unique_job",
]
