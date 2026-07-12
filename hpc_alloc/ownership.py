"""Strong v2 Slurm job identity encoding."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


TAG_PREFIX = "hpc-alloc:v2:"
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,62}\Z")
OPERATION_RE = re.compile(r"[0-9a-f]{32}\Z")


def normalize_host_label(raw_hostname: str) -> str:
    """Return a deterministic ownership-tag label for a system hostname.

    Valid short DNS labels remain human-readable and unchanged.  Every lossy
    normalization carries a digest of the raw hostname so distinct pathological
    names do not silently collapse to the same display label.  ``owner_id``,
    rather than this label, remains the authoritative machine identity.
    """

    raw = str(raw_hostname)
    first_label = raw.split(".", 1)[0]
    if IDENTIFIER_RE.fullmatch(first_label):
        return first_label
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", first_label).strip("_-")
    cleaned = re.sub(r"[-_]{2,}", "-", cleaned)
    if not cleaned or not cleaned[0].isalnum() or not cleaned[0].isascii():
        cleaned = "host"
    digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()[:8]
    suffix = f"-{digest}"
    label = cleaned[: 63 - len(suffix)].rstrip("_-") or "host"
    normalized = label + suffix
    assert IDENTIFIER_RE.fullmatch(normalized)
    return normalized


@dataclass(frozen=True)
class OwnershipTag:
    owner_id: str
    operation_id: str
    host: str
    kind: str
    logical_name: str | None


def slurm_job_name(kind: str, operation_id: str) -> str:
    if kind not in ("allocation", "run"):
        raise ValueError(f"unsupported job kind: {kind}")
    if not OPERATION_RE.fullmatch(operation_id):
        raise ValueError(f"invalid operation id: {operation_id!r}")
    short_kind = "alloc" if kind == "allocation" else "run"
    return f"hpcalloc-v2-{short_kind}-{operation_id}"


def format_tag(
    owner_id: str,
    operation_id: str,
    host: str,
    kind: str,
    logical_name: str | None,
) -> str:
    for label, value in (("owner id", owner_id), ("host", host)):
        if not IDENTIFIER_RE.fullmatch(value):
            raise ValueError(f"invalid {label}: {value!r}")
    if not OPERATION_RE.fullmatch(operation_id):
        raise ValueError(f"invalid operation id: {operation_id!r}")
    if kind not in ("allocation", "run"):
        raise ValueError(f"invalid job kind: {kind!r}")
    name = logical_name or "-"
    if name != "-" and not IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"invalid logical name: {name!r}")
    return f"{TAG_PREFIX}{owner_id}:{operation_id}:{host}:{kind}:{name}"


def parse_tag(comment: str) -> OwnershipTag | None:
    if not comment.startswith(TAG_PREFIX):
        return None
    fields = comment[len(TAG_PREFIX) :].split(":")
    if len(fields) != 5:
        return None
    owner_id, operation_id, host, kind, name = fields
    try:
        expected = format_tag(
            owner_id,
            operation_id,
            host,
            kind,
            None if name == "-" else name,
        )
    except ValueError:
        return None
    if expected != comment:
        return None
    return OwnershipTag(owner_id, operation_id, host, kind, None if name == "-" else name)
