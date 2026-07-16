"""Local partition-access eligibility from cached scheduler data.

Gates a submission before it is dispatched: a partition the user's account, QOS,
or groups cannot use is refused locally, so a pure access error never becomes an
ambiguous remote submission that must then be recovered.  This is a hard-access
SUBSET check (Allow/Deny by account, QOS, and group), NOT a resource-limit check
-- a request that is permitted here can still be rejected remotely for exceeding
a limit -- and it FAILS OPEN whenever its inputs are missing or unparsable: an
uncertain gate must never block a legitimate job.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AccessVerdict(Enum):
    """Whether a partition's access rules admit a user, from cached data.

    ``DENY`` is returned ONLY when the cached data PROVES the user cannot submit.
    Every uncertainty -- an empty or partition-scoped QOS set, unknown groups,
    absent data -- yields ``UNKNOWN`` so the caller falls open to the
    authoritative submit rather than blocking a legitimate job on a guess.
    ``ALLOW`` means proven clear.
    """

    ALLOW = "allow"
    DENY = "deny"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class UserAccess:
    accounts: frozenset[str]
    qos: frozenset[str]
    groups: frozenset[str]
    # True when any association is partition-scoped (a non-empty Partition
    # column): the flattened global QOS view cannot then be trusted to prove a
    # QOS denial, so QOS contributes only uncertainty, never a refusal.
    qos_scoped: bool = False


@dataclass(frozen=True, slots=True)
class PartitionRules:
    # ``None`` means an unrestricted allow-list (everyone), matching the
    # scheduler's ALL sentinel; a deny-list is a plain set (empty means none).
    allow_accounts: frozenset[str] | None
    allow_qos: frozenset[str] | None
    allow_groups: frozenset[str] | None
    deny_accounts: frozenset[str]
    deny_qos: frozenset[str]


def _csv(value: str) -> frozenset[str]:
    return frozenset(item for item in value.split(",") if item)


def _allow(value: str) -> frozenset[str] | None:
    return None if value in ("", "ALL", "(null)") else _csv(value)


def parse_user_access(text: str) -> UserAccess | None:
    """Parse the ``GROUPS ...`` / ``ASSOC`` / ``account|partition|qos`` snapshot.

    Returns ``None`` when no association rows are present, so the caller falls
    open rather than gating on absent data.
    """

    groups: frozenset[str] = frozenset()
    accounts: set[str] = set()
    qos: set[str] = set()
    qos_scoped = False
    in_assoc = False
    saw_assoc = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("GROUPS"):
            groups = frozenset(line[len("GROUPS") :].split())
        elif line == "ASSOC":
            in_assoc = True
        elif in_assoc and line:
            fields = line.split("|")
            if len(fields) >= 3 and fields[0]:
                saw_assoc = True
                accounts.add(fields[0])
                if fields[1]:
                    # Partition-scoped association: mark the uncertainty, but do
                    # NOT fold its QOS into the global set -- it applies only on
                    # that one partition, which this flattened view cannot place,
                    # so counting it globally would be a false grant.
                    qos_scoped = True
                else:
                    qos |= _csv(fields[2])
    if not saw_assoc:
        return None
    return UserAccess(frozenset(accounts), frozenset(qos), groups, qos_scoped)


def parse_partition_rules(text: str) -> dict[str, PartitionRules]:
    """Parse the one-line-per-partition access dump into rules by name."""

    rules: dict[str, PartitionRules] = {}
    for line in text.splitlines():
        if not line.startswith("PartitionName="):
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
        name = fields.get("PartitionName", "")
        if not name:
            continue
        rules[name] = PartitionRules(
            allow_accounts=_allow(fields.get("AllowAccounts", "ALL")),
            allow_qos=_allow(fields.get("AllowQos", "ALL")),
            allow_groups=_allow(fields.get("AllowGroups", "ALL")),
            deny_accounts=_csv(fields.get("DenyAccounts", "")),
            deny_qos=_csv(fields.get("DenyQos", "")),
        )
    return rules


def partition_eligibility(
    rules: PartitionRules, access: UserAccess
) -> tuple[AccessVerdict, str]:
    """Return ``(verdict, reason)``; ``reason`` is non-empty only on ``DENY``.

    Refuses (``DENY``) only when the cached data PROVES the user cannot submit --
    an account, QOS, or group barrier evaluated against complete data.  Any
    dimension we cannot resolve yields ``UNKNOWN`` instead, so the caller falls
    open to the authoritative submit rather than blocking a legitimate job:
    an empty QOS set (a blank accounting column), a partition-scoped QOS
    (granted only on some partitions), or unknown groups are all uncertainty,
    never a refusal.  Rules are checked account, then QOS, then group, so a DENY
    reason names the first decisive barrier.
    """

    # Account is always known: an association row implies a non-empty account.
    if rules.allow_accounts is not None and not (access.accounts & rules.allow_accounts):
        return AccessVerdict.DENY, f"requires an account in {sorted(rules.allow_accounts)}"
    denied_accounts = access.accounts & rules.deny_accounts
    if denied_accounts:
        return AccessVerdict.DENY, f"account {sorted(denied_accounts)} is denied"

    uncertain = False

    # QOS is a proven barrier only when the user's QOS view is complete: a
    # non-empty set with no partition-scoped rows.  Otherwise denial cannot be
    # proven, so QOS never blocks and only contributes uncertainty.
    usable = access.qos
    if rules.allow_qos is not None:
        usable = usable & rules.allow_qos
    usable = usable - rules.deny_qos
    if not usable:
        if access.qos and not access.qos_scoped:
            denied_qos = access.qos & rules.deny_qos
            if denied_qos:
                return AccessVerdict.DENY, f"QOS {sorted(denied_qos)} is denied"
            return AccessVerdict.DENY, f"requires a QOS in {sorted(rules.allow_qos or [])}"
        uncertain = True

    # Groups are a proven barrier only when we actually know the user's groups.
    if rules.allow_groups is not None and not (access.groups & rules.allow_groups):
        if access.groups:
            return AccessVerdict.DENY, f"requires membership in {sorted(rules.allow_groups)}"
        uncertain = True

    return (AccessVerdict.UNKNOWN if uncertain else AccessVerdict.ALLOW), ""
