from __future__ import annotations

import unittest

from hpc_alloc.eligibility import (
    AccessVerdict,
    UserAccess,
    parse_partition_rules,
    parse_user_access,
    partition_eligibility,
)


# The exact shapes the slurm adapter emits, captured live from Bouchet.
USER_ACCESS = "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||interactive,normal\n"

PARTITIONS = "\n".join(
    (
        "PartitionName=priority_gpu AllowGroups=ALL AllowAccounts=priority "
        "DenyQos=normal,nothrottle,interactive State=UP",
        "PartitionName=gpu_b200 AllowGroups=ALL AllowAccounts=ALL "
        "AllowQos=normal,nothrottle State=UP",
        "PartitionName=education_gpu AllowGroups=ALL "
        "AllowAccounts=admins,course01,course02 State=UP",
        "PartitionName=day AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL State=UP",
    )
)


class UserAccessParsingTests(unittest.TestCase):
    def test_parses_groups_accounts_and_qos(self) -> None:
        access = parse_user_access(USER_ACCESS)
        assert access is not None
        self.assertEqual(access.accounts, frozenset({"pi_lab01"}))
        self.assertEqual(access.qos, frozenset({"interactive", "normal"}))
        self.assertEqual(access.groups, frozenset({"ab1234", "pi_lab01"}))
        self.assertFalse(access.qos_scoped)  # unscoped associations (empty Partition)

    def test_absent_associations_fall_open_to_none(self) -> None:
        # No ASSOC rows -> None, so the caller does not gate on missing data.
        self.assertIsNone(parse_user_access("GROUPS ab1234 pi_lab01\nASSOC\n"))
        self.assertIsNone(parse_user_access(""))


class PartitionRuleParsingTests(unittest.TestCase):
    def test_allow_all_becomes_none_and_lists_parse(self) -> None:
        rules = parse_partition_rules(PARTITIONS)
        self.assertEqual(set(rules), {"priority_gpu", "gpu_b200", "education_gpu", "day"})
        pg = rules["priority_gpu"]
        self.assertEqual(pg.allow_accounts, frozenset({"priority"}))
        self.assertIsNone(pg.allow_qos)  # AllowQos absent -> unrestricted (None)
        self.assertEqual(pg.deny_qos, frozenset({"normal", "nothrottle", "interactive"}))
        b200 = rules["gpu_b200"]
        self.assertIsNone(b200.allow_accounts)  # ALL -> None
        self.assertEqual(b200.allow_qos, frozenset({"normal", "nothrottle"}))


class EligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.access = parse_user_access(USER_ACCESS)
        assert self.access is not None
        self.rules = parse_partition_rules(PARTITIONS)

    def test_priority_gpu_is_denied_on_account(self) -> None:
        # Double-locked live (AllowAccounts=priority AND DenyQos), but account is
        # checked first, so that is the decisive reason.
        verdict, reason = partition_eligibility(self.rules["priority_gpu"], self.access)
        self.assertIs(verdict, AccessVerdict.DENY)
        self.assertIn("priority", reason)

    def test_gpu_b200_and_day_are_allowed(self) -> None:
        for name in ("gpu_b200", "day"):
            with self.subTest(partition=name):
                verdict, reason = partition_eligibility(self.rules[name], self.access)
                self.assertIs(verdict, AccessVerdict.ALLOW, reason)
                self.assertEqual(reason, "")

    def test_education_gpu_is_denied_on_account(self) -> None:
        verdict, reason = partition_eligibility(self.rules["education_gpu"], self.access)
        self.assertIs(verdict, AccessVerdict.DENY)
        self.assertIn("account", reason)

    def test_deny_qos_blocks_even_when_allow_is_all(self) -> None:
        rules = parse_partition_rules(
            "PartitionName=q AllowGroups=ALL AllowAccounts=ALL "
            "DenyQos=normal,interactive State=UP"
        )["q"]
        verdict, reason = partition_eligibility(rules, self.access)
        self.assertIs(verdict, AccessVerdict.DENY)
        self.assertIn("denied", reason)

    def test_allow_qos_without_a_usable_member_is_denied(self) -> None:
        rules = parse_partition_rules(
            "PartitionName=q AllowGroups=ALL AllowAccounts=ALL AllowQos=priority State=UP"
        )["q"]
        verdict, reason = partition_eligibility(rules, self.access)
        self.assertIs(verdict, AccessVerdict.DENY)
        self.assertIn("priority", reason)

    def test_group_restriction_blocks_a_non_member(self) -> None:
        outsider = UserAccess(
            accounts=frozenset({"pi_lab01"}),
            qos=frozenset({"normal"}),
            groups=frozenset({"pi_lab01"}),
        )
        rules = parse_partition_rules(
            "PartitionName=q AllowGroups=other_lab AllowAccounts=ALL State=UP"
        )["q"]
        verdict, reason = partition_eligibility(rules, outsider)
        self.assertIs(verdict, AccessVerdict.DENY)
        self.assertIn("membership", reason)

    def test_empty_qos_is_unknown_not_denied(self) -> None:
        # A blank sacctmgr QOS column leaves qos empty; it must fall open to
        # UNKNOWN (never DENY), even against an AllowQos-restricted partition, so
        # the accelerator can never block a legitimate job on missing QOS data.
        access = parse_user_access("GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||\n")
        assert access is not None
        self.assertEqual(access.qos, frozenset())
        for name in ("day", "gpu_b200"):
            with self.subTest(partition=name):
                verdict, _ = partition_eligibility(self.rules[name], access)
                self.assertIs(verdict, AccessVerdict.UNKNOWN)
        # A provable account barrier still denies even with no QOS data.
        verdict, _ = partition_eligibility(self.rules["priority_gpu"], access)
        self.assertIs(verdict, AccessVerdict.DENY)

    def test_partition_scoped_qos_is_unknown_not_a_false_allow(self) -> None:
        # A QOS granted only on a specific partition must not read as globally
        # usable: the flattened view yields UNKNOWN, not a false ALLOW.
        access = parse_user_access(
            "GROUPS ab1234 pi_lab01\nASSOC\npi_lab01||normal\npi_lab01|gpu|gpu_qos\n"
        )
        assert access is not None
        self.assertTrue(access.qos_scoped)
        rules = parse_partition_rules(
            "PartitionName=bigmem AllowGroups=ALL AllowAccounts=ALL AllowQos=gpu_qos State=UP"
        )["bigmem"]
        verdict, _ = partition_eligibility(rules, access)
        self.assertIs(verdict, AccessVerdict.UNKNOWN)

    def test_empty_groups_is_unknown_not_denied(self) -> None:
        # A failed `id -Gn` leaves groups empty while associations remain; a
        # group-restricted partition must fall open to UNKNOWN, not refuse.
        access = UserAccess(
            accounts=frozenset({"pi_lab01"}),
            qos=frozenset({"normal"}),
            groups=frozenset(),
        )
        rules = parse_partition_rules(
            "PartitionName=q AllowGroups=other_lab AllowAccounts=ALL State=UP"
        )["q"]
        verdict, _ = partition_eligibility(rules, access)
        self.assertIs(verdict, AccessVerdict.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
