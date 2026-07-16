from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hpc_alloc.config import Config
from hpc_alloc.errors import ConfigInvalid


VALID_CONFIG = b"""\
[identity]
netid = "ab1234"

[ssh]
identity_file = "~/.ssh/id_ed25519"

[defaults]
cluster = "grace"
cpus = 2

[cluster.grace]
host = "grace.ycrc.yale.edu"
"""


class ConfigValidationTests(unittest.TestCase):
    def load_bytes(self, payload: bytes) -> Config:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_bytes(payload)
            return Config.load(path)

    def assert_invalid(self, payload: bytes, pattern: str) -> None:
        with self.assertRaisesRegex(ConfigInvalid, pattern):
            self.load_bytes(payload)

    def test_valid_config_resolves_only_configured_primary(self) -> None:
        config = self.load_bytes(VALID_CONFIG)
        self.assertEqual(config.resolve_cluster(), "grace")
        self.assertEqual(config.resolve_cluster("grace"), "grace")
        with self.assertRaisesRegex(ConfigInvalid, "not configured"):
            config.resolve_cluster("bouchet")

    def test_invalid_utf8_is_a_typed_config_error(self) -> None:
        self.assert_invalid(VALID_CONFIG + b"\xff", "not valid UTF-8")

    def test_malformed_toml_does_not_fall_back_to_builtins(self) -> None:
        self.assert_invalid(VALID_CONFIG.replace(b'cluster = "grace"', b'cluster = "grace'), "invalid TOML")

    def test_default_cluster_must_name_a_configured_cluster(self) -> None:
        self.assert_invalid(
            VALID_CONFIG.replace(b'cluster = "grace"', b'cluster = "typo"'),
            "unconfigured cluster",
        )

    def test_nondedicated_globs_cluster_overrides_defaults(self) -> None:
        payload = b"""\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
nondedicated_partition_globs = ["scavenge*"]
[cluster.grace]
host = "grace.ycrc.yale.edu"
nondedicated_partition_globs = ["preempt*", "*-short"]
[cluster.other]
host = "other.ycrc.yale.edu"
"""
        config = self.load_bytes(payload)
        self.assertEqual(config.nondedicated_globs("grace"), ("preempt*", "*-short"))
        self.assertEqual(config.nondedicated_globs("other"), ("scavenge*",))  # from defaults

    def test_nondedicated_globs_absent_is_none(self) -> None:
        # None means "unset"; the selection logic applies the built-in default.
        self.assertIsNone(self.load_bytes(VALID_CONFIG).nondedicated_globs("grace"))

    def test_nondedicated_globs_must_be_a_nonempty_string_list(self) -> None:
        self.assert_invalid(
            VALID_CONFIG + b'nondedicated_partition_globs = "scavenge*"\n',
            "must be a non-empty list",
        )
        self.assert_invalid(
            VALID_CONFIG + b"nondedicated_partition_globs = []\n",
            "must be a non-empty list",
        )

    def test_cluster_table_shape_is_validated(self) -> None:
        payload = b"""\
[identity]
netid = "ab1234"
[cluster]
grace = "grace.ycrc.yale.edu"
"""
        self.assert_invalid(payload, r"\[cluster\.grace\] must be a TOML table")

    def test_control_or_directive_injection_is_rejected(self) -> None:
        injected_host = VALID_CONFIG.replace(
            b'host = "grace.ycrc.yale.edu"',
            b'host = "grace.ycrc.yale.edu ProxyCommand=evil"',
        )
        self.assert_invalid(injected_host, "not a valid hostname")

    def test_bracketed_ip_hosts_drop_only_the_outer_brackets(self) -> None:
        cases = (
            (b"[192.0.2.7]", "192.0.2.7"),
            (b"[2001:0DB8:0000::7]", "2001:0DB8:0000::7"),
            (b"[FE80:0::1%eth0]", "FE80:0::1%eth0"),
            (b"2001:0DB8:0000::7", "2001:0DB8:0000::7"),
            (b"grace.ycrc.yale.edu.", "grace.ycrc.yale.edu."),
        )
        for configured, expected in cases:
            with self.subTest(configured=configured):
                payload = VALID_CONFIG.replace(b"grace.ycrc.yale.edu", configured)
                config = self.load_bytes(payload)
                self.assertEqual(config.clusters["grace"].host, expected)

    def test_brackets_must_be_balanced_and_enclose_an_ip_literal(self) -> None:
        for configured in (
            b"[grace.ycrc.yale.edu]",
            b"[2001:db8::7",
            b"2001:db8::7]",
        ):
            with self.subTest(configured=configured):
                payload = VALID_CONFIG.replace(b"grace.ycrc.yale.edu", configured)
                self.assert_invalid(payload, "not a valid hostname")

    def test_unknown_keys_fail_closed(self) -> None:
        payload = VALID_CONFIG.replace(b"cpus = 2", b"cpus = 2\nidentity_flie = \"typo\"")
        self.assert_invalid(payload, "unknown key 'identity_flie'")

    def test_boolean_and_nonpositive_resource_values_are_rejected(self) -> None:
        self.assert_invalid(VALID_CONFIG.replace(b"cpus = 2", b"cpus = true"), "positive integer")
        self.assert_invalid(VALID_CONFIG.replace(b"cpus = 2", b"cpus = 0"), "positive integer")

    def test_memory_accepts_only_ascii_digits(self) -> None:
        """`\\d` also matches full-width and Arabic-Indic digits.

        Those pass a `\\d`-based grammar but the scheduler rejects them, and
        because submission never retries, a value that slips past pre-flight
        validation fails remotely as an ambiguous submission ("may have
        committed") instead of a clean local ConfigInvalid.
        """

        for memory in ("６４G", "٦٤G"):
            with self.subTest(memory=memory):
                self.assert_invalid(
                    VALID_CONFIG.replace(b"cpus = 2", f'mem = "{memory}"'.encode()),
                    "mem",
                )
                with self.assertRaisesRegex(ConfigInvalid, "mem"):
                    Config.validate_resource_override("mem", memory)

        self.assertEqual(Config.validate_resource_override("mem", "64G"), "64G")

    def test_memory_matches_the_schedulers_mem_grammar(self) -> None:
        """The validator must be no more permissive than the scheduler.

        The scheduler's --mem is a whole number with an optional K/M/G/T unit.
        The old pattern also accepted a fractional part and B/iB/P/E suffixes;
        those pass pre-flight but the scheduler rejects them, so the submission
        fails remotely as the exact ambiguous "may have committed" outcome this
        check exists to avoid, over a pure typo.
        """

        for memory in ("1024", "500M", "16G", "64G", "2T", "500m", "16g"):
            with self.subTest(accepted=memory):
                self.assertEqual(
                    Config.validate_resource_override("mem", memory), memory
                )

        for memory in ("64GB", "64GiB", "1.5G", "64P", "64E", "16 G", "G", ""):
            with self.subTest(rejected=memory):
                with self.assertRaisesRegex(ConfigInvalid, "mem"):
                    Config.validate_resource_override("mem", memory)

    def test_all_documented_numeric_duration_forms_are_accepted(self) -> None:
        for duration in (
            b"90",
            b"90:30",
            b"4:30:00",
            b"2-00",
            b"2-04:30",
            b"2-04:30:00",
            b"0001",
            b"0:01",
            b"0:00:01",
            b"0-1",
            b"0-0:01",
            b"0-0:00:01",
        ):
            with self.subTest(duration=duration):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + duration + b'"'
                )
                self.assertEqual(self.load_bytes(payload).defaults.time, duration.decode())

    def test_all_zero_duration_forms_and_leading_zero_equivalents_are_rejected(self) -> None:
        for duration in (
            b"0",
            b"000",
            b"0:00",
            b"000:00",
            b"0:00:00",
            b"000:00:00",
            b"0-0",
            b"000-000",
            b"0-0:00",
            b"000-000:00",
            b"0-0:00:00",
            b"000-000:00:00",
        ):
            with self.subTest(duration=duration):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + duration + b'"'
                )
                self.assert_invalid(payload, "Slurm duration")

    def test_duration_minutes_and_seconds_are_range_checked(self) -> None:
        for duration in (
            b"4:60:00",
            b"4:00:60",
            b"1-2:99",
            b"1-02:00:60",
            b"1:2",
            b"1-02:3",
        ):
            with self.subTest(duration=duration):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + duration + b'"'
                )
                self.assert_invalid(payload, "Slurm duration")

    def test_duration_rejects_signs_whitespace_control_and_directive_text(self) -> None:
        for duration in (
            b"+90",
            b"-90",
            b" 90",
            b"90 ",
            b"90;--constraint=evil",
            b"INFINITE",
        ):
            with self.subTest(duration=duration):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + duration + b'"'
                )
                self.assert_invalid(payload, "Slurm duration")

        for escaped_control in (b"90\\t", b"90\\n"):
            with self.subTest(escaped_control=escaped_control):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + escaped_control + b'"'
                )
                self.assert_invalid(payload, "without control characters")


if __name__ == "__main__":
    unittest.main()
