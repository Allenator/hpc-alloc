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

    def test_unknown_keys_fail_closed(self) -> None:
        payload = VALID_CONFIG.replace(b"cpus = 2", b"cpus = 2\nidentity_flie = \"typo\"")
        self.assert_invalid(payload, "unknown key 'identity_flie'")

    def test_boolean_and_nonpositive_resource_values_are_rejected(self) -> None:
        self.assert_invalid(VALID_CONFIG.replace(b"cpus = 2", b"cpus = true"), "positive integer")
        self.assert_invalid(VALID_CONFIG.replace(b"cpus = 2", b"cpus = 0"), "positive integer")

    def test_duration_minutes_and_seconds_are_range_checked(self) -> None:
        for duration in (b"4:60:00", b"4:00:60", b"1-2:99"):
            with self.subTest(duration=duration):
                payload = VALID_CONFIG.replace(
                    b"cpus = 2", b'cpus = 2\ntime = "' + duration + b'"'
                )
                self.assert_invalid(payload, "Slurm duration")


if __name__ == "__main__":
    unittest.main()
