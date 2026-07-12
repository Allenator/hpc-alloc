from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from hpc_alloc.commands import _remote_home


class CommandRemoteHomeTests(unittest.TestCase):
    def context(self, cached: object) -> SimpleNamespace:
        state = SimpleNamespace(
            get_cluster_cache=Mock(return_value=cached),
            set_cluster_cache=Mock(),
        )
        config = SimpleNamespace(
            identity=SimpleNamespace(netid="ab1234"),
            clusters={"grace": SimpleNamespace(host="grace.example.edu")},
        )
        return SimpleNamespace(state=state, config=config)

    def assert_refreshes(self, cached: object) -> None:
        ctx = self.context(cached)
        client = SimpleNamespace(remote_home=Mock(return_value="/home/ab1234"))

        self.assertEqual(_remote_home(ctx, client, "grace"), "/home/ab1234")

        client.remote_home.assert_called_once_with()
        ctx.state.set_cluster_cache.assert_called_once_with(
            "grace",
            "remote_home",
            {
                "path": "/home/ab1234",
                "netid": "ab1234",
                "host": "grace.example.edu",
            },
        )

    def test_matching_scoped_cache_is_reused(self) -> None:
        cached = {
            "path": "/home/ab1234",
            "netid": "ab1234",
            "host": "grace.example.edu",
        }
        ctx = self.context(cached)
        client = SimpleNamespace(remote_home=Mock())

        self.assertEqual(_remote_home(ctx, client, "grace"), "/home/ab1234")

        client.remote_home.assert_not_called()
        ctx.state.set_cluster_cache.assert_not_called()

    def test_changed_login_scope_refreshes_cache(self) -> None:
        mismatches = (
            {
                "path": "/home/old-user",
                "netid": "old-user",
                "host": "grace.example.edu",
            },
            {
                "path": "/home/ab1234",
                "netid": "ab1234",
                "host": "old-grace.example.edu",
            },
        )
        for cached in mismatches:
            with self.subTest(cached=cached):
                self.assert_refreshes(cached)

    def test_legacy_string_refreshes_cache(self) -> None:
        self.assert_refreshes("/home/old-user")

    def test_malformed_records_refresh_cache(self) -> None:
        malformed = (
            None,
            {"path": "/home/ab1234", "netid": "ab1234"},
            {
                "path": "/home/ab1234",
                "netid": "ab1234",
                "host": "grace.example.edu",
                "extra": True,
            },
            {
                "path": "home/ab1234",
                "netid": "ab1234",
                "host": "grace.example.edu",
            },
            {
                "path": "/home/ab1234\nother",
                "netid": "ab1234",
                "host": "grace.example.edu",
            },
            {
                "path": "/home/ab1234\x00other",
                "netid": "ab1234",
                "host": "grace.example.edu",
            },
        )
        for cached in malformed:
            with self.subTest(cached=cached):
                self.assert_refreshes(cached)


if __name__ == "__main__":
    unittest.main()
