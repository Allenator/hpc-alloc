from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hpc_alloc.context import RuntimeContext
from hpc_alloc.state import StateRepository


CONFIG = """\
[identity]
netid = "ab1234"
[defaults]
cluster = "grace"
[cluster.grace]
host = "grace.ycrc.yale.edu"
"""


class DryRunContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        root = Path(self.dir.name)
        self.config_path = root / "config.toml"
        self.config_path.write_text(CONFIG)
        self.state_path = root / "state.db"

    def _load(self) -> RuntimeContext:
        return RuntimeContext.load(
            command="dry-run",
            config_path=self.config_path,
            state_path=self.state_path,
        )

    def test_dry_run_without_a_state_db_leaves_state_none_and_creates_nothing(self) -> None:
        # No prior setup -> no database -> a dry run must not create one.
        self.assertFalse(self.state_path.exists())
        ctx = self._load()
        self.assertIsNone(ctx.state)
        self.assertFalse(self.state_path.exists())

    def test_dry_run_reads_an_existing_warm_cache_without_initializing(self) -> None:
        # A prior run warmed the topology cache; the dry run reads it (so it can
        # resolve a typed GPU partition offline) without initializing the journal.
        repo = StateRepository(self.state_path).initialize()
        repo.set_cluster_cache("grace", "gpu_topology", {"gpu_h200": ["h200"]})
        ctx = self._load()
        self.assertIsNotNone(ctx.state)
        self.assertEqual(
            ctx.state.get_cluster_cache("grace", "gpu_topology"),
            {"gpu_h200": ["h200"]},
        )


if __name__ == "__main__":
    unittest.main()
