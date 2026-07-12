from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


class InstallerDeliveryTests(unittest.TestCase):
    @staticmethod
    def assert_no_delivery(home: Path) -> None:
        if home.joinpath(".local").exists():
            raise AssertionError("installer created ~/.local before validation completed")
        if home.joinpath(".claude").exists():
            raise AssertionError("installer created ~/.claude before validation completed")

    def test_incomplete_source_fails_before_creating_delivery_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incomplete-source"
            home = root / "home"
            source.mkdir()
            home.mkdir()
            installer = shutil.copy2(REPO / "install.sh", source / "install.sh")
            shutil.copy2(REPO / "hpc-alloc", source / "hpc-alloc")
            self.assertFalse((source / "hpc_alloc").exists())

            result = subprocess.run(
                ["bash", str(installer)],
                cwd=source,
                env={**os.environ, "HOME": str(home)},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("hpc-alloc installation is incomplete:", result.stderr)
            self.assertIn("/hpc_alloc is missing", result.stderr)
            self.assert_no_delivery(home)

    def test_caller_package_cannot_shadow_an_incomplete_adjacent_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incomplete-source"
            caller = root / "caller-with-valid-package"
            home = root / "home"
            source.joinpath("hpc_alloc").mkdir(parents=True)
            source.joinpath("skill").mkdir()
            caller.mkdir()
            home.mkdir()
            installer = shutil.copy2(REPO / "install.sh", source / "install.sh")
            shutil.copy2(REPO / "hpc-alloc", source / "hpc-alloc")
            shutil.copy2(
                REPO / "hpc_alloc" / "__init__.py",
                source / "hpc_alloc" / "__init__.py",
            )
            shutil.copy2(REPO / "skill" / "SKILL.md", source / "skill" / "SKILL.md")
            shutil.copytree(REPO / "hpc_alloc", caller / "hpc_alloc")
            self.assertFalse(source.joinpath("hpc_alloc", "cli.py").exists())

            result = subprocess.run(
                ["bash", str(installer)],
                cwd=caller,
                env={**os.environ, "HOME": str(home)},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn(
                "installation is incomplete: the adjacent Python package cannot be imported",
                result.stderr,
            )
            self.assert_no_delivery(home)

    def test_missing_skill_is_rejected_before_binary_link_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "incomplete-source"
            home = root / "home"
            source.mkdir()
            home.mkdir()
            installer = shutil.copy2(REPO / "install.sh", source / "install.sh")
            shutil.copy2(REPO / "hpc-alloc", source / "hpc-alloc")
            shutil.copytree(REPO / "hpc_alloc", source / "hpc_alloc")

            result = subprocess.run(
                ["bash", str(installer)],
                cwd=source,
                env={**os.environ, "HOME": str(home)},
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("/skill/SKILL.md is missing", result.stderr)
            self.assert_no_delivery(home)


if __name__ == "__main__":
    unittest.main()
