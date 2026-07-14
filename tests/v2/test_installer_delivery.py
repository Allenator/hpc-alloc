from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REQUIRED_SKILL_FILES = (
    "skill/SKILL.md",
    "skill/references/command-contracts.md",
    "skill/references/recovery-and-lifecycle.md",
)


class InstallerDeliveryTests(unittest.TestCase):
    @staticmethod
    def runtime_manifest() -> tuple[str, ...]:
        installer = (REPO / "install.sh").read_text()
        match = re.search(
            r"^runtime_modules=\(\n(?P<body>.*?)^\)$",
            installer,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("installer runtime-module manifest was not found")
        return tuple(line.strip() for line in match.group("body").splitlines())

    @staticmethod
    def skill_manifest() -> tuple[str, ...]:
        installer = (REPO / "install.sh").read_text()
        match = re.search(
            r"^skill_files=\(\n(?P<body>.*?)^\)$",
            installer,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError("installer skill-file manifest was not found")
        return tuple(shlex.split(match.group("body"), comments=True))

    @staticmethod
    def copy_complete_source(destination: Path) -> Path:
        destination.mkdir()
        installer = shutil.copy2(REPO / "install.sh", destination / "install.sh")
        shutil.copy2(REPO / "hpc-alloc", destination / "hpc-alloc")
        shutil.copytree(REPO / "hpc_alloc", destination / "hpc_alloc")
        shutil.copytree(REPO / "skill", destination / "skill")
        return installer

    @staticmethod
    def run_installer(
        installer: Path, *, cwd: Path, home: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(installer)],
            cwd=cwd,
            env={**os.environ, "HOME": str(home)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def assert_no_delivery(home: Path) -> None:
        if home.joinpath(".local").exists():
            raise AssertionError("installer created ~/.local before validation completed")
        if home.joinpath(".claude").exists():
            raise AssertionError("installer created ~/.claude before validation completed")

    def test_existing_real_skill_directory_is_replaced_not_nested(self) -> None:
        """`ln -sfn SRC DIR` descends into an existing real directory.

        It creates DIR/skill *inside* it rather than replacing it, so SKILL.md
        lands at hpc-alloc/skill/SKILL.md, Claude Code never loads the skill --
        and the installer still prints "linked" and exits 0.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home = root / "home"
            source = root / "source"
            installer = self.copy_complete_source(source)

            stale = home / ".claude" / "skills" / "hpc-alloc"
            stale.mkdir(parents=True)
            stale.joinpath("SKILL.md").write_text("a stale copied skill\n")

            result = self.run_installer(installer, cwd=source, home=home)
            self.assertEqual(result.returncode, 0, result.stderr)

            link = home / ".claude" / "skills" / "hpc-alloc"
            self.assertTrue(link.is_symlink(), "a real directory survived the install")
            self.assertEqual(link.resolve(), (source / "skill").resolve())
            self.assertTrue(link.joinpath("SKILL.md").exists())
            self.assertFalse(
                link.joinpath("skill").exists(),
                "installer nested the link inside the existing directory",
            )

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
            caller.mkdir()
            home.mkdir()
            installer = shutil.copy2(REPO / "install.sh", source / "install.sh")
            shutil.copy2(REPO / "hpc-alloc", source / "hpc-alloc")
            shutil.copy2(
                REPO / "hpc_alloc" / "__init__.py",
                source / "hpc_alloc" / "__init__.py",
            )
            shutil.copytree(REPO / "skill", source / "skill")
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
                "installation is incomplete: the adjacent Python package cannot be "
                "imported or does not match the runtime-module manifest",
                result.stderr,
            )
            self.assert_no_delivery(home)

    def test_each_missing_skill_file_is_rejected_before_delivery(self) -> None:
        for relative_path in REQUIRED_SKILL_FILES:
            with (
                self.subTest(relative_path=relative_path),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source = root / "incomplete-source"
                home = root / "home"
                home.mkdir()
                installer = self.copy_complete_source(source)
                source.joinpath(relative_path).unlink()

                result = self.run_installer(installer, cwd=source, home=home)

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn(f"/{relative_path} is missing", result.stderr)
                self.assert_no_delivery(home)

    def test_runtime_manifest_matches_every_adjacent_python_module(self) -> None:
        manifested_sources = {
            (
                "__init__.py"
                if name == "hpc_alloc"
                else f"{name.removeprefix('hpc_alloc.')}.py"
            )
            for name in self.runtime_manifest()
        }
        package_sources = {path.name for path in (REPO / "hpc_alloc").glob("*.py")}

        self.assertEqual(manifested_sources, package_sources)

    def test_skill_manifest_matches_required_bundled_documents(self) -> None:
        self.assertEqual(self.skill_manifest(), REQUIRED_SKILL_FILES)

    def test_each_missing_runtime_module_fails_before_creating_links(self) -> None:
        for module in self.runtime_manifest():
            source_name = (
                "__init__.py"
                if module == "hpc_alloc"
                else f"{module.removeprefix('hpc_alloc.')}.py"
            )
            with self.subTest(module=module), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "incomplete-source"
                home = root / "home"
                home.mkdir()
                installer = self.copy_complete_source(source)
                source.joinpath("hpc_alloc", source_name).unlink()

                result = self.run_installer(installer, cwd=source, home=home)

                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("hpc-alloc installation is incomplete:", result.stderr)
                self.assert_no_delivery(home)

    def test_complete_source_is_linked_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.mkdir()
            installer = self.copy_complete_source(source)

            result = self.run_installer(installer, cwd=root, home=home)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                home.joinpath(".local", "bin", "hpc-alloc").resolve(),
                source.joinpath("hpc-alloc").resolve(),
            )
            self.assertEqual(
                home.joinpath(".claude", "skills", "hpc-alloc").resolve(),
                source.joinpath("skill").resolve(),
            )
            installed_skill = home.joinpath(".claude", "skills", "hpc-alloc")
            for relative_path in REQUIRED_SKILL_FILES:
                skill_relative_path = Path(relative_path).relative_to("skill")
                with self.subTest(relative_path=relative_path):
                    self.assertEqual(
                        installed_skill.joinpath(skill_relative_path).read_bytes(),
                        source.joinpath(relative_path).read_bytes(),
                    )


if __name__ == "__main__":
    unittest.main()
