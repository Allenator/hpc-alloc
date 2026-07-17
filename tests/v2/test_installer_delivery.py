from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
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
    def isolated_path(root: Path) -> str:
        """Build a PATH carrying a supported python3 but no harness launcher.

        Skill targets are detected with `command -v claude|codex`, so a
        developer machine with either installed would otherwise mask every
        undetected-harness path under test.
        """

        bin_dir = root / "isolated-bin"
        bin_dir.mkdir(exist_ok=True)
        python = bin_dir / "python3"
        if not python.exists():
            python.symlink_to(sys.executable)
        return os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"])

    @staticmethod
    def run_installer(
        installer: Path,
        *,
        cwd: Path,
        home: Path,
        args: tuple[str, ...] = (),
        path: str | None = None,
        codex_home: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "HOME": str(home)}
        # An inherited CODEX_HOME would silently relocate the Codex skill away
        # from the temporary home every assertion is written against.
        environment.pop("CODEX_HOME", None)
        if path is not None:
            environment["PATH"] = path
        if codex_home is not None:
            environment["CODEX_HOME"] = str(codex_home)
        return subprocess.run(
            ["bash", str(installer), *args],
            cwd=cwd,
            env=environment,
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
        if home.joinpath(".codex").exists():
            raise AssertionError("installer created ~/.codex before validation completed")

    def test_existing_real_skill_directory_is_replaced_not_nested(self) -> None:
        """`ln -sfn SRC DIR` descends into an existing real directory.

        It creates DIR/skill *inside* it rather than replacing it, so SKILL.md
        lands at hpc-alloc/skill/SKILL.md, the harness never loads the skill --
        and the installer still prints "linked" and exits 0.  Every harness is
        linked through one helper, so each one is held to the guard.
        """

        for harness_dir, flag in ((".claude", "--claude"), (".codex", "--codex")):
            with self.subTest(harness=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                home = root / "home"
                source = root / "source"
                installer = self.copy_complete_source(source)

                stale = home / harness_dir / "skills" / "hpc-alloc"
                stale.mkdir(parents=True)
                stale.joinpath("SKILL.md").write_text("a stale copied skill\n")

                result = self.run_installer(
                    installer,
                    cwd=source,
                    home=home,
                    args=(flag,),
                    path=self.isolated_path(root),
                )
                self.assertEqual(result.returncode, 0, result.stderr)

                link = home / harness_dir / "skills" / "hpc-alloc"
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

    def assert_skill_package_delivered(self, installed_skill: Path, source: Path) -> None:
        self.assertEqual(installed_skill.resolve(), source.joinpath("skill").resolve())
        for relative_path in REQUIRED_SKILL_FILES:
            skill_relative_path = Path(relative_path).relative_to("skill")
            with self.subTest(relative_path=relative_path):
                self.assertEqual(
                    installed_skill.joinpath(skill_relative_path).read_bytes(),
                    source.joinpath(relative_path).read_bytes(),
                )

    def test_complete_source_is_linked_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.joinpath(".claude").mkdir(parents=True)
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer, cwd=root, home=home, path=self.isolated_path(root)
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                home.joinpath(".local", "bin", "hpc-alloc").resolve(),
                source.joinpath("hpc-alloc").resolve(),
            )
            self.assert_skill_package_delivered(
                home.joinpath(".claude", "skills", "hpc-alloc"), source
            )
            self.assertFalse(
                home.joinpath(".codex").exists(),
                "installer delivered a skill to an undetected harness",
            )

    def test_detected_codex_home_receives_the_same_skill_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.joinpath(".codex").mkdir(parents=True)
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer, cwd=root, home=home, path=self.isolated_path(root)
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_skill_package_delivered(
                home.joinpath(".codex", "skills", "hpc-alloc"), source
            )
            self.assertFalse(
                home.joinpath(".claude").exists(),
                "installer delivered a skill to an undetected harness",
            )

    def test_every_detected_harness_shares_one_skill_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.joinpath(".claude").mkdir(parents=True)
            home.joinpath(".codex").mkdir(parents=True)
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer, cwd=root, home=home, path=self.isolated_path(root)
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for harness_dir in (".claude", ".codex"):
                with self.subTest(harness=harness_dir):
                    self.assert_skill_package_delivered(
                        home.joinpath(harness_dir, "skills", "hpc-alloc"), source
                    )

    def test_explicit_target_installs_for_an_undetected_harness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.mkdir()
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer,
                cwd=root,
                home=home,
                args=("--codex",),
                path=self.isolated_path(root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_skill_package_delivered(
                home.joinpath(".codex", "skills", "hpc-alloc"), source
            )
            self.assertFalse(
                home.joinpath(".claude").exists(),
                "an explicit target installed for a harness it did not name",
            )

    def test_codex_home_relocates_the_codex_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.mkdir()
            relocated = root / "codex-home"
            relocated.mkdir()
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer,
                cwd=root,
                home=home,
                path=self.isolated_path(root),
                codex_home=relocated,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_skill_package_delivered(
                relocated.joinpath("skills", "hpc-alloc"), source
            )
            self.assertFalse(
                home.joinpath(".codex").exists(),
                "installer used the default Codex home despite CODEX_HOME",
            )

    def test_no_detected_harness_fails_before_creating_delivery_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.mkdir()
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer, cwd=root, home=home, path=self.isolated_path(root)
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertIn("found no agent harness", result.stderr)
            self.assert_no_delivery(home)

    def test_unknown_option_is_rejected_before_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "complete-source"
            home = root / "home"
            home.mkdir()
            installer = self.copy_complete_source(source)

            result = self.run_installer(
                installer,
                cwd=root,
                home=home,
                args=("--codexx",),
                path=self.isolated_path(root),
            )

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("unknown option: --codexx", result.stderr)
            self.assert_no_delivery(home)


if __name__ == "__main__":
    unittest.main()
