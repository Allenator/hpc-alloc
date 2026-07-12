from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "hpc_alloc"
SCHEDULER_COMMAND = re.compile(r"\b(?:sbatch|scancel|squeue|sacct|sinfo|scontrol|sprio)\b")


class ArchitectureGateTests(unittest.TestCase):
    def test_executable_is_only_a_versioned_launcher(self) -> None:
        launcher = (REPO / "hpc-alloc").read_text(encoding="utf-8")
        self.assertLessEqual(len(launcher.splitlines()), 35)
        self.assertIn("from hpc_alloc.cli import main", launcher)
        self.assertNotIn("class ", launcher)

    def test_scheduler_commands_are_owned_only_by_slurm_adapter(self) -> None:
        offenders: list[str] = []
        for path in PACKAGE.glob("*.py"):
            if path.name == "slurm.py":
                continue
            if SCHEDULER_COMMAND.search(path.read_text(encoding="utf-8")):
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_foundation_layers_do_not_depend_on_transport_or_commands(self) -> None:
        forbidden = {"commands", "ssh", "slurm", "streaming", "monitor"}
        for filename in ("config.py", "errors.py", "models.py", "ownership.py", "state.py"):
            tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
            dependencies: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    dependencies.add((node.module or "").split(".", 1)[0])
            with self.subTest(filename=filename):
                self.assertFalse(dependencies & forbidden)

    def test_only_cli_boundary_maps_application_errors_to_process_status(self) -> None:
        for path in PACKAGE.glob("*.py"):
            if path.name == "cli.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            forbidden_calls = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in {"exit", "quit"}:
                    forbidden_calls.append(node.lineno)
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys"
                    and node.func.attr == "exit"
                ):
                    forbidden_calls.append(node.lineno)
            with self.subTest(filename=path.name):
                self.assertEqual(forbidden_calls, [])

    def test_managed_ssh_projection_has_one_serialized_writer(self) -> None:
        commands = (PACKAGE / "commands.py").read_text(encoding="utf-8")
        self.assertNotIn("render_ssh_config", commands)
        self.assertNotRegex(
            commands,
            r"atomic_write_600\s*\(\s*paths\.managed_ssh_config",
        )
        self.assertIn("sync_managed_config", commands)


if __name__ == "__main__":
    unittest.main()
