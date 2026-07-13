from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from hpc_alloc.config import Config
from hpc_alloc.state import SCHEMA_VERSION


REPO = Path(__file__).resolve().parents[2]
CLI = REPO / "hpc-alloc"
CONFIG = """\
[identity]
netid = "ab1234"

[ssh]
identity_file = "~/.ssh/id_ed25519"

[defaults]
cluster = "grace"

[cluster.grace]
host = "grace.example.edu"
"""


class CliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.home = Path(self.directory.name)
        self.environment = {
            **os.environ,
            "HOME": str(self.home),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONWARNINGS": "error",
        }

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(CLI), *arguments],
            cwd=REPO,
            env=self.environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )

    def write_config(self, text: str = CONFIG) -> Path:
        path = self.home / ".config" / "hpc-alloc" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_config_and_dry_runs_use_real_launcher_without_creating_state(self) -> None:
        self.write_config()
        configured = self.run_cli("config", "--json")
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertEqual(json.loads(configured.stdout)["primary_cluster"], "grace")

        allocation = self.run_cli("up", "--dry-run", "--name", "smoke")
        command = self.run_cli(
            "run", "--dry-run", "--", "python3", "-c", 'print("ok")'
        )
        self.assertEqual(allocation.returncode, 0, allocation.stderr)
        self.assertEqual(command.returncode, 0, command.stderr)
        self.assertIn("sbatch --parsable", allocation.stdout)
        self.assertIn("hpc-alloc:v2:dryrun-", command.stdout)
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_run_dry_run_is_paste_ready_for_remote_home_and_chdir_paths(self) -> None:
        self.write_config()
        fake_bin = self.home / "fake-bin"
        fake_bin.mkdir()
        fake_sbatch = fake_bin / "sbatch"
        fake_sbatch.write_text(
            "#!/bin/sh\n"
            ': "${SBATCH_ARGS_FILE:?}"\n'
            'printf \'%s\\n\' "$@" > "$SBATCH_ARGS_FILE"\n',
            encoding="utf-8",
        )
        fake_sbatch.chmod(0o755)

        remote_home = self.home / "remote home's space"
        remote_home.mkdir()
        execution_cwd = self.home / "not-the-home-directory"
        execution_cwd.mkdir()
        absolute_chdir = self.home / "absolute project's directory"
        cases = (
            ("relative", "project dir/it's;$(touch INJECTED)"),
            ("tilde", "~/tilde dir/it's;$(touch INJECTED)"),
            ("home", "~"),
            ("absolute", str(absolute_chdir)),
        )

        for label, chdir in cases:
            with self.subTest(label=label):
                rendered = self.run_cli(
                    "run", "--dry-run", "--chdir", chdir, "--", "true"
                )
                self.assertEqual(rendered.returncode, 0, rendered.stderr)
                self.assertIn('"${HOME:?}"/.hpc-alloc', rendered.stdout)
                self.assertFalse(
                    (self.home / ".config" / "hpc-alloc" / "state.db").exists()
                )

                arguments_file = self.home / f"sbatch-{label}.args"
                execution = subprocess.run(
                    rendered.stdout,
                    shell=True,
                    executable="/bin/sh",
                    cwd=execution_cwd,
                    env={
                        **self.environment,
                        "HOME": str(remote_home),
                        "PATH": f"{fake_bin}:{self.environment['PATH']}",
                        "SBATCH_ARGS_FILE": str(arguments_file),
                    },
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(execution.returncode, 0, execution.stderr)
                self.assertEqual(execution.stderr, "")
                arguments = arguments_file.read_text(encoding="utf-8").splitlines()
                output = arguments[arguments.index("--output") + 1]
                self.assertEqual(Path(output).parent, remote_home / ".hpc-alloc")
                self.assertTrue(Path(output).name.startswith("run-"), output)
                self.assertTrue(Path(output).name.endswith(".log"), output)

                actual_chdir = next(
                    argument.removeprefix("--chdir=")
                    for argument in arguments
                    if argument.startswith("--chdir=")
                )
                if chdir == "~":
                    expected_chdir = str(remote_home)
                elif chdir.startswith("~/"):
                    expected_chdir = f"{remote_home}/{chdir[2:]}"
                elif Path(chdir).is_absolute():
                    expected_chdir = chdir
                else:
                    expected_chdir = f"{remote_home}/{chdir}"
                self.assertEqual(actual_chdir, expected_chdir)
                self.assertFalse((execution_cwd / "INJECTED").exists())

        self.assertTrue((remote_home / ".hpc-alloc").is_dir())
        self.assertFalse((execution_cwd / ".hpc-alloc").exists())
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_up_dry_run_uses_execution_time_home(self) -> None:
        self.write_config()
        fake_bin = self.home / "fake-bin"
        fake_bin.mkdir()
        fake_sbatch = fake_bin / "sbatch"
        fake_sbatch.write_text(
            "#!/bin/sh\n"
            'printf \'%s\\n\' "$@" > "$SBATCH_ARGS_FILE"\n',
            encoding="utf-8",
        )
        fake_sbatch.chmod(0o755)
        remote_home = self.home / "different remote home"
        remote_home.mkdir()
        arguments_file = self.home / "allocation.args"

        rendered = self.run_cli("up", "--dry-run", "--name", "pasteable")
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())
        execution = subprocess.run(
            rendered.stdout,
            shell=True,
            executable="/bin/sh",
            cwd=REPO,
            env={
                **self.environment,
                "HOME": str(remote_home),
                "PATH": f"{fake_bin}:{self.environment['PATH']}",
                "SBATCH_ARGS_FILE": str(arguments_file),
            },
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(execution.returncode, 0, execution.stderr)
        arguments = arguments_file.read_text(encoding="utf-8").splitlines()
        output = arguments[arguments.index("--output") + 1]
        self.assertEqual(Path(output).parent, remote_home / ".hpc-alloc")
        self.assertTrue(Path(output).name.startswith("alloc-"), output)
        self.assertTrue((remote_home / ".hpc-alloc").is_dir())
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_invalid_cli_resources_fail_before_state_or_remote_work(self) -> None:
        self.write_config()
        for arguments in (
            ("up", "--dry-run", "--time", "1:99"),
            ("up", "--dry-run", "--time", "90 "),
            ("up", "--dry-run", "--time", "90;--constraint=evil"),
            ("run", "--dry-run", "--gpus", "h200:0", "--", "true"),
            (
                "run",
                "--dry-run",
                "--chdir",
                "~someone/project",
                "--",
                "true",
            ),
        ):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_all_documented_duration_forms_are_accepted_as_cli_overrides(self) -> None:
        self.write_config()
        for duration in (
            "90",
            "90:30",
            "4:30:00",
            "2-00",
            "2-04:30",
            "2-04:30:00",
        ):
            with self.subTest(duration=duration):
                result = self.run_cli(
                    "up", "--dry-run", "--name", "duration", "--time", duration
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"--time={duration}", result.stdout)
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_all_zero_duration_forms_are_rejected_as_cli_overrides(self) -> None:
        self.write_config()
        for duration in (
            "0",
            "0:00",
            "0:00:00",
            "0-0",
            "0-0:00",
            "0-0:00:00",
        ):
            with self.subTest(duration=duration):
                result = self.run_cli(
                    "up", "--dry-run", "--name", "duration", "--time", duration
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Slurm duration", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_setup_with_existing_key_commits_v2_config_database_and_include(self) -> None:
        ssh = self.home / ".ssh"
        ssh.mkdir(mode=0o700)
        (ssh / "id_ed25519").write_text("fake private key for setup preflight\n")
        (ssh / "id_ed25519.pub").write_text("ssh-ed25519 AAAATEST user@test\n")

        result = self.run_cli(
            "setup",
            "--netid",
            "ab1234",
            "--cluster",
            "grace",
            "--host",
            "grace.example.edu",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ssh-ed25519 AAAATEST", result.stdout)
        config = self.home / ".config" / "hpc-alloc" / "config.toml"
        database = self.home / ".config" / "hpc-alloc" / "state.db"
        self.assertIn("[identity]", config.read_text())
        self.assertIn("[cluster.grace]", config.read_text())
        self.assertIn("Include ~/.config/hpc-alloc/ssh_config", (ssh / "config").read_text())
        with closing(sqlite3.connect(database)) as connection:
            self.assertEqual(
                connection.execute("SELECT schema_version FROM metadata").fetchone()[0],
                SCHEMA_VERSION,
            )
            self.assertIsNotNone(
                connection.execute("SELECT machine_id FROM machine").fetchone()[0]
            )

    def test_concurrent_setups_serialize_and_waiter_rechecks_existing_config(self) -> None:
        ssh = self.home / ".ssh"
        ssh.mkdir(mode=0o700)
        (ssh / "id_ed25519").write_text("fake private key\n")
        (ssh / "id_ed25519.pub").write_text(
            "ssh-ed25519 AAAATEST user@test\n"
        )
        command = [
            str(CLI),
            "setup",
            "--netid",
            "ab1234",
            "--cluster",
            "grace",
            "--host",
            "grace.example.edu",
        ]
        processes = [
            subprocess.Popen(
                command,
                cwd=REPO,
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        results = [process.communicate(timeout=10) for process in processes]

        self.assertEqual(sorted(process.returncode for process in processes), [0, 1])
        failed_stderr = next(
            stderr
            for process, (_stdout, stderr) in zip(processes, results, strict=True)
            if process.returncode == 1
        )
        self.assertIn("configuration already exists", failed_stderr)
        configured = Config.load(
            self.home / ".config" / "hpc-alloc" / "config.toml"
        )
        self.assertEqual(configured.identity.netid, "ab1234")

    def test_invalid_setup_preflight_creates_no_key_or_state(self) -> None:
        result = self.run_cli(
            "setup",
            "--netid",
            "!",
            "--cluster",
            "grace",
            "--host",
            "grace.example.edu",
        )
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.home / ".ssh" / "id_ed25519").exists())
        self.assertFalse((self.home / ".config" / "hpc-alloc" / "state.db").exists())

    def test_dangling_user_ssh_config_is_typed_and_precedes_key_generation(self) -> None:
        ssh = self.home / ".ssh"
        ssh.mkdir(mode=0o700)
        (ssh / "config").symlink_to(self.home / "missing" / "ssh-config")
        result = self.run_cli(
            "setup",
            "--netid",
            "ab1234",
            "--cluster",
            "grace",
            "--host",
            "grace.example.edu",
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("dangling or looping symlink", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((ssh / "id_ed25519").exists())

    def test_askpass_hook_is_independent_of_configuration(self) -> None:
        environment = {
            **self.environment,
            "HPC_ALLOC_ASKPASS": "1",
            "SSH_ASKPASS": str(CLI),
        }
        result = subprocess.run(
            [str(CLI)],
            cwd=REPO,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "1\n")

    def test_installer_style_symlink_imports_package_outside_repository(self) -> None:
        binary = self.home / "bin" / "hpc-alloc"
        binary.parent.mkdir()
        binary.symlink_to(CLI)
        result = subprocess.run(
            [str(binary), "--help"],
            cwd=self.home,
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: hpc-alloc", result.stdout)

    def test_target_command_help_documents_durable_operation_selectors(self) -> None:
        for command in ("why", "logs", "cancel", "down", "ssh", "sync"):
            with self.subTest(command=command):
                result = self.run_cli(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("@operation", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
