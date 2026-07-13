"""Filesystem locations managed by hpc-alloc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    home: Path
    config_dir: Path
    config_file: Path
    state_db: Path
    config_scope_lock: Path
    operation_locks_dir: Path
    managed_ssh_config: Path
    ssh_config_lock: Path
    known_hosts: Path
    ssh_dir: Path
    user_ssh_config: Path

    @classmethod
    def for_home(cls, home: Path | None = None) -> "AppPaths":
        root = (home or Path.home()).expanduser()
        config_dir = root / ".config" / "hpc-alloc"
        ssh_dir = root / ".ssh"
        return cls(
            home=root,
            config_dir=config_dir,
            config_file=config_dir / "config.toml",
            state_db=config_dir / "state.db",
            config_scope_lock=config_dir / ".config_scope.lock",
            operation_locks_dir=config_dir / "operation-locks",
            managed_ssh_config=config_dir / "ssh_config",
            ssh_config_lock=config_dir / ".ssh_config.lock",
            known_hosts=config_dir / "known_hosts",
            ssh_dir=ssh_dir,
            user_ssh_config=ssh_dir / "config",
        )
