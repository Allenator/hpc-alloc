"""Validated runtime context assembled once after CLI argument parsing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .config import ClusterConfig, Config
from .errors import ConfigInvalid
from .paths import AppPaths
from .state import StateRepository


RECOVERY_COMMANDS = frozenset({"help", "config", "setup"})


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    paths: AppPaths
    config: Config | None
    state: StateRepository | None
    cluster_name: str | None = None
    cluster: ClusterConfig | None = None
    config_error: ConfigInvalid | None = None

    @property
    def primary_cluster(self) -> str | None:
        """Resolved default cluster name used by command orchestration."""

        return self.cluster_name

    @classmethod
    def load(
        cls,
        *,
        command: str,
        explicit_cluster: str | None = None,
        paths: AppPaths | None = None,
        config_path: str | Path | None = None,
        state_path: str | Path | None = None,
    ) -> "RuntimeContext":
        """Build one authoritative context for an invocation.

        Operational commands fail closed.  Recovery commands receive the
        typed configuration error so they can display or replace bad input.
        ``setup`` deliberately does not initialize state before its filesystem
        preflight and configuration commit have succeeded.
        """

        app_paths = paths or AppPaths.for_home()
        if config_path is not None or state_path is not None:
            selected_config = Path(config_path) if config_path is not None else app_paths.config_file
            app_paths = replace(
                app_paths,
                config_dir=selected_config.parent,
                config_file=selected_config,
                state_db=Path(state_path) if state_path is not None else app_paths.state_db,
            )
        try:
            config = Config.load(app_paths.config_file)
        except ConfigInvalid as exc:
            if command not in RECOVERY_COMMANDS:
                raise
            return cls(paths=app_paths, config=None, state=None, config_error=exc)

        if command in {"help", "config", "setup"}:
            return cls(paths=app_paths, config=config, state=None)

        if command == "dry-run":
            cluster_name = config.resolve_cluster(explicit_cluster)
            return cls(
                paths=app_paths,
                config=config,
                state=None,
                cluster_name=cluster_name,
                cluster=config.clusters[cluster_name],
            )

        state = StateRepository(app_paths.state_db)
        state.initialize()
        cluster_name: str | None = None
        cluster: ClusterConfig | None = None
        if command not in {"help", "config", "status", "recover"} or explicit_cluster is not None:
            cluster_name = config.resolve_cluster(explicit_cluster)
            cluster = config.clusters[cluster_name]
        return cls(
            paths=app_paths,
            config=config,
            state=state,
            cluster_name=cluster_name,
            cluster=cluster,
        )

    def resolve_cluster(self, explicit: str | None = None) -> ClusterConfig:
        if self.config is None:
            if self.config_error is not None:
                raise self.config_error
            raise ConfigInvalid("configuration is unavailable", path=self.paths.config_file)
        return self.config.cluster(explicit)
