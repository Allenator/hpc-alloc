"""Validated runtime context assembled once after CLI argument parsing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import ClusterConfig, Config
from .errors import ConfigInvalid
from .paths import AppPaths
from .state import StateRepository


RECOVERY_COMMANDS = frozenset({"help", "config", "setup"})
IMPLICIT_CLUSTER_COMMANDS = frozenset(
    {"connect", "up", "run", "avail", "partitions"}
)


class SessionState:
    """Mutable scratch shared by the helpers within one CLI invocation.

    RuntimeContext is deliberately frozen, but a few things are worth doing
    exactly once per process rather than once per helper call.  They live here.
    """

    __slots__ = ("ssh_projection_repaired",)

    def __init__(self) -> None:
        # The managed SSH projection is derived from config and durable state.
        # Repairing it is idempotent recovery, not a post-mutation write -- the
        # commands that change state re-sync it explicitly -- so once per
        # invocation is enough.  It used to run on every _services() call: once
        # per cluster inside `status`, twice inside `up` and `run`, each time
        # re-parsing the TOML, re-reading the jobs table on a fresh SQLite
        # connection, and re-rendering the managed file under an exclusive lock.
        self.ssh_projection_repaired = False


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    paths: AppPaths
    config: Config | None
    state: StateRepository | None
    cluster_name: str | None = None
    cluster: ClusterConfig | None = None
    config_error: ConfigInvalid | None = None
    session: SessionState = field(default_factory=SessionState)

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
            # A dry run must not initialize or mutate the journal, but it may READ
            # the disposable per-cluster caches (GPU topology, access rules) to
            # resolve a typed GPU partition offline.  Open an already-existing
            # database WITHOUT initializing it -- never creating one -- so the read
            # stays journal-free; a missing database simply leaves the caches cold.
            dry_state = (
                StateRepository(app_paths.state_db)
                if Path(app_paths.state_db).exists()
                else None
            )
            return cls(
                paths=app_paths,
                config=config,
                state=dry_state,
                cluster_name=cluster_name,
                cluster=config.clusters[cluster_name],
            )

        state = StateRepository(app_paths.state_db)
        state.initialize()
        cluster_name: str | None = None
        cluster: ClusterConfig | None = None
        if explicit_cluster is not None or command in IMPLICIT_CLUSTER_COMMANDS:
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
