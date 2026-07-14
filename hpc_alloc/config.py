"""Strict Python 3.11+ TOML configuration for hpc-alloc v2."""

from __future__ import annotations

import ipaddress
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping

from .errors import ConfigInvalid
from .ownership import IDENTIFIER_RE


_NETID = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{1,63}$")
_PARTITION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SLURM_TIME = re.compile(
    r"^(?:"
    r"[0-9]+"
    r"|[0-9]+:[0-5][0-9](?::[0-5][0-9])?"
    r"|[0-9]+-[0-9]+(?::[0-5][0-9](?::[0-5][0-9])?)?"
    r")$"
)
# ASCII digits only.  `\d` also matches full-width and Arabic-Indic digits, which
# the scheduler rejects -- and because submission never retries, a value that
# slips past pre-flight validation here fails remotely as an ambiguous
# submission ("may have committed") instead of a clean ConfigInvalid.
_MEMORY = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:[KMGTPE](?:i?B)?)?$", re.IGNORECASE)
_IDENTITY_PATH = re.compile(r"^(?:~/|/)[A-Za-z0-9_./@%+=:,~-]+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _valid_slurm_time(value: str) -> bool:
    """Accept one finite numeric Slurm duration in a documented form."""

    return _SLURM_TIME.fullmatch(value) is not None and any(
        character in "123456789" for character in value
    )


def _table(value: Any, name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigInvalid(f"[{name}] must be a TOML table", path=path)
    return value


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], name: str, path: Path) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigInvalid(f"[{name}] contains unknown key {unknown[0]!r}", path=path)


def _string(
    table: Mapping[str, Any],
    key: str,
    section: str,
    path: Path,
    *,
    required: bool = False,
) -> str | None:
    value = table.get(key)
    if value is None:
        if required:
            raise ConfigInvalid(f"[{section}].{key} is required", path=path)
        return None
    if not isinstance(value, str) or not value or _CONTROL.search(value):
        raise ConfigInvalid(f"[{section}].{key} must be a non-empty string without control characters", path=path)
    return value


def _positive_int(
    table: Mapping[str, Any], key: str, section: str, path: Path, *, allow_zero: bool = False
) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ConfigInvalid(f"[{section}].{key} must be a {comparison} integer", path=path)
    return value


def _normalize_host(host: str) -> str | None:
    bracketed = host.startswith("[") or host.endswith("]")
    if bracketed:
        if not (host.startswith("[") and host.endswith("]")):
            return None
        candidate = host[1:-1]
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            return None
        return candidate

    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass

    candidate = host[:-1] if host.endswith(".") else host
    if len(candidate) > 253:
        return None
    labels = candidate.split(".")
    valid = bool(labels) and all(
        0 < len(label) <= 63
        and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
        for label in labels
    )
    return host if valid else None


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    netid: str


@dataclass(frozen=True, slots=True)
class SshConfig:
    identity_file: str | None = None


@dataclass(frozen=True, slots=True)
class DefaultsConfig:
    cluster: str | None = None
    partition: str | None = None
    gpu_partition: str | None = None
    time: str | None = None
    cpus: int | None = None
    mem: str | None = None
    idle_timeout: int | None = None


@dataclass(frozen=True, slots=True)
class ClusterConfig:
    name: str
    host: str
    partition: str | None = None
    gpu_partition: str | None = None
    time: str | None = None
    cpus: int | None = None
    mem: str | None = None
    idle_timeout: int | None = None


@dataclass(frozen=True, slots=True)
class Config:
    identity: IdentityConfig
    ssh: SshConfig
    defaults: DefaultsConfig
    clusters: dict[str, ClusterConfig]
    path: Path

    TOP_LEVEL: ClassVar[set[str]] = {"identity", "ssh", "defaults", "cluster"}
    RESOURCE_KEYS: ClassVar[set[str]] = {
        "partition",
        "gpu_partition",
        "time",
        "cpus",
        "mem",
        "idle_timeout",
    }

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        try:
            raw_bytes = config_path.read_bytes()
        except FileNotFoundError:
            raise ConfigInvalid("configuration does not exist", path=config_path) from None
        except OSError as exc:
            raise ConfigInvalid(f"cannot read configuration: {exc}", path=config_path) from exc
        try:
            raw = tomllib.loads(raw_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ConfigInvalid("configuration is not valid UTF-8", path=config_path) from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigInvalid(f"invalid TOML: {exc}", path=config_path) from exc

        _reject_unknown(raw, cls.TOP_LEVEL, "root", config_path)
        identity = _table(raw.get("identity"), "identity", config_path)
        _reject_unknown(identity, {"netid"}, "identity", config_path)
        netid = _string(identity, "netid", "identity", config_path, required=True)
        assert netid is not None
        if not _NETID.fullmatch(netid):
            raise ConfigInvalid("[identity].netid has invalid characters", path=config_path)

        ssh = _table(raw.get("ssh", {}), "ssh", config_path)
        _reject_unknown(ssh, {"identity_file"}, "ssh", config_path)
        identity_file = _string(ssh, "identity_file", "ssh", config_path)
        if identity_file is not None and not _IDENTITY_PATH.fullmatch(identity_file):
            raise ConfigInvalid(
                "[ssh].identity_file must be an absolute or ~/ path without whitespace",
                path=config_path,
            )

        defaults_raw = _table(raw.get("defaults", {}), "defaults", config_path)
        _reject_unknown(defaults_raw, cls.RESOURCE_KEYS | {"cluster"}, "defaults", config_path)
        default_cluster = _string(defaults_raw, "cluster", "defaults", config_path)
        if default_cluster is not None and not IDENTIFIER_RE.fullmatch(default_cluster):
            raise ConfigInvalid("[defaults].cluster is not a valid cluster name", path=config_path)
        defaults_values = cls._resource_values(defaults_raw, "defaults", config_path)
        defaults = DefaultsConfig(cluster=default_cluster, **defaults_values)

        cluster_tables = _table(raw.get("cluster"), "cluster", config_path)
        if not cluster_tables:
            raise ConfigInvalid("at least one [cluster.NAME] table is required", path=config_path)
        clusters: dict[str, ClusterConfig] = {}
        for name, value in cluster_tables.items():
            if not isinstance(name, str) or not IDENTIFIER_RE.fullmatch(name):
                raise ConfigInvalid(f"cluster name {name!r} is invalid", path=config_path)
            section = f"cluster.{name}"
            table = _table(value, section, config_path)
            _reject_unknown(table, cls.RESOURCE_KEYS | {"host"}, section, config_path)
            host = _string(table, "host", section, config_path, required=True)
            assert host is not None
            normalized_host = _normalize_host(host)
            if normalized_host is None:
                raise ConfigInvalid(f"[{section}].host is not a valid hostname or IP address", path=config_path)
            clusters[name] = ClusterConfig(
                name=name,
                host=normalized_host,
                **cls._resource_values(table, section, config_path),
            )

        if default_cluster is not None and default_cluster not in clusters:
            raise ConfigInvalid(
                f"[defaults].cluster names unconfigured cluster {default_cluster!r}", path=config_path
            )
        return cls(
            identity=IdentityConfig(netid),
            ssh=SshConfig(identity_file),
            defaults=defaults,
            clusters=clusters,
            path=config_path,
        )

    @classmethod
    def _resource_values(
        cls, table: Mapping[str, Any], section: str, path: Path
    ) -> dict[str, str | int | None]:
        partition = _string(table, "partition", section, path)
        gpu_partition = _string(table, "gpu_partition", section, path)
        for key, value in (("partition", partition), ("gpu_partition", gpu_partition)):
            if value is not None and not _PARTITION.fullmatch(value):
                raise ConfigInvalid(f"[{section}].{key} is not a valid partition name", path=path)
        slurm_time = _string(table, "time", section, path)
        if slurm_time is not None and not _valid_slurm_time(slurm_time):
            raise ConfigInvalid(f"[{section}].time must be a quoted Slurm duration", path=path)
        memory = _string(table, "mem", section, path)
        if memory is not None and not _MEMORY.fullmatch(memory):
            raise ConfigInvalid(f"[{section}].mem is not a valid Slurm memory value", path=path)
        return {
            "partition": partition,
            "gpu_partition": gpu_partition,
            "time": slurm_time,
            "cpus": _positive_int(table, "cpus", section, path),
            "mem": memory,
            "idle_timeout": _positive_int(table, "idle_timeout", section, path, allow_zero=True),
        }

    def resolve_cluster(self, explicit: str | None = None) -> str:
        if explicit is not None:
            if explicit not in self.clusters:
                raise ConfigInvalid(f"cluster {explicit!r} is not configured", path=self.path)
            return explicit
        if self.defaults.cluster is not None:
            return self.defaults.cluster
        if len(self.clusters) == 1:
            return next(iter(self.clusters))
        raise ConfigInvalid(
            "multiple clusters are configured; set [defaults].cluster or pass --cluster", path=self.path
        )

    def cluster(self, explicit: str | None = None) -> ClusterConfig:
        return self.clusters[self.resolve_cluster(explicit)]

    def resolve_option(self, key: str, cluster: str | None = None, *, fallback: Any = None) -> Any:
        if key not in self.RESOURCE_KEYS:
            raise KeyError(key)
        selected = self.cluster(cluster)
        cluster_value = getattr(selected, key)
        if cluster_value is not None:
            return cluster_value
        default_value = getattr(self.defaults, key)
        return default_value if default_value is not None else fallback

    @classmethod
    def validate_resource_override(cls, key: str, value: Any) -> Any:
        """Validate one CLI/resource override with the authoritative schema."""

        if key not in cls.RESOURCE_KEYS:
            raise KeyError(key)
        return cls._resource_values(
            {key: value}, "command line", Path("<command line>")
        )[key]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without the source path."""

        def values(instance: object, *, skip: set[str] = frozenset()) -> dict[str, Any]:
            return {f.name: getattr(instance, f.name) for f in fields(instance) if f.name not in skip}

        return {
            "identity": values(self.identity),
            "ssh": values(self.ssh),
            "defaults": values(self.defaults),
            "cluster": {name: values(cluster, skip={"name"}) for name, cluster in self.clusters.items()},
        }
