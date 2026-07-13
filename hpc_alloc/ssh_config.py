"""Rendering and atomic installation of hpc-alloc's managed SSH config."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Iterable

from .errors import ConfigInvalid
from .models import JobKind


INCLUDE_LINE = "Include ~/.config/hpc-alloc/ssh_config"
_COMPUTE_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z")
_CONTROL_CLUSTER_DIGEST_LENGTH = 8


def login_alias(cluster: str) -> str:
    # Dots cannot occur in validated cluster/allocation identifiers, making
    # the mapping injective even when several cluster/name pairs share text.
    return f"hpc-{cluster}.login"


def allocation_alias(cluster: str, name: str) -> str:
    return f"hpc-{cluster}.{name}"


def compute_host_key_alias(cluster: str, node: str) -> str:
    """Return the durable known-host identity for one physical cluster node."""

    return f"hpc-alloc-node.{cluster}.{node}"


def _compute_control_path(cluster: str) -> str:
    """Keep cross-cluster masters distinct without risking long socket paths."""

    digest = hashlib.sha256(cluster.encode("utf-8")).hexdigest()[
        :_CONTROL_CLUSTER_DIGEST_LENGTH
    ]
    return f"~/.ssh/hpc-alloc-{digest}-%C"


def _quoted_value(value: str | Path) -> str:
    """Quote an OpenSSH config value that may contain filesystem spaces."""

    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _hostname_value(host: str) -> str:
    """Serialize one validated host without triggering OpenSSH tokens."""

    if "%" not in host:
        return host
    # Hostname expands percent tokens even inside quotes.  A scoped IPv6
    # literal therefore needs both token escaping and quoting: ``%%`` reaches
    # the connector as one literal percent, while quotes keep scope IDs that
    # contain whitespace in a single config argument.
    return _quoted_value(host.replace("%", "%%"))


def atomic_write_600(path: Path, data: str) -> bool:
    """Atomically replace *path*, fsyncing both content and its directory."""

    try:
        return _atomic_write_600(path, data)
    except OSError as exc:
        raise ConfigInvalid(f"cannot update {path}: {exc}") from exc


def _atomic_write_600(path: Path, data: str) -> bool:
    """Unchecked implementation used behind the typed filesystem boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.exists() and path.read_text() == data:
            path.chmod(0o600)
            return False
    except (OSError, UnicodeError):
        pass
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        return True
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def render(config: object, jobs: Iterable[object], known_hosts: Path) -> str:
    """Render login aliases and currently-active allocation aliases."""

    identity = getattr(getattr(config, "ssh"), "identity_file", None)
    netid = getattr(getattr(config, "identity"), "netid")
    id_lines = [f"    IdentityFile {identity}", "    IdentitiesOnly yes"] if identity else []
    lines = [
        "# Managed by hpc-alloc v2 — regenerated; do not edit.",
        "",
    ]
    clusters = getattr(config, "clusters")
    for name, cluster in sorted(clusters.items()):
        lines += [
            f"Host {login_alias(name)}",
            f"    HostName {_hostname_value(cluster.host)}",
            f"    User {netid}",
            *id_lines,
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/hpc-alloc-%C",
            "    ControlPersist 4h",
            "    ServerAliveInterval 15",
            "    ServerAliveCountMax 3",
            "    StrictHostKeyChecking accept-new",
            "",
        ]
    for job in sorted(jobs, key=lambda item: (item.cluster, item.logical_name or "")):
        if getattr(job, "kind") != JobKind.ALLOCATION:
            continue
        if getattr(job, "cluster", None) not in clusters:
            continue
        node = getattr(job, "current_node", None)
        name = getattr(job, "logical_name", None)
        if not node or not name:
            continue
        if _COMPUTE_NODE.fullmatch(node) is None:
            raise ConfigInvalid(f"unsafe compute-node name in state: {node!r}")
        lines += [
            f"Host {allocation_alias(job.cluster, name)}",
            f"    HostName {node}",
            f"    HostKeyAlias {compute_host_key_alias(job.cluster, node)}",
            f"    User {netid}",
            *id_lines,
            f"    ProxyJump {login_alias(job.cluster)}",
            "    ControlMaster auto",
            f"    ControlPath {_compute_control_path(job.cluster)}",
            "    ControlPersist 4h",
            "    StrictHostKeyChecking accept-new",
            f"    UserKnownHostsFile {_quoted_value(known_hosts)}",
            "    ServerAliveInterval 15",
            "    ServerAliveCountMax 3",
            "",
        ]
    return "\n".join(lines)


def sync_managed_config(
    *,
    config_path: Path,
    repository: object,
    managed_path: Path,
    lock_path: Path,
    known_hosts: Path,
) -> bool:
    """Project current config/state into one serialized managed SSH file.

    The stable sibling lock is deliberately separate from ``managed_path``:
    atomic replacement changes the managed file's inode, which would make a
    lock on that file ineffective across concurrent writers.
    """

    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path.parent.chmod(0o700)
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ConfigInvalid(f"cannot open managed SSH-config lock {lock_path}: {exc}") from exc
    try:
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ConfigInvalid(f"managed SSH-config lock is not a regular file: {lock_path}")
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            raise ConfigInvalid(f"cannot lock managed SSH config {lock_path}: {exc}") from exc

        # Both authoritative inputs are loaded only after acquiring the lock.
        # A process that waited behind a newer writer therefore cannot publish
        # the stale config or job snapshot it started with.
        from .config import Config

        config = Config.load(config_path)
        jobs = repository.list_jobs(include_final=False)
        return atomic_write_600(managed_path, render(config, jobs, known_hosts))
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def resolve_user_ssh_config(path: Path) -> Path:
    """Resolve a live symlink, rejecting dangling/looping configurations."""

    if not path.is_symlink():
        return path
    try:
        target = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigInvalid(
            f"{path} is a dangling or looping symlink ({exc}); repair the dotfiles link "
            "before rerunning setup"
        ) from exc
    if not target.parent.is_dir():
        raise ConfigInvalid(f"the target directory for {path} does not exist: {target.parent}")
    return target


def ensure_include(user_config: Path) -> bool:
    """Prepend the managed Include once, writing through a valid symlink."""

    try:
        user_config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        user_config.parent.chmod(0o700)
    except OSError as exc:
        raise ConfigInvalid(f"cannot prepare {user_config.parent}: {exc}") from exc
    target = resolve_user_ssh_config(user_config)
    try:
        existing = target.read_text() if target.exists() else ""
    except (OSError, UnicodeError) as exc:
        raise ConfigInvalid(f"cannot read {target}: {exc}") from exc
    if re.search(
        r"^[ \t]*Include[ \t]+~/\.config/hpc-alloc/ssh_config[ \t]*$",
        existing,
        re.MULTILINE,
    ):
        return False
    atomic_write_600(target, f"# hpc-alloc managed hosts\n{INCLUDE_LINE}\n\n{existing}")
    return True
