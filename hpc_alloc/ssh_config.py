"""Rendering and atomic installation of hpc-alloc's managed SSH config."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .errors import ConfigInvalid
from .locking import configuration_scope_lock
from .models import JobKind


INCLUDE_LINE = "Include ~/.config/hpc-alloc/ssh_config"
_MANAGED_HEADER = "# Managed by hpc-alloc v2 — regenerated; do not edit."
_COMPUTE_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}\Z")
_MANAGED_ALLOCATION_ALIAS = re.compile(
    r"hpc-([A-Za-z0-9][A-Za-z0-9_-]{0,62})\."
    r"([A-Za-z0-9][A-Za-z0-9_-]{0,62})\Z"
)
_CONTROL_CLUSTER_DIGEST_LENGTH = 8


@dataclass(frozen=True, slots=True)
class ComputeMasterRetirement:
    """Old aliases and the unchanged subset that may share their masters."""

    old_aliases: tuple[str, ...]
    retained_aliases: tuple[str, ...]


def login_alias(cluster: str) -> str:
    # Dots cannot occur in validated cluster/allocation identifiers, making
    # the mapping injective even when several cluster/name pairs share text.
    return f"hpc-{cluster}.login"


def allocation_alias(cluster: str, name: str) -> str:
    return f"hpc-{cluster}.{name}"


def compute_host_key_alias(cluster: str, node: str) -> str:
    """Return the durable known-host identity for one physical cluster node."""

    return f"hpc-alloc-node.{cluster}.{node}"


def compute_control_socket_prefix(cluster: str) -> str:
    """Return the basename prefix reserved for one cluster's compute masters."""

    digest = hashlib.sha256(cluster.encode("utf-8")).hexdigest()[
        :_CONTROL_CLUSTER_DIGEST_LENGTH
    ]
    return f"hpc-alloc-{digest}-"


def _compute_control_path(cluster: str) -> str:
    """Keep cross-cluster masters distinct without risking long socket paths."""

    return f"~/.ssh/{compute_control_socket_prefix(cluster)}%C"


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


def _managed_compute_stanzas(text: str) -> dict[str, str]:
    """Extract only compute stanzas that match this renderer's safe shape."""

    if not text.startswith(f"{_MANAGED_HEADER}\n"):
        return {}
    stanzas: dict[str, str] = {}
    duplicate_aliases: set[str] = set()
    for block in text.split("\n\n"):
        lines = block.splitlines()
        if not lines or not lines[0].startswith("Host "):
            continue
        alias = lines[0].removeprefix("Host ")
        alias_match = _MANAGED_ALLOCATION_ALIAS.fullmatch(alias)
        if alias_match is None:
            continue
        if alias in stanzas or alias in duplicate_aliases:
            stanzas.pop(alias, None)
            duplicate_aliases.add(alias)
            continue
        cluster = alias_match.group(1)
        directives: dict[str, str] = {}
        valid = True
        for line in lines[1:]:
            match = re.fullmatch(r"    ([A-Za-z][A-Za-z0-9]*) (.+)", line)
            if match is None or match.group(1) in directives:
                valid = False
                break
            directives[match.group(1)] = match.group(2)
        node = directives.get("HostName", "")
        required = {
            "HostName",
            "HostKeyAlias",
            "User",
            "ProxyJump",
            "ControlMaster",
            "ControlPath",
            "ControlPersist",
            "StrictHostKeyChecking",
            "UserKnownHostsFile",
            "ServerAliveInterval",
            "ServerAliveCountMax",
        }
        allowed = required | {"IdentityFile", "IdentitiesOnly"}
        identity_shape = (
            "IdentityFile" in directives
            and directives.get("IdentitiesOnly") == "yes"
        ) or (
            "IdentityFile" not in directives
            and "IdentitiesOnly" not in directives
        )
        if (
            not valid
            or set(directives) - allowed
            or not required.issubset(directives)
            or not identity_shape
            or _COMPUTE_NODE.fullmatch(node) is None
            or directives["HostKeyAlias"]
            != compute_host_key_alias(cluster, node)
            or directives["ProxyJump"] != login_alias(cluster)
            or directives["ControlMaster"] != "auto"
            or directives["ControlPath"] != _compute_control_path(cluster)
            or directives["ControlPersist"] != "4h"
            or directives["StrictHostKeyChecking"] != "accept-new"
            or directives["ServerAliveInterval"] != "15"
            or directives["ServerAliveCountMax"] != "3"
        ):
            continue
        if alias not in duplicate_aliases:
            stanzas[alias] = block
    return stanzas


def _compute_master_retirement(
    previous: str, replacement: str
) -> ComputeMasterRetirement | None:
    old = _managed_compute_stanzas(previous)
    if not old:
        return None
    new = _managed_compute_stanzas(replacement)
    retained = tuple(
        sorted(alias for alias, block in old.items() if new.get(alias) == block)
    )
    old_aliases = tuple(sorted(old))
    if len(retained) == len(old_aliases):
        return None
    return ComputeMasterRetirement(old_aliases, retained)


def atomic_write_600(path: Path, data: str) -> bool:
    """Atomically replace *path*, fsyncing both content and its directory."""

    try:
        return _atomic_write_600(path, data)
    except OSError as exc:
        raise ConfigInvalid(f"cannot update {path}: {exc}") from exc


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _trusted_regular_file(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_nlink == 1
    )


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result] | None:
    """Open one stable, app-trusted inode without following the final path."""

    try:
        before = path.lstat()
    except OSError:
        return None
    if not _trusted_regular_file(before):
        return None

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
    except OSError:
        os.close(fd)
        return None
    except BaseException:
        os.close(fd)
        raise
    if not _trusted_regular_file(opened) or not _same_inode(before, opened):
        os.close(fd)
        return None
    return fd, opened


def _read_utf8_fd(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 64 * 1024)
        if not chunk:
            return b"".join(chunks).decode("utf-8", errors="strict")
        chunks.append(chunk)


def _write_utf8_fd(fd: int, data: str) -> None:
    remaining = memoryview(data.encode("utf-8"))
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("short write while updating managed configuration")
        remaining = remaining[written:]


def _read_regular_nofollow(path: Path) -> str | None:
    """Read UTF-8 only from a validated regular file at *path*."""

    opened = _open_regular_nofollow(path)
    if opened is None:
        return None
    fd, _metadata = opened
    try:
        return _read_utf8_fd(fd)
    except (OSError, UnicodeError):
        return None
    finally:
        os.close(fd)


def _atomic_write_600(path: Path, data: str) -> bool:
    """Unchecked implementation used behind the typed filesystem boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    opened = _open_regular_nofollow(path)
    if opened is not None:
        fd, metadata = opened
        try:
            unchanged = _read_utf8_fd(fd) == data
            if unchanged:
                current = path.lstat()
                if _trusted_regular_file(current) and _same_inode(metadata, current):
                    os.fchmod(fd, 0o600)
                    current = path.lstat()
                    opened_after = os.fstat(fd)
                    if (
                        _trusted_regular_file(current)
                        and _trusted_regular_file(opened_after)
                        and _same_inode(metadata, current)
                        and _same_inode(metadata, opened_after)
                    ):
                        return False
        except (OSError, UnicodeError):
            pass
        finally:
            os.close(fd)
    raw_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    fd: int | None = raw_fd
    tmp = Path(tmp_name)
    try:
        assert fd is not None
        os.fchmod(fd, 0o600)
        _write_utf8_fd(fd, data)
        os.fsync(fd)
        closing_fd, fd = fd, None
        os.close(closing_fd)
        tmp.replace(path)
        dirfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        return True
    finally:
        try:
            if fd is not None:
                os.close(fd)
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
        _MANAGED_HEADER,
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
    before_replace: Callable[[ComputeMasterRetirement], None] | None = None,
) -> bool:
    """Project current config/state into one serialized managed SSH file.

    The stable sibling lock is deliberately separate from ``managed_path``:
    atomic replacement changes the managed file's inode, which would make a
    lock on that file ineffective across concurrent writers.
    """

    with configuration_scope_lock(lock_path, exclusive=True):
        # Both authoritative inputs are loaded only after acquiring the lock.
        # A process that waited behind a newer writer therefore cannot publish
        # the stale config or job snapshot it started with.
        from .config import Config

        config = Config.load(config_path)
        jobs = repository.list_jobs(include_final=False)
        replacement = render(config, jobs, known_hosts)
        # Only the contents of the validated regular-file inode at the managed
        # path may authorize retirement.  A symlink (including one whose
        # target looks exactly like our projection) is untrusted input and is
        # repaired by the atomic replacement below without being followed.
        previous = _read_regular_nofollow(managed_path) or ""
        retirement = _compute_master_retirement(previous, replacement)
        if retirement is not None and before_replace is not None:
            before_replace(retirement)
        return atomic_write_600(managed_path, replacement)


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
