"""Typed OpenSSH transport with explicit authentication and retry policy.

This module deliberately knows nothing about Slurm.  It establishes and heals
the multiplexed SSH connection, classifies OpenSSH failures, and returns remote
exit status without turning it into a process exit.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .errors import (
    AuthRequired,
    HostKeyChanged,
    LocalToolUnavailable,
    SchedulerUnavailable,
    TransportLost,
)
from .ssh_config import (
    ComputeMasterRetirement,
    compute_control_socket_prefix,
    login_alias,
)


_MASTER_RETIREMENT_TIMEOUT_SECONDS = 10


class AuthMode(StrEnum):
    """Whether a call may establish an interactive login session."""

    INTERACTIVE_BOOTSTRAP = "interactive-bootstrap"
    NONINTERACTIVE = "noninteractive"


class RetryPolicy(StrEnum):
    """Remote retry policy; mutations must always use :attr:`NEVER`."""

    SAFE_READ = "safe-read"
    NEVER = "never"


class ProbeStatus(StrEnum):
    OK = "ok"
    AUTH = "auth"
    HOST_KEY = "host-key"
    NETWORK = "network"


@dataclass(frozen=True, slots=True)
class RemoteResult:
    """One completed remote command.

    ``stderr`` is always decoded with replacement.  ``stdout`` is bytes only
    when ``binary=True`` was requested from :meth:`SshTransport.run`.
    """

    returncode: int
    stdout: str | bytes
    stderr: str

    @property
    def stdout_text(self) -> str:
        if isinstance(self.stdout, bytes):
            return self.stdout.decode("utf-8", errors="replace")
        return self.stdout

    @property
    def stdout_bytes(self) -> bytes:
        if isinstance(self.stdout, bytes):
            return self.stdout
        return self.stdout.encode("utf-8")


def ssh_argv(
    alias: str,
    remote_command: str | None = None,
    *,
    batch: bool = True,
    connect_timeout: int = 10,
    extra_options: Sequence[str] = (),
) -> list[str]:
    """Build every SSH invocation in one place."""

    argv = ssh_transport_argv(
        batch=batch, connect_timeout=connect_timeout, extra_options=extra_options
    )
    argv += ["--", alias]
    if remote_command is not None:
        argv.append(remote_command)
    return argv


def ssh_transport_argv(
    *,
    batch: bool = True,
    connect_timeout: int = 10,
    extra_options: Sequence[str] = (),
) -> list[str]:
    """The ssh command and options, with no host and no remote command.

    rsync needs this shape for ``-e``.  Passing it a bare ``ssh`` instead left
    the transfer as the one SSH call site with none of the hardening every other
    one gets: no BatchMode, so a dead master could drop it to an interactive
    password prompt on the user's terminal, and no ConnectTimeout, so it could
    hang indefinitely where the rest of the tool bounds and raises a typed error.
    """

    argv = ["ssh", "-o", f"BatchMode={'yes' if batch else 'no'}"]
    for option in extra_options:
        argv += ["-o", option]
    argv += ["-o", f"ConnectTimeout={connect_timeout}"]
    return argv


def can_prompt() -> bool:
    """Return whether OpenSSH can reach a controlling terminal."""

    if sys.stdin.isatty():
        return True
    try:
        fd = os.open("/dev/tty", os.O_RDWR)
    except OSError:
        return False
    os.close(fd)
    return True


def _effective_compute_control_path(
    alias: str,
    *,
    ssh_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess],
) -> tuple[bool, Path | None, str | None]:
    """Resolve one old alias to an app-owned effective control socket."""

    try:
        result = runner(
            ["ssh", "-G", "-T", "--", alias],
            capture_output=True,
            text=True,
            check=False,
            timeout=_MASTER_RETIREMENT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return False, None, f"could not inspect obsolete SSH alias {alias}: {exc}"
    if result.returncode != 0:
        detail = str(result.stderr or "").strip().splitlines()
        reason = detail[0] if detail else f"ssh -G exited {result.returncode}"
        return False, None, f"could not inspect obsolete SSH alias {alias}: {reason}"
    stdout = result.stdout
    if not isinstance(stdout, str):
        return False, None, f"could not inspect obsolete SSH alias {alias}: non-text ssh -G output"
    values = [
        fields[1]
        for line in stdout.splitlines()
        if len(fields := line.split(None, 1)) == 2
        and fields[0].lower() == "controlpath"
    ]
    if len(values) != 1:
        return (
            False,
            None,
            f"could not inspect obsolete SSH alias {alias}: expected one controlpath",
        )
    if values[0].lower() == "none":
        return True, None, None

    body = alias.removeprefix("hpc-")
    cluster, separator, _name = body.partition(".")
    if not separator or not cluster:
        return False, None, f"could not inspect malformed obsolete SSH alias {alias}"
    root = Path(os.path.abspath(os.path.expanduser(str(ssh_dir))))
    path = Path(os.path.abspath(os.path.expanduser(values[0])))
    if path.parent != root or not path.name.startswith(
        compute_control_socket_prefix(cluster)
    ):
        return (
            False,
            None,
            f"refusing to retire non-managed SSH control path for {alias}: {path}",
        )
    return True, path, None


def retire_compute_masters(
    retirement: ComputeMasterRetirement,
    *,
    ssh_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[str, ...]:
    """Best-effort close old compute masters without disturbing shared ones."""

    warnings: list[str] = []
    resolved: dict[str, Path | None] = {}

    # Resolve retained aliases first.  If any cannot be inspected, do not risk
    # closing a socket still shared by that active allocation.
    for alias in retirement.retained_aliases:
        ok, path, warning = _effective_compute_control_path(
            alias, ssh_dir=ssh_dir, runner=runner
        )
        if warning is not None:
            warnings.append(warning)
        if not ok:
            warnings.append(
                "skipped obsolete SSH master retirement because a retained "
                "allocation's control path could not be protected"
            )
            return tuple(warnings)
        resolved[alias] = path
    retained_paths = {path for path in resolved.values() if path is not None}

    obsolete_paths: dict[Path, str] = {}
    for alias in retirement.old_aliases:
        if alias not in resolved:
            ok, path, warning = _effective_compute_control_path(
                alias, ssh_dir=ssh_dir, runner=runner
            )
            if warning is not None:
                warnings.append(warning)
            if not ok:
                continue
            resolved[alias] = path
        path = resolved[alias]
        if path is not None and path not in retained_paths:
            obsolete_paths.setdefault(path, alias)

    for path, alias in sorted(obsolete_paths.items()):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            warnings.append(f"could not inspect obsolete SSH control socket {path}: {exc}")
            continue
        if not stat.S_ISSOCK(metadata.st_mode):
            warnings.append(
                f"refusing to retire non-socket SSH control path {path}"
            )
            continue
        if metadata.st_uid != os.geteuid():
            warnings.append(
                f"refusing to retire SSH control socket not owned by the "
                f"current user: {path}"
            )
            continue
        try:
            result = runner(
                ["ssh", "-S", str(path), "-O", "exit", "--", alias],
                capture_output=True,
                text=True,
                check=False,
                timeout=_MASTER_RETIREMENT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            warnings.append(f"could not retire obsolete SSH control socket {path}: {exc}")
            continue
        if result.returncode == 0:
            continue
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            warnings.append(f"could not verify obsolete SSH control socket {path}: {exc}")
            continue
        detail = str(result.stderr or "").strip().splitlines()
        reason = detail[0] if detail else f"ssh -O exit returned {result.returncode}"
        warnings.append(f"could not retire obsolete SSH control socket {path}: {reason}")

    return tuple(warnings)


class SshTransport:
    """Multiplexed SSH transport for configured login aliases.

    ``state_repository`` is optional and used only to discover active compute
    aliases during healing.  Callers can instead provide ``alias_provider``;
    neither is consulted while a repository transaction is active.
    """

    def __init__(
        self,
        config: object,
        paths: object,
        *,
        entrypoint: Path | None = None,
        state_repository: object | None = None,
        alias_provider: Callable[[str], Iterable[str]] | None = None,
        info: Callable[[str], None] | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.entrypoint = (entrypoint or Path(sys.argv[0])).resolve()
        self.state_repository = state_repository
        self.alias_provider = alias_provider
        self.info = info or (lambda _message: None)
        self._runner = runner
        self._environ = dict(os.environ if environ is None else environ)
        self._verified: set[str] = set()
        self.last_probe: ProbeStatus | None = None
        self.probe_detail = ""
        # True when the last probe timed out rather than actively failing.  A
        # stall is evidence of slowness, never of a dead ControlMaster.
        self.probe_stalled = False

    def _run(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess:
        """Invoke a local OpenSSH tool without leaking local OS tracebacks."""

        try:
            return self._runner(*args, **kwargs)
        except OSError as exc:
            command = args[0] if args else ()
            executable = command[0] if isinstance(command, (list, tuple)) and command else "ssh"
            raise LocalToolUnavailable(f"cannot execute {executable}: {exc}") from exc

    def _cluster_known(self, cluster: str) -> None:
        clusters = getattr(self.config, "clusters", {})
        if cluster not in clusters:
            raise TransportLost(f"cluster {cluster!r} is not configured")

    @staticmethod
    def _classify(stderr: str) -> ProbeStatus:
        detail = stderr.lower()
        if "host key verification failed" in detail or "identification has changed" in detail:
            return ProbeStatus.HOST_KEY
        if (
            "permission denied" in detail
            or "authentication" in detail
            or "no supported authentication methods" in detail
        ):
            return ProbeStatus.AUTH
        return ProbeStatus.NETWORK

    @staticmethod
    def _probe_failure(
        alias: str, status: ProbeStatus, detail: str
    ) -> HostKeyChanged | AuthRequired | TransportLost:
        """Translate one classified SSH failure without losing its cause."""

        first = detail.strip().splitlines()[0] if detail.strip() else ""
        if status is ProbeStatus.HOST_KEY:
            reason = first or "stored key does not match"
            return HostKeyChanged(
                f"SSH host-key verification failed for {alias}: {reason}"
            )
        if status is ProbeStatus.AUTH:
            reason = first or "authentication was rejected"
            return AuthRequired(f"SSH authentication failed for {alias}: {reason}")
        reason = first or "connection failed"
        return TransportLost(f"cannot reach {alias}: {reason}")

    @classmethod
    def _raise_terminal_ssh_failure(cls, alias: str, detail: str) -> None:
        """Raise security/authentication rc-255 failures before retry policy."""

        status = cls._classify(detail)
        if status in {ProbeStatus.HOST_KEY, ProbeStatus.AUTH}:
            raise cls._probe_failure(alias, status, detail)

    def master_alive(self, cluster: str) -> bool:
        result = self._run(
            ["ssh", "-O", "check", "--", login_alias(cluster)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    def close_master(self, alias: str, *, drain: bool = True) -> None:
        """Retire a ControlMaster, by default letting its sessions finish.

        `-O exit` terminates the master immediately, taking every session
        multiplexed on it with it.  That is unsafe here: the login master is
        shared by every concurrent hpc-alloc process, so one process healing a
        blip would tear down another's in-flight submission -- which, because
        submissions never retry, surfaces as an ambiguous mutation and orphans a
        GPU job.  A compute master can likewise be carrying the user's
        interactive `ssh` shell or an in-flight `sync`.

        `-O stop` instead makes the master stop accepting *new* multiplexed
        sessions and exit once the existing ones drain, which heals just as well
        (a subsequent connection builds a fresh master) without killing live
        work.  A master too wedged to service `-O stop` is equally too wedged
        for `-O exit`; sweep_dead_sockets clears that case.
        """

        self._run(
            ["ssh", "-O", "stop" if drain else "exit", "--", alias],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )

    def sweep_dead_sockets(self) -> None:
        ssh_dir = Path(getattr(self.paths, "ssh_dir"))
        for socket in ssh_dir.glob("hpc-alloc-*"):
            result = self._run(
                ["ssh", "-S", str(socket), "-O", "check", "unused"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                try:
                    socket.unlink()
                except OSError:
                    pass

    def _compute_aliases(self, cluster: str) -> tuple[str, ...]:
        if self.alias_provider is not None:
            return tuple(self.alias_provider(cluster))
        repository = self.state_repository
        if repository is None:
            return ()
        # Foundation repositories expose snapshots outside a transaction.  Be
        # deliberately duck-typed so transport does not depend on persistence.
        for method_name in ("active_aliases", "compute_aliases"):
            method = getattr(repository, method_name, None)
            if method is not None:
                return tuple(method(cluster))
        return ()

    def heal(self, cluster: str, aliases: Iterable[str] | None = None) -> None:
        for alias in aliases if aliases is not None else self._compute_aliases(cluster):
            self.close_master(alias)
        self.close_master(login_alias(cluster))
        self.sweep_dead_sockets()
        self._verified.discard(cluster)

    def probe_alias(self, alias: str, *, timeout: int = 15) -> ProbeStatus:
        self.probe_stalled = False
        try:
            result = self._run(
                ssh_argv(alias, "true"),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            self.probe_detail = "connection stalled"
            self.probe_stalled = True
            return ProbeStatus.NETWORK
        self.probe_detail = (result.stderr or "").strip()
        if result.returncode == 0:
            self.probe_detail = ""
            return ProbeStatus.OK
        return self._classify(self.probe_detail)

    def probe(self, cluster: str, *, timeout: int = 15) -> ProbeStatus:
        self._cluster_known(cluster)
        status = self.probe_alias(login_alias(cluster), timeout=timeout)
        self.last_probe = status
        return status

    def probe_node(self, alias: str, *, timeout: int = 15) -> ProbeStatus:
        status = self.probe_alias(alias, timeout=timeout)
        if status is ProbeStatus.OK or self.probe_stalled:
            # A stalled probe means the node is slow -- a loaded allocation (the
            # very thing an allocation is for) or a briefly hung shared home --
            # not that its master is dead.  Retiring the master on that evidence
            # would disrupt every session multiplexed on it, including another
            # process's interactive shell or in-flight rsync, while the master's
            # connection was in fact healthy.
            return status
        self.close_master(alias)
        return self.probe_alias(alias, timeout=timeout)

    def require_node(self, alias: str, *, timeout: int = 15) -> None:
        """Require a usable compute alias and preserve its typed SSH failure."""

        status = self.probe_node(alias, timeout=timeout)
        if status is not ProbeStatus.OK:
            raise self._probe_failure(alias, status, self.probe_detail)

    def bootstrap(
        self,
        cluster: str,
        auth_mode: AuthMode = AuthMode.INTERACTIVE_BOOTSTRAP,
    ) -> None:
        """Ensure a login master exists, optionally prompting exactly once."""

        self._cluster_known(cluster)
        if cluster in self._verified:
            return
        if self.master_alive(cluster):
            self._verified.add(cluster)
            return
        status = self.probe(cluster)
        if status == ProbeStatus.OK:
            self._verified.add(cluster)
            return
        if status == ProbeStatus.HOST_KEY:
            first = self.probe_detail.splitlines()[0] if self.probe_detail else "host key mismatch"
            raise HostKeyChanged(f"SSH host-key verification failed for {cluster}: {first}")
        if status == ProbeStatus.AUTH:
            if auth_mode == AuthMode.NONINTERACTIVE or not can_prompt():
                raise AuthRequired(
                    f"{login_alias(cluster)} requires interactive authentication; "
                    f"run `hpc-alloc connect --cluster {cluster}`"
                )
            self.info(f"connecting to {login_alias(cluster)} (answer the Duo prompt)")
            result = self._run(
                ssh_argv(
                    login_alias(cluster),
                    "true",
                    batch=False,
                    connect_timeout=30,
                    extra_options=("NumberOfPasswordPrompts=1",),
                ),
                capture_output=True,
                text=True,
                errors="replace",
            )
            if result.returncode == 0:
                self._verified.add(cluster)
                return
            detail = (result.stderr or "").strip()
            classified = self._classify(detail)
            if classified == ProbeStatus.HOST_KEY:
                raise HostKeyChanged(detail or f"host key changed for {cluster}")
            if classified == ProbeStatus.AUTH:
                raise AuthRequired(detail or f"authentication failed for {cluster}")
            raise TransportLost(detail or f"cannot reach {login_alias(cluster)}")
        raise TransportLost(
            self.probe_detail or f"cannot reach {login_alias(cluster)}; check VPN connectivity"
        )

    def _identity_file(self) -> Path | None:
        identity = getattr(getattr(self.config, "ssh", None), "identity_file", None)
        return Path(identity).expanduser() if identity else None

    def _key_fingerprint(self, key: Path) -> str | None:
        public = Path(str(key) + ".pub")
        result = self._run(
            ["ssh-keygen", "-lf", str(public if public.exists() else key)],
            capture_output=True,
            text=True,
        )
        fields = (result.stdout or "").split()
        return fields[1] if result.returncode == 0 and len(fields) >= 2 else None

    def _key_in_agent(self, key: Path) -> bool:
        fingerprint = self._key_fingerprint(key)
        if not fingerprint:
            return False
        result = self._run(["ssh-add", "-l"], capture_output=True, text=True)
        return result.returncode == 0 and fingerprint in (result.stdout or "")

    def _key_needs_passphrase(self, key: Path) -> bool:
        result = self._run(
            ["ssh-keygen", "-y", "-P", "", "-f", str(key)],
            capture_output=True,
        )
        return result.returncode != 0

    def push_login(self, cluster: str, *, timeout: int = 90) -> None:
        """Authenticate through one askpass-triggered Duo push."""

        self._cluster_known(cluster)
        key = self._identity_file()
        if key and key.exists() and self._key_needs_passphrase(key) and not self._key_in_agent(key):
            raise AuthRequired(
                f"{key} is passphrase-protected and not loaded in ssh-agent; "
                f"run `ssh-add {key}` first"
            )
        environment = {
            **self._environ,
            "SSH_ASKPASS": str(self.entrypoint),
            "SSH_ASKPASS_REQUIRE": "force",
            "HPC_ALLOC_ASKPASS": "1",
            "DISPLAY": self._environ.get("DISPLAY", ":0"),
        }
        self.info(f"requesting one Duo push for {login_alias(cluster)}")
        try:
            result = self._run(
                ssh_argv(
                    login_alias(cluster),
                    "true",
                    batch=False,
                    connect_timeout=30,
                    extra_options=("NumberOfPasswordPrompts=1",),
                ),
                env=environment,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AuthRequired("Duo push was not approved before timeout") from exc
        if result.returncode != 0:
            detail = (result.stderr or "").strip()
            status = self._classify(detail)
            if status == ProbeStatus.HOST_KEY:
                raise HostKeyChanged(detail or f"host key changed for {cluster}")
            if status == ProbeStatus.NETWORK:
                raise TransportLost(detail or f"cannot reach {cluster}")
            raise AuthRequired(detail or "Duo push was denied")
        self._verified.add(cluster)

    def _invoke(
        self,
        cluster: str,
        command: str,
        *,
        timeout: float,
        binary: bool,
    ) -> RemoteResult | None:
        kwargs: dict[str, object] = {
            "capture_output": True,
            "timeout": timeout,
            # ssh forwards its own stdin to the remote command unless told
            # otherwise, and capture_output only redirects stdout/stderr.  These
            # polling invocations never consume input, so an inherited fd 0
            # would silently drain the caller's stdin one poll at a time --
            # eating a `while read ...; done < list` loop's remaining lines.
            "stdin": subprocess.DEVNULL,
        }
        if not binary:
            kwargs.update(text=True, errors="replace")
        try:
            result = self._run(
                ssh_argv(login_alias(cluster), command, connect_timeout=20),
                **kwargs,
            )
        except subprocess.TimeoutExpired:
            return None
        stderr = result.stderr or (b"" if binary else "")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return RemoteResult(result.returncode, result.stdout, stderr)

    def run(
        self,
        cluster: str,
        command: str,
        *,
        auth: AuthMode = AuthMode.NONINTERACTIVE,
        retry: RetryPolicy = RetryPolicy.SAFE_READ,
        timeout: float = 60,
        binary: bool = False,
    ) -> RemoteResult:
        """Run one remote command without interpreting its nonzero exit code."""

        self.bootstrap(cluster, auth)
        result = self._invoke(cluster, command, timeout=timeout, binary=binary)
        if result is not None and result.returncode != 255:
            return result

        self._verified.discard(cluster)
        detail = "remote command timed out" if result is None else result.stderr.strip() or "ssh failed"
        if result is not None:
            self._raise_terminal_ssh_failure(login_alias(cluster), detail)
        if retry == RetryPolicy.NEVER:
            raise TransportLost(detail)

        status = self.probe(cluster)
        if status is ProbeStatus.HOST_KEY:
            raise self._probe_failure(login_alias(cluster), status, self.probe_detail)
        if status != ProbeStatus.OK:
            self.heal(cluster)
            self.bootstrap(cluster, auth)
        else:
            self._verified.add(cluster)

        second = self._invoke(cluster, command, timeout=timeout, binary=binary)
        if second is not None and second.returncode != 255:
            return second
        self._verified.discard(cluster)
        second_detail = (
            "remote command timed out" if second is None else second.stderr.strip() or "ssh failed"
        )
        if second is not None:
            self._raise_terminal_ssh_failure(login_alias(cluster), second_detail)
        else:
            final_status = self.probe(cluster)
            if final_status is ProbeStatus.OK:
                self._verified.add(cluster)
                raise SchedulerUnavailable(
                    f"remote command repeatedly timed out while SSH remained reachable ({second_detail})"
                )
            if final_status in {ProbeStatus.HOST_KEY, ProbeStatus.AUTH}:
                raise self._probe_failure(
                    login_alias(cluster), final_status, self.probe_detail
                )
        self.heal(cluster)
        raise TransportLost(second_detail)


__all__ = [
    "AuthMode",
    "ProbeStatus",
    "RemoteResult",
    "RetryPolicy",
    "SshTransport",
    "can_prompt",
    "retire_compute_masters",
    "ssh_argv",
    "ssh_transport_argv",
]
