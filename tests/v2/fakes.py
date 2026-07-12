"""Strict, dependency-free test doubles for the v2 domain core."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class ExpectedCall:
    """One exact scripted call and the result (or exception) it produces."""

    method: str
    result: Any = None
    args: tuple[Any, ...] | None = None
    kwargs: dict[str, Any] | None = None


class StrictScript:
    """Consumes an exact call sequence; unknown and missing calls are failures."""

    def __init__(self, calls: Iterable[ExpectedCall] = ()) -> None:
        self._calls = deque(calls)
        self.ledger: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def invoke(self, method: str, *args: Any, **kwargs: Any) -> Any:
        self.ledger.append((method, args, kwargs))
        if not self._calls:
            raise AssertionError(f"unexpected call {method}{args!r}{kwargs!r}")
        expected = self._calls[0]
        if method != expected.method:
            raise AssertionError(f"expected {expected.method}, got {method}")
        if expected.args is not None and args != expected.args:
            raise AssertionError(f"{method}: expected args {expected.args!r}, got {args!r}")
        if expected.kwargs is not None and kwargs != expected.kwargs:
            raise AssertionError(
                f"{method}: expected kwargs {expected.kwargs!r}, got {kwargs!r}"
            )
        self._calls.popleft()
        if isinstance(expected.result, BaseException):
            raise expected.result
        if callable(expected.result):
            return expected.result(*args, **kwargs)
        return expected.result

    def assert_complete(self) -> None:
        if self._calls:
            remaining = ", ".join(call.method for call in self._calls)
            raise AssertionError(f"unconsumed expected calls: {remaining}")

    def count(self, method: str) -> int:
        return sum(1 for name, _args, _kwargs in self.ledger if name == method)


class StrictProxy:
    """Turns arbitrary method access into StrictScript calls."""

    def __init__(self, script: StrictScript) -> None:
        self.script = script

    def __getattr__(self, name: str) -> Callable[..., Any]:
        return lambda *args, **kwargs: self.script.invoke(name, *args, **kwargs)


@dataclass
class VirtualClock:
    """Monotonic clock/sleeper pair that never blocks tests."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise AssertionError(f"negative sleep: {seconds}")
        self.sleeps.append(seconds)
        self.now += seconds
