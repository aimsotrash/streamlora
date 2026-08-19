"""Telemetry source interface and shared helpers.

A source is a small, independently-failing unit. The contract is intentionally
narrow so that adding a signal later means writing one class:

    class FanSource(TelemetrySource):
        name = "fan"
        def signals(self): ...
        def probe(self): ...
        def read(self, now): ...

Nothing in the collector, forecaster or UI needs to change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..schema import SignalSpec


@dataclass(slots=True)
class ProbeResult:
    available: bool
    detail: str = ""
    #: Signals this source will actually emit on this machine. May be a subset
    #: of ``signals()`` -- e.g. a thermal source that finds a CPU sensor but no
    #: GPU sensor. Empty means "all declared signals".
    provides: tuple[str, ...] = ()


class SourceUnavailable(RuntimeError):
    """Raised by ``read`` when the source has become unusable mid-run."""


class TelemetrySource(abc.ABC):
    """One family of related signals from one mechanism.

    Sources own any cross-sample state they need (counter deltas, file
    handles). They are not required to be thread-safe; the collector reads
    them from a single thread.
    """

    #: Short stable identifier, used in logs, health records and the UI.
    name: str = "unnamed"
    #: Human-facing description shown in the settings/diagnostics view.
    description: str = ""

    @abc.abstractmethod
    def signals(self) -> list[SignalSpec]:
        """Every signal this source can produce, with canonical units."""

    @abc.abstractmethod
    def probe(self) -> ProbeResult:
        """Cheap capability check. Must not raise."""

    @abc.abstractmethod
    def read(self, now: float) -> dict[str, float | None]:
        """Return current values keyed by signal name.

        ``None`` means "this signal is genuinely unavailable right now" and is
        distinct from raising, which means "the whole source failed".
        Implementations should return partial results rather than raising when
        only some signals fail.
        """

    def close(self) -> None:  # pragma: no cover - most sources are stateless
        """Release any held resources."""


class RateTracker:
    """Convert a monotonic counter into a per-second rate.

    Handles the two failure modes that actually occur with OS counters:
    a counter reset (device re-enumerated, 32-bit wraparound) and a zero/
    negative time delta. Both yield ``None`` rather than a garbage spike.
    """

    __slots__ = ("_prev_val", "_prev_ts")

    def __init__(self) -> None:
        self._prev_val: float | None = None
        self._prev_ts: float | None = None

    def update(self, value: float | None, now: float) -> float | None:
        if value is None:
            return None
        prev_v, prev_t = self._prev_val, self._prev_ts
        self._prev_val, self._prev_ts = float(value), float(now)
        if prev_v is None or prev_t is None:
            return None                      # first observation: no rate yet
        dt = now - prev_t
        if dt <= 0:
            return None
        delta = float(value) - prev_v
        if delta < 0:
            return None                      # counter reset; skip this interval
        return delta / dt

    def reset(self) -> None:
        self._prev_val = None
        self._prev_ts = None


def read_sysfs_float(path: str, scale: float = 1.0) -> float | None:
    """Read a single numeric value from sysfs, or None if unreadable.

    sysfs reads fail for mundane reasons (device removed, permissions, driver
    reload). Callers should treat None as "signal unavailable this tick".
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read(64).strip()
        if not raw:
            return None
        return float(raw) * scale
    except (OSError, ValueError):
        return None
