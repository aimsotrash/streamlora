"""Time abstraction.

Every component that needs "now" or "sleep" takes a Clock. Nothing in
StreamLoRA calls time.time() or time.sleep() directly outside this module.

That single rule is what makes the streaming pipeline replayable and
deterministic: a replay swaps RealClock for SimulatedClock and the identical
collector / forecaster / drift / adaptation code runs over historical
timestamps at whatever speed we choose.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds since the Unix epoch."""

    def monotonic(self) -> float:
        """Monotonic seconds, for measuring durations."""

    def sleep(self, seconds: float) -> None:
        """Advance time by ``seconds``."""


class RealClock:
    """Wall-clock time. Used in live collection."""

    __slots__ = ()

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimulatedClock:
    """Virtual time that only moves when someone advances it.

    ``speed`` is informational for the player; the clock itself never blocks,
    so replays run as fast as the CPU allows and remain deterministic.
    """

    def __init__(self, start: float, speed: float = 0.0) -> None:
        self._t = float(start)
        self._mono = 0.0
        self.speed = float(speed)
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def monotonic(self) -> float:
        # Real monotonic so that latency/duration instrumentation stays truthful
        # even inside a simulated replay.
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> float:
        with self._lock:
            self._t += float(seconds)
            self._mono += float(seconds)
            return self._t

    def set(self, t: float) -> None:
        with self._lock:
            self._t = float(t)


def resolve_clock(kind: str = "real", start: float | None = None, speed: float = 0.0) -> Clock:
    if kind == "real":
        return RealClock()
    if kind in ("sim", "simulated", "replay"):
        return SimulatedClock(start if start is not None else time.time(), speed=speed)
    raise ValueError(f"unknown clock kind: {kind!r}")
