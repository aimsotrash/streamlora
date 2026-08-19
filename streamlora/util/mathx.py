"""Small numerical helpers used across forecasting, drift and evaluation.

Kept dependency-light on purpose: these are the pieces where a subtle bug
silently corrupts every downstream metric, so they are simple and unit tested.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def quantile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted sequence."""
    n = len(sorted_xs)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_xs[0])
    q = clamp(q, 0.0, 1.0)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_xs[lo])
    frac = pos - lo
    return float(sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac)


class Welford:
    """Streaming mean/variance. Numerically stable, O(1) memory."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def copy(self) -> "Welford":
        w = Welford()
        w.n, w.mean, w.m2 = self.n, self.mean, self.m2
        return w


class RingStats:
    """Fixed-capacity window with mean/std/quantile access.

    Used where a *bounded* history matters (drift reference windows,
    conformal residual pools) and O(window) recomputation is acceptable.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, x: float) -> None:
        self._buf.append(float(x))

    def extend(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.push(x)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def full(self) -> bool:
        return len(self._buf) == self.capacity

    def values(self) -> list[float]:
        return list(self._buf)

    def mean(self) -> float:
        return safe_mean(self._buf)

    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        m = self.mean()
        return math.sqrt(sum((x - m) ** 2 for x in self._buf) / (n - 1))

    def quantile(self, q: float) -> float:
        return quantile(sorted(self._buf), q)

    def clear(self) -> None:
        self._buf.clear()


def linear_slope(ys: Sequence[float], dt: float = 1.0) -> float:
    """Least-squares slope of ``ys`` against evenly spaced time, per unit time.

    Returns 0.0 for degenerate inputs rather than raising: telemetry windows
    are routinely short or constant and callers should not have to guard.
    """
    n = len(ys)
    if n < 2 or dt <= 0:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(ys):
        dx = i - mean_x
        num += dx * (y - mean_y)
        den += dx * dx
    if den == 0:
        return 0.0
    return (num / den) / dt


def ewma(xs: Sequence[float], alpha: float) -> float:
    """Exponentially weighted mean, most recent sample weighted ``alpha``."""
    if not xs:
        return float("nan")
    acc = float(xs[0])
    for x in xs[1:]:
        acc = alpha * float(x) + (1 - alpha) * acc
    return acc
