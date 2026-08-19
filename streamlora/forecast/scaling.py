"""Deterministic feature scaling derived from signal specs.

Raw telemetry spans wildly different magnitudes: percentages in [0, 100],
network throughput in [0, 20000] MB/s, temperatures around 60. Feeding those
directly into a least-squares update produces an information matrix with a
condition number in the millions, and the ridge penalty then falls almost
entirely on the small-scale features.

The obvious fix -- a running mean/std normaliser -- is subtly wrong for
*online* learning: as the normaliser drifts, the meaning of every previously
learned coefficient silently changes, so the model is chasing a moving
parameterisation as well as a moving world. That makes "did adaptation help?"
unanswerable.

So scaling here is **fixed and declarative**, computed from each signal's
declared range and kind. It never changes, which means a coefficient learned an
hour ago still means the same thing now, and a model file is portable.
"""

from __future__ import annotations

import math

import numpy as np

from ..telemetry.schema import SignalKind, SignalRegistry, SignalSpec

#: Divisor applied after log1p for rate signals, chosen so that a busy laptop's
#: throughput (~100 MB/s -> log1p ~4.6) lands near 0.5.
_RATE_DIV = 9.0


def scale_value(spec: SignalSpec | None, v: float) -> float:
    """Map a raw reading onto a roughly [0, 1] scale."""
    if spec is None:
        return float(v)
    if spec.kind == SignalKind.FLAG:
        return 1.0 if v > 0.5 else 0.0
    if spec.kind == SignalKind.RATE:
        # Rates are heavy-tailed and mostly zero; a log transform makes the
        # difference between 0 and 5 MB/s as visible as 100 vs 400 MB/s.
        return math.log1p(max(0.0, v)) / _RATE_DIV
    lo = spec.lo
    hi = spec.hi
    if lo is None or hi is None or hi <= lo:
        return float(v) / 100.0
    return (float(v) - lo) / (hi - lo)


def unscale_value(spec: SignalSpec | None, z: float) -> float:
    if spec is None:
        return float(z)
    if spec.kind == SignalKind.FLAG:
        return 1.0 if z > 0.5 else 0.0
    if spec.kind == SignalKind.RATE:
        return math.expm1(max(0.0, z) * _RATE_DIV)
    lo, hi = spec.lo, spec.hi
    if lo is None or hi is None or hi <= lo:
        return float(z) * 100.0
    return lo + float(z) * (hi - lo)


class Scaler:
    """Vectorised scaler for a fixed ordered list of signals."""

    def __init__(self, signals: list[str], registry: SignalRegistry) -> None:
        self.signals = list(signals)
        self.specs = [registry.get(s) for s in self.signals]
        self._kind = np.array(
            [0 if s is None else {"gauge": 0, "rate": 1, "flag": 2, "counter": 1}.get(str(s.kind), 0)
             for s in self.specs]
        )
        los, spans = [], []
        for s in self.specs:
            if s is None or s.lo is None or s.hi is None or s.hi <= s.lo:
                los.append(0.0)
                spans.append(100.0)
            else:
                los.append(float(s.lo))
                spans.append(float(s.hi - s.lo))
        self._lo = np.asarray(los)
        self._span = np.asarray(spans)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Scale a (n, n_signals) or (n_signals,) array, preserving NaN."""
        v = np.asarray(values, dtype=np.float64)
        out = (v - self._lo) / self._span
        rate = self._kind == 1
        if rate.any():
            with np.errstate(invalid="ignore"):
                out[..., rate] = np.log1p(np.clip(v[..., rate], 0.0, None)) / _RATE_DIV
        flag = self._kind == 2
        if flag.any():
            out[..., flag] = (v[..., flag] > 0.5).astype(np.float64)
            # Preserve NaN through the boolean cast, which would map it to 0.
            out[..., flag] = np.where(np.isnan(v[..., flag]), np.nan, out[..., flag])
        return out

    #: Scale factor converting a scaled-space delta back to native units.
    def native_span(self, signal: str) -> float:
        i = self.signals.index(signal)
        return float(self._span[i])
