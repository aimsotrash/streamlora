"""Baseline forecasters.

These exist to make the project's central claim falsifiable. On telemetry, the
naive baselines are strong: CPU utilisation is close to a noisy random walk at
short horizons, so persistence is genuinely hard to beat below a few minutes.
Reporting a learned model's MAE without them would be meaningless.

Every baseline is computed from the raw window in native units, never from
scaled features, so persistence is exactly the last measured value.
"""

from __future__ import annotations

import math

import numpy as np

from .base import ForecastContext, Forecaster


def _clean(ts: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = ~np.isnan(vals)
    return ts[m], vals[m]


def _clip(ctx: ForecastContext, signal: str, v: float) -> float:
    lo, hi = ctx.range_of(signal)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


class PersistenceForecaster(Forecaster):
    """y(t+h) = y(t). The reference every other model must beat."""

    kind = "persistence"

    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        return ctx.current(signal)


class MovingAverageForecaster(Forecaster):
    """y(t+h) = mean of the last ``window_s`` seconds.

    Beats persistence when the signal is mean-reverting noise, loses when it
    trends. Which of those a laptop signal is depends on the horizon, which is
    exactly why the comparison is reported per horizon.
    """

    kind = "moving_average"

    def __init__(self, window_s: float = 300.0) -> None:
        self.window_s = float(window_s)

    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        ts, vals = _clean(*ctx.raw_series(signal))
        if ts.size == 0:
            return None
        m = ts >= (ctx.ts - self.window_s)
        w = vals[m]
        if w.size == 0:
            return float(vals[-1])
        return _clip(ctx, signal, float(w.mean()))

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "window_s": self.window_s}


class EwmaForecaster(Forecaster):
    """Exponentially weighted mean of the window.

    Computed from the stored window rather than carried as running state, so a
    replay produces bit-identical output regardless of where it starts.
    ``half_life_s`` is specified in time, not samples, so the behaviour does not
    change when the collection cadence does.
    """

    kind = "ewma"

    def __init__(self, half_life_s: float = 120.0) -> None:
        self.half_life_s = float(half_life_s)

    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        ts, vals = _clean(*ctx.raw_series(signal))
        if ts.size == 0:
            return None
        if ts.size == 1:
            return float(vals[0])
        age = ctx.ts - ts
        w = np.exp(-math.log(2.0) * age / max(self.half_life_s, 1e-6))
        s = w.sum()
        if s <= 0:
            return float(vals[-1])
        return _clip(ctx, signal, float((w * vals).sum() / s))

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "half_life_s": self.half_life_s}


class LinearTrendForecaster(Forecaster):
    """y(t+h) = y(t) + slope * h, slope fitted over ``fit_window_s``.

    This is the physically-motivated battery baseline: state of charge really
    does fall roughly linearly under a steady load, so extrapolating the
    discharge rate is the forecast a competent engineer would write by hand.
    A learned model that cannot beat it on battery has not learned anything.
    Damping keeps it from producing absurd values at long horizons.
    """

    kind = "linear_trend"

    def __init__(self, fit_window_s: float = 300.0, damping: float = 1.0) -> None:
        self.fit_window_s = float(fit_window_s)
        self.damping = float(damping)

    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        ts, vals = _clean(*ctx.raw_series(signal))
        if ts.size < 3:
            return None if ts.size == 0 else float(vals[-1])
        m = ts >= (ctx.ts - self.fit_window_s)
        t, v = ts[m], vals[m]
        if t.size < 3:
            t, v = ts, vals
        t0 = t - t[0]
        var = float(((t0 - t0.mean()) ** 2).sum())
        if var <= 0:
            return float(v[-1])
        slope = float(((t0 - t0.mean()) * (v - v.mean())).sum() / var)
        return _clip(ctx, signal, float(v[-1] + slope * horizon_s * self.damping))

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "fit_window_s": self.fit_window_s, "damping": self.damping}


class SeasonalNaiveForecaster(Forecaster):
    """y(t+h) = y(t + h - 24h): the value at the same clock time yesterday.

    Requires a day of history, so it returns None until it has one rather than
    silently degrading to persistence -- a baseline that quietly becomes a
    different baseline is worse than a missing one. Needs the database, so it is
    only used in offline evaluation.
    """

    kind = "seasonal_naive"

    def __init__(self, period_s: float = 86400.0, tolerance_s: float = 60.0) -> None:
        self.period_s = float(period_s)
        self.tolerance_s = float(tolerance_s)
        self._ts: np.ndarray | None = None
        self._vals: dict[str, np.ndarray] = {}

    def load_history(self, ts: np.ndarray, series: dict[str, np.ndarray]) -> None:
        self._ts = np.asarray(ts)
        self._vals = {k: np.asarray(v) for k, v in series.items()}

    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        if self._ts is None or signal not in self._vals:
            return None
        want = ctx.ts + horizon_s - self.period_s
        if want < self._ts[0]:
            return None
        i = int(np.abs(self._ts - want).argmin())
        if abs(float(self._ts[i]) - want) > self.tolerance_s:
            return None
        v = float(self._vals[signal][i])
        return None if math.isnan(v) else _clip(ctx, signal, v)


def default_baselines(window_s: float = 300.0) -> list[Forecaster]:
    return [
        PersistenceForecaster(),
        MovingAverageForecaster(window_s=window_s),
        EwmaForecaster(half_life_s=window_s / 2.5),
        LinearTrendForecaster(fit_window_s=window_s),
    ]
