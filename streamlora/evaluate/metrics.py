"""Forecast error metrics.

Choices here are deliberate, because the wrong metric makes a bad model look
fine on telemetry:

* **MAE** is the headline. It is in the signal's own units ("CPU is wrong by 6
  percentage points"), which is what a user can act on, and it is robust to the
  spikes telemetry is full of.
* **RMSE** is reported alongside because it is what large errors show up in.
  MAE improving while RMSE worsens means a model got better on average by
  becoming occasionally catastrophic -- worth knowing.
* **MAPE is deliberately not used.** CPU utilisation is legitimately 0.4% on an
  idle laptop, and a 2-point error there is a 500% "percentage error". A single
  idle minute can dominate a day's MAPE. **sMAPE** with a symmetric denominator
  is reported instead, and it still degrades near zero, so it is never the
  headline.
* **Skill score** ``1 - MAE / MAE_reference`` against persistence is the number
  that actually answers "is this worth anything". An MAE of 4.2 means nothing
  without knowing that doing nothing scores 4.4.
* **Coverage and interval width** together, never separately: 100% coverage with
  an interval spanning 0-100% is worthless, and reporting only coverage hides it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class ErrorMetrics:
    n: int = 0
    mae: float | None = None
    rmse: float | None = None
    smape: float | None = None
    bias: float | None = None
    p90_abs_error: float | None = None
    max_abs_error: float | None = None
    #: Fraction of outcomes inside the predicted interval, where intervals exist.
    coverage: float | None = None
    n_intervals: int = 0
    mean_interval_width: float | None = None
    #: 1 - mae/mae_reference. Positive means better than the reference.
    skill: float | None = None
    reference: str | None = None

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None, k: int = 4) -> float | None:
            return None if x is None else round(x, k)

        return {
            "n": self.n, "mae": r(self.mae), "rmse": r(self.rmse), "smape": r(self.smape),
            "bias": r(self.bias), "p90_abs_error": r(self.p90_abs_error),
            "max_abs_error": r(self.max_abs_error), "coverage": r(self.coverage),
            "n_intervals": self.n_intervals,
            "mean_interval_width": r(self.mean_interval_width),
            "skill": r(self.skill), "reference": self.reference,
        }


def smape(actual: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE in percent, in [0, 200].

    Chosen over MAPE because telemetry signals genuinely reach zero. Still
    unstable when both actual and prediction are near zero, so it is reported as
    supporting information and never used to rank models.
    """
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    ok = denom > eps
    if not ok.any():
        return float("nan")
    return float(100.0 * np.mean(np.abs(actual[ok] - pred[ok]) / denom[ok]))


def compute(
    actual: Sequence[float],
    pred: Sequence[float],
    lo: Sequence[float | None] | None = None,
    hi: Sequence[float | None] | None = None,
    reference_mae: float | None = None,
    reference_name: str | None = None,
) -> ErrorMetrics:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    m = ErrorMetrics(n=int(a.size))
    if a.size == 0:
        return m
    err = p - a
    abs_err = np.abs(err)
    m.mae = float(abs_err.mean())
    m.rmse = float(math.sqrt(float((err**2).mean())))
    m.smape = smape(a, p)
    # Signed: positive means the model systematically over-predicts. A large
    # bias with a small MAE means a constant offset, which is easy to fix; the
    # reverse means noise, which is not.
    m.bias = float(err.mean())
    m.p90_abs_error = float(np.quantile(abs_err, 0.9))
    m.max_abs_error = float(abs_err.max())
    if lo is not None and hi is not None:
        L = np.array([np.nan if x is None else float(x) for x in lo], dtype=np.float64)[ok]
        H = np.array([np.nan if x is None else float(x) for x in hi], dtype=np.float64)[ok]
        have = np.isfinite(L) & np.isfinite(H)
        if have.any():
            inside = (a[have] >= L[have]) & (a[have] <= H[have])
            m.coverage = float(inside.mean())
            m.n_intervals = int(have.sum())
            m.mean_interval_width = float((H[have] - L[have]).mean())
    if reference_mae is not None and reference_mae > 0 and m.mae is not None:
        m.skill = float(1.0 - m.mae / reference_mae)
        m.reference = reference_name
    return m


@dataclass(slots=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    coverage: float


def calibration_curve(
    actual: Sequence[float], lo: Sequence[float | None], hi: Sequence[float | None],
    bins: int = 5,
) -> list[CalibrationBin]:
    """Coverage as a function of interval width.

    A single coverage number hides the failure mode that matters: intervals that
    are correct on average because they are far too wide when the machine is calm
    and far too narrow right after a regime change. Binning by width exposes it.
    """
    a = np.asarray(actual, dtype=np.float64)
    L = np.array([np.nan if x is None else float(x) for x in lo], dtype=np.float64)
    H = np.array([np.nan if x is None else float(x) for x in hi], dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(L) & np.isfinite(H)
    if ok.sum() < bins * 4:
        return []
    a, L, H = a[ok], L[ok], H[ok]
    width = H - L
    inside = (a >= L) & (a <= H)
    edges = np.quantile(width, np.linspace(0, 1, bins + 1))
    out: list[CalibrationBin] = []
    for i in range(bins):
        m = (width >= edges[i]) & (width <= edges[i + 1] if i == bins - 1 else width < edges[i + 1])
        if m.sum() == 0:
            continue
        out.append(
            CalibrationBin(
                lo=float(edges[i]), hi=float(edges[i + 1]), n=int(m.sum()),
                coverage=float(inside[m].mean()),
            )
        )
    return out


def diebold_mariano(
    err_a: Sequence[float], err_b: Sequence[float], loss: str = "abs"
) -> tuple[float, float] | None:
    """Test whether two forecasters differ significantly (Diebold-Mariano).

    Included because "model A had MAE 6.31 and model B 6.44" is not a result on
    2,000 autocorrelated samples -- the difference has to survive a test that
    accounts for the fact that consecutive forecast errors are not independent.
    Uses a Newey-West long-run variance with a rule-of-thumb lag.

    Returns ``(statistic, two-sided p-value)`` under a normal approximation, or
    None when there is too little data. The normal approximation is why this is
    reported as supporting evidence and not as an exact p-value.
    """
    a = np.asarray(err_a, dtype=np.float64)
    b = np.asarray(err_b, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 30:
        return None
    a, b = a[:n], b[:n]
    if loss == "sq":
        d = a**2 - b**2
    else:
        d = np.abs(a) - np.abs(b)
    dbar = float(d.mean())
    d0 = d - dbar
    # Newey-West with lag = floor(n^(1/3)), the standard rule of thumb.
    L = max(1, int(n ** (1.0 / 3.0)))
    gamma0 = float((d0 * d0).mean())
    var = gamma0
    for k in range(1, L + 1):
        cov = float((d0[k:] * d0[:-k]).mean())
        var += 2.0 * (1.0 - k / (L + 1.0)) * cov
    if var <= 0:
        return None
    stat = dbar / math.sqrt(var / n)
    # Two-sided normal p-value via the complementary error function.
    p = math.erfc(abs(stat) / math.sqrt(2.0))
    return float(stat), float(p)
