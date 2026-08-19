"""Prediction intervals by adaptive conformal inference.

The requirement is intervals that mean something, or an honest statement that
they do not. Three candidates:

* **Gaussian from the model's parameter covariance** (``sqrt(x' P x)``).
  Available for free from RLS, but it captures uncertainty about the
  *coefficients* only, not observation noise, and it assumes the noise is
  Gaussian and homoscedastic. Telemetry noise is neither: CPU error is tiny when
  idle and huge mid-build. Intervals from this route are badly miscalibrated.
* **Split conformal.** Distribution-free finite-sample coverage, but it assumes
  exchangeability -- which continual learning breaks by construction, since the
  model that produced old residuals no longer exists.
* **Adaptive conformal inference** (Gibbs & Candes, 2021). Keeps a pool of
  recent absolute residuals, and adjusts the working quantile level online:

      alpha_{t+1} = alpha_t + step * (alpha_target - miss_t)

  where ``miss_t`` is 1 if the last outcome fell outside its interval. Coverage
  converges to ``1 - alpha_target`` over time *regardless* of distribution
  shift or model change, which is exactly the setting here.

So this module implements adaptive conformal. Its guarantee is long-run average
coverage, not conditional coverage: intervals may be too wide in calm periods
and too narrow in the first minutes of a new regime, and the recorded coverage
metric will show that. That limitation is documented rather than hidden.

Residuals are pooled **per (signal, horizon)** because error scale differs by an
order of magnitude across both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import UncertaintyConfig
from ..util.mathx import RingStats, clamp


@dataclass
class ConformalState:
    """Residual pool and working miscoverage level for one (signal, horizon)."""

    alpha_target: float
    alpha_t: float
    step: float
    pool: RingStats
    n_observed: int = 0
    n_missed: int = 0
    #: Coverage over the pool's lifetime, for the calibration report.
    hits: int = 0

    def observe(self, abs_error: float, covered: bool | None) -> None:
        self.pool.push(abs_error)
        self.n_observed += 1
        if covered is None:
            return
        if covered:
            self.hits += 1
        else:
            self.n_missed += 1
        # ACI update: widen after a miss, narrow after a hit.
        miss = 0.0 if covered else 1.0
        self.alpha_t = clamp(
            self.alpha_t + self.step * (self.alpha_target - miss), 1e-4, 0.999
        )

    def width(self, min_residuals: int) -> float | None:
        if len(self.pool) < min_residuals:
            return None
        q = clamp(1.0 - self.alpha_t, 0.0, 1.0)
        return float(self.pool.quantile(q))

    @property
    def empirical_coverage(self) -> float | None:
        total = self.hits + self.n_missed
        return (self.hits / total) if total else None

    def as_dict(self) -> dict[str, object]:
        return {
            "alpha_target": self.alpha_target,
            "alpha_t": round(self.alpha_t, 4),
            "residuals": len(self.pool),
            "observed": self.n_observed,
            "empirical_coverage": (
                round(self.empirical_coverage, 4) if self.empirical_coverage is not None else None
            ),
        }


class ConformalCalibrator:
    """Adaptive conformal intervals, keyed by (model_kind, signal, horizon)."""

    def __init__(self, config: UncertaintyConfig | None = None) -> None:
        self.cfg = config or UncertaintyConfig()
        self._states: dict[tuple[str, str, float], ConformalState] = {}

    @property
    def enabled(self) -> bool:
        return self.cfg.method == "conformal"

    def _state(self, model_kind: str, signal: str, horizon_s: float) -> ConformalState:
        key = (model_kind, signal, float(horizon_s))
        st = self._states.get(key)
        if st is None:
            st = ConformalState(
                alpha_target=self.cfg.alpha,
                alpha_t=self.cfg.alpha,
                step=self.cfg.step,
                pool=RingStats(self.cfg.window),
            )
            self._states[key] = st
        return st

    def interval(
        self, model_kind: str, signal: str, horizon_s: float, point: float
    ) -> tuple[float | None, float | None, float | None]:
        """Return ``(lo, hi, alpha_t)``; all None before enough residuals.

        Emitting nothing is deliberate: a fabricated interval from five
        residuals is worse than an explicit "not yet calibrated".
        """
        if not self.enabled:
            return None, None, None
        st = self._state(model_kind, signal, horizon_s)
        w = st.width(self.cfg.min_residuals)
        if w is None:
            return None, None, st.alpha_t
        return point - w, point + w, st.alpha_t

    def observe(
        self, model_kind: str, signal: str, horizon_s: float,
        abs_error: float, covered: bool | None,
    ) -> None:
        if not self.enabled:
            return
        self._state(model_kind, signal, horizon_s).observe(abs_error, covered)

    def report(self) -> dict[str, dict[str, object]]:
        return {
            f"{k[0]}|{k[1]}@{int(k[2])}": v.as_dict() for k, v in sorted(self._states.items(), key=str)
        }

    def state_for(self, model_kind: str, signal: str, horizon_s: float) -> ConformalState | None:
        return self._states.get((model_kind, signal, float(horizon_s)))
