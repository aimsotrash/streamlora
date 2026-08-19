"""Forecaster interface and prediction record.

The one rule that keeps evaluation honest: a forecaster sees only a
``ForecastContext`` built from data at or before ``ts``. There is no path from a
model to future samples, so leakage has to be introduced deliberately rather
than by accident.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..telemetry.schema import SignalRegistry
from .features import FeatureExtractor, FeatureVector


@dataclass(slots=True)
class ForecastContext:
    """Everything a forecaster may look at when predicting."""

    ts: float
    fv: FeatureVector
    extractor: FeatureExtractor
    registry: SignalRegistry
    regime: str = "unknown"
    #: Baseline forecasts for this tick, keyed by (signal, horizon) then model
    #: kind. Populated by the engine before the learned model runs, so the
    #: learned model can consume them as inputs without recomputing them.
    baseline_preds: dict[tuple[str, float], dict[str, float | None]] = field(
        default_factory=dict
    )

    def raw_series(self, signal: str) -> tuple[np.ndarray, np.ndarray]:
        return self.extractor.raw_series(signal)

    def volatility(self, signal: str, span_s: float) -> float:
        """Std of ``signal`` over the last ``span_s`` seconds, in native units.

        This is the unit the learned model's target is expressed in, which is
        what makes that target comparable across an idle machine and a compiling
        one -- and therefore what lets one ridge penalty be correct for both.
        """
        ts, vals = self.extractor.raw_series(signal)
        if ts.size == 0:
            return 0.0
        m = (~np.isnan(vals)) & (ts >= self.ts - span_s)
        w = vals[m]
        if w.size < 3:
            return 0.0
        return float(w.std(ddof=0))

    def current(self, signal: str) -> float | None:
        ts, vals = self.extractor.raw_series(signal)
        for i in range(len(vals) - 1, -1, -1):
            if not np.isnan(vals[i]):
                return float(vals[i])
        return None

    def range_of(self, signal: str) -> tuple[float | None, float | None]:
        spec = self.registry.get(signal)
        return (None, None) if spec is None else (spec.lo, spec.hi)


@dataclass(slots=True)
class Prediction:
    signal: str
    horizon_s: float
    ts_made: float
    ts_target: float
    value: float
    model_kind: str
    model_version: str = "n/a"
    regime: str = "unknown"
    lo: float | None = None
    hi: float | None = None
    alpha: float | None = None
    anchor: float | None = None
    feature_id: int | None = None
    infer_ms: float = 0.0

    def as_row(self, run_id: str | None) -> dict[str, object]:
        return {
            "ts_made": self.ts_made, "ts_target": self.ts_target, "horizon_s": self.horizon_s,
            "signal": self.signal, "value": self.value, "lo": self.lo, "hi": self.hi,
            "alpha": self.alpha, "model_kind": self.model_kind,
            "model_version": self.model_version, "regime": self.regime, "anchor": self.anchor,
            "feature_id": self.feature_id, "infer_ms": self.infer_ms, "run_id": run_id,
        }


@dataclass(slots=True)
class TrainingExample:
    """A resolved (inputs -> outcome) pair used for incremental updates.

    Carries everything needed to reconstruct the model's input exactly as it was
    at ``ts_made``, which is what lets the promotion gate score a candidate on
    the same rows the active model saw.
    """

    ts_made: float
    ts_target: float
    signal: str
    horizon_s: float
    x: np.ndarray                    # shared per-tick feature vector at ts_made
    anchor: float                    # value of the target signal at ts_made
    actual: float                    # measured value at ts_target
    #: Baseline forecasts as normalised offsets from the anchor, in scope order.
    extras: np.ndarray | None = None
    #: Volatility scale used to normalise this example's target.
    vol: float = 1.0
    regime: str = "unknown"
    prediction_id: int | None = None
    weight: float = 1.0


class Forecaster(abc.ABC):
    """Numerical forecaster. No language model is involved anywhere below."""

    kind: str = "abstract"
    #: True if the forecaster supports incremental updates.
    learnable: bool = False

    @abc.abstractmethod
    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        """Point forecast for ``signal`` at ``ctx.ts + horizon_s``, or None."""

    def version(self, signal: str = "", horizon_s: float = 0.0) -> str:
        return f"{self.kind}-v001"

    def update(self, examples: Sequence[TrainingExample]) -> int:
        """Incorporate resolved outcomes. Returns the number applied."""
        return 0

    def describe(self) -> dict[str, object]:
        return {"kind": self.kind, "learnable": self.learnable}
