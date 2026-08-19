"""Streaming forecast engine.

One object owns the per-tick pipeline:

    sample -> regime -> feature window -> predictions (learned + baselines)
           -> resolve matured predictions -> errors
           -> drift detectors -> training buffer -> adaptation decision

Two invariants keep the numbers trustworthy:

* **Predictions are written before their outcome exists.** Nothing resolves a
  prediction using data available when it was made, so there is no path for
  hindsight to leak into a reported error.
* **Every arm sees the identical context.** The learned model and all four
  baselines are called with the same ``ForecastContext`` at the same tick, so a
  skill difference is a modelling difference, not a data difference.

Pending predictions are held in memory for resolution and mirrored to SQLite for
the lifecycle record. The database is never read on the hot path.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..adapt.controller import AdaptationController
from ..config import Config
from ..drift.controller import DriftController
from ..store.repo import Repos
from ..telemetry.schema import SignalRegistry, TelemetrySample
from ..util.ids import config_hash
from ..util.logging import get_logger
from .base import ForecastContext, Forecaster, Prediction, TrainingExample
from .baselines import (
    EwmaForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    PersistenceForecaster,
)
from .features import FeatureExtractor
from .linear import RLSForecaster, build_scope_specs
from .regime import RegimeClassifier
from .uncertainty import ConformalCalibrator

log = get_logger("forecast.engine")

_BASELINE_FACTORIES = {
    "persistence": lambda cfg: PersistenceForecaster(),
    "moving_average": lambda cfg: MovingAverageForecaster(window_s=cfg.features.window_s),
    "ewma": lambda cfg: EwmaForecaster(half_life_s=cfg.features.window_s / 2.5),
    "linear_trend": lambda cfg: LinearTrendForecaster(fit_window_s=cfg.features.window_s),
}


@dataclass(slots=True)
class _Pending:
    """A prediction awaiting its outcome."""

    db_id: int
    signal: str
    horizon_s: float
    ts_made: float
    ts_target: float
    value: float
    lo: float | None
    hi: float | None
    model_kind: str
    model_version: str
    regime: str
    anchor: float | None


@dataclass(slots=True)
class _PendingTarget:
    """A (features -> future value) pair awaiting its outcome.

    Deliberately independent of the prediction stream. Deriving training
    examples from the learned model's own predictions would deadlock: the model
    withholds predictions until it is trained, and it cannot train without
    examples. A supervised example needs only the inputs, the anchor and the
    future measurement -- no model is involved -- so the learning stream is
    generated on every tick regardless of which arms chose to predict.
    """

    signal: str
    horizon_s: float
    ts_made: float
    ts_target: float
    #: Shared reference to the tick's feature vector; one copy per tick.
    x: np.ndarray
    anchor: float
    regime: str
    #: Baseline forecasts as normalised offsets from the anchor, scope order.
    extras: np.ndarray | None = None
    #: Volatility scale in force when the inputs were captured.
    vol: float = 1.0
    prediction_id: int | None = None


@dataclass(slots=True)
class EngineStats:
    ticks: int = 0
    predictions: int = 0
    resolved: int = 0
    unresolvable: int = 0
    predict_ms: float = 0.0
    resolve_ms: float = 0.0
    adapt_ms: float = 0.0
    max_predict_ms: float = 0.0
    skipped_not_ready: int = 0
    skipped_low_coverage: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "ticks": self.ticks, "predictions": self.predictions, "resolved": self.resolved,
            "unresolvable": self.unresolvable,
            "avg_predict_ms": round(self.predict_ms, 3),
            "avg_resolve_ms": round(self.resolve_ms, 3),
            "avg_adapt_ms": round(self.adapt_ms, 3),
            "max_predict_ms": round(self.max_predict_ms, 3),
            "skipped_not_ready": self.skipped_not_ready,
            "skipped_low_coverage": self.skipped_low_coverage,
        }


class ForecastEngine:
    def __init__(
        self,
        config: Config,
        repos: Repos,
        signal_registry: SignalRegistry,
        run_id: str | None = None,
        adapt_controller: AdaptationController | None = None,
        learned: RLSForecaster | None = None,
        enable_drift: bool = True,
        enable_adapt: bool = True,
        record_predictions: bool = True,
    ) -> None:
        self.cfg = config
        self.repos = repos
        self.registry = signal_registry
        self.run_id = run_id
        self.record_predictions = record_predictions

        self.extractor = FeatureExtractor(
            signal_registry, config.features, interval_s=config.collect.interval_s
        )
        self.regimes = RegimeClassifier(config.regime)
        self.targets = [t for t in config.forecast.targets if t in signal_registry]
        self.missing_targets = [t for t in config.forecast.targets if t not in signal_registry]
        self.horizons = [float(h) for h in config.forecast.horizons_s]

        self.baselines: list[Forecaster] = [
            _BASELINE_FACTORIES[name](config)
            for name in config.forecast.baselines
            if name in _BASELINE_FACTORIES
        ]
        masks = {t: self.extractor.select_indices(t) for t in self.targets}
        ranges = {}
        for t in self.targets:
            spec = signal_registry.get(t)
            ranges[t] = (spec.lo, spec.hi) if spec is not None else (None, None)
        baseline_names = (
            tuple(b.kind for b in self.baselines) if config.forecast.baseline_features else ()
        )
        scopes = build_scope_specs(
            signals=self.targets, horizons=self.horizons, masks=masks,
            baseline_names=baseline_names, ranges=ranges,
            vol_floor_frac=config.forecast.vol_floor_frac,
        )
        self.learned = learned or RLSForecaster(
            feature_names=self.extractor.names, scopes=scopes,
            ridge_lambda=config.forecast.ridge_lambda,
            forgetting=config.forecast.forgetting,
            sigma_floor=config.forecast.sigma_floor,
            clip_z=config.forecast.clip_z,
            target_clip=config.forecast.target_clip,
            trust_k=config.forecast.trust_k,
            trust_region_scope=config.forecast.trust_region_scope,
            standardize=config.forecast.standardize,
            standardize_warmup=config.forecast.standardize_warmup,
            per_regime=config.forecast.per_regime_models,
            frozen=(config.forecast.model == "static"),
            vol_span_s=config.forecast.vol_span_s,
        )
        self.calibrator = ConformalCalibrator(config.uncertainty)
        self.drift = DriftController(config.drift) if enable_drift else None
        self.adapt = adapt_controller if enable_adapt else None

        self._pending: deque[_Pending] = deque()
        self._pending_targets: deque[_PendingTarget] = deque()
        #: Warm-up pool for the frozen feature standardiser.
        self._warmup_x: list[np.ndarray] = []
        self._actuals: deque[tuple[float, dict[str, float]]] = deque(maxlen=4096)
        self.stats = EngineStats()
        self._schema_registered = False
        self._fingerprint = config_hash(config.to_dict())
        if self.missing_targets:
            log.warning(
                "forecast targets unavailable on this machine; skipping them",
                targets=self.missing_targets,
            )

    # -- properties --------------------------------------------------------
    @property
    def all_forecasters(self) -> list[Forecaster]:
        return [self.learned, *self.baselines]

    def scope(self, signal: str, horizon_s: float) -> str:
        return f"{signal}@{int(horizon_s)}"

    # -- main entry point --------------------------------------------------
    def on_sample(self, sample: TelemetrySample) -> list[Prediction]:
        self.stats.ticks += 1
        regime = self.regimes.update(sample)
        self.extractor.push(sample, regime)
        self._remember_actuals(sample)

        t0 = time.perf_counter()
        resolved = self._resolve_due(sample.ts)
        self.stats.resolve_ms = _ema(self.stats.resolve_ms, (time.perf_counter() - t0) * 1000.0)

        preds: list[Prediction] = []
        if not self.extractor.ready:
            self.stats.skipped_not_ready += 1
        else:
            t1 = time.perf_counter()
            preds = self._predict_all(sample.ts, regime)
            dt = (time.perf_counter() - t1) * 1000.0
            self.stats.predict_ms = _ema(self.stats.predict_ms, dt)
            self.stats.max_predict_ms = max(self.stats.max_predict_ms, dt)

        if self.adapt is not None and resolved:
            t2 = time.perf_counter()
            self.adapt.maybe_adapt(sample.ts)
            self.adapt.check_regressions(sample.ts)
            self.stats.adapt_ms = _ema(self.stats.adapt_ms, (time.perf_counter() - t2) * 1000.0)
        return preds

    # -- prediction --------------------------------------------------------
    def _predict_all(self, now: float, regime: str) -> list[Prediction]:
        fv = self.extractor.extract(regime)
        if fv is None:
            return []
        if fv.coverage < self.cfg.features.min_coverage:
            # Predicting from a mostly-empty window would produce a confident
            # number from imputed inputs. Skipping is recorded and visible.
            self.stats.skipped_low_coverage += 1
            log.dedupe(
                "coverage", "feature window coverage below threshold; skipping tick",
                level="info", coverage=round(fv.coverage, 3),
                min_coverage=self.cfg.features.min_coverage, now=now,
            )
            return []

        if self.drift is not None:
            for ev in self.drift.observe_features(now, list(fv.values), fv.names):
                self.repos.events.add_drift(**ev.as_row(self.run_id))
                if self.adapt is not None:
                    self.adapt.note_drift(
                        None, None, ev.detector, feature=str(ev.detail.get("feature") or "")
                    )

        feature_id: int | None = None
        if self.record_predictions:
            if not self._schema_registered:
                self.repos.features.register_schema(fv.schema_hash, fv.names, now)
                self._schema_registered = True
            feature_id = self.repos.features.insert(
                now, fv.schema_hash, fv.values, fv.regime, fv.coverage, self.run_id
            )

        ctx = ForecastContext(
            ts=now, fv=fv, extractor=self.extractor, registry=self.registry, regime=regime
        )
        out: list[Prediction] = []
        link: list[tuple[_PendingTarget, int]] = []
        for signal in self.targets:
            anchor = ctx.current(signal)
            vol = self.learned.volatility(ctx, signal)
            for horizon in self.horizons:
                # Baselines run first: the learned model consumes their forecasts
                # as inputs, so they must exist before it is called. Running them
                # once and caching also avoids computing each baseline twice.
                bl: dict[str, float | None] = {}
                for b in self.baselines:
                    p = self._one_prediction(ctx, b, signal, horizon, anchor, feature_id)
                    bl[b.kind] = p.value if p is not None else None
                    if p is not None:
                        out.append(p)
                ctx.baseline_preds[(signal, float(horizon))] = bl

                extras = (
                    self.learned.build_extras(ctx, signal, horizon, anchor, vol)
                    if anchor is not None else None
                )
                target: _PendingTarget | None = None
                if anchor is not None and extras is not None:
                    # Feed the standardiser its warm-up sample, then enqueue the
                    # supervised target. Both happen regardless of whether the
                    # learned model was ready to predict.
                    self.learned.observe_warmup(signal, horizon, fv.values, extras)
                    target = _PendingTarget(
                        signal=signal, horizon_s=horizon, ts_made=now,
                        ts_target=now + horizon, x=fv.values, anchor=anchor,
                        regime=regime, extras=extras, vol=vol,
                    )
                    self._pending_targets.append(target)
                p = self._one_prediction(ctx, self.learned, signal, horizon, anchor, feature_id)
                if p is not None:
                    out.append(p)
                    # Remember which prediction this training target belongs to,
                    # so "was this prediction used for an adaptation update?" is
                    # answerable from the database.
                    if target is not None:
                        link.append((target, len(out) - 1))
        if out:
            ids = self._persist(out)
            for target, idx in link:
                if idx < len(ids) and ids[idx] > 0:
                    target.prediction_id = ids[idx]
        self.stats.predictions += len(out)
        return out

    def _one_prediction(
        self, ctx: ForecastContext, fc: Forecaster, signal: str, horizon_s: float,
        anchor: float | None, feature_id: int | None,
    ) -> Prediction | None:
        is_learned = fc is self.learned
        if is_learned:
            if self.learned.n_updates(signal, horizon_s) < self.cfg.forecast.min_train_before_predict:
                # Warm-up: an RLS model fitted on three examples produces
                # confident nonsense. Withholding is honest and keeps the
                # learned arm's metrics from being dominated by its first minute.
                # The count comes from the model itself, so it cannot drift out
                # of sync with what was actually trained.
                return None
        t0 = time.perf_counter()
        try:
            value = fc.predict(ctx, signal, horizon_s)
        except Exception:
            log.exception("forecaster raised", kind=fc.kind, signal=signal, horizon_s=horizon_s)
            return None
        infer_ms = (time.perf_counter() - t0) * 1000.0
        if value is None or not math.isfinite(value):
            return None
        lo = hi = alpha = None
        if is_learned or fc.kind == "persistence":
            # Intervals are calibrated for the learned model and for
            # persistence, the reference arm. Calibrating all six arms would
            # quadruple residual bookkeeping for numbers nobody compares.
            lo, hi, alpha = self.calibrator.interval(fc.kind, signal, horizon_s, value)
            if lo is not None and hi is not None:
                # Clip to the signal's physical range. A symmetric conformal
                # interval around a near-zero CPU forecast otherwise reaches
                # below 0%, and "CPU will be between -12% and 38%" is not a
                # statement about a CPU. Clipping is conservative for coverage:
                # it can only move the bound toward the truth, since the actual
                # value cannot fall outside the range either.
                s_lo, s_hi = ctx.range_of(signal)
                if s_lo is not None:
                    lo = max(lo, s_lo)
                    hi = max(hi, s_lo)
                if s_hi is not None:
                    hi = min(hi, s_hi)
                    lo = min(lo, s_hi)
        return Prediction(
            signal=signal, horizon_s=horizon_s, ts_made=ctx.ts,
            ts_target=ctx.ts + horizon_s, value=float(value), model_kind=fc.kind,
            model_version=fc.version(signal, horizon_s), regime=ctx.regime,
            lo=lo, hi=hi, alpha=alpha, anchor=anchor, feature_id=feature_id,
            infer_ms=infer_ms,
        )

    def _persist(self, preds: Sequence[Prediction]) -> list[int]:
        ids: list[int] = []
        if self.record_predictions:
            ids = self.repos.predictions.insert_many([p.as_row(self.run_id) for p in preds])
        else:
            ids = [-1] * len(preds)
        for p, db_id in zip(preds, ids):
            self._pending.append(
                _Pending(
                    db_id=db_id, signal=p.signal, horizon_s=p.horizon_s, ts_made=p.ts_made,
                    ts_target=p.ts_target, value=p.value, lo=p.lo, hi=p.hi,
                    model_kind=p.model_kind, model_version=p.model_version, regime=p.regime,
                    anchor=p.anchor,
                )
            )
        return ids

    # -- resolution --------------------------------------------------------
    def _remember_actuals(self, sample: TelemetrySample) -> None:
        vals = {}
        for t in self.targets:
            v = sample.value(t)
            if v is not None:
                vals[t] = float(v)
        if vals:
            self._actuals.append((sample.ts, vals))

    def _actual_at(self, ts_target: float, signal: str) -> tuple[float, float] | None:
        """Closest measured value to ``ts_target``, within tolerance."""
        tol = max(self.cfg.collect.interval_s * 1.5, 2.0)
        best: tuple[float, float] | None = None
        best_dt = float("inf")
        for ts, vals in reversed(self._actuals):
            if ts < ts_target - tol:
                break
            v = vals.get(signal)
            if v is None:
                continue
            dt = abs(ts - ts_target)
            if dt < best_dt:
                best_dt, best = dt, (v, ts)
        if best is None or best_dt > tol:
            return None
        return best

    def _resolve_due(self, now: float) -> list[_Pending]:
        if not self._pending:
            return []
        tol = max(self.cfg.collect.interval_s * 1.5, 2.0)
        resolved: list[_Pending] = []
        updates: list[dict[str, object]] = []
        keep: deque[_Pending] = deque()
        for p in self._pending:
            if p.ts_target > now:
                keep.append(p)
                continue
            hit = self._actual_at(p.ts_target, p.signal)
            if hit is None:
                if now - p.ts_target <= tol:
                    keep.append(p)          # actual may still arrive
                else:
                    # Telemetry gap over the target instant. Dropping is correct:
                    # resolving against a value from minutes later would invent
                    # an error the model never made.
                    self.stats.unresolvable += 1
                continue
            actual, _ts_actual = hit
            err = p.value - actual
            abs_err = abs(err)
            in_interval: int | None = None
            if p.lo is not None and p.hi is not None:
                in_interval = 1 if (p.lo <= actual <= p.hi) else 0
            if p.db_id > 0:
                updates.append({
                    "id": p.db_id, "ts_resolved": now, "actual": actual, "error": err,
                    "abs_error": abs_err, "in_interval": in_interval, "resolve_quality": 0,
                })
            self.calibrator.observe(
                p.model_kind, p.signal, p.horizon_s, abs_err,
                None if in_interval is None else bool(in_interval),
            )
            if p.model_kind == "persistence":
                # Persistence errors drive the drift detectors: they measure how
                # unpredictable the *world* is, independent of which model is
                # live, so a drift alarm cannot be caused by our own adaptation.
                self._observe_drift(now, p, abs_err)
            resolved.append(p)
        self._pending = keep
        if updates:
            self.repos.predictions.resolve_many(updates)
        self.stats.resolved += len(resolved)
        self._resolve_targets(now)
        return resolved

    def _resolve_targets(self, now: float) -> int:
        """Turn matured (features -> outcome) pairs into training examples."""
        if not self._pending_targets:
            return 0
        tol = max(self.cfg.collect.interval_s * 1.5, 2.0)
        keep: deque[_PendingTarget] = deque()
        n = 0
        for t in self._pending_targets:
            if t.ts_target > now:
                keep.append(t)
                continue
            hit = self._actual_at(t.ts_target, t.signal)
            if hit is None:
                if now - t.ts_target <= tol:
                    keep.append(t)
                continue
            actual, _ = hit
            ex = TrainingExample(
                ts_made=t.ts_made, ts_target=t.ts_target, signal=t.signal,
                horizon_s=t.horizon_s, x=t.x, anchor=t.anchor, actual=actual,
                extras=t.extras, vol=t.vol, regime=t.regime,
                prediction_id=t.prediction_id,
            )
            if self.adapt is not None:
                # abs_error here is the *learned model's* current error on this
                # example, used only for the error-budget policy. It is measured
                # against the live model, not against whatever version made the
                # original prediction.
                self.adapt.observe_outcome(ex, self._current_abs_error(ex))
            n += 1
        self._pending_targets = keep
        return n

    def _current_abs_error(self, ex: TrainingExample) -> float:
        if ex.extras is None:
            return 0.0
        pred = self.learned.predict_from_parts(
            ex.signal, ex.horizon_s, ex.x, ex.extras, ex.anchor, ex.vol, ex.regime
        )
        return 0.0 if (pred is None or not math.isfinite(pred)) else abs(pred - ex.actual)

    def _observe_drift(self, now: float, p: _Pending, abs_err: float) -> None:
        if self.drift is None:
            return
        for ev in self.drift.observe_error(now, p.signal, p.horizon_s, abs_err):
            self.repos.events.add_drift(**ev.as_row(self.run_id))
            if self.adapt is not None:
                self.adapt.note_drift(ev.signal, ev.horizon_s, ev.detector)

    # -- warm-up / bootstrap ----------------------------------------------
    def bootstrap_from_outcomes(self, examples: Sequence[TrainingExample], now: float) -> dict[str, int]:
        """Fit the learned model on already-resolved outcomes.

        Used by offline evaluation and by replay warm-up. Feeding these through
        the same buffer the online path uses keeps a single code path for
        training data, so a bug cannot affect one arm and not the other.
        """
        applied: dict[str, int] = {}
        by_scope: dict[str, list[TrainingExample]] = {}
        for ex in examples:
            by_scope.setdefault(self.scope(ex.signal, ex.horizon_s), []).append(ex)
        for scope, batch in by_scope.items():
            batch.sort(key=lambda e: e.ts_target)
            n = self.learned.fit(batch)
            applied[scope] = n
            if self.adapt is not None:
                for ex in batch:
                    self.adapt.buffer(scope).add(ex)
                sig, _, hor = scope.rpartition("@")
                self.adapt.register_initial(
                    sig, float(hor), now, n, {"source": "bootstrap"}
                )
        return applied

    # -- reporting ---------------------------------------------------------
    def report(self) -> dict[str, object]:
        return {
            "stats": self.stats.as_dict(),
            "pending": len(self._pending),
            "pending_targets": len(self._pending_targets),
            "regime": self.regimes.label,
            "regime_distribution": self.regimes.recent_distribution(),
            "targets": self.targets,
            "missing_targets": self.missing_targets,
            "horizons_s": self.horizons,
            "features": {
                "n": self.extractor.n_features,
                "inputs": self.extractor.inputs,
                "dropped": self.extractor.dropped_inputs,
                "schema_hash": self.extractor.schema_hash,
                "window_span_s": round(self.extractor.span_s, 1),
                "ready": self.extractor.ready,
            },
            "learned": self.learned.describe(),
            "baselines": [b.describe() for b in self.baselines],
            "calibration": self.calibrator.report(),
            "drift": self.drift.state() if self.drift else None,
            "adapt": self.adapt.report() if self.adapt else None,
        }


def _ema(prev: float, x: float, alpha: float = 0.1) -> float:
    return x if prev == 0.0 else (1 - alpha) * prev + alpha * x
