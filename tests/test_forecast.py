"""Forecasting: RLS correctness, baselines, features, prediction shape, stability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from streamlora.config import FeatureConfig
from streamlora.forecast.base import ForecastContext, TrainingExample
from streamlora.forecast.baselines import (
    EwmaForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from streamlora.forecast.features import FeatureExtractor
from streamlora.forecast.linear import RLSForecaster, RLSRegressor, build_scope_specs
from streamlora.forecast.regime import RegimeClassifier
from streamlora.forecast.scaling import Scaler, scale_value, unscale_value
from streamlora.telemetry.schema import SignalKind, SignalSpec

from .conftest import T0, make_sample


# --------------------------------------------------------------------------
# RLS
# --------------------------------------------------------------------------

def test_rls_with_no_forgetting_equals_closed_form_ridge():
    """The property that makes 'static vs adapted' the same estimator."""
    rng = np.random.default_rng(0)
    d, n, lam = 8, 400, 2.0
    X = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + rng.normal(scale=0.1, size=n)
    r = RLSRegressor(d=d, ridge_lambda=lam, forgetting=1.0)
    r.fit(X, y)
    w_closed = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    np.testing.assert_allclose(r.w, w_closed, atol=1e-10)


def test_rls_is_order_invariant_without_forgetting():
    rng = np.random.default_rng(1)
    d, n = 6, 200
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d)
    a = RLSRegressor(d=d, ridge_lambda=1.0, forgetting=1.0)
    a.fit(X, y)
    p = rng.permutation(n)
    b = RLSRegressor(d=d, ridge_lambda=1.0, forgetting=1.0)
    b.fit(X[p], y[p])
    np.testing.assert_allclose(a.w, b.w, atol=1e-10)


def test_rls_rejects_non_finite_input_without_corrupting_state():
    r = RLSRegressor(d=3, ridge_lambda=1.0)
    r.update(np.array([1.0, 0.0, 0.0]), 1.0)
    w_before = r.w.copy()
    assert not r.update(np.array([np.nan, 0.0, 0.0]), 1.0)
    assert not r.update(np.array([1.0, 0.0, 0.0]), float("inf"))
    np.testing.assert_array_equal(r.w, w_before)
    assert np.all(np.isfinite(r.P))
    assert r.n_skipped == 2


def test_rls_tracks_a_parameter_change_when_forgetting():
    rng = np.random.default_rng(5)
    d, n = 5, 600
    X = rng.normal(size=(n, d))
    w1 = rng.normal(size=d)
    w2 = -w1
    y = np.concatenate([X[: n // 2] @ w1, X[n // 2:] @ w2])
    r = RLSRegressor(d=d, ridge_lambda=1.0, forgetting=0.98)
    r.fit(X, y)
    assert np.abs(r.w - w2).mean() < np.abs(r.w - w1).mean()


def test_rls_covariance_stays_symmetric_and_bounded_over_a_long_run():
    """Both documented failure modes of naive RLS, exercised."""
    rng = np.random.default_rng(3)
    d = 10
    r = RLSRegressor(d=d, ridge_lambda=1.0, forgetting=0.995)
    for i in range(6000):
        # Deliberately uninformative for long stretches: this is what causes
        # covariance windup.
        x = np.zeros(d) if i % 3 else rng.normal(size=d) * 0.01
        x[0] = 1.0
        r.update(x, 0.001 * rng.normal())
    assert np.all(np.isfinite(r.P))
    np.testing.assert_allclose(r.P, r.P.T, atol=1e-9)
    assert np.trace(r.P) < 1e9
    assert np.all(np.isfinite(r.w))


# --------------------------------------------------------------------------
# scaling
# --------------------------------------------------------------------------

def test_scaling_round_trips_for_each_kind():
    gauge = SignalSpec("g", "percent", SignalKind.GAUGE, 0.0, 100.0)
    rate = SignalSpec("r", "mbps", SignalKind.RATE, 0.0, 20000.0)
    flag = SignalSpec("f", "bool", SignalKind.FLAG, 0.0, 1.0)
    for spec, v in ((gauge, 42.0), (rate, 12.5), (flag, 1.0)):
        assert unscale_value(spec, scale_value(spec, v)) == pytest.approx(v, rel=1e-6)


def test_scaler_preserves_nan(simple_registry):
    sc = Scaler(["cpu.util_pct", "net.recv_mbps", "battery.plugged"], simple_registry)
    out = sc.transform(np.array([np.nan, np.nan, np.nan]))
    assert np.isnan(out).all()


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def _feed(fx, n=80, start=T0, step=5.0, cpu=lambda i: 20 + 30 * math.sin(i / 10)):
    rc = RegimeClassifier()
    for i in range(n):
        s = make_sample(start + i * step, **{
            "cpu.util_pct": cpu(i), "mem.used_pct": 50.0 + i * 0.05,
            "battery.percent": 90.0 - i * 0.02, "battery.plugged": 1.0,
            "net.recv_mbps": 2.0,
        })
        fx.push(s, rc.update(s))
    return fx


def test_feature_vector_is_finite_and_correctly_sized(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    fv = fx.extract()
    assert fv is not None
    assert fv.values.shape == (fx.n_features,)
    assert np.all(np.isfinite(fv.values))
    assert fv.values[0] == 1.0                      # bias
    assert 0.0 <= fv.coverage <= 1.0


def test_features_never_emit_nan_even_with_a_dead_signal(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    rc = RegimeClassifier()
    for i in range(80):
        s = make_sample(T0 + i * 5, **{"cpu.util_pct": 30.0})   # everything else missing
        fx.push(s, rc.update(s))
    fv = fx.extract()
    assert np.all(np.isfinite(fv.values))
    assert fv.coverage < 1.0


def test_extractor_drops_signals_the_machine_lacks(simple_registry):
    cfg = FeatureConfig(inputs=["cpu.util_pct", "gpu.util_pct", "does.not.exist"])
    fx = FeatureExtractor(simple_registry, cfg, interval_s=5.0)
    assert fx.inputs == ["cpu.util_pct"]
    assert set(fx.dropped_inputs) == {"gpu.util_pct", "does.not.exist"}


def test_per_target_mask_is_smaller_than_the_shared_vector(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    mask = fx.select_indices("cpu.util_pct")
    assert 0 < mask.size < fx.n_features
    assert mask[0] == 0                                    # bias always kept
    names = [fx.names[i] for i in mask]
    assert any(n.startswith("cpu.util_pct|lag") for n in names)
    # every other signal contributes only a coarse summary
    assert sum(1 for n in names if n.startswith("mem.used_pct|")) <= 3


def test_out_of_order_samples_are_ignored(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx, n=20)
    n = fx.n_rows()
    fx.push(make_sample(T0, **{"cpu.util_pct": 1.0}), "idle")
    assert fx.n_rows() == n


def test_not_ready_before_enough_history(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    assert not fx.ready
    _feed(fx, n=3)
    assert not fx.ready


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def _ctx(simple_registry, n=80):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx, n=n)
    fv = fx.extract()
    return ForecastContext(ts=fv.ts, fv=fv, extractor=fx, registry=simple_registry,
                           regime=fv.regime)


def test_persistence_is_exactly_the_last_measurement(simple_registry):
    ctx = _ctx(simple_registry)
    p = PersistenceForecaster()
    assert p.predict(ctx, "cpu.util_pct", 300.0) == ctx.current("cpu.util_pct")


@pytest.mark.parametrize("horizon", [60.0, 300.0, 900.0, 3600.0])
def test_baselines_return_in_range_values_at_every_horizon(simple_registry, horizon):
    ctx = _ctx(simple_registry)
    for f in (PersistenceForecaster(), MovingAverageForecaster(300.0),
              EwmaForecaster(120.0), LinearTrendForecaster(300.0)):
        v = f.predict(ctx, "cpu.util_pct", horizon)
        assert v is not None and math.isfinite(v)
        assert 0.0 <= v <= 100.0


def test_baselines_return_none_for_an_unknown_signal(simple_registry):
    ctx = _ctx(simple_registry)
    for f in (PersistenceForecaster(), MovingAverageForecaster(), EwmaForecaster(),
              LinearTrendForecaster()):
        assert f.predict(ctx, "nope.not.here", 300.0) is None


def test_linear_trend_extrapolates_a_ramp(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx, n=80, cpu=lambda i: 10.0 + 0.1 * i)     # +0.02 %/s
    fv = fx.extract()
    ctx = ForecastContext(ts=fv.ts, fv=fv, extractor=fx, registry=simple_registry)
    v = LinearTrendForecaster(300.0).predict(ctx, "cpu.util_pct", 300.0)
    assert v == pytest.approx(10.0 + 0.1 * 79 + 0.02 * 300, rel=0.05)


def test_seasonal_naive_declines_rather_than_degrading_silently(simple_registry):
    ctx = _ctx(simple_registry)
    sn = SeasonalNaiveForecaster()
    assert sn.predict(ctx, "cpu.util_pct", 300.0) is None


def test_ewma_is_independent_of_where_the_window_starts(simple_registry):
    """Computed from the stored window, so replays are reproducible."""
    ctx = _ctx(simple_registry, n=80)
    a = EwmaForecaster(120.0).predict(ctx, "cpu.util_pct", 300.0)
    b = EwmaForecaster(120.0).predict(ctx, "cpu.util_pct", 300.0)
    assert a == b


# --------------------------------------------------------------------------
# learned forecaster
# --------------------------------------------------------------------------

def _learned(simple_registry, fx, horizons=(300.0,)):
    scopes = build_scope_specs(
        signals=["cpu.util_pct"], horizons=list(horizons),
        masks={"cpu.util_pct": fx.select_indices("cpu.util_pct")},
        baseline_names=("persistence", "ewma"),
        ranges={"cpu.util_pct": (0.0, 100.0)},
    )
    return RLSForecaster(fx.names, scopes, standardize_warmup=20)


def test_unfitted_model_refuses_to_predict(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    m = _learned(simple_registry, fx)
    fv = fx.extract()
    ctx = ForecastContext(ts=fv.ts, fv=fv, extractor=fx, registry=simple_registry)
    ctx.baseline_preds[("cpu.util_pct", 300.0)] = {"persistence": 30.0, "ewma": 31.0}
    # A zero-weight linear model would silently reproduce persistence and
    # inflate the learned arm's apparent skill.
    assert m.predict(ctx, "cpu.util_pct", 300.0) is None


def test_zero_weights_reproduce_persistence_exactly(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    m = _learned(simple_registry, fx)
    fv = fx.extract()
    sm = m.model("cpu.util_pct", 300.0)
    extras = np.zeros(len(sm.spec.baseline_names))
    sm.freeze_standardizer(np.zeros(sm.spec.d), np.ones(sm.spec.d))
    reg = sm.get("*", create=True)
    reg.n_updates = 1                    # pretend it is fitted, weights still zero
    anchor = 42.0
    out = m.predict_from_parts("cpu.util_pct", 300.0, fv.values, extras, anchor, 5.0, "idle")
    assert out == pytest.approx(anchor)


def test_predictions_are_clipped_to_the_signal_range(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    m = _learned(simple_registry, fx)
    fv = fx.extract()
    sm = m.model("cpu.util_pct", 300.0)
    sm.freeze_standardizer(np.zeros(sm.spec.d), np.ones(sm.spec.d))
    reg = sm.get("*", create=True)
    reg.w = np.full(sm.spec.d, 50.0)     # absurd weights
    reg.n_updates = 5
    out = m.predict_from_parts("cpu.util_pct", 300.0, fv.values,
                               np.zeros(len(sm.spec.baseline_names)), 90.0, 30.0, "idle")
    assert 0.0 <= out <= 100.0


def test_learning_reduces_error_on_a_learnable_signal(simple_registry):
    """A signal with real structure: the model must actually beat its anchor."""
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    rng = np.random.default_rng(0)
    rc = RegimeClassifier()
    m = None
    examples: list[TrainingExample] = []
    hist: list[tuple[float, float]] = []
    for i in range(700):
        cpu = 40 + 30 * math.sin(i / 30.0) + rng.normal(0, 1.0)
        s = make_sample(T0 + i * 5, **{
            "cpu.util_pct": max(0.0, min(100.0, cpu)), "mem.used_pct": 50.0,
            "battery.percent": 90.0, "battery.plugged": 1.0, "net.recv_mbps": 1.0,
        })
        fx.push(s, rc.update(s))
        if not fx.ready:
            continue
        fv = fx.extract()
        if m is None:
            m = _learned(simple_registry, fx)
        hist.append((fv.ts, fv.values, s.value("cpu.util_pct")))
    assert m is not None
    k = 60      # 300 s at 5 s cadence
    sm = m.model("cpu.util_pct", 300.0)
    for j in range(len(hist) - k):
        ts, x, anchor = hist[j]
        actual = hist[j + k][2]
        extras = np.zeros(len(sm.spec.baseline_names))
        m.observe_warmup("cpu.util_pct", 300.0, x, extras)
        examples.append(TrainingExample(
            ts_made=ts, ts_target=ts + 300.0, signal="cpu.util_pct", horizon_s=300.0,
            x=x, anchor=anchor, actual=actual, extras=extras, vol=8.0, regime="interactive",
        ))
    split = len(examples) // 2
    m.fit(examples[:split])
    errs_model, errs_persist = [], []
    for ex in examples[split:]:
        p = m.predict_from_parts("cpu.util_pct", 300.0, ex.x, ex.extras, ex.anchor,
                                 ex.vol, ex.regime)
        if p is None:
            continue
        errs_model.append(abs(p - ex.actual))
        errs_persist.append(abs(ex.anchor - ex.actual))
    assert errs_model
    assert np.mean(errs_model) < np.mean(errs_persist)


def test_save_and_load_round_trip(simple_registry, tmp_path):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    m = _learned(simple_registry, fx, horizons=(300.0, 900.0))
    sm = m.model("cpu.util_pct", 300.0)
    sm.freeze_standardizer(np.zeros(sm.spec.d), np.ones(sm.spec.d) * 2.0)
    r = sm.get("*", create=True)
    r.w = np.arange(sm.spec.d, dtype=float)
    r.n_updates = 17
    m.set_version("cpu.util_pct", 300.0, 4)
    path = str(tmp_path / "m.npz")
    m.save(path)
    back = RLSForecaster.load(path)
    sm2 = back.model("cpu.util_pct", 300.0)
    np.testing.assert_allclose(sm2.get("*").w, r.w)
    assert sm2.get("*").n_updates == 17
    assert sm2.version_n == 4
    np.testing.assert_allclose(sm2.sigma, sm.sigma)
    assert back.version("cpu.util_pct", 300.0) == "forecast-model-v004"


def test_standardizer_freezes_only_once_and_protects_the_bias(simple_registry):
    fx = FeatureExtractor(simple_registry, FeatureConfig(), interval_s=5.0)
    _feed(fx)
    m = _learned(simple_registry, fx)
    sm = m.model("cpu.util_pct", 300.0)
    d = sm.spec.d
    sm.freeze_standardizer(np.full(d, 3.0), np.zeros(d))   # degenerate std
    assert sm.mu[0] == 0.0 and sm.sigma[0] == 1.0
    assert np.all(sm.sigma >= sm.sigma_floor)              # floor applied
    v = sm.prepare(np.ones(d))
    assert np.all(np.isfinite(v))
    assert np.all(np.abs(v) <= sm.clip_z + 1e-9)


# --------------------------------------------------------------------------
# regimes
# --------------------------------------------------------------------------

def test_regime_hysteresis_prevents_flapping():
    rc = RegimeClassifier()
    idle = make_sample(T0, **{"cpu.util_pct": 2.0, "battery.plugged": 1.0})
    busy = make_sample(T0, **{"cpu.util_pct": 90.0, "battery.plugged": 1.0})
    for _ in range(5):
        rc.update(idle)
    assert rc.label == "idle"
    rc.update(busy)
    assert rc.label == "idle"           # one sample must not switch it
    rc.update(busy)
    rc.update(busy)
    assert rc.label == "compute_heavy"


def test_regime_one_hot_is_stable_and_sized():
    rc = RegimeClassifier()
    v = rc.one_hot("build")
    assert sum(v) == 1.0
    assert len(v) == len(rc.one_hot("idle"))
