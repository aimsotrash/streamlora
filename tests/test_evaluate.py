"""Evaluation: metric correctness, chronological splits, embargo, reporting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from streamlora.evaluate import metrics as M
from streamlora.evaluate import report as R
from streamlora.evaluate.splits import chronological_split, rolling_origin_splits
from streamlora.store.repo import PredictionRecord

from .conftest import T0


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_metrics_match_hand_computed_values():
    actual = [10.0, 20.0, 30.0]
    pred = [12.0, 18.0, 33.0]
    m = M.compute(actual, pred)
    assert m.n == 3
    assert m.mae == pytest.approx((2 + 2 + 3) / 3)
    assert m.rmse == pytest.approx(math.sqrt((4 + 4 + 9) / 3))
    assert m.bias == pytest.approx((2 - 2 + 3) / 3)
    assert m.max_abs_error == pytest.approx(3.0)


def test_skill_is_relative_to_the_reference():
    m = M.compute([0.0] * 10, [1.0] * 10, reference_mae=2.0, reference_name="persistence")
    assert m.mae == pytest.approx(1.0)
    assert m.skill == pytest.approx(0.5)
    assert m.reference == "persistence"


def test_coverage_and_width_are_reported_together():
    actual = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 2.0, 3.0, 4.0]
    lo = [0.0, 1.0, 2.0, 10.0]
    hi = [2.0, 3.0, 4.0, 12.0]
    m = M.compute(actual, pred, lo, hi)
    assert m.coverage == pytest.approx(0.75)
    assert m.n_intervals == 4
    assert m.mean_interval_width == pytest.approx(2.0)


def test_intervals_are_optional():
    m = M.compute([1.0, 2.0], [1.0, 2.0], [None, None], [None, None])
    assert m.coverage is None and m.n_intervals == 0


def test_non_finite_rows_are_dropped_not_propagated():
    m = M.compute([1.0, float("nan"), 3.0], [1.0, 5.0, 3.0])
    assert m.n == 2
    assert m.mae == pytest.approx(0.0)


def test_smape_is_bounded_and_survives_zeros():
    v = M.smape(np.array([0.0, 5.0]), np.array([4.0, 5.0]))
    assert 0.0 <= v <= 200.0
    assert math.isfinite(v)


def test_diebold_mariano_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    assert M.diebold_mariano(a, a) is None            # identical -> zero variance
    b = rng.normal(0, 4, 500)
    stat, p = M.diebold_mariano(a, b)
    assert p < 0.01 and stat < 0
    # Two independent draws from the same distribution should not look different.
    c = rng.normal(0, 1, 500)
    d = rng.normal(0, 1, 500)
    res = M.diebold_mariano(c, d)
    assert res is None or res[1] > 0.01


def test_diebold_mariano_needs_enough_samples():
    assert M.diebold_mariano([1.0] * 10, [2.0] * 10) is None


def test_calibration_curve_bins_by_width():
    rng = np.random.default_rng(2)
    n = 200
    actual = rng.normal(0, 1, n)
    width = np.linspace(0.5, 6.0, n)
    lo = -width / 2
    hi = width / 2
    bins = M.calibration_curve(actual, list(lo), list(hi), bins=4)
    assert len(bins) >= 2
    # Wider intervals must cover more; that is the whole point of the view.
    assert bins[-1].coverage >= bins[0].coverage


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def test_chronological_split_never_reorders():
    ts = np.arange(1000) * 5.0 + T0
    s = chronological_split(ts, 0.6, 0.2, horizon_s=0.0)
    assert s.train.max() < s.validation.min() < s.test.min()
    assert s.train.size + s.validation.size + s.test.size == 1000


def test_embargo_removes_rows_whose_target_lands_in_the_next_segment():
    ts = np.arange(1000) * 5.0 + T0
    horizon = 900.0
    s = chronological_split(ts, 0.6, 0.2, horizon_s=horizon)
    assert s.embargoed > 0
    gap = ts[s.validation.min()] - ts[s.train.max()]
    assert gap >= horizon
    gap2 = ts[s.test.min()] - ts[s.validation.max()]
    assert gap2 >= horizon


def test_split_rejects_nonsense_fractions():
    ts = np.arange(100) * 5.0
    with pytest.raises(ValueError):
        chronological_split(ts, 0.9, 0.2)
    with pytest.raises(ValueError):
        chronological_split(ts[:5], 0.6, 0.2)


def test_rolling_origin_always_trains_before_testing():
    ts = np.arange(1200) * 5.0 + T0
    folds = list(rolling_origin_splits(ts, n_folds=4, horizon_s=300.0))
    assert len(folds) == 4
    prev_end = -1
    for f in folds:
        assert f.train.max() < f.test.min()
        assert ts[f.test.min()] - ts[f.train.max()] >= 300.0
        assert f.test.min() > prev_end
        prev_end = f.test.min()


def test_expanding_window_grows_and_sliding_does_not():
    ts = np.arange(1200) * 5.0 + T0
    exp = list(rolling_origin_splits(ts, n_folds=3, expanding=True))
    sli = list(rolling_origin_splits(ts, n_folds=3, expanding=False))
    assert exp[-1].train.size > exp[0].train.size
    assert sli[-1].train.size == pytest.approx(sli[0].train.size, rel=0.1)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _rec(i, kind, err, signal="cpu.util_pct", horizon=300.0, regime="idle"):
    return PredictionRecord(
        id=i, ts_made=T0 + i * 5, ts_target=T0 + i * 5 + horizon, horizon_s=horizon,
        signal=signal, value=50.0 + err, lo=None, hi=None, model_kind=kind,
        model_version="v1", regime=regime, anchor=50.0, feature_id=None,
        resolved=True, actual=50.0, error=err, abs_error=abs(err),
    )


def test_summarize_groups_and_scores_against_the_reference():
    recs = [_rec(i, "persistence", 4.0) for i in range(50)]
    recs += [_rec(i + 100, "rls", 2.0) for i in range(50)]
    rows = R.summarize(recs, reference="persistence")
    by_arm = {r.arm: r for r in rows}
    assert by_arm["persistence"].m.mae == pytest.approx(4.0)
    assert by_arm["rls"].m.mae == pytest.approx(2.0)
    assert by_arm["rls"].m.skill == pytest.approx(0.5)
    # The reference scores exactly 0.0 against itself, by definition.
    assert by_arm["persistence"].m.skill == pytest.approx(0.0)
    assert by_arm["rls"].m.reference == "persistence"


def test_summarize_by_regime_slices_correctly():
    recs = [_rec(i, "rls", 1.0, regime="idle") for i in range(40)]
    recs += [_rec(i + 100, "rls", 9.0, regime="build") for i in range(40)]
    recs += [_rec(i + 200, "persistence", 5.0, regime="idle") for i in range(40)]
    rows = R.summarize(recs, by_regime=True, dm_test=False)
    got = {(r.arm, r.regime): r.m.mae for r in rows}
    assert got[("rls", "idle")] == pytest.approx(1.0)
    assert got[("rls", "build")] == pytest.approx(9.0)


def test_summarize_separates_horizons():
    recs = [_rec(i, "rls", 1.0, horizon=300.0) for i in range(30)]
    recs += [_rec(i + 100, "rls", 5.0, horizon=1800.0) for i in range(30)]
    rows = R.summarize(recs, dm_test=False)
    got = {r.horizon_s: r.m.mae for r in rows}
    assert got[300.0] == pytest.approx(1.0)
    assert got[1800.0] == pytest.approx(5.0)


def test_over_time_windows_show_a_trend():
    recs = []
    for i in range(200):
        err = 10.0 if i < 100 else 1.0     # model improves halfway through
        recs.append(_rec(i, "rls", err))
    rows = R.summarize_over_time(recs, n_windows=4)
    maes = [r.m.mae for r in sorted(rows, key=lambda r: r.window)]
    assert maes[0] > maes[-1]


def test_table_renders_without_data():
    assert "no resolved" in R.to_table([])


def test_unresolved_predictions_are_excluded():
    r = _rec(1, "rls", 1.0)
    r.resolved = False
    r.actual = None
    assert R.summarize([r]) == []
