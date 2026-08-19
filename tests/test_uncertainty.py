"""Prediction intervals: coverage, adaptation to variance shift, honesty."""

from __future__ import annotations

import numpy as np
import pytest

from streamlora.config import UncertaintyConfig
from streamlora.forecast.uncertainty import ConformalCalibrator


def _drive(cal, truths, kind="rls", sig="cpu.util_pct", hor=300.0):
    covered = []
    for t in truths:
        lo, hi, _a = cal.interval(kind, sig, hor, 0.0)
        if lo is None:
            cal.observe(kind, sig, hor, abs(t), None)
            continue
        c = bool(lo <= t <= hi)
        covered.append(c)
        cal.observe(kind, sig, hor, abs(t), c)
    return np.array(covered)


def test_no_interval_before_enough_residuals():
    cal = ConformalCalibrator(UncertaintyConfig(min_residuals=30))
    lo, hi, _ = cal.interval("rls", "cpu.util_pct", 300.0, 50.0)
    # Fabricating an interval from five residuals is worse than saying nothing.
    assert lo is None and hi is None


def test_long_run_coverage_matches_the_nominal_level():
    rng = np.random.default_rng(0)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=0.1, window=300, step=0.02,
                                                min_residuals=30))
    cov = _drive(cal, rng.normal(0, 1, 4000))
    assert 0.86 <= cov.mean() <= 0.94


@pytest.mark.parametrize("alpha,lo,hi", [(0.05, 0.91, 0.99), (0.2, 0.74, 0.86)])
def test_coverage_tracks_the_requested_alpha(alpha, lo, hi):
    rng = np.random.default_rng(1)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=alpha, window=400, step=0.02,
                                                min_residuals=30))
    cov = _drive(cal, rng.normal(0, 1, 5000))
    assert lo <= cov.mean() <= hi


def test_coverage_recovers_after_a_variance_shift():
    """The reason adaptive conformal was chosen over split conformal."""
    rng = np.random.default_rng(3)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=0.1, window=300, step=0.02,
                                                min_residuals=30))
    truths = np.concatenate([rng.standard_t(3, 1500), rng.standard_t(3, 1500) * 6.0])
    cov = _drive(cal, truths)
    n = len(cov)
    late = cov[int(n * 0.75):]
    assert late.mean() >= 0.85


def test_intervals_widen_under_heavier_noise():
    rng = np.random.default_rng(4)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=0.1, window=200, min_residuals=20))
    for t in rng.normal(0, 1, 400):
        lo, hi, _ = cal.interval("rls", "s", 300.0, 0.0)
        cal.observe("rls", "s", 300.0, abs(t), None if lo is None else bool(lo <= t <= hi))
    narrow = cal.interval("rls", "s", 300.0, 0.0)
    for t in rng.normal(0, 10, 400):
        lo, hi, _ = cal.interval("rls", "s", 300.0, 0.0)
        cal.observe("rls", "s", 300.0, abs(t), None if lo is None else bool(lo <= t <= hi))
    wide = cal.interval("rls", "s", 300.0, 0.0)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0]) * 3


def test_intervals_are_centred_on_the_point_forecast():
    rng = np.random.default_rng(5)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=0.1, min_residuals=20, window=100))
    for t in rng.normal(0, 1, 200):
        cal.observe("rls", "s", 300.0, abs(t), True)
    lo, hi, _ = cal.interval("rls", "s", 300.0, 42.0)
    assert (lo + hi) / 2 == pytest.approx(42.0)


def test_scopes_are_calibrated_independently():
    rng = np.random.default_rng(6)
    cal = ConformalCalibrator(UncertaintyConfig(alpha=0.1, min_residuals=20, window=100))
    for t in rng.normal(0, 1, 200):
        cal.observe("rls", "cpu.util_pct", 300.0, abs(t), True)
    for t in rng.normal(0, 20, 200):
        cal.observe("rls", "cpu.util_pct", 1800.0, abs(t), True)
    short = cal.interval("rls", "cpu.util_pct", 300.0, 0.0)
    long_ = cal.interval("rls", "cpu.util_pct", 1800.0, 0.0)
    assert (long_[1] - long_[0]) > (short[1] - short[0]) * 3


def test_disabling_uncertainty_yields_no_intervals():
    cal = ConformalCalibrator(UncertaintyConfig(method="none"))
    assert cal.interval("rls", "s", 300.0, 1.0) == (None, None, None)
    cal.observe("rls", "s", 300.0, 1.0, True)
    assert cal.report() == {}


def test_report_is_serialisable():
    import json
    cal = ConformalCalibrator(UncertaintyConfig(min_residuals=5, window=50))
    for i in range(60):
        cal.observe("rls", "s", 300.0, float(i % 7), True)
    json.dumps(cal.report())
