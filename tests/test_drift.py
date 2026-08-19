"""Drift detection: no false alarms on noise, reliable alarms on real change."""

from __future__ import annotations

import numpy as np
import pytest

from streamlora.config import DriftConfig
from streamlora.drift.controller import DriftController
from streamlora.drift.detectors import AdwinLite, FeatureShiftDetector, PageHinkley

from .conftest import T0


def _fires(det, xs):
    return [i for i, x in enumerate(xs) if det.update(x).fired]


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_page_hinkley_no_false_alarms_on_stationary_noise(seed):
    rng = np.random.default_rng(seed)
    xs = np.abs(rng.normal(5.0, 1.5, size=3000))
    assert _fires(PageHinkley(), xs) == []


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_adwin_no_false_alarms_on_stationary_noise(seed):
    """A plain Hoeffding bound gave ~11 false alarms here; the variance-aware
    ADWIN2 bound gives none."""
    rng = np.random.default_rng(seed)
    xs = np.abs(rng.normal(5.0, 1.5, size=3000))
    assert _fires(AdwinLite(), xs) == []


@pytest.mark.parametrize("detector", [PageHinkley, AdwinLite])
@pytest.mark.parametrize("seed", [1, 7, 23])
def test_detectors_fire_promptly_on_a_sustained_error_increase(detector, seed):
    rng = np.random.default_rng(seed)
    xs = np.concatenate([
        np.abs(rng.normal(5.0, 1.5, size=600)),
        np.abs(rng.normal(11.0, 1.5, size=600)),
    ])
    fires = _fires(detector(), xs)
    assert fires, f"{detector.__name__} missed a 4-sigma shift"
    delay = fires[0] - 600
    assert 0 <= delay <= 60, f"{detector.__name__} delay {delay} samples"


def test_detectors_catch_a_subtler_one_sigma_shift_eventually():
    rng = np.random.default_rng(5)
    xs = np.concatenate([
        np.abs(rng.normal(5.0, 1.5, size=600)),
        np.abs(rng.normal(6.5, 1.5, size=1200)),
    ])
    for det in (PageHinkley(), AdwinLite()):
        fires = _fires(det, xs)
        assert fires and fires[0] >= 600
        assert fires[0] - 600 <= 300


def test_a_constant_stream_never_alarms():
    """Perfect prediction gives zero variance; without a floor on the reference
    std the first non-zero error is an infinite z-score."""
    xs = [0.0] * 500 + [0.0001] * 500
    assert _fires(PageHinkley(), xs) == []
    assert _fires(AdwinLite(), xs) == []


def test_detectors_re_reference_after_an_alarm():
    rng = np.random.default_rng(11)
    xs = np.concatenate([
        np.abs(rng.normal(5.0, 1.0, size=400)),
        np.abs(rng.normal(15.0, 1.0, size=1500)),
    ])
    ph = PageHinkley()
    fires = _fires(ph, xs)
    # After the shift becomes the new normal, it must stop alarming about it.
    assert fires
    assert len([f for f in fires if f > fires[0] + 400]) == 0


def test_feature_shift_detects_a_moved_input_and_names_it():
    rng = np.random.default_rng(3)
    fs = FeatureShiftDetector(window=100, threshold=4.0)
    names = ["bias", "cpu|lag0", "mem|lag0", "flag"]
    fired = []
    for i in range(1200):
        mu = 0.2 if i < 600 else 0.7
        v = [1.0, rng.normal(mu, 0.05), rng.normal(0.5, 0.05), 0.0]
        s = fs.update(v, names)
        if s.fired:
            fired.append((i, s.detail.get("feature")))
    assert fired
    assert fired[0][0] > 600
    assert fired[0][1] == "cpu|lag0"


def test_feature_shift_ignores_constant_columns():
    """A constant column has std at the floor; dividing by it manufactures
    enormous z-scores out of numerical dust."""
    rng = np.random.default_rng(4)
    fs = FeatureShiftDetector(window=80, threshold=4.0)
    fired = []
    for i in range(600):
        v = [1.0, 0.0, rng.normal(0.5, 0.05)]
        if fs.update(v, ["bias", "always_zero", "noisy"]).fired:
            fired.append(i)
    assert fired == []


def test_feature_shift_restarts_if_the_feature_space_changes():
    fs = FeatureShiftDetector(window=30, threshold=4.0)
    for _ in range(60):
        fs.update([1.0, 0.5], ["bias", "a"])
    s = fs.update([1.0, 0.5, 0.2], ["bias", "a", "b"])   # new hardware appeared
    assert not s.fired
    assert s.detail.get("state") == "warmup"


# --------------------------------------------------------------------------
# controller
# --------------------------------------------------------------------------

def test_controller_cooldown_collapses_an_alarm_storm():
    cfg = DriftConfig(cooldown_s=300.0, ph_threshold=5.0, adwin_enabled=False)
    c = DriftController(cfg)
    rng = np.random.default_rng(0)
    for i in range(200):
        c.observe_error(T0 + i, "cpu.util_pct", 300.0, abs(rng.normal(2.0, 0.5)))
    events = []
    for i in range(200, 400):
        events += c.observe_error(T0 + i, "cpu.util_pct", 300.0, abs(rng.normal(40.0, 0.5)))
    # 200 samples at 1 s with a 300 s cooldown can produce at most one alarm.
    assert len(events) <= 1
    assert c.n_suppressed >= 0


def test_controller_records_scope_and_can_be_disabled():
    c = DriftController(DriftConfig(cooldown_s=0.0, ph_threshold=1.0, adwin_enabled=False))
    rng = np.random.default_rng(1)
    for i in range(120):
        c.observe_error(T0 + i, "cpu.util_pct", 300.0, abs(rng.normal(1.0, 0.2)))
    got = []
    for i in range(120, 300):
        got += c.observe_error(T0 + i, "cpu.util_pct", 300.0, abs(rng.normal(20.0, 0.2)))
    assert got
    assert got[0].signal == "cpu.util_pct" and got[0].horizon_s == 300.0
    assert got[0].scope == "error"

    off = DriftController(DriftConfig(enabled=False))
    assert off.observe_error(T0, "cpu.util_pct", 300.0, 1e6) == []
    assert off.observe_features(T0, [1.0, 2.0]) == []


def test_controller_state_is_serialisable():
    c = DriftController(DriftConfig())
    c.observe_error(T0, "cpu.util_pct", 300.0, 1.0)
    st = c.state()
    assert "error" in st and "feature" in st
    import json
    json.dumps(st)      # must not contain numpy scalars or objects
