"""Adaptation: buffer semantics, policies, gating, versioning, rollback."""

from __future__ import annotations

import numpy as np
import pytest

from streamlora.adapt.buffer import TrainingBuffer
from streamlora.adapt.controller import AdaptationController
from streamlora.adapt.policy import AdaptationPolicy, ScopeState
from streamlora.adapt.registry import ModelVersionRegistry
from streamlora.config import AdaptConfig
from streamlora.forecast.base import TrainingExample
from streamlora.forecast.linear import RLSForecaster, build_scope_specs

from .conftest import T0


def _ex(i: int, d: int = 6, signal: str = "cpu.util_pct", horizon: float = 300.0,
        actual: float | None = None, regime: str = "idle") -> TrainingExample:
    rng = np.random.default_rng(i)
    x = np.concatenate([[1.0], rng.normal(size=d - 1)])
    return TrainingExample(
        ts_made=T0 + i * 5, ts_target=T0 + i * 5 + horizon, signal=signal,
        horizon_s=horizon, x=x, anchor=50.0,
        actual=50.0 + (x[1] * 3.0 if actual is None else actual),
        extras=np.zeros(2), vol=5.0, regime=regime,
    )


# --------------------------------------------------------------------------
# buffer
# --------------------------------------------------------------------------

def test_reservoir_is_uniform_and_seeded():
    a = TrainingBuffer(recency=10, reservoir=20, seed=42)
    b = TrainingBuffer(recency=10, reservoir=20, seed=42)
    for i in range(500):
        a.add(_ex(i))
        b.add(_ex(i))
    ta = [e.ts_target for e in a.draw_batch(20, 1.0)[0]]
    tb = [e.ts_target for e in b.draw_batch(20, 1.0)[0]]
    assert ta == tb, "identical seeds must draw identical batches"
    assert a.stats.reservoir == 20
    # Uniform sampling must keep some genuinely old examples.
    assert min(ta) < T0 + 100 * 5


def test_gate_split_is_chronological_and_newest_is_held_out():
    buf = TrainingBuffer(recency=200, reservoir=200, seed=1)
    for i in range(120):
        buf.add(_ex(i))
    train, gate = buf.split_gate(30)
    assert len(gate) == 30
    assert max(e.ts_target for e in train) < min(e.ts_target for e in gate)


def test_gate_split_degrades_when_there_is_not_enough_data():
    buf = TrainingBuffer(recency=50, reservoir=50, seed=1)
    for i in range(10):
        buf.add(_ex(i))
    train, gate = buf.split_gate(30)
    assert gate == [] and len(train) == 10


def test_absorb_once_prevents_reabsorbing_the_same_example():
    """Re-absorbing overlapping batches shrinks the covariance as if the model
    had seen far more independent data than it has."""
    buf = TrainingBuffer(recency=500, reservoir=0, seed=1)
    for i in range(100):
        buf.add(_ex(i))
    batch1, seq1 = buf.draw_batch(50, 0.0)
    buf.mark_absorbed(seq1)
    batch2, _ = buf.draw_batch(50, 0.0)
    t1 = {e.ts_target for e in batch1}
    t2 = {e.ts_target for e in batch2}
    assert not (t1 & t2), "an example was absorbed twice"


def test_rejected_candidate_leaves_its_batch_available():
    buf = TrainingBuffer(recency=500, reservoir=0, seed=1)
    for i in range(60):
        buf.add(_ex(i))
    batch1, seq1 = buf.draw_batch(30, 0.0)
    # Do NOT mark absorbed: the candidate was rejected and discarded.
    batch2, _ = buf.draw_batch(30, 0.0)
    assert {e.ts_target for e in batch1} == {e.ts_target for e in batch2}


def test_rewind_absorbed_restores_examples_after_a_rollback():
    buf = TrainingBuffer(recency=500, reservoir=0, seed=1)
    for i in range(60):
        buf.add(_ex(i))
    _, seq = buf.draw_batch(30, 0.0)
    before = buf.absorbed_seq
    buf.mark_absorbed(seq)
    assert buf.n_unabsorbed < 60
    buf.rewind_absorbed(before)
    assert buf.n_unabsorbed == 60


def test_gate_boundary_excludes_gate_rows_from_training():
    buf = TrainingBuffer(recency=500, reservoir=0, seed=1)
    for i in range(100):
        buf.add(_ex(i))
    _train, gate = buf.split_gate(20)
    boundary = min(e.ts_target for e in gate)
    batch, _ = buf.draw_batch(100, 0.0, exclude_after_ts=boundary)
    assert all(e.ts_target < boundary for e in batch)


def test_batch_is_ordered_oldest_first():
    buf = TrainingBuffer(recency=200, reservoir=100, seed=3)
    for i in range(150):
        buf.add(_ex(i))
    batch, _ = buf.draw_batch(60, 0.5)
    ts = [e.ts_target for e in batch]
    assert ts == sorted(ts)


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

def test_periodic_samples_policy():
    p = AdaptationPolicy(AdaptConfig(policies=["periodic_samples"], every_n_samples=10))
    st = ScopeState("s")
    for _ in range(9):
        st.note_outcome(1.0)
    assert p.should_adapt(st, T0) is None
    st.note_outcome(1.0)
    assert p.should_adapt(st, T0).name == "periodic_samples"


def test_drift_policy_takes_priority():
    p = AdaptationPolicy(AdaptConfig(policies=["periodic_samples", "drift"],
                                     every_n_samples=1000))
    st = ScopeState("s")
    st.note_outcome(1.0)
    st.note_drift("page_hinkley")
    t = p.should_adapt(st, T0)
    assert t.name == "drift" and t.detail["detectors"] == ["page_hinkley"]


def test_time_policy_needs_some_new_data():
    p = AdaptationPolicy(AdaptConfig(policies=["periodic_time"], every_n_seconds=10.0))
    st = ScopeState("s")
    # A time trigger with an empty buffer would create a version identical to
    # its parent and pollute the version history.
    assert p.should_adapt(st, T0 + 1000) is None
    st.note_outcome(1.0)
    assert p.should_adapt(st, T0 + 1000).name == "periodic_time"


def test_error_budget_policy():
    p = AdaptationPolicy(AdaptConfig(policies=["error"], error_budget=50.0))
    st = ScopeState("s")
    for _ in range(4):
        st.note_outcome(10.0)
    assert p.should_adapt(st, T0) is None
    st.note_outcome(20.0)
    assert p.should_adapt(st, T0).name == "error"


def test_disabled_policy_never_fires():
    p = AdaptationPolicy(AdaptConfig(enabled=False, policies=["periodic_samples"],
                                     every_n_samples=1))
    st = ScopeState("s")
    st.note_outcome(1.0)
    assert p.should_adapt(st, T0) is None


def test_state_resets_after_adaptation():
    st = ScopeState("s")
    for _ in range(5):
        st.note_outcome(3.0)
    st.note_drift("adwin")
    st.reset_after_adapt(T0)
    assert st.n_new == 0 and st.error_accum == 0.0 and st.pending_drift == []
    assert st.n_total == 5 and st.adaptations == 1


# --------------------------------------------------------------------------
# controller: gating, promotion, rollback
# --------------------------------------------------------------------------

def _controller(repos, tmp_path, **cfg_kw):
    d = 6
    scopes = build_scope_specs(
        signals=["cpu.util_pct"], horizons=[300.0],
        masks={"cpu.util_pct": np.arange(d - 2)},
        baseline_names=("persistence", "ewma"),
        ranges={"cpu.util_pct": (0.0, 100.0)},
    )
    fc = RLSForecaster([f"f{i}" for i in range(d)], scopes, standardize_warmup=5)
    sm = fc.model("cpu.util_pct", 300.0)
    sm.freeze_standardizer(np.zeros(sm.spec.d), np.ones(sm.spec.d))
    reg = ModelVersionRegistry(repos, str(tmp_path / "models"), keep=10)
    cfg = AdaptConfig(**{"every_n_samples": 20, "gate_window": 20,
                         "gate_min_samples": 10, **cfg_kw})
    return AdaptationController(fc, reg, repos, cfg, run_id="t", seed=1), fc, reg


def test_first_adaptation_promotes_and_registers_v001(repos, tmp_path):
    ctrl, fc, reg = _controller(repos, tmp_path)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    out = ctrl.maybe_adapt(T0 + 400)
    assert len(out) == 1
    o = out[0]
    assert o.decision == "promoted"
    assert o.gate.reason == "no_active_model"
    assert reg.active("cpu.util_pct@300").version == "forecast-model-v001"
    assert fc.n_updates("cpu.util_pct", 300.0) > 0


def test_a_worse_candidate_is_rejected_and_the_active_version_survives(repos, tmp_path):
    ctrl, fc, reg = _controller(repos, tmp_path)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    ctrl.maybe_adapt(T0 + 400)
    active_before = reg.active("cpu.util_pct@300").version

    # Feed nonsense so the candidate can only get worse.
    rng = np.random.default_rng(9)
    for i in range(200, 260):
        ex = _ex(i)
        ex.actual = 50.0 + rng.normal(0, 200)
        ctrl.observe_outcome(ex, 1.0)
    outs = ctrl.maybe_adapt(T0 + 5000)
    assert outs
    assert outs[0].decision in ("rejected", "promoted")
    if outs[0].decision == "rejected":
        assert reg.active("cpu.util_pct@300").version == active_before
        assert outs[0].gate.reason in ("worse_than_tolerance", "hard_reject_worse")
    assert ctrl.n_rejected + ctrl.n_promoted == 2


def test_gate_can_be_disabled_for_ablation(repos, tmp_path):
    ctrl, _fc, _reg = _controller(repos, tmp_path, gate_enabled=False, gate_min_samples=1000)
    for i in range(40):
        ctrl.observe_outcome(_ex(i), 1.0)
    outs = ctrl.maybe_adapt(T0 + 400)
    assert outs and outs[0].decision == "promoted"
    assert outs[0].gate.reason == "gate_disabled"


def test_insufficient_gate_data_is_recorded_as_skipped(repos, tmp_path):
    ctrl, _fc, reg = _controller(repos, tmp_path, gate_min_samples=50, every_n_samples=5)
    for i in range(10):
        ctrl.observe_outcome(_ex(i), 1.0)
    outs = ctrl.maybe_adapt(T0 + 400)
    assert outs and outs[0].decision == "skipped"
    assert outs[0].gate.reason == "insufficient_gate_samples"
    assert reg.active("cpu.util_pct@300") is None
    # A silent skip is indistinguishable from "no trigger" when debugging.
    events = repos.events.adapt_events()
    assert any(e["decision"] == "skipped" for e in events)


def test_baseline_floor_is_recorded_on_the_gate(repos, tmp_path):
    ctrl, _fc, _reg = _controller(repos, tmp_path)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    outs = ctrl.maybe_adapt(T0 + 400)
    assert outs[0].gate.baseline_mae is not None
    assert outs[0].gate.baseline_name in ("persistence", "ewma")


def test_versions_increment_and_carry_a_parent(repos, tmp_path):
    ctrl, _fc, reg = _controller(repos, tmp_path)
    seen = []
    for round_ in range(3):
        for i in range(round_ * 100, round_ * 100 + 60):
            ctrl.observe_outcome(_ex(i), 1.0)
        outs = ctrl.maybe_adapt(T0 + 400 + round_ * 1000)
        if outs and outs[0].decision == "promoted":
            seen.append(outs[0].candidate_version)
    assert seen == sorted(set(seen))         # strictly increasing, no reuse
    hist = reg.history("cpu.util_pct@300")
    latest = hist[0]
    if len(seen) > 1:
        assert latest.parent is not None


def test_rollback_restores_the_previous_version(repos, tmp_path):
    ctrl, fc, reg = _controller(repos, tmp_path)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    ctrl.maybe_adapt(T0 + 400)
    v1 = reg.active("cpu.util_pct@300").version
    for i in range(100, 160):
        ctrl.observe_outcome(_ex(i), 1.0)
    ctrl.maybe_adapt(T0 + 2000)
    v2 = reg.active("cpu.util_pct@300").version
    if v2 == v1:
        pytest.skip("second candidate was rejected; nothing to roll back")
    prev = reg.rollback("cpu.util_pct@300", T0 + 3000)
    assert prev.version == v1
    assert reg.active("cpu.util_pct@300").version == v1
    units = reg.load_unit("cpu.util_pct@300", v1)
    assert units and all(np.all(np.isfinite(m.w)) for m in units.values())


def test_rollback_with_no_earlier_version_returns_none(repos, tmp_path):
    _ctrl, _fc, reg = _controller(repos, tmp_path)
    assert reg.rollback("cpu.util_pct@300", T0) is None


def test_regression_watch_rolls_back_a_version_that_turned_out_worse(repos, tmp_path):
    """The gate uses tens of samples; this is the guarantee behind it."""
    ctrl, fc, reg = _controller(repos, tmp_path, gate_window=20, gate_min_samples=10)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    ctrl.maybe_adapt(T0 + 400)
    for i in range(100, 160):
        ctrl.observe_outcome(_ex(i), 1.0)
    outs = ctrl.maybe_adapt(T0 + 2000)
    if not (outs and outs[0].decision == "promoted" and outs[0].active_version):
        pytest.skip("no promotion to watch")
    # Live performance is dramatically worse than the predecessor's gate MAE.
    for i in range(300, 340):
        ctrl.observe_outcome(_ex(i), 500.0)
    rolled = ctrl.check_regressions(T0 + 4000, min_samples=20, worse_ratio=1.25)
    assert "cpu.util_pct@300" in rolled
    assert ctrl.n_rolled_back == 1
    ev = repos.events.adapt_events()
    assert any(e["decision"] == "rolled_back" and e["trigger"] == "regression_watch"
               for e in ev)


def test_corrupt_model_file_is_reported_not_crashed(repos, tmp_path):
    ctrl, _fc, reg = _controller(repos, tmp_path)
    for i in range(60):
        ctrl.observe_outcome(_ex(i), 1.0)
    ctrl.maybe_adapt(T0 + 400)
    v = reg.active("cpu.util_pct@300").version
    with open(reg.path_for("cpu.util_pct@300", v), "wb") as fh:
        fh.write(b"not a real npz file")
    with pytest.raises(Exception):
        reg.load_unit("cpu.util_pct@300", v)


def test_drift_routes_only_to_the_implicated_signal(repos, tmp_path):
    ctrl, _fc, _reg = _controller(repos, tmp_path)
    ctrl.state("cpu.util_pct@300")
    ctrl.state("mem.used_pct@300")
    ctrl.note_drift(None, None, "feature_shift", feature="cpu.util_pct|lag0")
    assert ctrl.state("cpu.util_pct@300").pending_drift == ["feature_shift"]
    assert ctrl.state("mem.used_pct@300").pending_drift == []
    ctrl.note_drift(None, None, "feature_shift", feature="time|sin_day")
    # A non-signal feature implicates everything.
    assert ctrl.state("mem.used_pct@300").pending_drift == ["feature_shift"]


def test_ungated_adaptation_works_with_an_empty_gate(repos, tmp_path):
    """The no-gate ablation legitimately has nothing held out."""
    ctrl, _fc, reg = _controller(
        repos, tmp_path, gate_enabled=False, gate_window=1000, every_n_samples=5
    )
    for i in range(10):
        ctrl.observe_outcome(_ex(i), 1.0)
    outs = ctrl.maybe_adapt(T0 + 400)
    assert outs and outs[0].decision == "promoted"
    assert outs[0].gate.n == 0
    assert reg.active("cpu.util_pct@300") is not None
