"""Storage: schema, round-trips, pivots, version registry semantics."""

from __future__ import annotations

import numpy as np
import pytest

from streamlora.store.db import Database, SCHEMA_VERSION
from streamlora.store.repo import Repos
from streamlora.telemetry.schema import Quality, Reading, SignalSpec, TelemetrySample

from .conftest import T0, make_sample


def test_schema_version_is_recorded(tmp_path):
    db = Database(str(tmp_path / "a.sqlite"))
    row = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(row["value"]) == SCHEMA_VERSION
    db.close()


def test_sample_round_trip_preserves_quality(repos):
    repos.signals.register([SignalSpec("cpu.util_pct", "percent", lo=0.0, hi=100.0)])
    s = TelemetrySample(ts=T0, readings={
        "cpu.util_pct": Reading(42.0, Quality.SUSPECT),
    })
    repos.telemetry.insert_samples([s], run_id="r1")
    got = list(repos.telemetry.iter_samples())
    assert len(got) == 1
    r = got[0].readings["cpu.util_pct"]
    assert r.value == 42.0
    assert r.quality is Quality.SUSPECT


def test_window_pivot_uses_nan_for_unusable(repos):
    repos.signals.register([
        SignalSpec("a", "percent", lo=0.0, hi=100.0),
        SignalSpec("b", "percent", lo=0.0, hi=100.0),
    ])
    samples = [make_sample(T0 + i * 5, a=float(i)) for i in range(4)]
    samples[2].readings["a"] = Reading(None, Quality.MISSING)
    samples[1].readings["b"] = Reading(7.0)
    repos.telemetry.insert_samples(samples)
    w = repos.telemetry.window(["a", "b"])
    assert w.values.shape == (4, 2)
    assert np.isnan(w.values[2, 0])
    assert w.values[1, 1] == 7.0
    assert np.isnan(w.values[0, 1])
    assert list(w.ts) == [T0, T0 + 5, T0 + 10, T0 + 15]


def test_window_is_empty_when_nothing_matches(repos):
    repos.signals.register([SignalSpec("a", "percent")])
    w = repos.telemetry.window(["a"])
    assert len(w) == 0
    assert w.values.shape == (0, 1)


def test_adding_a_new_signal_needs_no_migration(repos):
    """The long-format layout is what makes 'add GPU temp later' a non-event."""
    repos.signals.register([SignalSpec("a", "percent")])
    repos.telemetry.insert_samples([make_sample(T0, a=1.0)])
    repos.signals.register([SignalSpec("gpu.temp_c", "celsius", lo=-20.0, hi=130.0)])
    repos.telemetry.insert_samples([make_sample(T0 + 5, a=2.0, **{"gpu.temp_c": 60.0})])
    w = repos.telemetry.window(["a", "gpu.temp_c"])
    assert np.isnan(w.values[0, 1])
    assert w.values[1, 1] == 60.0


def test_prediction_lifecycle(repos):
    ids = repos.predictions.insert_many([{
        "ts_made": T0, "ts_target": T0 + 300, "horizon_s": 300.0, "signal": "cpu.util_pct",
        "value": 50.0, "lo": 40.0, "hi": 60.0, "alpha": 0.1, "model_kind": "rls",
        "model_version": "forecast-model-v001", "regime": "idle", "anchor": 45.0,
        "feature_id": None, "infer_ms": 0.2, "run_id": "r",
    }])
    assert len(repos.predictions.due(T0 + 100)) == 0
    due = repos.predictions.due(T0 + 400)
    assert len(due) == 1 and not due[0].resolved
    repos.predictions.resolve_many([{
        "id": ids[0], "ts_resolved": T0 + 300, "actual": 55.0, "error": -5.0,
        "abs_error": 5.0, "in_interval": 1, "resolve_quality": 0,
    }])
    got = repos.predictions.resolved()[0]
    assert got.actual == 55.0 and got.abs_error == 5.0 and got.in_interval == 1
    assert not got.used_for_adapt
    repos.predictions.mark_used_for_adapt(ids)
    assert repos.predictions.get(ids[0]).used_for_adapt


def test_feature_vectors_round_trip_as_float32(repos):
    repos.features.register_schema("h", ["a", "b", "c"], T0)
    vec = np.array([1.5, -2.25, 1e-3])
    fid = repos.features.insert(T0, "h", vec, "idle", 1.0, "r")
    got, schema, regime = repos.features.get(fid)
    assert schema == "h" and regime == "idle"
    np.testing.assert_allclose(got, vec, rtol=1e-6)
    assert repos.features.schema_names("h") == ["a", "b", "c"]


def test_version_counter_is_monotonic_across_pruning(repos):
    """Counting rows breaks the moment pruning removes any."""
    kind, scope = "forecast-model", "cpu.util_pct@300"
    for i in range(1, 6):
        v = f"forecast-model-v{repos.models.next_version_number(kind, scope):03d}"
        repos.models.register(kind, scope, v, T0 + i, parent=None, path=f"/tmp/{v}")
    repos.models.activate(kind, scope, "forecast-model-v005", T0 + 10)
    repos.models.prune(kind, scope, keep=1)
    assert repos.models.next_version_number(kind, scope) == 6


def test_prune_protects_the_active_version_and_its_parent(repos):
    kind, scope = "forecast-model", "s"
    for i in range(1, 8):
        v = f"v{i:03d}"
        repos.models.register(kind, scope, v, T0 + i,
                              parent=f"v{i - 1:03d}" if i > 1 else None, path=f"/tmp/{v}")
    repos.models.activate(kind, scope, "v007", T0 + 20)
    doomed = repos.models.prune(kind, scope, keep=1)
    assert "v007" not in doomed          # active
    assert "v006" not in doomed          # rollback target
    remaining = {r["version"] for r in repos.models.history(kind, scope)}
    assert {"v006", "v007"} <= remaining


def test_activate_leaves_exactly_one_active(repos):
    kind, scope = "forecast-model", "s"
    for v in ("v1", "v2", "v3"):
        repos.models.register(kind, scope, v, T0)
    for v in ("v1", "v2", "v3"):
        repos.models.activate(kind, scope, v, T0)
        active = [r for r in repos.models.history(kind, scope) if r["active"]]
        assert len(active) == 1 and active[0]["version"] == v


def test_feedback_consumption_tracking(repos):
    a = repos.feedback.add(ts=T0, kind="label", text="normal for me")
    b = repos.feedback.add(ts=T0 + 1, kind="note", text="ml work")
    assert len(repos.feedback.list(unconsumed_only=True)) == 2
    repos.feedback.mark_consumed([a], "adapter-v001")
    assert len(repos.feedback.list(unconsumed_only=True)) == 1


def test_retention_deletes_old_telemetry(repos):
    repos.signals.register([SignalSpec("a", "percent")])
    repos.telemetry.insert_samples([make_sample(T0 + i * 5, a=1.0) for i in range(10)])
    removed = repos.db.vacuum_older_than(T0 + 25)
    assert removed == 5
    assert repos.telemetry.count() == 5


def test_readings_cascade_on_sample_delete(repos):
    repos.signals.register([SignalSpec("a", "percent")])
    repos.telemetry.insert_samples([make_sample(T0, a=1.0)])
    repos.db.vacuum_older_than(T0 + 1)
    n = repos.db.conn.execute("SELECT COUNT(*) AS n FROM readings").fetchone()["n"]
    assert n == 0
