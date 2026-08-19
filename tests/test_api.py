"""API surface: every endpoint the dashboard depends on, plus error states."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from streamlora.api.app import create_app
from streamlora.replay.player import ReplayPlayer
from streamlora.replay.synthetic import build_scenario, generate
from streamlora.store.repo import Repos


@pytest.fixture
def client(config):
    """A server over stored data: no collector thread, so tests stay hermetic."""
    samples, reg = generate(build_scenario("idle_to_build", minutes=90))
    repos = Repos.open(config.db_file)
    repos.signals.register(reg.as_mapping().values())
    repos.telemetry.insert_samples(samples, run_id="api")
    player = ReplayPlayer(config, repos, reg, run_id="api")
    player.run(samples)
    repos.close()
    app = create_app(config, collect=False)
    with TestClient(app) as c:
        c.last_ts = samples[-1].ts
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["collecting"] is False
    assert d["samples_total"] > 500
    assert d["telemetry_span"]["hours"] > 1.0


def test_settings_lists_signals_and_config(client):
    d = client.get("/api/settings").json()
    assert d["targets"]
    assert any(s["name"] == "cpu.util_pct" for s in d["signals"])
    assert d["config"]["forecast"]["horizons_s"]
    # Units must be reported so the UI never has to guess from a name suffix.
    cpu = next(s for s in d["signals"] if s["name"] == "cpu.util_pct")
    assert cpu["unit"] == "percent"


def test_overview_has_everything_the_dashboard_needs(client):
    d = client.get("/api/overview").json()
    assert d["current"]
    assert d["regime"] != "unknown", "regime must be recoverable from stored predictions"
    assert d["forecasts"]
    assert d["accuracy"]
    arms = {r["arm"] for r in d["accuracy"]}
    assert "persistence" in arms
    for r in d["accuracy"]:
        assert r["n"] > 0


def test_telemetry_series_and_nan_handling(client):
    d = client.get("/api/telemetry?signals=cpu.util_pct,mem.used_pct&window_s=3600").json()
    assert len(d["ts"]) == len(d["series"]["cpu.util_pct"])
    for v in d["series"]["cpu.util_pct"]:
        # NaN must cross the wire as null so charts draw a gap.
        assert v is None or isinstance(v, (int, float))
    raw = client.get("/api/telemetry?signals=cpu.util_pct&window_s=3600").text
    assert "NaN" not in raw, "invalid JSON: bare NaN in payload"


def test_telemetry_decimates_large_windows(client):
    d = client.get("/api/telemetry?signals=cpu.util_pct&window_s=86400&max_points=50").json()
    assert len(d["ts"]) <= 51
    assert d["decimation"] >= 1


def test_telemetry_with_unknown_signal_is_not_an_error(client):
    d = client.get("/api/telemetry?signals=nope.nope").json()
    assert d["series"] == {} and d["ts"] == []
    assert "available" in d


def test_forecasts_returns_actual_resolved_and_pending(client):
    d = client.get("/api/forecasts?signal=cpu.util_pct&horizon_s=300&window_s=7200").json()
    assert d["actual"]["ts"]
    assert d["resolved"]
    row = d["resolved"][0]
    assert {"ts_target", "value", "actual", "error", "model_version"} <= set(row)


def test_prediction_detail_exposes_its_features(client):
    d = client.get("/api/forecasts?signal=cpu.util_pct&horizon_s=300").json()
    ov = client.get("/api/overview").json()
    pid = ov["forecasts"][0]["id"]
    r = client.get(f"/api/prediction/{pid}")
    assert r.status_code == 200
    body = r.json()
    assert body["signal"] and body["model_version"]
    if body.get("feature_id"):
        assert "features" in body
        assert body["features"]["top_by_magnitude"]


def test_prediction_not_found(client):
    assert client.get("/api/prediction/99999999").status_code == 404


def test_metrics_endpoint_returns_rows_and_a_table(client):
    d = client.get("/api/metrics").json()
    assert d["n_resolved"] > 0
    assert d["rows"]
    assert "scope" in d["table"]


def test_metrics_by_regime_and_over_time(client):
    d = client.get("/api/metrics?by_regime=true").json()
    assert any(r["regime"] for r in d["rows"])
    d2 = client.get("/api/metrics?over_time=3").json()
    assert any(r["window"] for r in d2["rows"])


def test_models_and_rollback(client):
    d = client.get("/api/models?kind=forecast-model").json()
    versions = d["versions"]
    assert versions
    active = [v for v in versions if v["active"]]
    assert active
    scope = active[0]["scope"]
    r = client.post("/api/models/rollback", json={"scope": scope, "kind": "forecast-model"})
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        body = r.json()
        assert body["to"] != body["from"]
        after = client.get("/api/models?kind=forecast-model").json()["versions"]
        now_active = [v for v in after if v["active"] and v["scope"] == scope]
        assert now_active and now_active[0]["version"] == body["to"]


def test_rollback_unknown_scope_is_a_conflict_not_a_crash(client):
    r = client.post("/api/models/rollback", json={"scope": "no.such@300"})
    assert r.status_code == 409


def test_events_endpoint(client):
    d = client.get("/api/events?window_s=1000000").json()
    assert "drift" in d and "adaptation" in d and "ops" in d
    assert isinstance(d["adaptation"], list)


def test_chat_is_grounded_and_reports_its_backend(client):
    r = client.post("/api/chat", json={"question": "What is my machine doing right now?"})
    assert r.status_code == 200
    d = r.json()
    assert d["text"]
    assert d["grounded"] is True
    assert d["backend"] in ("template", "hf_local")
    assert d["evidence_text"]


def test_chat_rejects_an_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_feedback_round_trip_updates_the_persona(client):
    before = client.get("/api/feedback").json()["persona"]["revision"]
    r = client.post("/api/feedback", json={
        "kind": "label", "text": "That happens when I'm compiling. Normal for me.",
    })
    assert r.status_code == 200
    assert r.json()["persona_revision"] > before
    d = client.get("/api/feedback").json()
    assert d["items"]
    assert any(rule["regime"] == "build" for rule in d["persona"]["rules"])


def test_feedback_rejects_unknown_kind(client):
    r = client.post("/api/feedback", json={"kind": "nonsense", "text": "x"})
    assert r.status_code == 422


def test_experiments_listing_and_traversal_guard(client):
    d = client.get("/api/experiments").json()
    assert "experiments" in d
    assert client.get("/api/experiments/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    assert client.get("/api/experiments/does_not_exist").status_code == 404


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "StreamLoRA" in r.text


def test_response_time_header_present(client):
    r = client.get("/api/health")
    assert "X-Response-Time-ms" in r.headers


def test_forecast_rows_carry_prediction_ids_for_feedback(client):
    d = client.get("/api/forecasts?signal=cpu.util_pct&horizon_s=300").json()
    assert d["resolved"] and "id" in d["resolved"][0]
    pid = d["resolved"][0]["id"]
    r = client.post("/api/feedback", json={
        "kind": "forecast_useful", "label": "yes", "prediction_id": pid,
    })
    assert r.status_code == 200
    items = client.get("/api/feedback").json()["items"]
    row = next(i for i in items if i["prediction_id"] == pid)
    # The signal is derived from the referenced prediction, so per-scope
    # usefulness is answerable without asking the user twice.
    assert row["signal"] == "cpu.util_pct"


def test_event_happened_feedback_is_aggregated_per_signal(client):
    d = client.get("/api/forecasts?signal=cpu.util_pct&horizon_s=300").json()
    pid = d["resolved"][0]["id"]
    client.post("/api/feedback", json={
        "kind": "event_happened", "label": "yes", "prediction_id": pid,
    })
    client.post("/api/feedback", json={
        "kind": "event_happened", "label": "no", "prediction_id": pid,
    })
    acc = client.get("/api/feedback").json()["user_judged_accuracy"]
    assert acc["cpu.util_pct"] == {"yes": 1, "no": 1}


def test_health_snapshots_are_persisted_for_later_diagnosis(config, tmp_path):
    """/api/health is the live picture and dies with the process.

    Answering "was the collector healthy when that drift alarm fired?" needs the
    history, so a snapshot is written to ops_events on a slow cadence.
    """
    from streamlora.api.state import AppState

    state = AppState.create(config, collect=False)
    try:
        # Drive the snapshot directly rather than waiting minutes for the
        # collector to reach the cadence.
        from streamlora.pipeline import build_telemetry_pipeline
        from streamlora.telemetry.schema import Reading, TelemetrySample

        state.pipeline = build_telemetry_pipeline(
            config, run_id="ops", origin="live", repos=state.repos
        )
        state.pipeline.collector.stats.ticks = AppState.OPS_SNAPSHOT_EVERY
        state._record_ops_health(TelemetrySample(ts=1.0, readings={"a": Reading(1.0)}))
        events = state.repos.events.ops_events()
        assert events, "no health snapshot recorded"
        ev = events[0]
        assert ev["kind"] == "health"
        detail = json.loads(ev["detail"])
        assert "collector" in detail and "sources" in detail and "normalizer" in detail
        # A snapshot at a non-cadence tick must not be written.
        before = len(events)
        state.pipeline.collector.stats.ticks = AppState.OPS_SNAPSHOT_EVERY + 1
        state._record_ops_health(TelemetrySample(ts=2.0, readings={}))
        assert len(state.repos.events.ops_events()) == before
    finally:
        state.close()
