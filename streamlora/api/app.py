"""FastAPI application: JSON API plus the static dashboard.

Endpoints are shaped around the questions the UI asks, not around the database
tables, so the front end stays thin and every view has exactly one request.
Everything is local: the server binds 127.0.0.1 by default and no route reaches
out to a network service.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import Config
from ..evaluate import report as R
from ..util.logging import get_logger
from .state import AppState

log = get_logger("api")

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "web")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    window_s: float | None = None
    backend: str | None = None


class FeedbackRequest(BaseModel):
    kind: str
    text: str = ""
    label: str | None = None
    prediction_id: int | None = None
    signal: str | None = None


class RollbackRequest(BaseModel):
    scope: str
    kind: str = "forecast-model"


def create_app(config: Config, collect: bool = True) -> FastAPI:
    state = AppState.create(config, collect=collect)
    app = FastAPI(
        title="StreamLoRA",
        description="Continuous learning and forecasting for personal computer telemetry",
        version="0.1.0",
    )
    app.state.sl = state

    @app.on_event("shutdown")
    def _shutdown() -> None:  # pragma: no cover - lifecycle
        state.close()

    @app.middleware("http")
    async def _timing(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000.0
        response.headers["X-Response-Time-ms"] = f"{dt:.1f}"
        if dt > 750.0:
            log.dedupe(
                f"slow:{request.url.path}", "slow API response", level="info",
                path=request.url.path, ms=round(dt, 1),
            )
        return response

    # -- health / meta -----------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return state.health()

    @app.get("/api/settings")
    def settings() -> dict[str, Any]:
        specs = state.repos.signals.all_specs()
        return {
            "config": state.config.to_dict(),
            "signals": [
                {"name": s.name, "unit": s.unit, "kind": str(s.kind), "lo": s.lo, "hi": s.hi,
                 "description": s.description}
                for s in specs
            ],
            "targets": state.config.forecast.targets,
            "horizons_s": state.config.forecast.horizons_s,
            "collecting": bool(state.pipeline and state.pipeline.collector.running),
            "language": state.language.describe() if state.language else None,
        }

    # -- overview ----------------------------------------------------------
    @app.get("/api/overview")
    def overview(window_s: float = Query(default=None)) -> dict[str, Any]:
        now, is_live = state.reference_now()
        w = window_s or state.config.api.default_window_s
        latest = state.repos.telemetry.latest(1)
        current: dict[str, Any] = {}
        sample_ts = None
        if latest:
            s = latest[0]
            sample_ts = s.ts
            current = {
                k: {"value": r.value, "quality": int(r.quality)}
                for k, r in sorted(s.readings.items())
            }
        engine_report = state.engine.report() if state.engine else None
        regime = (engine_report or {}).get("regime", "unknown")
        regime_dist = (engine_report or {}).get("regime_distribution", {})
        if regime == "unknown" and state.language is not None:
            # Serving stored data with no live engine: the regime is still
            # recoverable, because it is recorded on every prediction. Reporting
            # "unknown" over a database full of labelled predictions would be a
            # UI artefact, not a fact about the machine.
            regime, _since, regime_dist = state.language.grounding.regime_info(
                now, max(w, 3600.0)
            )
        forecasts = [
            {
                "id": p.id, "signal": p.signal, "horizon_s": p.horizon_s, "value": p.value,
                "lo": p.lo, "hi": p.hi, "model_kind": p.model_kind,
                "model_version": p.model_version, "ts_made": p.ts_made,
                "ts_target": p.ts_target, "anchor": p.anchor, "regime": p.regime,
            }
            for p in state.repos.predictions.latest_unresolved(limit=200)
        ]
        recs = state.repos.predictions.resolved(ts_from=now - max(w, 3600.0), ts_to=now)
        rows = R.summarize(recs, reference="persistence", dm_test=False)
        return {
            "now": now,
            "is_live": is_live,
            "sample_ts": sample_ts,
            "sample_age_s": None if sample_ts is None else round(now - sample_ts, 1),
            "current": current,
            "regime": regime,
            "regime_distribution": regime_dist,
            "forecasts": forecasts,
            "accuracy": [r.as_dict() for r in rows],
            "drift_events": state.repos.events.drift_events(ts_from=now - 86400.0, limit=20),
            "adapt_events": state.repos.events.adapt_events(ts_from=now - 86400.0, limit=20),
            "collecting": bool(state.pipeline and state.pipeline.collector.running),
            "engine": {
                "ready": bool(engine_report and engine_report["features"]["ready"]),
                "stats": (engine_report or {}).get("stats"),
                "features": (engine_report or {}).get("features"),
                "targets": (engine_report or {}).get("targets", []),
                "horizons_s": (engine_report or {}).get("horizons_s", []),
            } if engine_report else None,
        }

    # -- telemetry ---------------------------------------------------------
    @app.get("/api/telemetry")
    def telemetry(
        signals: str = Query(default=""),
        window_s: float = Query(default=3600.0),
        max_points: int = Query(default=1200, ge=10, le=20000),
    ) -> dict[str, Any]:
        now, _is_live = state.reference_now()
        names = [s for s in signals.split(",") if s.strip()] or state.config.forecast.targets
        available = set(state.repos.signals.registry().names())
        names = [n for n in names if n in available]
        if not names:
            return {"ts": [], "series": {}, "requested": signals, "available": sorted(available)}
        w = state.repos.telemetry.window(names, ts_from=now - window_s, ts_to=now + 1.0)
        # Decimate by striding rather than averaging: the dashboard shows
        # measurements, and an averaged point is not a measurement.
        # Ceiling division: floor division overshoots the cap (1080 rows at
        # max_points=50 gives step 21 and 52 points).
        step = max(1, -(-len(w) // max_points))
        idx = list(range(0, len(w), step))

        def cell(i: int, j: int) -> float | None:
            # NaN marks an unusable reading and must cross the wire as null, so
            # the chart draws a gap instead of interpolating through it.
            v = float(w.values[i, j])
            return None if v != v else v

        return {
            "ts": [float(w.ts[i]) for i in idx],
            "series": {
                name: [cell(i, j) for i in idx]
                for j, name in enumerate(w.signals)
            },
            "n_total": len(w),
            "decimation": step,
            "window_s": window_s,
        }

    # -- forecasts ---------------------------------------------------------
    @app.get("/api/forecasts")
    def forecasts(
        signal: str = Query(...),
        horizon_s: float = Query(default=None),
        window_s: float = Query(default=7200.0),
        model_kind: str = Query(default="rls"),
    ) -> dict[str, Any]:
        now, _is_live = state.reference_now()
        resolved = state.repos.predictions.resolved(
            signal=signal, horizon_s=horizon_s, model_kind=model_kind,
            ts_from=now - window_s, ts_to=now,
        )
        pending = [
            p for p in state.repos.predictions.latest_unresolved(limit=400)
            if p.signal == signal and (horizon_s is None or p.horizon_s == horizon_s)
            and p.model_kind == model_kind
        ]
        actual = state.repos.telemetry.window([signal], ts_from=now - window_s, ts_to=now + 1.0)
        col = actual.column(signal)
        return {
            "signal": signal,
            "horizon_s": horizon_s,
            "model_kind": model_kind,
            "now": now,
            "actual": {
                "ts": [float(t) for t in actual.ts],
                "values": [
                    None if float(v) != float(v) else float(v)
                    for v in (col if col is not None else [])
                ],
            },
            "resolved": [
                {"id": p.id, "ts_made": p.ts_made, "ts_target": p.ts_target, "value": p.value,
                 "lo": p.lo, "hi": p.hi, "actual": p.actual, "error": p.error,
                 "in_interval": p.in_interval, "model_version": p.model_version,
                 "horizon_s": p.horizon_s, "regime": p.regime}
                for p in resolved
            ],
            "pending": [
                {"id": p.id, "ts_made": p.ts_made, "ts_target": p.ts_target, "value": p.value,
                 "lo": p.lo, "hi": p.hi, "model_version": p.model_version,
                 "horizon_s": p.horizon_s}
                for p in pending
            ],
        }

    @app.get("/api/prediction/{pid}")
    def prediction(pid: int) -> dict[str, Any]:
        p = state.repos.predictions.get(pid)
        if p is None:
            raise HTTPException(status_code=404, detail=f"no prediction {pid}")
        out = {
            "id": p.id, "signal": p.signal, "horizon_s": p.horizon_s, "ts_made": p.ts_made,
            "ts_target": p.ts_target, "value": p.value, "lo": p.lo, "hi": p.hi,
            "model_kind": p.model_kind, "model_version": p.model_version, "regime": p.regime,
            "anchor": p.anchor, "resolved": p.resolved, "actual": p.actual,
            "error": p.error, "abs_error": p.abs_error, "in_interval": p.in_interval,
            "used_for_adapt": p.used_for_adapt, "feature_id": p.feature_id,
        }
        if p.feature_id:
            got = state.repos.features.get(p.feature_id)
            if got is not None:
                vec, schema_hash, regime = got
                names = state.repos.features.schema_names(schema_hash)
                pairs = sorted(
                    zip(names, (float(x) for x in vec)), key=lambda kv: -abs(kv[1])
                )[:15]
                out["features"] = {"schema_hash": schema_hash, "regime": regime,
                                   "top_by_magnitude": pairs}
        return out

    # -- metrics / experiments --------------------------------------------
    @app.get("/api/metrics")
    def metrics(
        run_id: str = Query(default=None),
        window_s: float = Query(default=None),
        by_regime: bool = Query(default=False),
        over_time: int = Query(default=0, ge=0, le=12),
    ) -> dict[str, Any]:
        now, _is_live = state.reference_now()
        ts_from = (now - window_s) if window_s else None
        recs = state.repos.predictions.resolved(run_id=run_id, ts_from=ts_from, ts_to=now)
        rows = (
            R.summarize_over_time(recs, n_windows=over_time) if over_time
            else R.summarize(recs, reference="persistence", by_regime=by_regime)
        )
        return {
            "rows": [r.as_dict() for r in rows],
            "n_resolved": len(recs),
            "table": R.to_table(rows),
        }

    @app.get("/api/experiments")
    def experiments() -> dict[str, Any]:
        out = []
        d = state.config.runs_dir
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d), reverse=True):
                if not fn.endswith(".json") or fn == "latest.json":
                    continue
                path = os.path.join(d, fn)
                try:
                    with open(path) as fh:
                        data = json.load(fh)
                except (OSError, json.JSONDecodeError):
                    continue
                out.append({
                    "id": data.get("experiment_id", fn[:-5]),
                    "name": data.get("name") or data.get("model_id", "language"),
                    "dataset": data.get("dataset") or data.get("dataset_notes", ""),
                    "kind": "language" if "arms" in data and data.get("model_id") else "forecast",
                    "started_ts": data.get("started_ts"),
                    "duration_s": data.get("duration_s"),
                    "n_arms": len(data.get("arms", [])),
                    "file": fn,
                })
        return {"experiments": out, "runs_dir": d}

    @app.get("/api/experiments/{name}")
    def experiment(name: str) -> dict[str, Any]:
        # Reject path traversal explicitly rather than relying on the framework.
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(status_code=400, detail="invalid experiment id")
        path = os.path.join(state.config.runs_dir, name if name.endswith(".json") else name + ".json")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail=f"no experiment report {name}")
        with open(path) as fh:
            return json.load(fh)

    # -- models / adaptation ----------------------------------------------
    @app.get("/api/models")
    def models(kind: str = Query(default=None), scope: str = Query(default=None),
               limit: int = Query(default=200, ge=1, le=2000)) -> dict[str, Any]:
        hist = state.repos.models.history(kind=kind, scope=scope, limit=limit)
        for h in hist:
            try:
                h["metrics"] = json.loads(h.get("metrics") or "{}")
            except json.JSONDecodeError:
                h["metrics"] = {}
        return {"versions": hist}

    @app.post("/api/models/rollback")
    def rollback(req: RollbackRequest) -> dict[str, Any]:
        from ..adapt.registry import ModelVersionRegistry

        if req.kind == "adapter":
            from ..language.lora import LanguageTrainer

            trainer = LanguageTrainer(state.config, state.repos)
            target = trainer.rollback()
            if target is None:
                raise HTTPException(status_code=409, detail="no earlier adapter to roll back to")
            if state.language is not None:
                state.language = None
                from ..language.service import LanguageService

                state.language = LanguageService(state.config, state.repos)
            return {"kind": "adapter", "to": target["version"]}
        reg = ModelVersionRegistry(
            state.repos, state.config.models_dir, keep=state.config.adapt.keep_versions
        )
        cur = reg.active(req.scope)
        prev = reg.rollback(req.scope, time.time())
        if prev is None:
            raise HTTPException(
                status_code=409, detail=f"no earlier version for scope {req.scope!r}"
            )
        # Load the restored weights into the running engine so the rollback takes
        # effect immediately rather than at the next restart.
        if state.engine is not None:
            sig, _, hor = req.scope.rpartition("@")
            try:
                units = reg.load_unit(req.scope, prev.version)
                state.engine.learned.install_unit(sig, float(hor), units)
                state.engine.learned.set_version(
                    sig, float(hor), int(prev.version.rsplit("v", 1)[-1])
                )
            except (OSError, ValueError, KeyError) as exc:
                raise HTTPException(
                    status_code=500, detail=f"rollback registered but reload failed: {exc}"
                ) from exc
        state.repos.events.add_adapt(
            ts=time.time(), scope=req.scope, kind="forecast", trigger="manual",
            decision="rolled_back", active_version=cur.version if cur else None,
            candidate_version=prev.version, metric_name="mae", metric_before=None,
            metric_after=None, gate_n=0, n_train=0, duration_ms=0.0,
            detail={"reason": "manual rollback via API"}, run_id=state.run_id,
        )
        return {"kind": "forecast-model", "scope": req.scope,
                "from": cur.version if cur else None, "to": prev.version}

    @app.get("/api/events")
    def events(window_s: float = Query(default=86400.0)) -> dict[str, Any]:
        now, _is_live = state.reference_now()
        return {
            "drift": state.repos.events.drift_events(ts_from=now - window_s, limit=500),
            "adaptation": state.repos.events.adapt_events(ts_from=now - window_s, limit=500),
            "ops": state.repos.events.ops_events(ts_from=now - window_s, limit=300),
        }

    # -- chat / feedback ---------------------------------------------------
    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        if state.language is None:
            raise HTTPException(status_code=503, detail="language service unavailable")
        now, _is_live = state.reference_now()
        ans = state.language.ask(
            req.question, now=now, window_s=req.window_s, force_backend=req.backend
        )
        return ans.as_dict()

    @app.post("/api/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, Any]:
        from ..language.feedback import VALID_KINDS, FeedbackStore

        if req.kind not in VALID_KINDS:
            raise HTTPException(
                status_code=422, detail=f"kind must be one of {list(VALID_KINDS)}"
            )
        store = FeedbackStore(state.repos)
        fid = store.add(
            kind=req.kind, text=req.text, label=req.label,
            prediction_id=req.prediction_id, signal=req.signal, run_id=state.run_id,
        )
        persona = state.language.refresh_persona() if state.language else None
        return {
            "id": fid,
            "persona_revision": persona.revision if persona else None,
            "summary": store.summary().as_dict(),
        }

    @app.get("/api/feedback")
    def feedback_list(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
        from ..language.feedback import FeedbackStore

        store = FeedbackStore(state.repos)
        return {
            "items": state.repos.feedback.list(limit=limit),
            "summary": store.summary().as_dict(),
            "persona": store.persona().to_dict(),
            "user_judged_accuracy": store.accuracy_by_user(),
        }

    # -- static UI ---------------------------------------------------------
    if os.path.isdir(os.path.join(WEB_DIR, "static")):
        app.mount(
            "/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static"
        )

    @app.get("/")
    def index() -> Any:
        idx = os.path.join(WEB_DIR, "index.html")
        if os.path.isfile(idx):
            return FileResponse(idx)
        return JSONResponse({
            "name": "StreamLoRA",
            "note": "dashboard assets not found; the JSON API is available under /api",
            "endpoints": ["/api/health", "/api/overview", "/api/telemetry", "/api/forecasts",
                          "/api/metrics", "/api/models", "/api/events", "/api/chat"],
        })

    return app


def serve(config: Config, host: str = "127.0.0.1", port: int = 8765,
          collect: bool = True, reload: bool = False) -> int:
    import uvicorn

    app = create_app(config, collect=collect)
    log.info("serving", host=host, port=port, collect=collect, db=config.db_file)
    print(f"\n  StreamLoRA dashboard: http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
