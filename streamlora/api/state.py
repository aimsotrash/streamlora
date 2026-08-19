"""Shared application state for the API process.

Holds the database handles and, optionally, a live collector plus forecasting
engine. Two modes:

* **serve + collect** (default): the process owns a collector thread, so opening
  the dashboard on a fresh machine starts producing telemetry immediately.
* **serve only** (``--no-collect``): read-only over stored data. Used when a
  separate ``streamlora collect`` process is already running -- two collectors
  would each compute correct rates (the OS counters are cumulative and each
  process keeps its own previous reading) but would write duplicate sample rows
  and pay the process-enumeration cost twice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..adapt.controller import AdaptationController
from ..adapt.registry import ModelVersionRegistry
from ..config import Config
from ..forecast.engine import ForecastEngine
from ..language.service import LanguageService
from ..pipeline import TelemetryPipeline, build_telemetry_pipeline
from ..store.repo import Repos
from ..util.ids import config_hash
from ..util.logging import get_logger

log = get_logger("api.state")


@dataclass
class AppState:
    config: Config
    repos: Repos
    pipeline: TelemetryPipeline | None = None
    engine: ForecastEngine | None = None
    language: LanguageService | None = None
    run_id: str | None = None
    started_ts: float = field(default_factory=time.time)
    collect: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def create(cls, config: Config, collect: bool = True) -> "AppState":
        config.ensure_dirs()
        repos = Repos.open(config.db_file)
        state = cls(config=config, repos=repos, collect=collect)
        if collect:
            state._start_live()
        else:
            # Register whatever signals exist so the settings view is populated
            # even with no collector running.
            log.info("serving stored data only; collector not started")
        state.language = LanguageService(config, repos)
        return state

    def _start_live(self) -> None:
        cfg = self.config
        run_id = f"serve-{time.strftime('%Y%m%d-%H%M%S')}"
        self.run_id = run_id
        pipe = build_telemetry_pipeline(cfg, run_id=run_id, origin="live", repos=self.repos)
        self.pipeline = pipe
        pipe.repos.runs.start(
            run_id, "serve", time.time(), cfg.to_dict(), config_hash(cfg.to_dict()),
            notes="API process with live collection",
        )
        engine = ForecastEngine(
            cfg, self.repos, pipe.registry.signals, run_id=run_id,
            enable_drift=cfg.drift.enabled, enable_adapt=cfg.adapt.enabled,
        )
        if cfg.adapt.enabled:
            engine.adapt = AdaptationController(
                forecaster=engine.learned,
                registry=ModelVersionRegistry(
                    self.repos, cfg.models_dir, keep=cfg.adapt.keep_versions
                ),
                repos=self.repos, config=cfg.adapt, run_id=run_id,
                seed=cfg.general.seed, config_fingerprint=config_hash(cfg.to_dict()),
            )
        self.engine = engine
        pipe.collector.subscribe(engine.on_sample)
        pipe.collector.subscribe(self._record_ops_health)
        pipe.collector.start_background()
        log.info("live collection started", run_id=run_id, interval_s=cfg.collect.interval_s)

    #: Ticks between durable health snapshots. Five minutes at the default
    #: cadence -- frequent enough to reconstruct an incident, sparse enough that
    #: the ops table stays readable.
    OPS_SNAPSHOT_EVERY = 60

    def _record_ops_health(self, sample: Any) -> None:
        """Persist a periodic health snapshot.

        ``/api/health`` reports the live picture, which is gone the moment the
        process restarts. Answering "was the collector healthy when that drift
        alarm fired?" needs the history, so a snapshot is written to
        ``ops_events`` on a slow cadence.
        """
        pipe = self.pipeline
        if pipe is None:
            return
        stats = pipe.collector.stats
        if stats.ticks % self.OPS_SNAPSHOT_EVERY:
            return
        detail: dict[str, Any] = {
            "collector": stats.as_dict(),
            "normalizer": pipe.normalizer.stats.as_dict(),
            "sources": {
                n: {"available": h.available, "failures": h.failures,
                    "avg_read_ms": round(h.avg_read_ms, 2),
                    "quarantined": h.quarantined_until > 0}
                for n, h in pipe.registry.health.items()
            },
        }
        level = "info"
        if self.engine is not None:
            e = self.engine.stats
            detail["engine"] = e.as_dict()
            detail["pending"] = len(self.engine._pending)
            if e.max_predict_ms > pipe.collector.interval_s * 1000.0 * 0.5:
                level = "warning"
        if any(h.quarantined_until > 0 for h in pipe.registry.health.values()):
            level = "warning"
        try:
            self.repos.events.add_ops(sample.ts, "health", level, detail, self.run_id)
        except Exception:  # pragma: no cover - never let telemetry die for a log
            log.exception("could not record health snapshot")

    # -- time anchoring ----------------------------------------------------
    #: A database whose newest sample is older than this is treated as historical.
    STALE_AFTER_S = 300.0

    def reference_now(self) -> tuple[float, bool]:
        """The instant every window should be measured back from.

        Wall clock when telemetry is arriving now; otherwise the newest stored
        sample. Without this, opening the dashboard on an imported dataset or a
        finished replay shows an empty page: every window is anchored to today
        while the data sits in the past. Returns ``(now, is_live)`` so the UI can
        say which it is rather than implying the machine is idle.
        """
        wall = time.time()
        rng = self.repos.telemetry.time_range()
        if rng is None:
            return wall, True
        latest = float(rng[1])
        if wall - latest > self.STALE_AFTER_S:
            return latest, False
        return wall, True

    # -- health ------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ok": True,
            "uptime_s": round(time.time() - self.started_ts, 1),
            "collecting": bool(self.pipeline and self.pipeline.collector.running),
            "run_id": self.run_id,
            "db": self.config.db_file,
            "samples_total": self.repos.telemetry.count(),
        }
        if self.pipeline is not None:
            out["telemetry"] = self.pipeline.collector.health()
        if self.engine is not None:
            out["engine"] = self.engine.report()
        if self.language is not None:
            out["language"] = self.language.describe()
        rng = self.repos.telemetry.time_range()
        out["telemetry_span"] = (
            {"from": rng[0], "to": rng[1], "hours": round((rng[1] - rng[0]) / 3600.0, 2)}
            if rng else None
        )
        return out

    def close(self) -> None:
        with self._lock:
            if self.pipeline is not None:
                try:
                    n = self.repos.telemetry.count(run_id=self.run_id)
                    self.repos.runs.finish(
                        self.run_id or "", time.time(), "stopped", summary={"samples": n}
                    )
                except Exception:  # pragma: no cover
                    pass
                self.pipeline.close()
                self.pipeline = None
            else:
                self.repos.close()
