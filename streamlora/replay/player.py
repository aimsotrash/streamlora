"""Replay a telemetry stream through the live pipeline.

The replay path deliberately reuses the *same* Normalizer, FeatureExtractor,
RegimeClassifier, ForecastEngine, DriftController and AdaptationController that
the live collector drives. Only two things differ:

* time comes from a ``SimulatedClock`` instead of the wall clock, and
* samples come from a file or the database instead of sensors.

That is the whole point: a bug that only shows up in production is a bug the
replay can reproduce, and a result measured in replay is a result about the
system that actually runs.

Determinism. Given the same dataset and config, a replay produces identical
predictions, identical adaptation decisions and identical metrics. Achieved by:
seeded reservoir sampling, no wall-clock reads in any model path, order-stable
iteration, and a forecaster whose update rule has no stochastic component. The
test suite asserts this rather than assuming it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from ..adapt.controller import AdaptationController
from ..adapt.registry import ModelVersionRegistry
from ..config import Config
from ..forecast.engine import ForecastEngine
from ..store.repo import Repos
from ..telemetry.normalizer import Normalizer
from ..telemetry.schema import SignalRegistry, TelemetrySample
from ..util.clock import SimulatedClock
from ..util.ids import config_hash
from ..util.logging import get_logger

log = get_logger("replay.player")


@dataclass(slots=True)
class ReplayResult:
    run_id: str
    n_samples: int
    n_predictions: int
    n_resolved: int
    ts_start: float
    ts_end: float
    wall_s: float
    engine_report: dict[str, object] = field(default_factory=dict)
    normalizer_stats: dict[str, int] = field(default_factory=dict)

    @property
    def speedup(self) -> float:
        span = self.ts_end - self.ts_start
        return span / self.wall_s if self.wall_s > 0 else float("inf")

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id, "n_samples": self.n_samples,
            "n_predictions": self.n_predictions, "n_resolved": self.n_resolved,
            "ts_start": self.ts_start, "ts_end": self.ts_end,
            "simulated_s": round(self.ts_end - self.ts_start, 1),
            "wall_s": round(self.wall_s, 2), "speedup": round(self.speedup, 1),
            "normalizer": self.normalizer_stats,
        }


class ReplayPlayer:
    """Drives a sample stream through a freshly assembled pipeline."""

    def __init__(
        self,
        config: Config,
        repos: Repos,
        signal_registry: SignalRegistry,
        run_id: str,
        speed: float = 0.0,
        store_telemetry: bool = False,
        record_predictions: bool = True,
        renormalize: bool = True,
        enable_drift: bool = True,
        enable_adapt: bool = True,
        freeze_after_ts: float | None = None,
    ) -> None:
        self.cfg = config
        self.repos = repos
        self.registry = signal_registry
        self.run_id = run_id
        #: 0 means "no waiting". Any positive value replays at that multiple of
        #: real time, which is only useful for demonstrating the live UI.
        self.speed = float(speed)
        self.store_telemetry = store_telemetry
        self.renormalize = renormalize
        self.clock = SimulatedClock(0.0, speed=self.speed)
        self.normalizer = Normalizer(
            signal_registry, expected_interval_s=config.collect.interval_s,
            gap_factor=config.collect.gap_factor,
        )
        self.model_registry = ModelVersionRegistry(
            repos, config.models_dir, keep=config.adapt.keep_versions
        )
        fingerprint = config_hash(config.to_dict())
        engine = ForecastEngine(
            config, repos, signal_registry, run_id=run_id, enable_drift=enable_drift,
            enable_adapt=enable_adapt, record_predictions=record_predictions,
        )
        if enable_adapt:
            engine.adapt = AdaptationController(
                forecaster=engine.learned, registry=self.model_registry, repos=repos,
                config=config.adapt, run_id=run_id, seed=config.general.seed,
                config_fingerprint=fingerprint,
            )
        self.engine = engine
        #: Simulated timestamp at which learning stops. This is how the "static
        #: trained model" arm is realised: identical code, identical data, but
        #: no updates past the boundary. Both arms are then scored on exactly the
        #: same held-out tail, which is the only fair comparison.
        self.freeze_after_ts = freeze_after_ts
        self._frozen = False
        self.on_sample: list[Callable[[TelemetrySample], None]] = []

    def _freeze(self) -> None:
        """Stop all learning; keep predicting."""
        self._frozen = True
        self.engine.learned.frozen = True
        if self.engine.adapt is not None:
            self.engine.adapt.cfg.enabled = False
        log.info(
            "learning frozen", run_id=self.run_id,
            at_ts=self.freeze_after_ts,
            updates={
                m.spec.name: m.n_updates for m in self.engine.learned.scopes.values()
            },
        )

    def run(self, samples: Iterable[TelemetrySample], limit: int | None = None) -> ReplayResult:
        t_wall = time.perf_counter()
        n = 0
        ts_start = ts_end = 0.0
        write_buf: list[TelemetrySample] = []
        prev_ts: float | None = None
        for raw in samples:
            if limit is not None and n >= limit:
                break
            # Simulated time is set from the data, so a gap in the dataset is a
            # gap in simulated time -- the normalizer and feature window see the
            # discontinuity exactly as they would live.
            self.clock.set(raw.ts)
            if self.renormalize:
                sample = self.normalizer.normalize(
                    raw.ts,
                    {k: (r.value if r.usable else None) for k, r in raw.readings.items()},
                    origin=raw.origin or "replay",
                )
            else:
                sample = raw
            if n == 0:
                ts_start = sample.ts
            ts_end = sample.ts

            if (
                not self._frozen
                and self.freeze_after_ts is not None
                and sample.ts >= self.freeze_after_ts
            ):
                self._freeze()

            self.engine.on_sample(sample)
            for cb in self.on_sample:
                cb(sample)

            if self.store_telemetry:
                write_buf.append(sample)
                if len(write_buf) >= 500:
                    self.repos.telemetry.insert_samples(write_buf, run_id=self.run_id)
                    write_buf = []
            n += 1
            if self.speed > 0 and prev_ts is not None:
                # Pace against real time for live demonstrations only.
                time.sleep(max(0.0, (sample.ts - prev_ts) / self.speed))
            prev_ts = sample.ts
        if write_buf:
            self.repos.telemetry.insert_samples(write_buf, run_id=self.run_id)

        wall = time.perf_counter() - t_wall
        report = self.engine.report()
        result = ReplayResult(
            run_id=self.run_id, n_samples=n,
            n_predictions=self.engine.stats.predictions,
            n_resolved=self.engine.stats.resolved,
            ts_start=ts_start, ts_end=ts_end, wall_s=wall,
            engine_report=report, normalizer_stats=self.normalizer.stats.as_dict(),
        )
        log.info(
            "replay complete", run_id=self.run_id, samples=n,
            predictions=result.n_predictions, resolved=result.n_resolved,
            simulated_min=round((ts_end - ts_start) / 60.0, 1),
            wall_s=round(wall, 2), speedup=round(result.speedup, 1),
        )
        return result


def samples_from_db(
    repos: Repos, run_id: str | None = None, ts_from: float | None = None,
    ts_to: float | None = None, origin: str | None = None,
) -> Iterator[TelemetrySample]:
    return repos.telemetry.iter_samples(
        ts_from=ts_from, ts_to=ts_to, run_id=run_id, origin=origin
    )
