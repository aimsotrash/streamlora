"""Wiring: build the standard telemetry -> storage pipeline from a Config.

Everything above this module takes its dependencies as arguments, so this is
the only place that knows the default assembly order. Experiments and tests
compose the same parts differently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .store.repo import Repos
from .telemetry.collector import Collector
from .telemetry.normalizer import Normalizer
from .telemetry.registry import SourceRegistry, default_sources
from .telemetry.schema import TelemetrySample
from .util.clock import Clock, RealClock
from .util.logging import get_logger

log = get_logger("pipeline")


@dataclass(slots=True)
class TelemetryPipeline:
    config: Config
    repos: Repos
    registry: SourceRegistry
    normalizer: Normalizer
    collector: Collector
    run_id: str | None = None

    def close(self) -> None:
        self.collector.stop()
        self.registry.close()
        self.repos.close()


def build_telemetry_pipeline(
    config: Config,
    clock: Clock | None = None,
    run_id: str | None = None,
    origin: str = "live",
    repos: Repos | None = None,
) -> TelemetryPipeline:
    config.ensure_dirs()
    clock = clock or RealClock()
    repos = repos or Repos.open(config.db_file)

    enable = set(config.collect.sources) if config.collect.sources else None
    if not config.collect.process_attribution and enable is None:
        # Privacy switch: drop the process source entirely rather than
        # collecting and discarding.
        srcs = [s for s in default_sources(None) if s.name != "process"]
    else:
        srcs = default_sources(enable)
    registry = SourceRegistry(srcs, clock)
    health = registry.probe_all()
    repos.signals.register(registry.signals.as_mapping().values())

    normalizer = Normalizer(
        registry.signals,
        expected_interval_s=config.collect.interval_s,
        gap_factor=config.collect.gap_factor,
    )

    def sink(batch: list[TelemetrySample]) -> None:
        repos.telemetry.insert_samples(batch, run_id=run_id)

    collector = Collector(
        registry, normalizer, clock=clock, interval_s=config.collect.interval_s,
        origin=origin, sink=sink, write_batch=config.collect.write_batch,
    )
    available = [n for n, h in health.items() if h.available]
    unavailable = {n: h.detail for n, h in health.items() if not h.available}
    log.info(
        "telemetry pipeline ready",
        sources_ok=len(available), signals=len(registry.signals),
        interval_s=config.collect.interval_s,
    )
    if unavailable:
        log.info("sources unavailable", **unavailable)
    return TelemetryPipeline(config, repos, registry, normalizer, collector, run_id)
