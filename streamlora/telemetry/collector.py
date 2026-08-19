"""The collection loop.

Responsibilities kept deliberately narrow: sample on a fixed grid, normalise,
hand the sample to subscribers, and persist in batches. It knows nothing about
forecasting -- subscribers do that -- which is what lets the identical loop run
under a simulated clock during replay.

Scheduling detail worth stating: ticks are aligned to a fixed grid derived from
the start time, not scheduled as "sleep(interval) after work". The latter
accumulates the per-tick work cost into the cadence, so a 5.000 s interval
becomes 5.045 s and, over a day, silently loses ~13 minutes of samples and
makes rate-derived signals subtly wrong.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from ..util.clock import Clock, RealClock
from ..util.logging import get_logger
from .normalizer import Normalizer
from .registry import SourceRegistry
from .schema import TelemetrySample

log = get_logger("telemetry.collector")

SampleHandler = Callable[[TelemetrySample], None]


@dataclass(slots=True)
class CollectorStats:
    ticks: int = 0
    written: int = 0
    handler_errors: int = 0
    overruns: int = 0
    skipped_ticks: int = 0
    last_tick_ms: float = 0.0
    avg_tick_ms: float = 0.0
    max_tick_ms: float = 0.0
    last_ts: float | None = None
    source_errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "ticks": self.ticks,
            "written": self.written,
            "handler_errors": self.handler_errors,
            "overruns": self.overruns,
            "skipped_ticks": self.skipped_ticks,
            "last_tick_ms": round(self.last_tick_ms, 2),
            "avg_tick_ms": round(self.avg_tick_ms, 2),
            "max_tick_ms": round(self.max_tick_ms, 2),
            "last_ts": self.last_ts,
            "source_errors": dict(self.source_errors),
        }


class Collector:
    """Samples every available source on a fixed cadence."""

    def __init__(
        self,
        registry: SourceRegistry,
        normalizer: Normalizer,
        clock: Clock | None = None,
        interval_s: float = 5.0,
        origin: str = "live",
        sink: Callable[[list[TelemetrySample]], None] | None = None,
        write_batch: int = 6,
    ) -> None:
        self.registry = registry
        self.normalizer = normalizer
        self.clock = clock or RealClock()
        self.interval_s = float(interval_s)
        self.origin = origin
        self.sink = sink
        self.write_batch = max(1, int(write_batch))
        self.stats = CollectorStats()
        self._handlers: list[SampleHandler] = []
        self._buffer: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_tick: float | None = None

    def subscribe(self, handler: SampleHandler) -> None:
        """Register a callback invoked with each normalised sample.

        Handler exceptions are logged and counted but never propagate: a broken
        forecaster must not stop telemetry collection, which is the one thing we
        can never recover retroactively.
        """
        self._handlers.append(handler)

    # -- single tick -------------------------------------------------------
    def tick(self, now: float | None = None) -> TelemetrySample:
        t_wall = self.clock.now() if now is None else now
        t0 = self.clock.monotonic()
        raw, errors = self.registry.read_all(t_wall)
        collect_ms = (self.clock.monotonic() - t0) * 1000.0
        sample = self.normalizer.normalize(t_wall, raw, origin=self.origin, collect_ms=collect_ms)

        s = self.stats
        s.ticks += 1
        s.last_tick_ms = collect_ms
        s.max_tick_ms = max(s.max_tick_ms, collect_ms)
        s.avg_tick_ms = collect_ms if s.ticks == 1 else 0.95 * s.avg_tick_ms + 0.05 * collect_ms
        s.last_ts = t_wall
        for name in errors:
            s.source_errors[name] = s.source_errors.get(name, 0) + 1

        for h in self._handlers:
            try:
                h(sample)
            except Exception:
                s.handler_errors += 1
                log.exception("sample handler failed", handler=getattr(h, "__name__", repr(h)))

        self._buffer.append(sample)
        if len(self._buffer) >= self.write_batch:
            self.flush()
        return sample

    def flush(self) -> int:
        if not self._buffer or self.sink is None:
            n = len(self._buffer)
            self._buffer.clear()
            return n
        batch, self._buffer = self._buffer, []
        try:
            self.sink(batch)
        except Exception:
            log.exception("telemetry sink failed", batch=len(batch))
            return 0
        self.stats.written += len(batch)
        return len(batch)

    # -- loops -------------------------------------------------------------
    def run_for_ticks(self, n: int) -> list[TelemetrySample]:
        """Synchronous fixed-count loop. Used by tests and by replay."""
        out = []
        for _ in range(n):
            if self._stop.is_set():
                break
            out.append(self.tick())
            self._sleep_to_next_tick()
        self.flush()
        return out

    def run(self, duration_s: float | None = None) -> None:
        """Run until stopped, or for ``duration_s`` of clock time."""
        start = self.clock.now()
        while not self._stop.is_set():
            self.tick()
            if duration_s is not None and (self.clock.now() - start) >= duration_s:
                break
            self._sleep_to_next_tick()
        self.flush()

    def _sleep_to_next_tick(self) -> None:
        """Sleep to the next grid point, skipping any we have already missed."""
        now = self.clock.now()
        if self._next_tick is None:
            self._next_tick = now + self.interval_s
        else:
            self._next_tick += self.interval_s
        if self._next_tick <= now:
            # Work took longer than the interval. Realign to the grid and
            # record it: silently drifting or bursting to catch up would both
            # corrupt rate-derived signals.
            missed = int((now - self._next_tick) // self.interval_s) + 1
            self.stats.overruns += 1
            self.stats.skipped_ticks += missed
            self._next_tick += missed * self.interval_s
            log.dedupe(
                "overrun", "tick overran its interval; realigning",
                level="warning", interval_s=self.interval_s,
                tick_ms=round(self.stats.last_tick_ms, 1), skipped=missed, now=now,
            )
        self.clock.sleep(max(0.0, self._next_tick - self.clock.now()))

    def start_background(self, duration_s: float | None = None) -> None:
        if self._thread is not None:
            raise RuntimeError("collector already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_guarded, args=(duration_s,), name="streamlora-collector", daemon=True
        )
        self._thread.start()

    def _run_guarded(self, duration_s: float | None) -> None:
        try:
            self.run(duration_s)
        except Exception:  # pragma: no cover - last-resort guard
            log.exception("collector loop crashed")

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
            self._thread = None
        self.flush()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def health(self) -> dict[str, object]:
        return {
            "collector": self.stats.as_dict(),
            "normalizer": self.normalizer.stats.as_dict(),
            "sources": {n: h.as_dict() for n, h in self.registry.health.items()},
            "interval_s": self.interval_s,
            "signals": self.registry.expected_signals(),
        }
