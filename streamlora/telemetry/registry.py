"""Source discovery, capability probing and per-source health tracking.

Two responsibilities that the spec calls out explicitly:

* **Discover what telemetry exists** rather than assuming a platform. Every
  source is probed; unavailable ones are recorded with a reason and skipped.
* **Degrade gracefully.** A source that starts failing mid-run is quarantined
  with exponential backoff and retried later. A missing battery must never stop
  CPU forecasting, so failures are contained per source, not per collector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..util.clock import Clock
from ..util.logging import get_logger
from .schema import SignalRegistry, SignalSpec
from .sources.base import ProbeResult, TelemetrySource

log = get_logger("telemetry.registry")

#: Backoff schedule (seconds) applied after consecutive read failures. The first
#: entry is non-zero on purpose: reaching the failure threshold and then
#: retrying immediately is not a backoff, and a source failing on every tick
#: would keep paying its full read cost forever.
_BACKOFF = (5.0, 15.0, 60.0, 300.0)
#: Consecutive failures before a source is quarantined at all.
_FAIL_THRESHOLD = 3


@dataclass(slots=True)
class SourceHealth:
    """Live operational state for one source. Surfaced in /api/health."""

    name: str
    available: bool
    detail: str = ""
    provides: tuple[str, ...] = ()
    reads: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_ok_ts: float | None = None
    last_error: str = ""
    last_read_ms: float = 0.0
    #: Exponential moving average of read cost, for the observability view.
    avg_read_ms: float = 0.0
    quarantined_until: float = 0.0
    #: Native cadence; 0 means "every tick".
    min_interval_s: float = 0.0
    last_sampled_ts: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "provides": list(self.provides),
            "reads": self.reads,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "last_ok_ts": self.last_ok_ts,
            "last_error": self.last_error,
            "avg_read_ms": round(self.avg_read_ms, 3),
            "quarantined": self.quarantined_until > 0.0,
            "min_interval_s": self.min_interval_s,
        }


def default_sources(enable: set[str] | None = None) -> list[TelemetrySource]:
    """Construct the built-in sources.

    Construction must be cheap and must not raise on unsupported platforms --
    availability is decided by ``probe()``, not by import or __init__ errors.
    Import failures are contained here so that, for example, a machine without
    psutil's sensor support still gets CPU and memory.
    """
    out: list[TelemetrySource] = []
    factories: list[tuple[str, object]] = []
    from .sources.cpu import CpuSource
    from .sources.memory import MemorySource
    factories.append(("cpu", CpuSource))
    factories.append(("memory", MemorySource))
    try:
        from .sources.battery import BatterySource
        factories.append(("battery", BatterySource))
    except Exception as exc:  # pragma: no cover
        log.warning("battery source import failed", error=str(exc))
    try:
        from .sources.thermal import ThermalSource
        factories.append(("thermal", ThermalSource))
    except Exception as exc:  # pragma: no cover
        log.warning("thermal source import failed", error=str(exc))
    try:
        from .sources.gpu import NvidiaGpuSource
        factories.append(("gpu_nvidia", NvidiaGpuSource))
    except Exception as exc:  # pragma: no cover
        log.warning("gpu source import failed", error=str(exc))
    try:
        from .sources.io import DiskSource, NetworkSource
        factories.append(("disk", DiskSource))
        factories.append(("network", NetworkSource))
    except Exception as exc:  # pragma: no cover
        log.warning("io source import failed", error=str(exc))
    try:
        from .sources.process import ProcessSource
        factories.append(("process", ProcessSource))
    except Exception as exc:  # pragma: no cover
        log.warning("process source import failed", error=str(exc))

    for name, factory in factories:
        if enable is not None and name not in enable:
            continue
        try:
            out.append(factory())  # type: ignore[operator]
        except Exception as exc:
            log.warning("source construction failed", source=name, error=str(exc))
    return out


class SourceRegistry:
    """Owns the live sources, their health, and the merged signal registry."""

    def __init__(self, sources: list[TelemetrySource], clock: Clock) -> None:
        self._sources = sources
        self._clock = clock
        self.signals = SignalRegistry()
        self.health: dict[str, SourceHealth] = {}
        self._active: list[TelemetrySource] = []
        self._last_values: dict[str, float | None] = {}

    # -- lifecycle ---------------------------------------------------------
    def probe_all(self) -> dict[str, SourceHealth]:
        """Probe every source once. Safe to call again to re-detect hardware."""
        self._active = []
        for src in self._sources:
            try:
                result = src.probe()
            except Exception as exc:
                result = ProbeResult(False, f"probe raised {type(exc).__name__}: {exc}")
                log.warning("source probe raised", source=src.name, error=str(exc))
            provides = result.provides or tuple(s.name for s in self._safe_signals(src))
            h = SourceHealth(
                name=src.name,
                available=result.available,
                detail=result.detail,
                provides=provides,
                min_interval_s=float(getattr(src, "min_interval_s", 0.0) or 0.0),
            )
            self.health[src.name] = h
            if not result.available:
                log.info("source unavailable", source=src.name, reason=result.detail)
                continue
            # Register only the specs the source says it will actually emit.
            for spec in self._safe_signals(src):
                if spec.name in provides:
                    try:
                        self.signals.add(spec)
                    except ValueError as exc:
                        log.error("signal spec conflict", source=src.name, error=str(exc))
            self._active.append(src)
            log.info(
                "source ready", source=src.name, detail=result.detail, signals=len(provides)
            )
        return dict(self.health)

    @staticmethod
    def _safe_signals(src: TelemetrySource) -> list[SignalSpec]:
        try:
            return src.signals()
        except Exception:  # pragma: no cover
            return []

    @property
    def active(self) -> list[TelemetrySource]:
        return list(self._active)

    def expected_signals(self) -> list[str]:
        return self.signals.names()

    # -- reading -----------------------------------------------------------
    def read_all(self, now: float) -> tuple[dict[str, float | None], dict[str, str]]:
        """Read every healthy source.

        Returns ``(values, errors)``. ``values`` maps signal -> raw value, with
        ``None`` for signals a source could not produce. ``errors`` maps source
        name -> message for sources that failed entirely this tick.
        """
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for src in self._active:
            h = self.health[src.name]
            if h.quarantined_until > now:
                for sig in h.provides:
                    values.setdefault(sig, None)
                continue
            # Honour a source's declared native cadence: reuse the previous
            # value instead of paying the read cost every tick. This is
            # intentional decimation, not staleness, so quality stays OK.
            if h.min_interval_s > 0 and (now - h.last_sampled_ts) < h.min_interval_s:
                for sig in h.provides:
                    values[sig] = self._last_values.get(sig)
                continue
            t0 = self._clock.monotonic()
            try:
                out = src.read(now)
            except Exception as exc:
                dt = (self._clock.monotonic() - t0) * 1000.0
                self._record_failure(h, now, f"{type(exc).__name__}: {exc}", dt)
                errors[src.name] = h.last_error
                for sig in h.provides:
                    values.setdefault(sig, None)
                continue
            dt = (self._clock.monotonic() - t0) * 1000.0
            self._record_success(h, now, dt)
            for sig in h.provides:
                v = out.get(sig)
                values[sig] = v
                if v is not None:
                    self._last_values[sig] = float(v)
            # A source may emit a signal it did not declare (new hardware seen
            # after probe). Accept it if we have a spec, otherwise ignore.
            for sig, v in out.items():
                if sig not in h.provides and sig in self.signals:
                    values.setdefault(sig, v)
        return values, errors

    def _record_success(self, h: SourceHealth, now: float, ms: float) -> None:
        h.reads += 1
        h.consecutive_failures = 0
        h.quarantined_until = 0.0
        h.last_ok_ts = now
        h.last_read_ms = ms
        h.last_sampled_ts = now
        h.avg_read_ms = ms if h.reads == 1 else 0.9 * h.avg_read_ms + 0.1 * ms

    def _record_failure(self, h: SourceHealth, now: float, msg: str, ms: float) -> None:
        h.failures += 1
        h.consecutive_failures += 1
        h.last_error = msg
        h.last_read_ms = ms
        if h.consecutive_failures >= _FAIL_THRESHOLD:
            idx = min(h.consecutive_failures - _FAIL_THRESHOLD, len(_BACKOFF) - 1)
            h.quarantined_until = now + _BACKOFF[idx]
            log.dedupe(
                f"quarantine:{h.name}",
                "source quarantined after repeated failures",
                source=h.name,
                consecutive_failures=h.consecutive_failures,
                backoff_s=_BACKOFF[idx],
                error=msg,
                now=now,
            )
        else:
            log.dedupe(
                f"readfail:{h.name}", "source read failed", source=h.name, error=msg, now=now
            )

    def close(self) -> None:
        for src in self._sources:
            try:
                src.close()
            except Exception:  # pragma: no cover
                pass
