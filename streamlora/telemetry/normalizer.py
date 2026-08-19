"""Turn raw source output into validated, quality-tagged samples.

Everything here exists because of a specific way real telemetry misbehaves:

* **Malformed values.** NaN and inf reach us from sensor drivers and from
  divisions by a zero total. They must never enter a model: one NaN in a
  least-squares update poisons every subsequent coefficient permanently.
* **Out-of-range values.** A CPU percent of 100.4 (rounding in psutil) is
  benign and should be clipped. A battery percent of 6000 is a driver bug and
  should also be clipped rather than dropped, but flagged.
* **Physically impossible jumps.** Battery leaping 40% in one tick is a sensor
  glitch, not a discharge. Flagged SUSPECT so evaluation can exclude it while
  the value stays visible for debugging.
* **Momentary unavailability.** A sensor that misses one tick should not create
  a hole in the feature window; holding the last value for a bounded time is
  correct and is marked STALE so nothing pretends it was measured.
* **Gaps.** Suspend/resume produces multi-hour jumps in timestamps. Holding a
  value across such a gap would be a lie, so hold state is discarded and the
  gap is recorded on the sample for the feature layer to see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..util.logging import get_logger
from .schema import Quality, Reading, SignalRegistry, SignalSpec, TelemetrySample

log = get_logger("telemetry.normalizer")


@dataclass(slots=True)
class _Held:
    value: float
    ts: float


@dataclass(slots=True)
class NormalizerStats:
    samples: int = 0
    clamped: int = 0
    suspect: int = 0
    stale: int = 0
    missing: int = 0
    malformed: int = 0
    gaps: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "samples": self.samples,
            "clamped": self.clamped,
            "suspect": self.suspect,
            "stale": self.stale,
            "missing": self.missing,
            "malformed": self.malformed,
            "gaps": self.gaps,
        }


class Normalizer:
    """Stateful per-signal validation. One instance per telemetry stream."""

    def __init__(
        self,
        signals: SignalRegistry,
        expected_interval_s: float = 5.0,
        gap_factor: float = 3.0,
    ) -> None:
        self.signals = signals
        self.expected_interval_s = float(expected_interval_s)
        #: A gap larger than ``gap_factor`` x expected interval is treated as a
        #: discontinuity rather than jitter.
        self.gap_factor = float(gap_factor)
        self._held: dict[str, _Held] = {}
        self._last_ts: float | None = None
        self.stats = NormalizerStats()

    @property
    def gap_threshold_s(self) -> float:
        return self.expected_interval_s * self.gap_factor

    def normalize(
        self, ts: float, raw: dict[str, float | None], origin: str = "live", collect_ms: float = 0.0
    ) -> TelemetrySample:
        gap_s: float | None = None
        discontinuity = False
        if self._last_ts is not None:
            gap_s = ts - self._last_ts
            if gap_s > self.gap_threshold_s or gap_s < 0:
                discontinuity = True
                self.stats.gaps += 1
                log.dedupe(
                    "gap",
                    "timestamp gap detected; hold state cleared",
                    level="info",
                    gap_s=round(gap_s, 2),
                    threshold_s=round(self.gap_threshold_s, 2),
                    now=ts,
                )
        if discontinuity:
            # Never carry a held value across a discontinuity.
            self._held.clear()

        readings: dict[str, Reading] = {}
        # Iterate over the union so a signal that vanished from source output is
        # still represented (as STALE or MISSING) instead of silently absent.
        names = set(raw) | set(self._held) | set(self.signals.names())
        for name in names:
            spec = self.signals.get(name)
            if spec is None:
                continue
            readings[name] = self._normalize_one(spec, raw.get(name, None), ts)

        self._last_ts = ts
        self.stats.samples += 1
        return TelemetrySample(
            ts=ts, readings=readings, collect_ms=collect_ms, origin=origin, gap_s=gap_s
        )

    def _normalize_one(self, spec: SignalSpec, value: float | None, ts: float) -> Reading:
        if value is None:
            return self._hold_or_missing(spec, ts)
        try:
            v = float(value)
        except (TypeError, ValueError):
            self.stats.malformed += 1
            log.dedupe(f"malformed:{spec.name}", "non-numeric reading", signal=spec.name, now=ts)
            return self._hold_or_missing(spec, ts)
        if not math.isfinite(v):
            self.stats.malformed += 1
            log.dedupe(
                f"nonfinite:{spec.name}", "non-finite reading", signal=spec.name, value=v, now=ts
            )
            return self._hold_or_missing(spec, ts)

        quality = Quality.OK
        if not spec.in_range(v):
            lo = spec.lo if spec.lo is not None else -math.inf
            hi = spec.hi if spec.hi is not None else math.inf
            clipped = float(min(max(v, lo), hi))
            # Sub-unit overshoot (100.4% CPU) is rounding, not a fault; clip it
            # without spending a log line or a quality downgrade.
            if abs(v - clipped) > 1e-6:
                self.stats.clamped += 1
                quality = Quality.CLAMPED
                log.dedupe(
                    f"range:{spec.name}",
                    "reading outside declared range; clamped",
                    signal=spec.name, value=v, lo=spec.lo, hi=spec.hi, now=ts,
                )
            v = clipped

        held = self._held.get(spec.name)
        if spec.max_rate_per_s is not None and held is not None:
            dt = ts - held.ts
            if dt > 0:
                rate = abs(v - held.value) / dt
                if rate > spec.max_rate_per_s:
                    self.stats.suspect += 1
                    quality = max(quality, Quality.SUSPECT)
                    log.dedupe(
                        f"spike:{spec.name}",
                        "implausible rate of change",
                        signal=spec.name, rate_per_s=round(rate, 4),
                        limit=spec.max_rate_per_s, value=v, prev=held.value, now=ts,
                    )
        self._held[spec.name] = _Held(v, ts)
        return Reading(v, quality)

    def _hold_or_missing(self, spec: SignalSpec, ts: float) -> Reading:
        held = self._held.get(spec.name)
        if held is not None and (ts - held.ts) <= spec.max_hold_s:
            self.stats.stale += 1
            return Reading(held.value, Quality.STALE)
        self.stats.missing += 1
        log.dedupe(
            f"missing:{spec.name}", "signal unavailable", level="debug",
            signal=spec.name, now=ts,
        )
        return Reading(None, Quality.MISSING)

    def reset(self) -> None:
        self._held.clear()
        self._last_ts = None
