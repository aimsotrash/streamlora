"""Telemetry data model.

Design goals
------------
1. **Extensible.** Adding GPU temperature or fan RPM later must not require a
   schema migration or a change to the forecaster. A signal is identified by a
   dotted string name and described by a ``SignalSpec``; storage is long-format.
2. **Honest about quality.** A reading is never silently faked. Every value
   carries a ``Quality`` so downstream code can distinguish "CPU was at 0%"
   from "we could not read the CPU".
3. **Canonical units.** Sources convert to the unit declared in the spec. No
   downstream component ever has to guess whether memory is in MB or bytes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, Mapping


class Quality(enum.IntEnum):
    """Why a value should or should not be trusted.

    Ordered by increasing severity so ``max()`` over a set of qualities gives a
    sensible aggregate for a whole sample.
    """

    OK = 0
    CLAMPED = 1        # value was outside the declared range and was clipped
    SUSPECT = 2        # value changed faster than physically plausible
    STALE = 3          # last known value reused because the source returned nothing
    IMPUTED = 4        # value filled in by the normalizer (short gap)
    MISSING = 5        # no value available

    @property
    def usable(self) -> bool:
        """True if a forecaster may consume the value as a real measurement."""
        return self <= Quality.SUSPECT


class SignalKind(enum.StrEnum):
    GAUGE = "gauge"        # instantaneous level, e.g. cpu.util_pct
    RATE = "rate"          # per-second flow derived from a counter, e.g. disk.read_bps
    FLAG = "flag"          # 0/1 state, e.g. battery.plugged
    COUNTER = "counter"    # monotonically increasing total


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """Static description of one telemetry signal."""

    name: str
    unit: str
    kind: SignalKind = SignalKind.GAUGE
    lo: float | None = None
    hi: float | None = None
    #: Largest plausible absolute change per second. Used for spike screening.
    #: ``None`` disables the check (correct for genuinely bursty rates).
    max_rate_per_s: float | None = None
    #: Longest gap (seconds) over which the normalizer may hold the last value.
    max_hold_s: float = 30.0
    description: str = ""

    def in_range(self, v: float) -> bool:
        if self.lo is not None and v < self.lo:
            return False
        if self.hi is not None and v > self.hi:
            return False
        return True


@dataclass(frozen=True, slots=True)
class Reading:
    """One value for one signal at one instant."""

    value: float | None
    quality: Quality = Quality.OK

    @property
    def usable(self) -> bool:
        return self.value is not None and self.quality.usable

    def as_float(self, default: float = 0.0) -> float:
        return default if self.value is None else float(self.value)


@dataclass(slots=True)
class TelemetrySample:
    """All readings collected at one tick.

    ``ts`` is the collector's timestamp for the tick, not per-signal read time:
    signals within one tick are treated as simultaneous, which is accurate at
    our sampling cadence (seconds) relative to sensor read cost (microseconds).
    """

    ts: float
    readings: dict[str, Reading] = field(default_factory=dict)
    #: Wall-clock seconds spent collecting this sample.
    collect_ms: float = 0.0
    #: Free-form provenance: "live", "replay", "synthetic:<scenario>".
    origin: str = "live"
    #: Seconds since the previous sample, or None for the first sample.
    gap_s: float | None = None

    def get(self, signal: str) -> Reading:
        return self.readings.get(signal, Reading(None, Quality.MISSING))

    def value(self, signal: str) -> float | None:
        r = self.readings.get(signal)
        return r.value if r is not None and r.usable else None

    def usable_signals(self) -> list[str]:
        return sorted(k for k, r in self.readings.items() if r.usable)

    @property
    def worst_quality(self) -> Quality:
        if not self.readings:
            return Quality.MISSING
        return max(r.quality for r in self.readings.values())

    def to_row(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "collect_ms": self.collect_ms,
            "origin": self.origin,
            "gap_s": self.gap_s,
            "worst_quality": int(self.worst_quality),
        }


class SignalRegistry:
    """Name -> SignalSpec lookup, populated by whichever sources are available.

    Deliberately mutable at runtime: a source that appears later (external GPU
    plugged in, battery present after a reboot) can register its signals
    without restarting the process.
    """

    def __init__(self, specs: Iterable[SignalSpec] = ()) -> None:
        self._specs: dict[str, SignalSpec] = {}
        for s in specs:
            self.add(s)

    def add(self, spec: SignalSpec) -> None:
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(
                f"conflicting spec for signal {spec.name!r}: {existing} vs {spec}"
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> SignalSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> SignalSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"unknown signal: {name!r}")
        return spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def as_mapping(self) -> Mapping[str, SignalSpec]:
        return dict(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)
