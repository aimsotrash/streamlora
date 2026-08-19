"""CPU utilisation, frequency and load average via psutil."""

from __future__ import annotations

import os

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource

_SPECS = [
    SignalSpec(
        "cpu.util_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
        max_rate_per_s=None, description="Aggregate CPU utilisation since the previous tick",
    ),
    SignalSpec(
        "cpu.util_max_core_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
        description="Utilisation of the busiest logical core; separates one pinned thread from a parallel build",
    ),
    SignalSpec(
        "cpu.iowait_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
        description="Fraction of CPU time blocked on I/O",
    ),
    SignalSpec(
        "cpu.freq_mhz", "megahertz", SignalKind.GAUGE, 0.0, 12000.0,
        description="Current average core frequency",
    ),
    SignalSpec(
        "cpu.load1_per_core", "ratio", SignalKind.GAUGE, 0.0, 64.0,
        description="1-minute load average divided by logical core count",
    ),
]


class CpuSource(TelemetrySource):
    name = "cpu"
    description = "CPU utilisation, per-core peak, frequency and load average"

    def __init__(self) -> None:
        self._ncpu = os.cpu_count() or 1
        self._have_freq = True
        self._have_percpu = True
        self._have_load = hasattr(os, "getloadavg")
        # psutil's cpu_percent is delta-based: the first call after construction
        # returns time since boot, which is meaningless here. Prime it so the
        # first collected sample already reflects a real interval.
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        provides = ["cpu.util_pct", "cpu.util_max_core_pct"]
        try:
            psutil.cpu_times_percent(interval=None)
            provides.append("cpu.iowait_pct")
        except Exception:
            pass
        try:
            if psutil.cpu_freq() is not None:
                provides.append("cpu.freq_mhz")
            else:
                self._have_freq = False
        except Exception:
            self._have_freq = False
        if self._have_load:
            provides.append("cpu.load1_per_core")
        return ProbeResult(True, f"{self._ncpu} logical cores", tuple(provides))

    def read(self, now: float) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        # Aggregate utilisation. psutil computes this against its own previous
        # snapshot, which is exactly our previous tick.
        try:
            out["cpu.util_pct"] = float(psutil.cpu_percent(interval=None))
        except Exception:
            out["cpu.util_pct"] = None
        try:
            per = psutil.cpu_percent(interval=None, percpu=True)
            out["cpu.util_max_core_pct"] = float(max(per)) if per else None
        except Exception:
            out["cpu.util_max_core_pct"] = None
        try:
            t = psutil.cpu_times_percent(interval=None)
            out["cpu.iowait_pct"] = float(getattr(t, "iowait", 0.0) or 0.0)
        except Exception:
            out["cpu.iowait_pct"] = None
        if self._have_freq:
            try:
                f = psutil.cpu_freq()
                out["cpu.freq_mhz"] = float(f.current) if f else None
            except Exception:
                out["cpu.freq_mhz"] = None
        if self._have_load:
            try:
                out["cpu.load1_per_core"] = float(os.getloadavg()[0]) / self._ncpu
            except OSError:
                out["cpu.load1_per_core"] = None
        return out
