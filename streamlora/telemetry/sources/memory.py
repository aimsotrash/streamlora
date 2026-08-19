"""Memory and swap utilisation via psutil."""

from __future__ import annotations

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource

_SPECS = [
    SignalSpec("mem.used_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
               description="Fraction of physical RAM unavailable to new work"),
    SignalSpec("mem.available_gb", "gigabyte", SignalKind.GAUGE, 0.0, 4096.0,
               description="Memory available without swapping"),
    SignalSpec("mem.cached_gb", "gigabyte", SignalKind.GAUGE, 0.0, 4096.0,
               description="Page cache plus reclaimable slab; drops sharply under memory pressure"),
    SignalSpec("mem.swap_used_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
               description="Swap occupancy"),
]
_GB = float(1 << 30)


class MemorySource(TelemetrySource):
    name = "memory"
    description = "Physical memory, page cache and swap usage"

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        try:
            vm = psutil.virtual_memory()
        except Exception as exc:
            return ProbeResult(False, f"virtual_memory failed: {exc}")
        provides = ["mem.used_pct", "mem.available_gb"]
        if hasattr(vm, "cached"):
            provides.append("mem.cached_gb")
        try:
            if psutil.swap_memory().total > 0:
                provides.append("mem.swap_used_pct")
        except Exception:
            pass
        return ProbeResult(True, f"{vm.total / _GB:.1f} GiB total", tuple(provides))

    def read(self, now: float) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        try:
            vm = psutil.virtual_memory()
            out["mem.used_pct"] = float(vm.percent)
            out["mem.available_gb"] = float(vm.available) / _GB
            cached = getattr(vm, "cached", None)
            slab = getattr(vm, "slab", 0.0) or 0.0
            out["mem.cached_gb"] = (float(cached) + float(slab)) / _GB if cached is not None else None
        except Exception:
            out.update({"mem.used_pct": None, "mem.available_gb": None, "mem.cached_gb": None})
        try:
            sw = psutil.swap_memory()
            out["mem.swap_used_pct"] = float(sw.percent) if sw.total > 0 else None
        except Exception:
            out["mem.swap_used_pct"] = None
        return out
