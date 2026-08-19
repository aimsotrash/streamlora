"""Disk and network throughput, derived from OS counters.

The kernel exposes cumulative byte counters, not rates. Converting them is
stateful and has two real failure modes -- counter resets and non-positive time
deltas -- both handled by ``RateTracker`` returning ``None`` for the affected
interval rather than emitting a fabricated spike. The first tick after startup
legitimately has no rate, which is also ``None``.
"""

from __future__ import annotations

import psutil

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, RateTracker, TelemetrySource

_MB = float(1 << 20)

_DISK_SPECS = [
    SignalSpec("disk.read_mbps", "megabyte_per_second", SignalKind.RATE, 0.0, 20000.0,
               description="Disk read throughput"),
    SignalSpec("disk.write_mbps", "megabyte_per_second", SignalKind.RATE, 0.0, 20000.0,
               description="Disk write throughput"),
    SignalSpec("disk.busy_pct", "percent", SignalKind.RATE, 0.0, 100.0,
               description="Fraction of wall time the device had I/O in flight"),
]
_NET_SPECS = [
    SignalSpec("net.recv_mbps", "megabyte_per_second", SignalKind.RATE, 0.0, 20000.0,
               description="Network receive throughput"),
    SignalSpec("net.sent_mbps", "megabyte_per_second", SignalKind.RATE, 0.0, 20000.0,
               description="Network transmit throughput"),
]


class DiskSource(TelemetrySource):
    name = "disk"
    description = "Disk read/write throughput and device busy time"

    def __init__(self) -> None:
        self._read = RateTracker()
        self._write = RateTracker()
        self._busy = RateTracker()
        self._have_busy = True

    def signals(self) -> list[SignalSpec]:
        return list(_DISK_SPECS)

    def probe(self) -> ProbeResult:
        try:
            c = psutil.disk_io_counters()
        except Exception as exc:
            return ProbeResult(False, f"disk_io_counters failed: {exc}")
        if c is None:
            return ProbeResult(False, "no disk counters exposed")
        self._have_busy = hasattr(c, "busy_time")
        provides = ["disk.read_mbps", "disk.write_mbps"]
        if self._have_busy:
            provides.append("disk.busy_pct")
        return ProbeResult(True, "psutil disk_io_counters", tuple(provides))

    def read(self, now: float) -> dict[str, float | None]:
        try:
            c = psutil.disk_io_counters()
        except Exception:
            c = None
        if c is None:
            return {"disk.read_mbps": None, "disk.write_mbps": None, "disk.busy_pct": None}
        r = self._read.update(c.read_bytes, now)
        w = self._write.update(c.write_bytes, now)
        out: dict[str, float | None] = {
            "disk.read_mbps": None if r is None else r / _MB,
            "disk.write_mbps": None if w is None else w / _MB,
            "disk.busy_pct": None,
        }
        if self._have_busy:
            # busy_time is milliseconds of device-busy per unit wall time, so
            # its rate is already a fraction; x100 for percent.
            b = self._busy.update(getattr(c, "busy_time", 0.0), now)
            if b is not None:
                out["disk.busy_pct"] = min(100.0, b / 10.0)
        return out


class NetworkSource(TelemetrySource):
    name = "network"
    description = "Network receive/transmit throughput"

    def __init__(self) -> None:
        self._rx = RateTracker()
        self._tx = RateTracker()

    def signals(self) -> list[SignalSpec]:
        return list(_NET_SPECS)

    def probe(self) -> ProbeResult:
        try:
            c = psutil.net_io_counters()
        except Exception as exc:
            return ProbeResult(False, f"net_io_counters failed: {exc}")
        if c is None:
            return ProbeResult(False, "no network counters exposed")
        return ProbeResult(True, "psutil net_io_counters", ("net.recv_mbps", "net.sent_mbps"))

    def read(self, now: float) -> dict[str, float | None]:
        try:
            c = psutil.net_io_counters()
        except Exception:
            c = None
        if c is None:
            return {"net.recv_mbps": None, "net.sent_mbps": None}
        rx = self._rx.update(c.bytes_recv, now)
        tx = self._tx.update(c.bytes_sent, now)
        return {
            "net.recv_mbps": None if rx is None else rx / _MB,
            "net.sent_mbps": None if tx is None else tx / _MB,
        }
