"""NVIDIA GPU telemetry.

Two backends. NVML (``nvidia-ml-py``) is used when importable because a full
read costs ~0.02 ms; shelling out to ``nvidia-smi`` costs ~45 ms, which at a
5 s cadence is a 1% duty cycle spent on one optional signal. The subprocess
path therefore declares a slower native cadence (``min_interval_s``) and the
collector decimates it, holding the previous value in between.

This source is NVIDIA-specific by necessity, but nothing above it is: the
signal names are vendor-neutral, so an AMD or Apple implementation can be
added as a sibling class emitting the same names.
"""

from __future__ import annotations

import shutil
import subprocess

from ..schema import SignalKind, SignalSpec
from .base import ProbeResult, TelemetrySource

_SPECS = [
    SignalSpec("gpu.util_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
               description="Fraction of the last sampling period with a kernel resident"),
    SignalSpec("gpu.mem_used_pct", "percent", SignalKind.GAUGE, 0.0, 100.0,
               description="VRAM occupancy"),
    SignalSpec("gpu.mem_used_gb", "gigabyte", SignalKind.GAUGE, 0.0, 256.0,
               description="VRAM in use"),
    SignalSpec("gpu.temp_c", "celsius", SignalKind.GAUGE, -20.0, 130.0, max_rate_per_s=15.0,
               description="GPU die temperature"),
    SignalSpec("gpu.power_w", "watt", SignalKind.GAUGE, 0.0, 800.0,
               description="Board power draw"),
]

_QUERY = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"


class NvidiaGpuSource(TelemetrySource):
    name = "gpu_nvidia"
    description = "NVIDIA GPU utilisation, VRAM, temperature and power"

    #: Native cadence hint honoured by the collector. NVML is cheap enough to
    #: read every tick; the subprocess fallback is not.
    min_interval_s: float = 0.0

    def __init__(self) -> None:
        self._nvml = None
        self._handle = None
        self._mode = "none"
        self._mem_total_gb: float | None = None

    def signals(self) -> list[SignalSpec]:
        return list(_SPECS)

    def probe(self) -> ProbeResult:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml = pynvml
            self._mode = "nvml"
            self.min_interval_s = 0.0
            raw = pynvml.nvmlDeviceGetName(self._handle)
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._mem_total_gb = mem.total / float(1 << 30)
            return ProbeResult(True, f"NVML: {name}", tuple(s.name for s in _SPECS))
        except Exception:
            self._nvml = None
        if shutil.which("nvidia-smi"):
            probe = self._read_smi()
            if probe:
                self._mode = "smi"
                # 45 ms per read: sample at most every 10 s and let the
                # collector hold the value between reads.
                self.min_interval_s = 10.0
                return ProbeResult(
                    True, "nvidia-smi subprocess (decimated to 10 s)",
                    tuple(s.name for s in _SPECS),
                )
        self._mode = "none"
        return ProbeResult(False, "no NVML and no usable nvidia-smi")

    def read(self, now: float) -> dict[str, float | None]:
        if self._mode == "nvml":
            return self._read_nvml()
        if self._mode == "smi":
            return self._read_smi() or dict.fromkeys(s.name for s in _SPECS)
        return dict.fromkeys(s.name for s in _SPECS)

    def _read_nvml(self) -> dict[str, float | None]:
        p, h = self._nvml, self._handle
        out: dict[str, float | None] = dict.fromkeys(s.name for s in _SPECS)
        if p is None or h is None:
            return out
        # Each metric is queried independently: on laptop GPUs in low-power
        # states individual queries return NOT_SUPPORTED while others succeed.
        try:
            out["gpu.util_pct"] = float(p.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass
        try:
            m = p.nvmlDeviceGetMemoryInfo(h)
            out["gpu.mem_used_gb"] = m.used / float(1 << 30)
            out["gpu.mem_used_pct"] = 100.0 * m.used / m.total if m.total else None
        except Exception:
            pass
        try:
            out["gpu.temp_c"] = float(p.nvmlDeviceGetTemperature(h, 0))
        except Exception:
            pass
        try:
            out["gpu.power_w"] = p.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pass
        return out

    def _read_smi(self) -> dict[str, float | None] | None:
        try:
            proc = subprocess.run(
                ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 5:
            return None

        def num(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None  # "[N/A]" on unsupported metrics

        util, used, total, temp, power = (num(p) for p in parts[:5])
        out: dict[str, float | None] = {
            "gpu.util_pct": util,
            "gpu.mem_used_gb": None if used is None else used / 1024.0,
            "gpu.mem_used_pct": None if (used is None or not total) else 100.0 * used / total,
            "gpu.temp_c": temp,
            "gpu.power_w": power,
        }
        return out

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
