"""Workload regime labelling.

Why this exists: 70% CPU during a compile and 70% CPU during a video call have
different futures. The compile ends abruptly; the call decays slowly. A single
homogeneous time series model cannot express that, so the regime label is fed
to the forecaster as a feature and used to slice every metric.

Design stance on the labels themselves
--------------------------------------
The labels are threshold rules over interpretable features, and the thresholds
live in ``RegimeConfig``. They are deliberately *not* treated as ground truth:

* they are data, not control flow -- adding a label requires editing one table;
* they are evaluated, not assumed -- ``streamlora eval`` reports metrics per
  regime, so a useless label shows up as a slice with no accuracy difference;
* user feedback can override them (``feedback kind='label'``), which is the
  documented path toward learned regimes.

Hysteresis matters more than it looks: without it, CPU oscillating around a
threshold relabels the regime every tick, which turns the one-hot feature into
noise and produces a stream of spurious "regime change" drift alarms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..config import RegimeConfig
from ..telemetry.schema import TelemetrySample

#: Stable ordering, used for the one-hot encoding. Extend at the end so an
#: existing trained model's feature indices remain valid.
REGIMES: tuple[str, ...] = (
    "unknown",
    "idle",
    "interactive",
    "compute_heavy",
    "gpu_heavy",
    "io_heavy",
    "build",
    "battery_constrained",
    "charging",
)
REGIME_INDEX = {r: i for i, r in enumerate(REGIMES)}


@dataclass(slots=True)
class RegimeState:
    label: str = "unknown"
    candidate: str = "unknown"
    streak: int = 0
    since_ts: float = 0.0
    changes: int = 0


class RegimeClassifier:
    """Assigns one label per sample, with hysteresis on transitions."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.cfg = config or RegimeConfig()
        self.state = RegimeState()
        self._history: deque[str] = deque(maxlen=256)

    def _raw_label(self, s: TelemetrySample) -> str:
        cpu = s.value("cpu.util_pct")
        gpu = s.value("gpu.util_pct")
        disk = s.value("disk.busy_pct")
        build = s.value("proc.cpu_build_pct")
        top1 = s.value("proc.top1_cpu_pct")
        batt = s.value("battery.percent")
        plugged = s.value("battery.plugged")
        charging = s.value("battery.charging")
        c = self.cfg

        if cpu is None:
            return "unknown"

        # Power state dominates when it constrains behaviour: a laptop at 15%
        # battery behaves differently at every CPU level, and an actively
        # charging machine has a rising battery signal regardless of workload.
        if charging is not None and charging > 0.5:
            return "charging"
        if batt is not None and plugged is not None and plugged < 0.5 and batt <= c.low_battery_pct:
            return "battery_constrained"

        # Build detection uses share of total CPU, not absolute percent, so it
        # fires on a 4-core laptop and a 32-core desktop alike.
        if build is not None and cpu > c.interactive_cpu_pct:
            total = max(cpu, 1e-6)
            if (build / (total * 1.0)) >= c.build_share or (
                top1 is not None and build >= top1 * 0.8 and build > c.interactive_cpu_pct
            ):
                return "build"

        if gpu is not None and gpu >= c.gpu_active_pct and gpu > cpu:
            return "gpu_heavy"
        if disk is not None and disk >= 40.0 and cpu < c.heavy_cpu_pct:
            return "io_heavy"
        if cpu >= c.heavy_cpu_pct:
            return "compute_heavy"
        if cpu <= c.idle_cpu_pct:
            return "idle"
        return "interactive"

    def update(self, sample: TelemetrySample) -> str:
        raw = self._raw_label(sample)
        st = self.state
        if not self.cfg.enabled:
            st.label = "unknown"
            return st.label
        if st.label == "unknown" and st.streak == 0:
            # Adopt the first observed label immediately; waiting for
            # hysteresis at startup would leave every early feature vector
            # labelled "unknown".
            st.label = raw
            st.candidate = raw
            st.streak = 1
            st.since_ts = sample.ts
            self._history.append(raw)
            return raw
        if raw == st.label:
            st.candidate = raw
            st.streak = 0
        elif raw == st.candidate:
            st.streak += 1
            if st.streak >= self.cfg.hysteresis:
                st.label = raw
                st.streak = 0
                st.since_ts = sample.ts
                st.changes += 1
        else:
            st.candidate = raw
            st.streak = 1
        self._history.append(st.label)
        return st.label

    @property
    def label(self) -> str:
        return self.state.label

    def recent_distribution(self) -> dict[str, float]:
        if not self._history:
            return {}
        n = len(self._history)
        out: dict[str, float] = {}
        for r in self._history:
            out[r] = out.get(r, 0.0) + 1.0 / n
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def one_hot(self, label: str | None = None) -> list[float]:
        lab = label or self.state.label
        vec = [0.0] * len(REGIMES)
        vec[REGIME_INDEX.get(lab, 0)] = 1.0
        return vec
