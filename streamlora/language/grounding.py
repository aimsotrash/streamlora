"""Evidence packs: the facts a language answer is allowed to use.

This module contains no language model and never will. It reads telemetry,
predictions, drift events and adaptation history out of SQLite and produces a
structured ``EvidencePack``: a list of typed facts with values, units and
timestamps, plus a few derived analyses (discharge rate, driver correlations,
recent forecast accuracy).

The reason for the hard split is the spec's requirement that the model must not
invent measurements. Numbers exist only here. The language layer receives them,
is asked to phrase them, and is then *checked* against them by
``language/verify.py``. If a number appears in an answer that is not in the
pack, that is a measurable hallucination, not a matter of opinion.

A secondary benefit: every answer the system gives can show its evidence, which
is what makes the explanation view real rather than decorative.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..config import Config
from ..evaluate import report as R
from ..store.repo import PredictionRecord, Repos
from ..telemetry.schema import SignalRegistry

#: Signals offered as candidate explanations for a change in another signal.
DRIVER_CANDIDATES: tuple[str, ...] = (
    "cpu.util_pct", "cpu.util_max_core_pct", "cpu.iowait_pct", "gpu.util_pct",
    "gpu.power_w", "mem.used_pct", "disk.busy_pct", "net.recv_mbps",
    "thermal.cpu_c", "proc.cpu_build_pct", "proc.cpu_browser_pct",
    "proc.cpu_media_pct", "proc.cpu_ml_pct", "proc.cpu_container_pct",
    "proc.cpu_editor_pct", "proc.top1_cpu_pct", "battery.power_w",
)

#: Human-facing names, so an answer says "GPU utilisation" not "gpu.util_pct".
SIGNAL_LABELS: dict[str, str] = {
    "cpu.util_pct": "CPU utilisation",
    "cpu.util_max_core_pct": "busiest CPU core",
    "cpu.iowait_pct": "CPU I/O wait",
    "cpu.load1_per_core": "load average per core",
    "cpu.freq_mhz": "CPU frequency",
    "mem.used_pct": "memory usage",
    "mem.available_gb": "available memory",
    "mem.cached_gb": "page cache",
    "mem.swap_used_pct": "swap usage",
    "battery.percent": "battery charge",
    "battery.power_w": "battery power draw",
    "battery.plugged": "AC power",
    "battery.charging": "charging",
    "gpu.util_pct": "GPU utilisation",
    "gpu.power_w": "GPU power",
    "gpu.temp_c": "GPU temperature",
    "gpu.mem_used_pct": "GPU memory",
    "thermal.cpu_c": "CPU temperature",
    "thermal.gpu_c": "GPU temperature",
    "thermal.max_c": "hottest sensor",
    "disk.busy_pct": "disk busy time",
    "disk.read_mbps": "disk reads",
    "disk.write_mbps": "disk writes",
    "net.recv_mbps": "network download",
    "net.sent_mbps": "network upload",
    "proc.cpu_build_pct": "build/compile activity",
    "proc.cpu_browser_pct": "browser activity",
    "proc.cpu_media_pct": "media activity",
    "proc.cpu_ml_pct": "ML/Python activity",
    "proc.cpu_container_pct": "container activity",
    "proc.cpu_editor_pct": "editor activity",
    "proc.top1_cpu_pct": "busiest single process",
    "proc.concentration": "workload concentration",
    "proc.count": "process count",
}

UNIT_SUFFIX: dict[str, str] = {
    "percent": "%", "celsius": " degC", "watt": " W", "gigabyte": " GB",
    "megahertz": " MHz", "megabyte_per_second": " MB/s", "ratio": "", "count": "",
    "bool": "",
}


def label_for(signal: str) -> str:
    return SIGNAL_LABELS.get(signal, signal)


def fmt(value: float | None, unit: str = "", digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}{UNIT_SUFFIX.get(unit, '')}"


def fmt_duration(seconds: float) -> str:
    s = abs(float(seconds))
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


@dataclass(slots=True)
class Fact:
    """One checkable statement. ``value`` is what the verifier matches against."""

    key: str
    text: str
    value: float | None = None
    unit: str = ""
    source: str = "telemetry"
    ts: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key, "text": self.text, "value": self.value,
            "unit": self.unit, "source": self.source, "ts": self.ts,
        }


@dataclass(slots=True)
class ForecastFact:
    signal: str
    horizon_s: float
    value: float
    lo: float | None
    hi: float | None
    model_kind: str
    model_version: str
    ts_made: float
    ts_target: float
    unit: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal, "horizon_s": self.horizon_s, "value": self.value,
            "lo": self.lo, "hi": self.hi, "model_kind": self.model_kind,
            "model_version": self.model_version, "ts_made": self.ts_made,
            "ts_target": self.ts_target, "unit": self.unit,
        }

    def describe(self) -> str:
        base = f"{label_for(self.signal)} in {fmt_duration(self.horizon_s)}: {fmt(self.value, self.unit)}"
        if self.lo is not None and self.hi is not None:
            base += f" (likely {fmt(self.lo, self.unit)} to {fmt(self.hi, self.unit)})"
        return base


@dataclass(slots=True)
class Correlation:
    signal: str
    r: float
    lag_s: float
    change: float
    unit: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"signal": self.signal, "r": self.r, "lag_s": self.lag_s,
                "change": self.change, "unit": self.unit}


@dataclass
class EvidencePack:
    """Everything a grounded answer may draw on."""

    now: float
    question: str = ""
    intent: str = "status"
    window_s: float = 3600.0
    regime: str = "unknown"
    regime_since_s: float | None = None
    regime_distribution: dict[str, float] = field(default_factory=dict)
    current: list[Fact] = field(default_factory=list)
    window: list[Fact] = field(default_factory=list)
    forecasts: list[ForecastFact] = field(default_factory=list)
    accuracy: list[Fact] = field(default_factory=list)
    drivers: list[Correlation] = field(default_factory=list)
    events: list[Fact] = field(default_factory=list)
    #: What this signal usually did next in the same regime.
    history: list[Fact] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    user_context: list[str] = field(default_factory=list)
    #: Target signal the question is about, when identifiable.
    focus_signal: str | None = None

    # -- numeric surface, for the verifier ---------------------------------
    def numeric_values(self) -> list[float]:
        out: list[float] = []
        for f in (*self.current, *self.window, *self.accuracy, *self.events, *self.history):
            if f.value is not None and math.isfinite(f.value):
                out.append(float(f.value))
        for fc in self.forecasts:
            out.append(float(fc.value))
            if fc.lo is not None:
                out.append(float(fc.lo))
            if fc.hi is not None:
                out.append(float(fc.hi))
            out.append(float(fc.horizon_s))
            out.append(float(fc.horizon_s / 60.0))
        for c in self.drivers:
            out.append(float(c.change))
            out.append(float(round(c.r, 2)))
        out.append(float(round(self.window_s / 60.0, 1)))
        if self.regime_since_s is not None:
            out.append(float(round(self.regime_since_s / 60.0, 1)))
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "now": self.now, "question": self.question, "intent": self.intent,
            "window_s": self.window_s, "regime": self.regime,
            "regime_since_s": self.regime_since_s,
            "regime_distribution": self.regime_distribution,
            "current": [f.as_dict() for f in self.current],
            "window": [f.as_dict() for f in self.window],
            "forecasts": [f.as_dict() for f in self.forecasts],
            "accuracy": [f.as_dict() for f in self.accuracy],
            "drivers": [c.as_dict() for c in self.drivers],
            "events": [f.as_dict() for f in self.events],
            "history": [f.as_dict() for f in self.history],
            "notes": list(self.notes),
            "user_context": list(self.user_context),
            "focus_signal": self.focus_signal,
        }

    def render(self, max_lines: int = 60) -> str:
        """Compact text rendering, used as prompt context and as UI evidence."""
        lines: list[str] = []
        lines.append(f"TIME: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.now))}")
        reg = f"REGIME: {self.regime}"
        if self.regime_since_s is not None:
            reg += f" (for {fmt_duration(self.regime_since_s)})"
        lines.append(reg)
        if self.current:
            lines.append("NOW:")
            lines += [f"  {f.text}" for f in self.current]
        if self.window:
            lines.append(f"LAST {fmt_duration(self.window_s)}:")
            lines += [f"  {f.text}" for f in self.window]
        if self.drivers:
            lines.append("CORRELATED ACTIVITY:")
            for c in self.drivers:
                lines.append(
                    f"  {label_for(c.signal)}: r={c.r:+.2f}, changed "
                    f"{c.change:+.1f}{UNIT_SUFFIX.get(c.unit, '')} over the window"
                )
        if self.forecasts:
            lines.append("FORECASTS:")
            lines += [f"  {f.describe()} [{f.model_kind} {f.model_version}]"
                      for f in self.forecasts]
        if self.accuracy:
            lines.append("RECENT FORECAST ACCURACY:")
            lines += [f"  {f.text}" for f in self.accuracy]
        if self.history:
            lines.append("COMPARABLE PERIODS:")
            lines += [f"  {f.text}" for f in self.history]
        if self.events:
            lines.append("EVENTS:")
            lines += [f"  {f.text}" for f in self.events]
        if self.user_context:
            lines.append("USER CONTEXT (previously told to us):")
            lines += [f"  {c}" for c in self.user_context]
        if self.notes:
            lines.append("NOTES:")
            lines += [f"  {n}" for n in self.notes]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"  ... ({len(lines) - max_lines} more lines omitted)"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# intent routing
# ---------------------------------------------------------------------------

_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Order matters: more specific patterns first. "Why was the last forecast
    # wrong?" contains "why", so forecast_error has to be checked before
    # why_change or it is misrouted to a driver-correlation answer.
    ("forecast_error", ("forecast wrong", "prediction wrong", "was wrong", "inaccurate",
                        "off by", "mispredict", "bad forecast", "accuracy",
                        "how accurate", "forecast was")),
    ("battery", ("battery", "charge", "unplug", "plugged", "power last", "last another",
                 "run out", "discharge")),
    ("why_change", ("why", "cause", "because", "reason", "what made", "driving",
                    "responsible")),
    ("anomaly", ("unusual", "weird", "strange", "anomal", "abnormal", "different than",
                 "changed compared", "compared with", "compared to")),
    ("adaptation", ("learn", "adapt", "model", "version", "rollback", "retrain",
                    "improving")),
    ("forecast", ("will", "predict", "forecast", "expect", "next", "going to")),
    ("trend", ("trend", "rising", "falling", "increasing", "decreasing", "climbing",
               "dropping", "spike")),
)


def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, keys in _INTENT_RULES:
        for k in keys:
            if k in q:
                return intent
    return "status"


def detect_focus_signal(question: str, available: Iterable[str]) -> str | None:
    """Best-effort mapping from words in the question to a signal name."""
    q = question.lower()
    avail = set(available)
    # Longest label first so "GPU temperature" wins over "GPU".
    keyed: list[tuple[str, str]] = []
    for sig, lab in SIGNAL_LABELS.items():
        if sig in avail:
            keyed.append((lab.lower(), sig))
    keyed.sort(key=lambda kv: -len(kv[0]))
    for lab, sig in keyed:
        if lab in q:
            return sig
    aliases = {
        "cpu": "cpu.util_pct", "processor": "cpu.util_pct",
        "memory": "mem.used_pct", "ram": "mem.used_pct",
        "battery": "battery.percent", "gpu": "gpu.util_pct",
        "temperature": "thermal.cpu_c", "temp": "thermal.cpu_c", "hot": "thermal.cpu_c",
        "disk": "disk.busy_pct", "network": "net.recv_mbps", "swap": "mem.swap_used_pct",
    }
    for word, sig in aliases.items():
        if word in q and sig in avail:
            return sig
    return None


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------

class GroundingEngine:
    """Assembles evidence packs from stored records."""

    def __init__(self, config: Config, repos: Repos) -> None:
        self.cfg = config
        self.repos = repos
        self.registry: SignalRegistry = repos.signals.registry()

    def _unit(self, signal: str) -> str:
        spec = self.registry.get(signal)
        return spec.unit if spec else ""

    def refresh_registry(self) -> None:
        self.registry = self.repos.signals.registry()

    # -- pieces ------------------------------------------------------------
    def _latest_sample(self):
        got = self.repos.telemetry.latest(1)
        return got[0] if got else None

    def _window_arrays(self, signals: Sequence[str], now: float, window_s: float):
        return self.repos.telemetry.window(signals, ts_from=now - window_s, ts_to=now + 1.0)

    def current_facts(self, now: float, focus: str | None = None) -> tuple[list[Fact], float | None]:
        sample = self._latest_sample()
        if sample is None:
            return [], None
        priority = [
            "cpu.util_pct", "mem.used_pct", "battery.percent", "battery.plugged",
            "gpu.util_pct", "thermal.cpu_c", "battery.power_w", "disk.busy_pct",
        ]
        if focus and focus not in priority:
            priority.insert(0, focus)
        out: list[Fact] = []
        for sig in priority:
            v = sample.value(sig)
            if v is None:
                continue
            unit = self._unit(sig)
            if sig in ("battery.plugged", "battery.charging"):
                txt = f"{label_for(sig)}: {'yes' if v > 0.5 else 'no'}"
            else:
                txt = f"{label_for(sig)}: {fmt(v, unit)}"
            out.append(Fact(key=sig, text=txt, value=float(v), unit=unit, ts=sample.ts))
        age = now - sample.ts
        return out, age

    def window_facts(
        self, now: float, window_s: float, signals: Sequence[str] | None = None
    ) -> list[Fact]:
        sigs = list(signals) if signals else [
            s for s in ("cpu.util_pct", "mem.used_pct", "battery.percent", "gpu.util_pct",
                        "thermal.cpu_c") if s in self.registry
        ]
        if not sigs:
            return []
        w = self._window_arrays(sigs, now, window_s)
        if len(w) < 2:
            return []
        out: list[Fact] = []
        for i, sig in enumerate(w.signals):
            col = w.values[:, i]
            good = col[~np.isnan(col)]
            if good.size < 2:
                continue
            unit = self._unit(sig)
            change = float(good[-1] - good[0])
            out.append(Fact(
                key=f"{sig}.mean", unit=unit,
                text=(f"{label_for(sig)}: mean {fmt(float(good.mean()), unit)}, "
                      f"range {fmt(float(good.min()), unit)} to {fmt(float(good.max()), unit)}, "
                      f"net change {change:+.1f}{UNIT_SUFFIX.get(unit, '')}"),
                value=float(good.mean()), ts=now,
            ))
            out.append(Fact(
                key=f"{sig}.change", text=f"{label_for(sig)} changed {change:+.1f}"
                f"{UNIT_SUFFIX.get(unit, '')} over the last {fmt_duration(window_s)}",
                value=change, unit=unit, ts=now,
            ))
        return out

    def battery_analysis(self, now: float, window_s: float) -> tuple[list[Fact], list[str]]:
        """Discharge rate and time-to-threshold, computed not guessed."""
        facts: list[Fact] = []
        notes: list[str] = []
        if "battery.percent" not in self.registry:
            notes.append("This machine exposes no battery sensor.")
            return facts, notes
        sigs = [s for s in ("battery.percent", "battery.plugged", "battery.power_w")
                if s in self.registry]
        w = self._window_arrays(sigs, now, max(window_s, 1800.0))
        if len(w) < 4:
            notes.append("Not enough battery history yet to estimate a rate.")
            return facts, notes
        pct = w.column("battery.percent")
        plugged = w.column("battery.plugged")
        if pct is None:
            return facts, notes
        ok = ~np.isnan(pct)
        if ok.sum() < 4:
            return facts, notes
        t = w.ts[ok]
        p = pct[ok]
        # Least-squares slope, in percent per minute.
        t0 = t - t[0]
        var = float(((t0 - t0.mean()) ** 2).sum())
        slope_per_min = 0.0
        if var > 0:
            slope_per_min = float(((t0 - t0.mean()) * (p - p.mean())).sum() / var) * 60.0
        is_plugged = bool(plugged is not None and not np.isnan(plugged[-1]) and plugged[-1] > 0.5)
        facts.append(Fact(
            key="battery.rate_pct_per_min", unit="percent",
            text=(f"Battery is {'rising' if slope_per_min > 0 else 'falling'} at "
                  f"{abs(slope_per_min):.3f}%/min over the last {fmt_duration(t[-1] - t[0])}"),
            value=round(slope_per_min, 4), ts=now,
        ))
        facts.append(Fact(
            key="battery.plugged", value=1.0 if is_plugged else 0.0,
            text=f"AC power is {'connected' if is_plugged else 'disconnected'}", ts=now,
        ))
        power = w.column("battery.power_w")
        if power is not None:
            good = power[~np.isnan(power)]
            if good.size and float(good.mean()) > 0:
                facts.append(Fact(
                    key="battery.power_w", unit="watt", value=float(good.mean()),
                    text=f"Mean battery power draw {fmt(float(good.mean()), 'watt')}", ts=now,
                ))
        if not is_plugged and slope_per_min < -1e-4:
            cur = float(p[-1])
            for threshold in (20.0, 10.0, 0.0):
                if cur <= threshold:
                    continue
                secs = (cur - threshold) / abs(slope_per_min) * 60.0
                facts.append(Fact(
                    key=f"battery.time_to_{int(threshold)}", unit="", value=round(secs / 60.0, 1),
                    text=(f"At the current rate, battery reaches {threshold:.0f}% in about "
                          f"{fmt_duration(secs)}"),
                    source="derived", ts=now,
                ))
            notes.append(
                "Time-to-threshold assumes the current discharge rate continues; a change "
                "in workload changes it."
            )
        elif is_plugged:
            notes.append("On AC power, so discharge projections do not apply.")
        return facts, notes

    def driver_correlations(
        self, target: str, now: float, window_s: float, top_k: int = 4,
        max_lag_s: float = 120.0,
    ) -> list[Correlation]:
        """Which signals moved with ``target`` over the window.

        Explicitly correlation, not causation, and the wording in every answer
        says "correlated", never "caused". Lags up to ``max_lag_s`` are tried so
        that a driver leading the target (build activity leading CPU) is found
        rather than missed.
        """
        candidates = [s for s in DRIVER_CANDIDATES if s in self.registry and s != target]
        if not candidates or target not in self.registry:
            return []
        w = self._window_arrays([target, *candidates], now, window_s)
        if len(w) < 12:
            return []
        y = w.column(target)
        if y is None:
            return []
        oky = ~np.isnan(y)
        if oky.sum() < 12:
            return []
        interval = float(np.median(np.diff(w.ts))) if len(w) > 2 else 5.0
        max_lag = max(0, int(max_lag_s / max(interval, 1e-6)))
        out: list[Correlation] = []
        for sig in candidates:
            x = w.column(sig)
            if x is None:
                continue
            best_r, best_lag = 0.0, 0.0
            for lag in range(0, max_lag + 1, max(1, max_lag // 6 or 1)):
                xa = x[: len(x) - lag] if lag else x
                ya = y[lag:] if lag else y
                m = (~np.isnan(xa)) & (~np.isnan(ya))
                if m.sum() < 12:
                    continue
                xv, yv = xa[m], ya[m]
                if float(xv.std()) < 1e-9 or float(yv.std()) < 1e-9:
                    continue
                r = float(np.corrcoef(xv, yv)[0, 1])
                if math.isfinite(r) and abs(r) > abs(best_r):
                    best_r, best_lag = r, lag * interval
            if abs(best_r) < 0.3:
                continue
            xg = x[~np.isnan(x)]
            change = float(xg[-1] - xg[0]) if xg.size >= 2 else 0.0
            out.append(Correlation(signal=sig, r=round(best_r, 3), lag_s=best_lag,
                                   change=round(change, 2), unit=self._unit(sig)))
        out.sort(key=lambda c: -abs(c.r))
        return out[:top_k]

    def active_forecasts(self, now: float, focus: str | None = None,
                         model_kind: str | None = None) -> list[ForecastFact]:
        """The most recent unresolved forecast per (signal, horizon)."""
        recs = self.repos.predictions.latest_unresolved(limit=400)
        prefer = model_kind or "rls"
        best: dict[tuple[str, float], PredictionRecord] = {}
        for r in recs:
            if focus and r.signal != focus:
                continue
            key = (r.signal, r.horizon_s)
            cur = best.get(key)
            # Prefer the learned model; fall back to persistence so the user
            # always gets a forecast, clearly labelled with which produced it.
            if cur is None:
                best[key] = r
                continue
            cur_pref = cur.model_kind == prefer
            new_pref = r.model_kind == prefer
            if new_pref and not cur_pref:
                best[key] = r
            elif new_pref == cur_pref and r.ts_made > cur.ts_made:
                best[key] = r
        out = [
            ForecastFact(
                signal=r.signal, horizon_s=r.horizon_s, value=round(r.value, 2),
                lo=None if r.lo is None else round(r.lo, 2),
                hi=None if r.hi is None else round(r.hi, 2),
                model_kind=r.model_kind, model_version=r.model_version,
                ts_made=r.ts_made, ts_target=r.ts_target, unit=self._unit(r.signal),
            )
            for r in best.values()
        ]
        out.sort(key=lambda f: (f.signal, f.horizon_s))
        return out

    def accuracy_facts(self, now: float, window_s: float = 7200.0,
                       focus: str | None = None) -> list[Fact]:
        recs = self.repos.predictions.resolved(ts_from=now - window_s, ts_to=now)
        if focus:
            recs = [r for r in recs if r.signal == focus]
        if not recs:
            return []
        rows = R.summarize(recs, reference="persistence", dm_test=False)
        out: list[Fact] = []
        for row in rows:
            if row.m.mae is None or row.m.n < 10:
                continue
            unit = self._unit(row.signal)
            txt = (f"{label_for(row.signal)} at {fmt_duration(row.horizon_s)}: "
                   f"{row.arm} MAE {fmt(row.m.mae, unit, 2)} over {row.m.n} forecasts")
            if row.m.skill is not None:
                txt += f" ({row.m.skill:+.0%} vs persistence)"
            out.append(Fact(key=f"mae.{row.arm}.{row.signal}.{int(row.horizon_s)}",
                            text=txt, value=round(row.m.mae, 3), unit=unit, source="evaluation",
                            ts=now))
            if row.m.coverage is not None and row.arm == "rls":
                out.append(Fact(
                    key=f"coverage.{row.signal}.{int(row.horizon_s)}",
                    text=(f"{label_for(row.signal)} at {fmt_duration(row.horizon_s)}: "
                          f"{row.m.coverage:.0%} of outcomes fell inside the predicted range"),
                    value=round(row.m.coverage, 3), source="evaluation", ts=now,
                ))
        return out

    def historical_comparison(
        self, signal: str, regime: str, now: float, horizon_s: float,
        lookback_s: float = 604800.0, min_samples: int = 12,
    ) -> list[Fact]:
        """What this signal usually did next, the last times the machine looked
        like this.

        Answers the "historical comparable periods" part of an explanation with
        measurements rather than an impression: over all stored history in the
        same regime, what was the distribution of the h-step change in this
        signal? Reported only when there are enough comparable periods to be
        worth stating -- the alternative is an authoritative-sounding claim built
        on three samples.
        """
        if signal not in self.registry:
            return []
        recs = self.repos.predictions.resolved(
            signal=signal, horizon_s=horizon_s, model_kind="persistence",
            ts_from=now - lookback_s, ts_to=now, regime=regime,
        )
        # Persistence rows carry the anchor and the actual, so the realised
        # h-step change is exactly (actual - anchor); no extra query needed.
        deltas = [
            r.actual - r.anchor for r in recs
            if r.actual is not None and r.anchor is not None
        ]
        if len(deltas) < min_samples:
            return []
        arr = np.asarray(deltas)
        unit = self._unit(signal)
        lo, hi = (float(np.quantile(arr, 0.1)), float(np.quantile(arr, 0.9)))
        med = float(np.median(arr))
        out = [Fact(
            key=f"history.{signal}.{int(horizon_s)}.median", unit=unit, value=round(med, 3),
            source="history", ts=now,
            text=(f"Historically, when this machine was in the '{regime}' regime, "
                  f"{label_for(signal)} changed by a median of {med:+.1f}"
                  f"{UNIT_SUFFIX.get(unit, '')} over {fmt_duration(horizon_s)} "
                  f"({len(deltas)} comparable periods)"),
        )]
        out.append(Fact(
            key=f"history.{signal}.{int(horizon_s)}.range", unit=unit, value=round(hi, 3),
            source="history", ts=now,
            text=(f"80% of those periods fell between {lo:+.1f} and {hi:+.1f}"
                  f"{UNIT_SUFFIX.get(unit, '')}"),
        ))
        return out

    def event_facts(self, now: float, window_s: float = 21600.0) -> list[Fact]:
        out: list[Fact] = []
        for ev in self.repos.events.drift_events(ts_from=now - window_s, limit=20):
            out.append(Fact(
                key="drift", source="drift", ts=ev["ts"],
                text=(f"{fmt_duration(now - ev['ts'])} ago: behaviour change detected by "
                      f"{ev['detector']} ({ev['scope']}"
                      + (f", {ev['signal']}" if ev["signal"] else "") + ")"),
            ))
        for ev in self.repos.events.adapt_events(ts_from=now - window_s, limit=20):
            if ev["decision"] == "skipped":
                continue
            before, after = ev["metric_before"], ev["metric_after"]
            delta = ""
            if before is not None and after is not None and before > 0:
                delta = f", MAE {before:.3f} -> {after:.3f}"
            out.append(Fact(
                key=f"adapt.{ev['decision']}", source="adaptation", ts=ev["ts"],
                text=(f"{fmt_duration(now - ev['ts'])} ago: model update {ev['decision']} for "
                      f"{ev['scope']} (trigger: {ev['trigger']}{delta})"),
            ))
        out.sort(key=lambda f: -(f.ts or 0.0))
        return out[:12]

    def user_context_lines(self, limit: int = 8) -> list[str]:
        """Things the user has told us, fed back into every answer."""
        rows = self.repos.feedback.list(limit=60)
        lines: list[str] = []
        for r in rows:
            txt = (r.get("text") or "").strip()
            if r["kind"] == "label" and txt:
                lines.append(f'User labels this behaviour: "{txt}"')
            elif r["kind"] == "note" and txt:
                lines.append(f'User note: "{txt}"')
            if len(lines) >= limit:
                break
        return lines

    def regime_info(self, now: float, window_s: float) -> tuple[str, float | None, dict[str, float]]:
        """Current regime and its recent distribution, from stored predictions.

        Regime is recorded on every prediction and feature window, so it is
        recoverable after the fact without re-running the classifier.
        """
        recs = self.repos.predictions.latest_unresolved(limit=40)
        regime = recs[0].regime if recs else "unknown"
        hist = self.repos.predictions.resolved(ts_from=now - window_s, ts_to=now, limit=4000)
        dist: dict[str, float] = {}
        since: float | None = None
        if hist:
            seen = [(r.ts_made, r.regime) for r in hist if r.model_kind == "persistence"]
            seen.sort()
            if seen:
                total = len(seen)
                for _, reg in seen:
                    dist[reg] = dist.get(reg, 0.0) + 1.0 / total
                if not recs:
                    regime = seen[-1][1]
                boundary = seen[-1][0]
                for ts_made, reg in reversed(seen):
                    if reg != regime:
                        break
                    boundary = ts_made
                since = max(0.0, now - boundary)
        return regime, since, dict(sorted(dist.items(), key=lambda kv: -kv[1]))

    # -- assembly ----------------------------------------------------------
    def build(
        self, question: str = "", now: float | None = None, window_s: float | None = None,
        intent: str | None = None,
    ) -> EvidencePack:
        now = time.time() if now is None else now
        window_s = window_s or self.cfg.api.default_window_s
        self.refresh_registry()
        intent = intent or detect_intent(question)
        focus = detect_focus_signal(question, self.registry.names()) if question else None
        if intent == "battery":
            focus = focus or ("battery.percent" if "battery.percent" in self.registry else None)

        pack = EvidencePack(
            now=now, question=question, intent=intent, window_s=window_s, focus_signal=focus
        )
        pack.regime, pack.regime_since_s, pack.regime_distribution = self.regime_info(now, window_s)
        pack.current, age = self.current_facts(now, focus)
        if not pack.current:
            pack.notes.append(
                "No telemetry has been collected yet. Start the collector "
                "(`streamlora collect`) and ask again."
            )
            return pack
        if age is not None and age > max(30.0, self.cfg.collect.interval_s * 6):
            pack.notes.append(
                f"The most recent sample is {fmt_duration(age)} old, so 'now' may be stale."
            )

        wsigs = [focus] if focus else None
        pack.window = self.window_facts(now, window_s, wsigs)
        if focus and focus != "battery.percent":
            # Add general context around the focus signal, skipping anything
            # already covered: the focus signal appears in both lists otherwise,
            # and a duplicated fact reads as sloppy and wastes prompt budget.
            have = {f.key for f in pack.window}
            pack.window += [
                f for f in self.window_facts(now, window_s, None) if f.key not in have
            ][:6]

        if intent in ("battery", "status", "forecast"):
            bfacts, bnotes = self.battery_analysis(now, window_s)
            pack.window += bfacts
            pack.notes += bnotes
        # A "why did the battery drop" question routes to the battery intent but
        # still wants correlated activity, so driver analysis is triggered by the
        # question as well as by the intent.
        wants_drivers = intent in ("why_change", "trend", "anomaly") or (
            any(k in question.lower() for k in ("why", "cause", "reason", "what made"))
        )
        if wants_drivers and focus:
            pack.drivers = self.driver_correlations(focus, now, window_s)
            if not pack.drivers:
                pack.notes.append(
                    "No other signal moved with it strongly enough (|r| >= 0.3) to be worth "
                    "reporting."
                )
        if intent in ("forecast", "battery", "status", "trend"):
            pack.forecasts = self.active_forecasts(now, focus)
            if not pack.forecasts:
                pack.notes.append(
                    "No forecasts are active yet; the model needs more history before it "
                    "will predict."
                )
        if intent in ("forecast_error", "adaptation", "status", "anomaly"):
            pack.accuracy = self.accuracy_facts(now, max(window_s, 7200.0), focus)
        if focus and intent in ("why_change", "trend", "anomaly", "forecast", "battery"):
            horizon = (
                self.cfg.forecast.horizons_s[0] if self.cfg.forecast.horizons_s else 300.0
            )
            pack.history = self.historical_comparison(focus, pack.regime, now, horizon)
        if intent in ("anomaly", "adaptation", "forecast_error", "status"):
            pack.events = self.event_facts(now)
        pack.user_context = self.user_context_lines()
        return pack
