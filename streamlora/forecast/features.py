"""Windowing and feature extraction.

One feature vector is built per tick and shared by every model, then each model
selects the columns it needs via a fixed index mask (see ``select_indices``).
That split matters for two reasons:

* **Storage.** One vector per tick is stored once and referenced by all nine
  (target x horizon) predictions made from it, which is what makes gated
  promotion possible -- a candidate model can be re-scored on exactly the inputs
  the active model saw.
* **Statistics.** A shared vector of ~200 columns would give each model a
  parameter-to-data ratio that guarantees overfitting under a bounded online
  memory. Per-target masks bring each model to ~70 columns: its own signal in
  full detail plus a coarse summary of everything else.

The window is held in memory as a ring buffer. The database is never on the
prediction hot path.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import FeatureConfig
from ..telemetry.schema import SignalRegistry, TelemetrySample
from ..util.ids import config_hash
from .regime import REGIMES
from .scaling import Scaler

#: Signals used as model inputs by default. Curated rather than "everything":
#: each one is either a forecast target, a known driver of one (thermal
#: throttling, GPU load, build activity), or a regime discriminator. The full
#: 38-signal set is still collected and stored; this only bounds what the
#: linear models see.
DEFAULT_INPUTS: tuple[str, ...] = (
    "cpu.util_pct",
    "cpu.util_max_core_pct",
    "cpu.iowait_pct",
    "cpu.load1_per_core",
    "cpu.freq_mhz",
    "mem.used_pct",
    "mem.available_gb",
    "battery.percent",
    "battery.power_w",
    "battery.plugged",
    "gpu.util_pct",
    "gpu.power_w",
    "thermal.cpu_c",
    "disk.busy_pct",
    "net.recv_mbps",
    "proc.concentration",
    "proc.cpu_build_pct",
    "proc.top1_cpu_pct",
)

#: Feature families generated per input signal.
_LAG = "lag"
_MEAN = "mean"
_STD = "std"
_SLOPE = "slope"
_DELTA = "delta"


@dataclass(slots=True)
class FeatureVector:
    ts: float
    names: list[str]
    values: np.ndarray
    regime: str
    #: Fraction of the requested window actually covered by usable samples.
    coverage: float
    #: Raw (unscaled) current value per input signal, for anchors and display.
    current: dict[str, float]
    schema_hash: str = ""

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.values)}


@dataclass(slots=True)
class _Row:
    ts: float
    scaled: np.ndarray   # (n_inputs,) scaled values, NaN where unusable
    raw: np.ndarray      # (n_inputs,) native values, NaN where unusable
    regime: str


class FeatureExtractor:
    """Maintains the rolling window and builds feature vectors from it."""

    def __init__(
        self,
        registry: SignalRegistry,
        config: FeatureConfig | None = None,
        interval_s: float = 5.0,
    ) -> None:
        self.cfg = config or FeatureConfig()
        self.interval_s = float(interval_s)
        requested = list(self.cfg.inputs) if self.cfg.inputs else list(DEFAULT_INPUTS)
        available = set(registry.names())
        # Missing signals are dropped from the feature space rather than
        # imputed forever: a machine with no GPU should not carry 12 dead
        # columns. The schema hash changes accordingly, so models are never
        # loaded against a mismatched feature space.
        self.inputs = [s for s in requested if s in available]
        self.dropped_inputs = [s for s in requested if s not in available]
        self.registry = registry
        self.scaler = Scaler(self.inputs, registry)
        self.lags = sorted(set(float(x) for x in self.cfg.lags_s))
        self.spans = sorted(set(float(x) for x in self.cfg.agg_spans_s))
        self.window_s = float(self.cfg.window_s)
        maxlen = max(8, int(self.window_s / max(self.interval_s, 1e-6)) + 4)
        self._rows: deque[_Row] = deque(maxlen=maxlen)
        self.names = self._build_names()
        self._index = {n: i for i, n in enumerate(self.names)}
        self.schema_hash = config_hash(
            {
                "names": self.names,
                "inputs": self.inputs,
                "lags": self.lags,
                "spans": self.spans,
                "window_s": self.window_s,
            }
        )

    # -- schema ------------------------------------------------------------
    def _build_names(self) -> list[str]:
        names: list[str] = ["bias"]
        for sig in self.inputs:
            for lag in self.lags:
                names.append(f"{sig}|{_LAG}{int(lag)}")
            for span in self.spans:
                names.append(f"{sig}|{_MEAN}{int(span)}")
            names.append(f"{sig}|{_STD}{int(self.spans[-1])}")
            for span in self.spans[-2:] if len(self.spans) >= 2 else self.spans:
                names.append(f"{sig}|{_SLOPE}{int(span)}")
            names.append(f"{sig}|{_DELTA}{int(self.lags[-1])}")
        if self.cfg.time_of_day:
            names += ["time|sin_day", "time|cos_day", "time|sin_week", "time|cos_week", "time|weekend"]
        if self.cfg.regime_features:
            names += [f"regime|{r}" for r in REGIMES]
        names.append("meta|coverage")
        return names

    @property
    def n_features(self) -> int:
        return len(self.names)

    # -- ingestion ---------------------------------------------------------
    def push(self, sample: TelemetrySample, regime: str = "unknown") -> None:
        raw = np.full(len(self.inputs), np.nan)
        for i, sig in enumerate(self.inputs):
            v = sample.value(sig)
            if v is not None:
                raw[i] = float(v)
        scaled = self.scaler.transform(raw)
        if self._rows and sample.ts <= self._rows[-1].ts:
            # Out-of-order or duplicate timestamp: keep the window monotone,
            # which every downstream lag/slope computation assumes.
            return
        self._rows.append(_Row(sample.ts, scaled, raw, regime))

    def reset(self) -> None:
        self._rows.clear()

    @property
    def ready(self) -> bool:
        """True once the window holds enough history for the longest lag."""
        if len(self._rows) < 3:
            return False
        span = self._rows[-1].ts - self._rows[0].ts
        needed = max(self.lags[-1], self.spans[-1]) * self.cfg.min_coverage
        return span >= needed

    @property
    def span_s(self) -> float:
        if len(self._rows) < 2:
            return 0.0
        return self._rows[-1].ts - self._rows[0].ts

    def current_raw(self) -> dict[str, float]:
        if not self._rows:
            return {}
        last = self._rows[-1]
        return {
            sig: float(last.raw[i])
            for i, sig in enumerate(self.inputs)
            if not math.isnan(last.raw[i])
        }

    def last_ts(self) -> float | None:
        return self._rows[-1].ts if self._rows else None

    def raw_series(self, signal: str) -> tuple[np.ndarray, np.ndarray]:
        """Native-unit window for one signal, NaN where unusable.

        Baselines consume this rather than the scaled feature vector: a
        persistence forecast must be exactly the last measured value, not a
        value round-tripped through a scaling transform.
        """
        try:
            i = self.inputs.index(signal)
        except ValueError:
            return np.empty(0), np.empty(0)
        rows = list(self._rows)
        return (
            np.array([r.ts for r in rows]),
            np.array([r.raw[i] for r in rows]),
        )

    def n_rows(self) -> int:
        return len(self._rows)

    # -- extraction --------------------------------------------------------
    def extract(self, regime: str | None = None) -> FeatureVector | None:
        if not self._rows:
            return None
        rows = list(self._rows)
        now = rows[-1].ts
        ts = np.array([r.ts for r in rows])
        mat = np.vstack([r.scaled for r in rows])          # (n, n_inputs)
        lab = regime if regime is not None else rows[-1].regime

        # Column means over the whole window are the imputation fallback.
        # Imputing 0.0 would mean "at the bottom of the signal's range", which
        # is a specific and usually wrong claim; the window mean is neutral.
        # An all-NaN column (a signal absent for the whole window) is expected on
        # a machine missing that sensor, so the empty-slice warning is silenced
        # rather than left to spam the logs; the NaN it returns is handled below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(mat, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.5, col_mean)

        usable = (~np.isnan(mat)).sum()
        coverage_frac = usable / float(mat.size) if mat.size else 0.0
        span_cov = min(1.0, (now - ts[0]) / max(self.window_s, 1e-6))
        coverage = float(coverage_frac * span_cov)

        out = np.empty(len(self.names))
        out[0] = 1.0  # bias
        k = 1
        for i, _sig in enumerate(self.inputs):
            series = mat[:, i]
            fill = col_mean[i]
            for lag in self.lags:
                out[k] = self._at_lag(ts, series, now, lag, fill)
                k += 1
            for span in self.spans:
                out[k] = self._span_stat(ts, series, now, span, fill, "mean")
                k += 1
            out[k] = self._span_stat(ts, series, now, self.spans[-1], 0.0, "std")
            k += 1
            for span in (self.spans[-2:] if len(self.spans) >= 2 else self.spans):
                out[k] = self._span_stat(ts, series, now, span, 0.0, "slope")
                k += 1
            # Change over the longest lag: the single most informative feature
            # for a differenced target, and it makes trend explicit rather than
            # something the model must infer from two collinear lags.
            out[k] = self._at_lag(ts, series, now, 0.0, fill) - self._at_lag(
                ts, series, now, self.lags[-1], fill
            )
            k += 1

        if self.cfg.time_of_day:
            out[k : k + 5] = self._time_features(now)
            k += 5
        if self.cfg.regime_features:
            oh = np.zeros(len(REGIMES))
            try:
                oh[REGIMES.index(lab)] = 1.0
            except ValueError:
                oh[0] = 1.0
            out[k : k + len(REGIMES)] = oh
            k += len(REGIMES)
        out[k] = coverage
        k += 1
        assert k == len(self.names), f"feature count mismatch {k} != {len(self.names)}"

        # Any residual NaN would silently poison a least-squares update forever.
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return FeatureVector(
            ts=now, names=self.names, values=out, regime=lab, coverage=coverage,
            current=self.current_raw(), schema_hash=self.schema_hash,
        )

    def _at_lag(
        self, ts: np.ndarray, series: np.ndarray, now: float, lag: float, fill: float
    ) -> float:
        """Value ``lag`` seconds ago, from the nearest sample within tolerance."""
        target = now - lag
        idx = int(np.abs(ts - target).argmin())
        # Tolerance scales with the lag: being 5 s off on a 300 s lag is
        # harmless, being 5 s off on a 0 s lag is not.
        tol = max(self.interval_s * 1.5, lag * 0.25)
        if abs(ts[idx] - target) > tol:
            return float(fill)
        v = series[idx]
        if math.isnan(v):
            # Walk outward for the closest usable sample before giving up.
            order = np.argsort(np.abs(ts - target))
            for j in order[:6]:
                if not math.isnan(series[j]) and abs(ts[j] - target) <= tol * 2:
                    return float(series[j])
            return float(fill)
        return float(v)

    def _span_stat(
        self, ts: np.ndarray, series: np.ndarray, now: float, span: float,
        fill: float, stat: str,
    ) -> float:
        mask = ts >= (now - span)
        vals = series[mask]
        good = vals[~np.isnan(vals)]
        if good.size == 0:
            return float(fill)
        if stat == "mean":
            return float(good.mean())
        if stat == "std":
            return float(good.std(ddof=0)) if good.size > 1 else 0.0
        if stat == "slope":
            if good.size < 2:
                return 0.0
            t = ts[mask][~np.isnan(vals)]
            t0 = t - t[0]
            var = float(((t0 - t0.mean()) ** 2).sum())
            if var <= 0:
                return 0.0
            cov = float(((t0 - t0.mean()) * (good - good.mean())).sum())
            # Per-minute slope keeps the feature O(0.1) instead of O(1e-4),
            # which matters for the ridge penalty being comparable across
            # feature families.
            return (cov / var) * 60.0
        raise ValueError(stat)

    @staticmethod
    def _time_features(now: float) -> list[float]:
        """Cyclical time encodings.

        Laptop workloads are strongly diurnal and weekly; a raw hour number
        would tell a linear model that 23:00 and 00:00 are maximally distant.
        """
        import time as _time

        lt = _time.localtime(now)
        sec_of_day = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
        day_frac = sec_of_day / 86400.0
        week_frac = (lt.tm_wday + day_frac) / 7.0
        return [
            math.sin(2 * math.pi * day_frac),
            math.cos(2 * math.pi * day_frac),
            math.sin(2 * math.pi * week_frac),
            math.cos(2 * math.pi * week_frac),
            1.0 if lt.tm_wday >= 5 else 0.0,
        ]

    # -- per-target column selection ---------------------------------------
    def select_indices(self, target: str) -> np.ndarray:
        """Columns a model for ``target`` should use.

        Own signal: every feature family. Other signals: current value, a
        medium-span mean and the long slope -- enough to express "GPU is busy
        and rising" without three collinear lags per signal.
        """
        keep: list[int] = [0]  # bias
        mid_span = int(self.spans[len(self.spans) // 2])
        long_span = int(self.spans[-1])
        for i, name in enumerate(self.names):
            if i == 0:
                continue
            if "|" not in name:
                continue
            head, feat = name.split("|", 1)
            if head in ("time", "regime", "meta"):
                keep.append(i)
                continue
            if head == target:
                keep.append(i)
                continue
            if feat in (f"{_LAG}0", f"{_MEAN}{mid_span}", f"{_SLOPE}{long_span}"):
                keep.append(i)
        return np.asarray(sorted(set(keep)), dtype=np.int64)
