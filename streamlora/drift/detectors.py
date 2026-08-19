"""Concept-drift detectors.

Two complementary questions get asked, because they fail in different ways:

* **Is the model getting worse?** (error-based: Page-Hinkley, ADWIN.) Directly
  actionable, but blind to a shift the model happens to handle well, and it
  only fires *after* accuracy has already degraded.
* **Does the input look different?** (feature-based.) Fires at the moment the
  machine's behaviour changes, before errors accumulate, but a shift in inputs
  is not automatically a problem.

Scale-free by construction
--------------------------
Every detector consumes a *standardised* stream: raw values are converted to
``z = (x - ref_mean) / ref_std`` using statistics from a warm-up reference
period. Without this, ``ph_threshold`` would have to be retuned per signal
(battery errors ~0.3 %, CPU errors ~12 %) and per machine, and a config that
worked on one laptop would silently never fire on another.

After an alarm each detector re-establishes its reference, because the point of
detecting a regime change is that the new regime is now normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..util.mathx import RingStats, Welford

#: Floors on the reference standard deviation, applied as
#: ``max(std, _MIN_STD_ABS, _MIN_STD_REL * |mean|)``.
#:
#: An absolute floor alone is not enough. A perfectly predicted signal (an idle
#: laptop) gives a warm-up std of exactly 0; with a floor of 1e-6 the next error
#: of 1e-4 becomes a 100-sigma event and ADWIN alarms on nothing. The absolute
#: floor is therefore set in *error units* -- errors below a thousandth of a
#: percentage point are not a behaviour change on any signal this system
#: forecasts. The relative floor covers the opposite case: a large-mean,
#: low-variance stream where a proportionally tiny wobble is also not news.
_MIN_STD_ABS = 1e-3
_MIN_STD_REL = 0.01
#: Retained for the feature detector, where inputs are already scaled to ~[0, 1]
#: and a much smaller floor is appropriate.
_MIN_STD = 1e-6


def _reference_std(std: float, mean: float) -> float:
    return max(std, _MIN_STD_ABS, _MIN_STD_REL * abs(mean))


@dataclass
class DriftSignal:
    """One detector's verdict at one point in time."""

    detector: str
    fired: bool
    statistic: float
    threshold: float
    #: 0 at threshold, growing with exceedance. Used to rank simultaneous alarms.
    severity: float = 0.0
    detail: dict[str, object] = field(default_factory=dict)


class _Standardiser:
    """Rolling warm-up standardiser shared by the error detectors."""

    def __init__(self, warmup: int) -> None:
        self.warmup = max(5, int(warmup))
        self._w = Welford()
        self.ref_mean = 0.0
        self.ref_std = 1.0
        self.ready = False

    def push(self, x: float) -> float | None:
        if not self.ready:
            self._w.push(x)
            if self._w.n >= self.warmup:
                self.ref_mean = self._w.mean
                self.ref_std = _reference_std(self._w.std, self._w.mean)
                self.ready = True
            return None
        return (x - self.ref_mean) / self.ref_std

    def restart(self) -> None:
        self._w = Welford()
        self.ready = False


class PageHinkley:
    """One-sided Page-Hinkley test for an increase in mean error.

    Accumulates ``z - delta`` and alarms when the running sum exceeds its own
    minimum by ``threshold``. ``delta`` is the shift magnitude to ignore, in
    standard deviations, so 0.5 means "do not react to drifts under half a
    sigma". With ``threshold`` 25 that requires roughly 50 consecutive samples
    of 0.5-sigma excess, or 17 of 2-sigma -- about 1.5 to 4 minutes at a 5 s
    cadence, which is the timescale a laptop workload actually changes on.
    """

    name = "page_hinkley"

    def __init__(self, delta: float = 0.5, threshold: float = 25.0, warmup: int = 60) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self._std = _Standardiser(warmup)
        self._cum = 0.0
        self._min = 0.0
        self.n = 0
        self.alarms = 0

    def update(self, x: float) -> DriftSignal:
        self.n += 1
        z = self._std.push(x)
        if z is None:
            return DriftSignal(self.name, False, 0.0, self.threshold, detail={"state": "warmup"})
        self._cum += z - self.delta
        self._min = min(self._min, self._cum)
        stat = self._cum - self._min
        if stat > self.threshold:
            self.alarms += 1
            self._reset_after_alarm()
            return DriftSignal(
                self.name, True, stat, self.threshold,
                severity=(stat - self.threshold) / self.threshold,
                detail={"ref_mean": self._std.ref_mean, "ref_std": self._std.ref_std},
            )
        return DriftSignal(self.name, False, stat, self.threshold)

    def _reset_after_alarm(self) -> None:
        self._cum = 0.0
        self._min = 0.0
        self._std.restart()

    def state(self) -> dict[str, object]:
        return {
            "n": self.n, "alarms": self.alarms, "statistic": round(self._cum - self._min, 3),
            "ready": self._std.ready, "ref_mean": round(self._std.ref_mean, 4),
            "ref_std": round(self._std.ref_std, 4),
        }


class AdwinLite:
    """ADWIN's statistical test over a bounded sliding window.

    Maintains a window of recent standardised errors and looks for a split point
    where the two sub-windows' means differ by more than the bound from the
    ADWIN2 paper (Bifet & Gavalda, 2007):

        eps = sqrt( (2 / m) * var_W * ln(2 / delta') ) + (2 / 3m) * ln(2 / delta')
        m   = 1 / (1/n0 + 1/n1),        delta' = delta / n

    The variance term is essential and is the reason a plain Hoeffding bound was
    rejected here: Hoeffding requires a bounded range, and the standardised
    error stream is unbounded. Using ``sqrt(ln(4/delta') / 2m)`` -- correct only
    for variables in [0, 1] -- under-estimates the bound by roughly 3x on unit
    variance data, which produced a false alarm every ~180 samples on pure
    stationary noise during development. The Bernstein-style bound above adapts
    to the observed variance and eliminated them.

    This is ADWIN's test, not ADWIN2's data structure: the original keeps an
    exponential histogram to support an unbounded window in logarithmic memory.
    Here the window is capped (``max_window``) and cut points are subsampled, so
    the cost per update is O(cuts) with a small constant. The cap is not a
    limitation in practice -- a drift that needs more than an hour of context to
    see is not one we would want to react to anyway -- and it keeps the detector
    simple enough to test exhaustively.
    """

    name = "adwin"

    def __init__(
        self, delta: float = 0.002, min_window: int = 40, max_window: int = 400,
        warmup: int = 60, cut_stride: int = 8,
    ) -> None:
        self.delta = float(delta)
        self.min_window = max(10, int(min_window))
        self.max_window = max(self.min_window * 2, int(max_window))
        self.cut_stride = max(1, int(cut_stride))
        self._std = _Standardiser(warmup)
        self._buf: list[float] = []
        self.n = 0
        self.alarms = 0
        self._last_stat = 0.0

    def update(self, x: float) -> DriftSignal:
        self.n += 1
        z = self._std.push(x)
        if z is None:
            return DriftSignal(self.name, False, 0.0, 0.0, detail={"state": "warmup"})
        self._buf.append(z)
        if len(self._buf) > self.max_window:
            del self._buf[: len(self._buf) - self.max_window]
        n = len(self._buf)
        if n < 2 * self.min_window:
            return DriftSignal(self.name, False, 0.0, 0.0)

        # Prefix sums of value and value^2 make every candidate cut O(1) and
        # give the window variance the bound needs.
        total = 0.0
        prefix = [0.0]
        for v in self._buf:
            total += v
            prefix.append(total)
        mean_w = total / n
        var_w = sum((v - mean_w) ** 2 for v in self._buf) / n
        best_stat = 0.0
        best_eps = 0.0
        best_cut = -1
        delta_prime = self.delta / n
        log_term = math.log(2.0 / max(delta_prime, 1e-12))
        for cut in range(self.min_window, n - self.min_window + 1, self.cut_stride):
            n0, n1 = cut, n - cut
            mean0 = prefix[cut] / n0
            mean1 = (total - prefix[cut]) / n1
            m_harm = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt((2.0 / m_harm) * var_w * log_term) + (
                2.0 / (3.0 * m_harm)
            ) * log_term
            stat = abs(mean0 - mean1)
            if stat - eps > best_stat - best_eps:
                best_stat, best_eps, best_cut = stat, eps, cut
        self._last_stat = best_stat
        if best_cut > 0 and best_stat > best_eps:
            self.alarms += 1
            # Drop the stale half and re-standardise: the newer window is the
            # new normal.
            self._buf = self._buf[best_cut:]
            self._std.restart()
            return DriftSignal(
                self.name, True, best_stat, best_eps,
                severity=(best_stat - best_eps) / max(best_eps, 1e-9),
                detail={"cut": best_cut, "window": n},
            )
        return DriftSignal(self.name, False, best_stat, best_eps)

    def state(self) -> dict[str, object]:
        return {
            "n": self.n, "alarms": self.alarms, "window": len(self._buf),
            "statistic": round(self._last_stat, 4), "ready": self._std.ready,
        }


class FeatureShiftDetector:
    """Detects a change in the input distribution, before errors accumulate.

    Compares the mean of a recent window against a frozen reference window,
    per feature, in reference standard deviations, and reports the largest
    standardised shift. Fires when that exceeds ``threshold``.

    Deliberately crude: a full multivariate two-sample test (MMD, energy
    distance) would be more powerful but needs far more data to be reliable at
    ~200 dimensions, and its statistic is not interpretable. Here the alarm
    always comes with "which feature moved and by how many sigma", which is what
    makes it usable in the UI and in an explanation.
    """

    name = "feature_shift"

    def __init__(self, window: int = 120, threshold: float = 4.0, min_features: int = 1) -> None:
        self.window = max(20, int(window))
        self.threshold = float(threshold)
        self.min_features = int(min_features)
        self._ref: list[Welford] | None = None
        self._ref_n = 0
        self._recent: list[RingStats] | None = None
        self.n = 0
        self.alarms = 0
        self._names: list[str] = []
        self._last: dict[str, object] = {}

    def update(self, x: list[float] | tuple[float, ...], names: list[str] | None = None) -> DriftSignal:
        self.n += 1
        d = len(x)
        if self._ref is None:
            self._ref = [Welford() for _ in range(d)]
            self._recent = [RingStats(self.window) for _ in range(d)]
            self._names = list(names) if names else [f"f{i}" for i in range(d)]
        if d != len(self._ref):
            # Feature space changed (hardware appeared or disappeared). Start
            # over rather than compare incomparable vectors.
            self._ref = [Welford() for _ in range(d)]
            self._recent = [RingStats(self.window) for _ in range(d)]
            self._ref_n = 0
            self._names = list(names) if names else [f"f{i}" for i in range(d)]
        assert self._ref is not None and self._recent is not None

        if self._ref_n < self.window:
            for i in range(d):
                self._ref[i].push(float(x[i]))
            self._ref_n += 1
            return DriftSignal(self.name, False, 0.0, self.threshold, detail={"state": "warmup"})

        for i in range(d):
            self._recent[i].push(float(x[i]))
        if not self._recent[0].full:
            return DriftSignal(self.name, False, 0.0, self.threshold, detail={"state": "filling"})

        worst = 0.0
        worst_i = -1
        shifts: list[tuple[str, float]] = []
        for i in range(d):
            ref = self._ref[i]
            std = max(ref.std, _MIN_STD)
            # A constant feature (bias, or a flag that never changes) has std at
            # the floor; skip it rather than let 1e-6 in the denominator
            # manufacture a huge z-score.
            if ref.std < 1e-9:
                continue
            z = abs(self._recent[i].mean() - ref.mean) / std
            shifts.append((self._names[i], z))
            if z > worst:
                worst, worst_i = z, i
        shifts.sort(key=lambda kv: -kv[1])
        self._last = {"top": [(n, round(v, 3)) for n, v in shifts[:5]]}
        if worst > self.threshold:
            self.alarms += 1
            self._reset_reference()
            return DriftSignal(
                self.name, True, worst, self.threshold,
                severity=(worst - self.threshold) / self.threshold,
                detail={"feature": self._names[worst_i] if worst_i >= 0 else "?", **self._last},
            )
        return DriftSignal(self.name, False, worst, self.threshold, detail=self._last)

    def _reset_reference(self) -> None:
        assert self._recent is not None
        # New reference = the window that just triggered the alarm.
        d = len(self._recent)
        self._ref = []
        for i in range(d):
            w = Welford()
            for v in self._recent[i].values():
                w.push(v)
            self._ref.append(w)
        self._ref_n = self.window
        for r in self._recent:
            r.clear()

    def state(self) -> dict[str, object]:
        return {"n": self.n, "alarms": self.alarms, "ready": self._ref_n >= self.window, **self._last}
