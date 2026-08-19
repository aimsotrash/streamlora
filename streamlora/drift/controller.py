"""Combines detectors into drift events and applies alarm hygiene.

Per (signal, horizon) the controller runs Page-Hinkley and ADWIN on the stream
of absolute errors; one shared detector watches the feature distribution. Two
pieces of hygiene that matter more than the detectors themselves:

* **Cooldown.** A regime change trips several detectors within seconds. Without
  a cooldown the adaptation controller would be handed a dozen "adapt now"
  triggers for one event, retrain repeatedly on nearly identical data, and
  burn the promotion gate's statistical power.
* **Warm-up gating.** Detectors return non-firing signals until they have a
  reference, so an alarm at sample 3 is impossible by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DriftConfig
from ..util.logging import get_logger
from .detectors import AdwinLite, DriftSignal, FeatureShiftDetector, PageHinkley

log = get_logger("drift")


@dataclass(slots=True)
class DriftEvent:
    ts: float
    detector: str
    scope: str                 # "error" | "feature"
    statistic: float
    threshold: float
    severity: float
    signal: str | None = None
    horizon_s: float | None = None
    detail: dict[str, object] = field(default_factory=dict)

    def as_row(self, run_id: str | None) -> dict[str, object]:
        return {
            "ts": self.ts, "detector": self.detector, "scope": self.scope,
            "signal": self.signal, "horizon_s": self.horizon_s,
            "statistic": self.statistic, "threshold": self.threshold,
            "severity": self.severity, "detail": self.detail, "run_id": run_id,
        }


class DriftController:
    def __init__(self, config: DriftConfig | None = None) -> None:
        self.cfg = config or DriftConfig()
        self._ph: dict[tuple[str, float], PageHinkley] = {}
        self._adwin: dict[tuple[str, float], AdwinLite] = {}
        self._feature = FeatureShiftDetector(
            window=self.cfg.feature_window, threshold=self.cfg.feature_threshold
        )
        self._last_alarm_ts: dict[str, float] = {}
        self.events: list[DriftEvent] = []
        self.n_suppressed = 0

    # -- error stream ------------------------------------------------------
    def observe_error(
        self, ts: float, signal: str, horizon_s: float, abs_error: float
    ) -> list[DriftEvent]:
        if not self.cfg.enabled:
            return []
        key = (signal, float(horizon_s))
        ph = self._ph.get(key)
        if ph is None:
            ph = PageHinkley(delta=self.cfg.ph_delta, threshold=self.cfg.ph_threshold)
            self._ph[key] = ph
        signals: list[DriftSignal] = [ph.update(abs_error)]
        if self.cfg.adwin_enabled:
            ad = self._adwin.get(key)
            if ad is None:
                ad = AdwinLite(delta=self.cfg.adwin_delta, min_window=self.cfg.adwin_min_window)
                self._adwin[key] = ad
            signals.append(ad.update(abs_error))
        out = []
        for s in signals:
            if not s.fired:
                continue
            ev = self._emit(
                ts, s, scope="error", signal=signal, horizon_s=float(horizon_s),
                cooldown_key=f"error|{signal}@{int(horizon_s)}",
            )
            if ev is not None:
                out.append(ev)
        return out

    # -- feature stream ----------------------------------------------------
    def observe_features(
        self, ts: float, values: list[float], names: list[str] | None = None
    ) -> list[DriftEvent]:
        if not self.cfg.enabled:
            return []
        s = self._feature.update(values, names)
        if not s.fired:
            return []
        ev = self._emit(ts, s, scope="feature", cooldown_key="feature")
        return [ev] if ev is not None else []

    def _emit(
        self, ts: float, s: DriftSignal, scope: str, cooldown_key: str,
        signal: str | None = None, horizon_s: float | None = None,
    ) -> DriftEvent | None:
        last = self._last_alarm_ts.get(cooldown_key)
        if last is not None and (ts - last) < self.cfg.cooldown_s:
            self.n_suppressed += 1
            return None
        self._last_alarm_ts[cooldown_key] = ts
        ev = DriftEvent(
            ts=ts, detector=s.detector, scope=scope, statistic=float(s.statistic),
            threshold=float(s.threshold), severity=float(s.severity), signal=signal,
            horizon_s=horizon_s, detail=dict(s.detail),
        )
        self.events.append(ev)
        if len(self.events) > 2000:
            del self.events[:1000]
        log.info(
            "drift detected", detector=ev.detector, scope=scope, signal=signal,
            horizon_s=horizon_s, statistic=round(ev.statistic, 3),
            threshold=round(ev.threshold, 3), severity=round(ev.severity, 3),
        )
        return ev

    def recent(self, since_ts: float) -> list[DriftEvent]:
        return [e for e in self.events if e.ts >= since_ts]

    def state(self) -> dict[str, object]:
        return {
            "enabled": self.cfg.enabled,
            "events": len(self.events),
            "suppressed": self.n_suppressed,
            "feature": self._feature.state(),
            "error": {
                f"{k[0]}@{int(k[1])}": {"page_hinkley": v.state(), "adwin": (
                    self._adwin[k].state() if k in self._adwin else None
                )}
                for k, v in sorted(self._ph.items(), key=str)
            },
        }
