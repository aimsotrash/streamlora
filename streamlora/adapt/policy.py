"""When to adapt.

The spec's requirement is that the trigger be configurable and that we do not
update after every sample. The four policies below OR together; the controller
records which one fired so the ablation between "periodic" and "drift-triggered"
adaptation is answerable from stored data rather than from the config file.

Rationale for the default (``periodic_samples`` + ``drift``): periodic alone
adapts pointlessly during long idle stretches and too slowly right after a
change; drift alone never adapts on a machine whose behaviour is genuinely
stable, so slow accuracy decay goes uncorrected. Together, the periodic trigger
is the floor and drift is the fast path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import AdaptConfig


@dataclass(slots=True)
class ScopeState:
    """Adaptation bookkeeping for one (signal, horizon)."""

    scope: str
    n_new: int = 0
    n_total: int = 0
    last_adapt_ts: float | None = None
    #: Sum of absolute errors since the last adaptation, for the error policy.
    error_accum: float = 0.0
    #: Drift events since the last adaptation, with their detectors.
    pending_drift: list[str] = field(default_factory=list)
    adaptations: int = 0
    promotions: int = 0
    rejections: int = 0
    rollbacks: int = 0

    def note_outcome(self, abs_error: float) -> None:
        self.n_new += 1
        self.n_total += 1
        self.error_accum += abs_error

    def note_drift(self, detector: str) -> None:
        if detector not in self.pending_drift:
            self.pending_drift.append(detector)

    def reset_after_adapt(self, ts: float) -> None:
        self.n_new = 0
        self.error_accum = 0.0
        self.pending_drift.clear()
        self.last_adapt_ts = ts
        self.adaptations += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope, "n_new": self.n_new, "n_total": self.n_total,
            "last_adapt_ts": self.last_adapt_ts, "error_accum": round(self.error_accum, 3),
            "pending_drift": list(self.pending_drift), "adaptations": self.adaptations,
            "promotions": self.promotions, "rejections": self.rejections,
            "rollbacks": self.rollbacks,
        }


@dataclass(slots=True)
class Trigger:
    name: str
    detail: dict[str, object] = field(default_factory=dict)


class AdaptationPolicy:
    """Evaluates the configured triggers against a scope's state."""

    def __init__(self, config: AdaptConfig | None = None) -> None:
        self.cfg = config or AdaptConfig()
        self.enabled_policies = set(self.cfg.policies)

    def should_adapt(self, state: ScopeState, now: float) -> Trigger | None:
        if not self.cfg.enabled:
            return None
        c = self.cfg
        # Drift first: it is the reason to adapt *sooner* than the schedule.
        if "drift" in self.enabled_policies and state.pending_drift:
            return Trigger("drift", {"detectors": list(state.pending_drift)})
        if "periodic_samples" in self.enabled_policies and state.n_new >= c.every_n_samples:
            return Trigger("periodic_samples", {"n_new": state.n_new})
        if "periodic_time" in self.enabled_policies:
            last = state.last_adapt_ts
            if last is None or (now - last) >= c.every_n_seconds:
                # Still require *some* new data: a time trigger with an empty
                # buffer would create a version identical to its parent and
                # pollute the version history.
                if state.n_new > 0:
                    return Trigger(
                        "periodic_time",
                        {"elapsed_s": None if last is None else round(now - last, 1)},
                    )
        if "error" in self.enabled_policies and state.error_accum >= c.error_budget:
            return Trigger("error", {"error_accum": round(state.error_accum, 2)})
        return None

    def describe(self) -> dict[str, object]:
        return {
            "enabled": self.cfg.enabled,
            "policies": sorted(self.enabled_policies),
            "every_n_samples": self.cfg.every_n_samples,
            "every_n_seconds": self.cfg.every_n_seconds,
            "error_budget": self.cfg.error_budget,
        }
