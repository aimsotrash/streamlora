"""The predefined ablation arms.

Each arm exists to answer one documented research question, and the arm's
``description`` states which. Anything that cannot be phrased that way does not
belong here.
"""

from __future__ import annotations

from .runner import Arm

#: A: no adaptation. Learns during the training prefix, then frozen.
#: Question: does continual adaptation beat a static model trained on the past?
STATIC = Arm(
    name="static",
    description="Trained on the chronological prefix, then frozen. The control arm.",
    overrides={"adapt.policies": "periodic_samples", "adapt.every_n_samples": "60"},
    freeze_after_train=True,
)

#: B: periodic adaptation only.
PERIODIC = Arm(
    name="online_periodic",
    description="Adapts every N resolved outcomes, ignoring drift signals.",
    overrides={"adapt.policies": "periodic_samples"},
)

#: C: drift-triggered adaptation only.
#: Question: does drift-triggered adaptation improve efficiency (fewer updates
#: for comparable accuracy) over a fixed schedule?
DRIFT = Arm(
    name="online_drift",
    description="Adapts only when a drift detector fires.",
    overrides={"adapt.policies": "drift"},
)

#: D: both, the shipped default.
PERIODIC_DRIFT = Arm(
    name="online_periodic_drift",
    description="Periodic floor plus a drift fast path. The default policy.",
    overrides={"adapt.policies": "periodic_samples,drift"},
)

#: E: online learning with the promotion gate disabled.
#: Question: what does the safety machinery actually buy?
NO_GATE = Arm(
    name="online_no_gate",
    description="Promotes every candidate unconditionally; no evaluation gate.",
    overrides={"adapt.policies": "periodic_samples,drift", "adapt.gate_enabled": "false"},
)

#: F: a separate model per workload regime.
#: Question: is regime-conditioned modelling worth the data fragmentation?
PER_REGIME = Arm(
    name="per_regime",
    description="One model per workload regime instead of regime one-hot features.",
    overrides={
        "adapt.policies": "periodic_samples,drift",
        "forecast.per_regime_models": "true",
    },
)

#: G: no trust region, to show what bounds extrapolation damage.
NO_TRUST_REGION = Arm(
    name="no_trust_region",
    description="Trust-region shrinkage effectively disabled (very large trust_k).",
    overrides={"adapt.policies": "periodic_samples,drift", "forecast.trust_k": "1000000"},
)

#: I: trust region applied to the whole output rather than the correction block.
#: Question: does backing off toward the baseline stack beat backing off toward
#: persistence?
TRUST_CORRECTION = Arm(
    name="trust_region_correction",
    description="Trust region spares the intercept and baseline inputs (shrinks only the telemetry block).",
    overrides={
        "adapt.policies": "periodic_samples,drift",
        "forecast.trust_region_scope": "correction",
    },
)

#: H: no baseline inputs, so the learned model must stand alone.
NO_BASELINE_FEATURES = Arm(
    name="no_baseline_features",
    description="Baseline forecasts removed from the input vector.",
    overrides={
        "adapt.policies": "periodic_samples,drift",
        "forecast.baseline_features": "false",
    },
)

SUITES: dict[str, list[Arm]] = {
    "core": [STATIC, PERIODIC, DRIFT, PERIODIC_DRIFT],
    "safety": [PERIODIC_DRIFT, NO_GATE],
    "design": [PERIODIC_DRIFT, NO_TRUST_REGION, TRUST_CORRECTION, NO_BASELINE_FEATURES],
    "regime": [PERIODIC_DRIFT, PER_REGIME],
    "full": [
        STATIC, PERIODIC, DRIFT, PERIODIC_DRIFT, NO_GATE, PER_REGIME,
        NO_TRUST_REGION, TRUST_CORRECTION, NO_BASELINE_FEATURES,
    ],
}

ALL_ARMS = {
    a.name: a
    for a in [STATIC, PERIODIC, DRIFT, PERIODIC_DRIFT, NO_GATE, PER_REGIME,
              NO_TRUST_REGION, TRUST_CORRECTION, NO_BASELINE_FEATURES]
}
