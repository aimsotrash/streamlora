"""The learned forecaster: an online linear correction on top of the baselines.

Why this shape
--------------
The requirement is incremental learning without retraining, and a result that is
honestly better than simple baselines. Getting there took three iterations, all
of which are recorded in ``docs/experiments.md`` because the dead ends are the
interesting part:

1. **Predict the level.** A linear model scores well by learning "output roughly
   equals input", which is persistence with extra steps.
2. **Predict the change, scaled by the signal's declared range.** Fails for a
   different reason: battery percent varies by ~0.5 over five minutes on a
   0-100 range, so its informative slope feature has magnitude ~1e-3 and its
   contribution to the information matrix is ~400x smaller than the ridge
   penalty. Measured battery@900 MAE was 0.55 against a 0.076 linear-trend
   baseline.
3. **Predict the change in units of the signal's current volatility, with the
   baselines as inputs.** This is what shipped.

The final target is

    y = clip( (actual(t+h) - value(t)) / vol(t),  +/- target_clip )

where ``vol(t)`` is the signal's standard deviation over the last
``vol_span_s`` seconds, floored. Two properties follow, and both matter:

* **The target is stationary across regimes.** An idle laptop and a compiling
  one produce targets of similar magnitude, so one ridge penalty is correct for
  both, and a coefficient learned in one regime is not nonsense in the other.
* **Zero weights mean persistence, not nonsense.** Ridge shrinkage therefore
  degrades the model gracefully towards a real forecast. Combined with the
  baseline inputs below, a single unit coefficient reproduces EWMA or
  linear-trend exactly, so shrinkage lands somewhere in the convex hull of the
  baselines rather than at an arbitrary point.

Baselines as inputs
-------------------
Each baseline's forecast enters as ``(baseline_pred - value(t)) / vol(t)``. The
learned model is then a *stack*: it can learn "trust EWMA at 5 minutes, trust
linear-trend for battery at 30 minutes" and correct from there. This is stated
plainly because it changes how the headline number should be read -- the learned
arm is not an independent competitor that happens to win, it is a correction
layer, and the experiment measures how much the correction is worth.

Trust region
------------
Out-of-distribution inputs are where linear extrapolation does real damage: 85
small coefficients all pushed the same way sum to a huge correction. RLS already
tracks the parameter covariance, so ``s = sqrt(x' P x)`` is available for free
and is exactly the "how unfamiliar is this input" signal. The correction is
shrunk by ``1 / (1 + (s / (trust_k * s_typical))^2)``, with ``s_typical`` the
running median of recent values. On the idle-to-build scenario this cut the
static model's worst-case MAE from 28.3 to 11.2.

Numerical care
--------------
Naive RLS fails in two documented ways over long runs, both guarded here:
``P`` drifts out of symmetry until it loses positive definiteness (fixed by
symmetrising every update), and with a forgetting factor below 1 and
uninformative inputs ``P`` grows like ``lambda^-n`` until one informative sample
causes an enormous jump (bounded by a trace cap, logged when it triggers).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..util.logging import get_logger
from ..util.mathx import RingStats
from .base import ForecastContext, Forecaster, TrainingExample

log = get_logger("forecast.rls")

#: Bound on trace(P) relative to its initial value, guarding covariance windup.
_TRACE_GROWTH_LIMIT = 1e4
#: Recent sqrt(x'Px) values kept for the trust region's typical scale.
_UNC_POOL = 200
#: Minimum pool size before the trust region is applied at all.
_UNC_MIN = 20


@dataclass
class RLSRegressor:
    """Exponentially weighted recursive least squares with ridge initialisation.

    With ``forgetting == 1`` this is exactly batch ridge regression, order
    independent, verified against the closed-form solution to 1e-15 in the test
    suite. That equivalence is what makes the "static vs continually adapted"
    comparison meaningful: both arms are the same estimator, differing only in
    the forgetting factor and the update schedule.
    """

    d: int
    ridge_lambda: float = 1.0
    forgetting: float = 0.999
    w: np.ndarray = field(default=None)  # type: ignore[assignment]
    P: np.ndarray = field(default=None)  # type: ignore[assignment]
    n_updates: int = 0
    n_skipped: int = 0
    n_windup: int = 0
    sse: float = 0.0

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros(self.d, dtype=np.float64)
        if self.P is None:
            self.P = np.eye(self.d, dtype=np.float64) / max(self.ridge_lambda, 1e-12)
        self._trace0 = float(np.trace(self.P))

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def uncertainty(self, x: np.ndarray) -> float:
        """sqrt(x' P x): parameter uncertainty at ``x``, the trust-region input."""
        return math.sqrt(max(float(x @ self.P @ x), 0.0))

    def update(self, x: np.ndarray, y: float, weight: float = 1.0) -> bool:
        if not np.all(np.isfinite(x)) or not math.isfinite(y):
            self.n_skipped += 1
            return False
        lam = self.forgetting
        Px = self.P @ x
        denom = lam + float(x @ Px) * weight
        if denom <= 1e-12 or not math.isfinite(denom):
            self.n_skipped += 1
            return False
        k = (Px * weight) / denom
        err = y - float(np.dot(self.w, x))
        self.w = self.w + k * err
        self.P = (self.P - np.outer(k, Px)) / lam
        self.P = 0.5 * (self.P + self.P.T)
        tr = float(np.trace(self.P))
        if not math.isfinite(tr) or tr <= 0:
            self.P = np.eye(self.d) / max(self.ridge_lambda, 1e-12)
            self._trace0 = float(np.trace(self.P))
            self.n_skipped += 1
            log.warning("RLS covariance reset", reason="non-finite trace", d=self.d)
            return False
        limit = self._trace0 * _TRACE_GROWTH_LIMIT
        if tr > limit:
            self.P *= limit / tr
            self.n_windup += 1
            log.dedupe("windup", "RLS covariance windup bounded", level="debug",
                       trace=tr, limit=limit, updates=self.n_updates)
        self.n_updates += 1
        self.sse += err * err
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> int:
        applied = 0
        for i in range(X.shape[0]):
            if self.update(X[i], float(y[i])):
                applied += 1
        return applied

    def copy(self) -> "RLSRegressor":
        r = RLSRegressor(d=self.d, ridge_lambda=self.ridge_lambda,
                         forgetting=self.forgetting, w=self.w.copy(), P=self.P.copy())
        r.n_updates, r.n_skipped, r.n_windup, r.sse = (
            self.n_updates, self.n_skipped, self.n_windup, self.sse
        )
        r._trace0 = self._trace0
        return r

    def top_coefficients(self, names: Sequence[str], k: int = 8) -> list[tuple[str, float]]:
        if len(names) != self.d:
            names = [f"f{i}" for i in range(self.d)]
        order = np.argsort(-np.abs(self.w))[:k]
        return [(names[int(i)], float(self.w[int(i)])) for i in order]


@dataclass(slots=True)
class ScopeSpec:
    """Static description of one (signal, horizon) model."""

    signal: str
    horizon_s: float
    #: Column indices into the shared per-tick feature vector.
    mask: np.ndarray
    #: Names of the baselines whose forecasts are appended as inputs, in order.
    baseline_names: tuple[str, ...]
    #: Floor on the volatility scale, in native units.
    vol_floor: float
    lo: float | None = None
    hi: float | None = None

    @property
    def key(self) -> tuple[str, float]:
        return (self.signal, float(self.horizon_s))

    @property
    def name(self) -> str:
        return f"{self.signal}@{int(self.horizon_s)}"

    @property
    def d(self) -> int:
        return int(self.mask.size) + len(self.baseline_names)


@dataclass
class ScopeModel:
    """Everything learned for one (signal, horizon).

    Standardisation is **per scope** rather than global because each scope has a
    different input vector: its own signal's features in full detail, a coarse
    summary of the others, and its own horizon's baseline forecasts. A shared
    standardiser would have to ignore the baseline inputs entirely.
    """

    spec: ScopeSpec
    ridge_lambda: float = 10.0
    forgetting: float = 0.999
    sigma_floor: float = 0.02
    clip_z: float = 5.0
    target_clip: float = 8.0
    trust_k: float = 3.0
    #: "correction" shrinks only the telemetry-derived block; "all" shrinks the
    #: whole output. See ForecastConfig.trust_region_scope.
    trust_region_scope: str = "correction"
    standardize: bool = True
    per_regime: bool = False
    models: dict[str, RLSRegressor] = field(default_factory=dict)
    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    version_n: int = 1
    _warmup: list[np.ndarray] = field(default_factory=list)
    _unc: RingStats = field(default_factory=lambda: RingStats(_UNC_POOL))

    # -- input assembly ----------------------------------------------------
    def raw_input(self, x_shared: np.ndarray, extras: np.ndarray) -> np.ndarray:
        return np.concatenate([x_shared[self.spec.mask], extras])

    @property
    def standardizer_ready(self) -> bool:
        return (not self.standardize) or self.mu is not None

    def observe_warmup(self, v: np.ndarray, need: int) -> bool:
        """Collect a warm-up input; freeze and return True when complete."""
        if self.standardizer_ready:
            return False
        self._warmup.append(np.asarray(v, dtype=np.float64).copy())
        if len(self._warmup) < need:
            return False
        M = np.vstack(self._warmup)
        self.freeze_standardizer(M.mean(axis=0), M.std(axis=0, ddof=0))
        self._warmup.clear()
        return True

    def freeze_standardizer(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        mu = np.asarray(mu, dtype=np.float64).copy()
        sigma = np.asarray(sigma, dtype=np.float64).copy()
        bad = ~np.isfinite(sigma)
        sigma[bad] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        sigma = np.maximum(sigma, self.sigma_floor)
        # The bias column must survive untouched or the intercept disappears.
        mu[0] = 0.0
        sigma[0] = 1.0
        self.mu = mu
        self.sigma = sigma

    def prepare(self, v: np.ndarray) -> np.ndarray | None:
        if not self.standardize:
            return v
        if self.mu is None or self.sigma is None:
            return None
        return np.clip((v - self.mu) / self.sigma, -self.clip_z, self.clip_z)

    # -- regime routing ----------------------------------------------------
    def _key(self, regime: str) -> str:
        return regime if self.per_regime else "*"

    def get(self, regime: str, create: bool = False) -> RLSRegressor | None:
        k = self._key(regime)
        m = self.models.get(k)
        if m is None and self.per_regime:
            m = self.models.get("*")
        if m is None and create:
            base = self.models.get("*")
            # A newly seen regime starts from the shared model when one exists:
            # starting a rare regime from zero discards everything general.
            m = base.copy() if base is not None else RLSRegressor(
                d=self.spec.d, ridge_lambda=self.ridge_lambda, forgetting=self.forgetting
            )
            self.models[k] = m
        return m

    @property
    def n_updates(self) -> int:
        return sum(m.n_updates for m in self.models.values())

    # -- inference ---------------------------------------------------------
    @property
    def n_stack(self) -> int:
        """Width of the un-shrunk block: the intercept plus baseline forecasts."""
        return 1 + len(self.spec.baseline_names)

    def shrink_factor(self, m: RLSRegressor, v_std: np.ndarray) -> float:
        """Trust-region multiplier in (0, 1]. 1.0 means "familiar input"."""
        if len(self._unc) < _UNC_MIN:
            # Not enough training-time uncertainties to know what typical looks
            # like. Applying a factor derived from two samples would be noise.
            return 1.0
        ref = max(self._unc.quantile(0.5), 1e-12)
        s = m.uncertainty(v_std)
        return 1.0 / (1.0 + (s / max(self.trust_k * ref, 1e-12)) ** 2)

    def correction(self, v_std: np.ndarray, regime: str,
                   m: RLSRegressor | None = None) -> float | None:
        """Normalised, trust-shrunk, clipped correction. None if unusable."""
        if m is None:
            m = self.get(regime)
        if m is None or m.n_updates == 0:
            return None
        shrink = self.shrink_factor(m, v_std)
        if self.trust_region_scope == "all":
            raw = m.predict(v_std) * shrink
        else:
            # Two blocks. The stack (intercept + baseline forecasts) is
            # low-dimensional and well conditioned, so it is trusted as-is; the
            # telemetry-derived correction is what extrapolates badly and gets
            # shrunk. Column order is fixed by ScopeModel.raw_input: masked
            # features first (index 0 is the bias), then the baselines.
            k = len(self.spec.baseline_names)
            n = v_std.size
            stack_idx = np.concatenate(
                [np.array([0], dtype=np.int64), np.arange(n - k, n, dtype=np.int64)]
            ) if k else np.array([0], dtype=np.int64)
            stack = float(np.dot(m.w[stack_idx], v_std[stack_idx]))
            raw = stack + (float(np.dot(m.w, v_std)) - stack) * shrink
        if not math.isfinite(raw):
            return None
        return float(np.clip(raw, -self.target_clip, self.target_clip))

    def target(self, actual: float, anchor: float, vol: float) -> float:
        return float(np.clip((actual - anchor) / max(vol, 1e-9),
                             -self.target_clip, self.target_clip))

    def apply(self, correction: float, anchor: float, vol: float) -> float:
        out = anchor + correction * max(vol, 1e-9)
        if self.spec.lo is not None:
            out = max(out, self.spec.lo)
        if self.spec.hi is not None:
            out = min(out, self.spec.hi)
        return float(out)

    # -- learning ----------------------------------------------------------
    def note_uncertainty(self, s: float) -> None:
        """Record a sqrt(x'Px) observed at training time.

        The trust region compares a prediction's uncertainty against the typical
        uncertainty of inputs the model was *trained* on, so this pool must be
        fed from the training path -- including the adaptation controller, which
        trains cloned regressors and therefore cannot rely on ``update`` below.
        """
        if math.isfinite(s):
            self._unc.push(float(s))

    def update(self, v_std: np.ndarray, y: float, regime: str, weight: float = 1.0) -> bool:
        m = self.get(regime, create=True)
        if m is None:
            return False
        # Track the typical uncertainty *before* the update, so the trust region
        # reference reflects inputs the model has actually been trained on.
        self.note_uncertainty(m.uncertainty(v_std))
        return m.update(v_std, y, weight=weight)

    def clone_models(self) -> dict[str, RLSRegressor]:
        return {k: v.copy() for k, v in self.models.items()}

    def install_models(self, models: dict[str, RLSRegressor]) -> None:
        self.models = dict(models)

    def input_names(self, feature_names: Sequence[str]) -> list[str]:
        return [feature_names[int(i)] for i in self.spec.mask] + [
            f"baseline|{b}" for b in self.spec.baseline_names
        ]

    def describe(self, feature_names: Sequence[str]) -> dict[str, object]:
        m = self.get("*") or next(iter(self.models.values()), None)
        return {
            "scope": self.spec.name, "d": self.spec.d, "version": self.version_n,
            "n_updates": self.n_updates, "standardizer_ready": self.standardizer_ready,
            "regimes": sorted(self.models),
            "unc_pool": len(self._unc),
            "top_coefficients": (
                m.top_coefficients(self.input_names(feature_names)) if m else []
            ),
        }


class RLSForecaster(Forecaster):
    """Bank of per-scope online models. No language model is involved."""

    kind = "rls"
    learnable = True

    def __init__(
        self,
        feature_names: Sequence[str],
        scopes: dict[tuple[str, float], ScopeSpec],
        ridge_lambda: float = 10.0,
        forgetting: float = 0.999,
        sigma_floor: float = 0.02,
        clip_z: float = 5.0,
        target_clip: float = 8.0,
        trust_k: float = 3.0,
        trust_region_scope: str = "correction",
        standardize: bool = True,
        standardize_warmup: int = 120,
        per_regime: bool = False,
        frozen: bool = False,
        vol_span_s: float = 300.0,
    ) -> None:
        self.feature_names = list(feature_names)
        self.ridge_lambda = float(ridge_lambda)
        self.forgetting = float(forgetting)
        self.standardize_warmup = int(standardize_warmup)
        self.per_regime = bool(per_regime)
        #: When frozen, updates are ignored: the "static trained model" arm.
        self.frozen = bool(frozen)
        self.vol_span_s = float(vol_span_s)
        self.scopes: dict[tuple[str, float], ScopeModel] = {
            k: ScopeModel(
                spec=v, ridge_lambda=ridge_lambda, forgetting=forgetting,
                sigma_floor=sigma_floor, clip_z=clip_z, target_clip=target_clip,
                trust_k=trust_k, trust_region_scope=trust_region_scope,
                standardize=standardize, per_regime=per_regime,
            )
            for k, v in scopes.items()
        }

    # -- lookup ------------------------------------------------------------
    def scope(self, signal: str, horizon_s: float) -> str:
        return f"{signal}@{int(horizon_s)}"

    def model(self, signal: str, horizon_s: float) -> ScopeModel | None:
        return self.scopes.get((signal, float(horizon_s)))

    def signals(self) -> list[str]:
        return sorted({s for s, _ in self.scopes})

    def standardizer_ready(self, signal: str, horizon_s: float) -> bool:
        m = self.model(signal, horizon_s)
        return bool(m and m.standardizer_ready)

    def n_updates(self, signal: str, horizon_s: float) -> int:
        m = self.model(signal, horizon_s)
        return m.n_updates if m else 0

    # -- input assembly ----------------------------------------------------
    def volatility(self, ctx: ForecastContext, signal: str) -> float:
        sm = next((m for (s, _), m in self.scopes.items() if s == signal), None)
        floor = sm.spec.vol_floor if sm else 0.5
        return max(ctx.volatility(signal, self.vol_span_s), floor)

    def build_extras(
        self, ctx: ForecastContext, signal: str, horizon_s: float, anchor: float, vol: float
    ) -> np.ndarray | None:
        """Baseline forecasts, expressed as normalised offsets from the anchor."""
        sm = self.model(signal, horizon_s)
        if sm is None:
            return None
        out = np.zeros(len(sm.spec.baseline_names))
        preds = ctx.baseline_preds.get((signal, float(horizon_s)), {})
        for i, name in enumerate(sm.spec.baseline_names):
            p = preds.get(name)
            if p is None or not math.isfinite(p):
                continue
            out[i] = float(np.clip((p - anchor) / max(vol, 1e-9),
                                   -sm.target_clip, sm.target_clip))
        return out

    # -- inference ---------------------------------------------------------
    def predict(self, ctx: ForecastContext, signal: str, horizon_s: float) -> float | None:
        sm = self.model(signal, horizon_s)
        if sm is None:
            return None
        anchor = ctx.current(signal)
        if anchor is None:
            return None
        vol = self.volatility(ctx, signal)
        extras = self.build_extras(ctx, signal, horizon_s, anchor, vol)
        if extras is None:
            return None
        v_std = sm.prepare(sm.raw_input(ctx.fv.values, extras))
        if v_std is None:
            return None
        corr = sm.correction(v_std, ctx.regime)
        if corr is None:
            # An unfitted model must not predict: a zero-weight linear model
            # would silently impersonate persistence and inflate the learned
            # arm's apparent skill.
            return None
        return sm.apply(corr, anchor, vol)

    def predict_from_parts(
        self, signal: str, horizon_s: float, x_shared: np.ndarray, extras: np.ndarray,
        anchor: float, vol: float, regime: str,
        models: dict[str, RLSRegressor] | None = None,
    ) -> float | None:
        """Predict from stored parts, optionally with an alternative model set.

        Used by the promotion gate and the error-budget accounting, which need to
        score candidate and active models on identical stored inputs without
        copying an 85x85 covariance per row.
        """
        sm = self.model(signal, horizon_s)
        if sm is None:
            return None
        v_std = sm.prepare(sm.raw_input(x_shared, extras))
        if v_std is None:
            return None
        if models is None:
            corr = sm.correction(v_std, regime)
        else:
            key = regime if self.per_regime else "*"
            m = models.get(key) or models.get("*")
            if m is None or m.n_updates == 0:
                return None
            corr = sm.correction(v_std, regime, m=m)
            if corr is None:
                return float("nan")
        if corr is None:
            return None
        return sm.apply(corr, anchor, vol)

    # -- learning ----------------------------------------------------------
    def observe_warmup(self, ex_signal: str, horizon_s: float, x_shared: np.ndarray,
                       extras: np.ndarray) -> bool:
        sm = self.model(ex_signal, horizon_s)
        if sm is None:
            return False
        return sm.observe_warmup(sm.raw_input(x_shared, extras), self.standardize_warmup)

    def update(self, examples: Sequence[TrainingExample]) -> int:
        if self.frozen:
            return 0
        applied = 0
        for ex in examples:
            sm = self.model(ex.signal, ex.horizon_s)
            if sm is None or ex.extras is None:
                continue
            v_std = sm.prepare(sm.raw_input(ex.x, ex.extras))
            if v_std is None:
                continue
            y = sm.target(ex.actual, ex.anchor, ex.vol)
            if sm.update(v_std, y, ex.regime, weight=ex.weight):
                applied += 1
        return applied

    def fit(self, examples: Sequence[TrainingExample]) -> int:
        was = self.frozen
        self.frozen = False
        try:
            return self.update(examples)
        finally:
            self.frozen = was

    # -- versioning --------------------------------------------------------
    def version(self, signal: str = "", horizon_s: float = 0.0) -> str:
        sm = self.model(signal, horizon_s) if signal else None
        n = sm.version_n if sm else 1
        return f"forecast-model-v{n:03d}"

    def set_version(self, signal: str, horizon_s: float, n: int) -> None:
        sm = self.model(signal, horizon_s)
        if sm is not None:
            sm.version_n = int(n)

    def clone_unit(self, signal: str, horizon_s: float) -> dict[str, RLSRegressor]:
        sm = self.model(signal, horizon_s)
        return sm.clone_models() if sm else {}

    def install_unit(self, signal: str, horizon_s: float, models: dict[str, RLSRegressor]) -> None:
        sm = self.model(signal, horizon_s)
        if sm is not None:
            sm.install_models(models)

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind, "learnable": self.learnable, "frozen": self.frozen,
            "ridge_lambda": self.ridge_lambda, "forgetting": self.forgetting,
            "per_regime": self.per_regime, "vol_span_s": self.vol_span_s,
            "scopes": {
                m.spec.name: m.describe(self.feature_names)
                for m in sorted(self.scopes.values(), key=lambda x: x.spec.name)
            },
        }

    # -- serialisation -----------------------------------------------------
    def save(self, path: str) -> None:
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, object] = {
            "kind": self.kind, "feature_names": self.feature_names,
            "ridge_lambda": self.ridge_lambda, "forgetting": self.forgetting,
            "per_regime": self.per_regime, "frozen": self.frozen,
            "vol_span_s": self.vol_span_s, "standardize_warmup": self.standardize_warmup,
            "scopes": [],
        }
        for i, (key, sm) in enumerate(sorted(self.scopes.items(), key=lambda kv: str(kv[0]))):
            arrays[f"mask::{i}"] = sm.spec.mask
            entry: dict[str, object] = {
                "i": i, "signal": sm.spec.signal, "horizon_s": sm.spec.horizon_s,
                "baseline_names": list(sm.spec.baseline_names),
                "vol_floor": sm.spec.vol_floor, "lo": sm.spec.lo, "hi": sm.spec.hi,
                "version_n": sm.version_n, "standardize": sm.standardize,
                "sigma_floor": sm.sigma_floor, "clip_z": sm.clip_z,
                "target_clip": sm.target_clip, "trust_k": sm.trust_k,
                "trust_region_scope": sm.trust_region_scope,
                "has_std": sm.mu is not None, "unc": sm._unc.values(),
                "models": [],
            }
            if sm.mu is not None and sm.sigma is not None:
                arrays[f"mu::{i}"] = sm.mu
                arrays[f"sg::{i}"] = sm.sigma
            for j, (regime, m) in enumerate(sorted(sm.models.items())):
                arrays[f"w::{i}::{j}"] = m.w
                arrays[f"P::{i}::{j}"] = m.P
                entry["models"].append({  # type: ignore[union-attr]
                    "j": j, "regime": regime, "d": m.d, "n_updates": m.n_updates,
                    "n_skipped": m.n_skipped, "n_windup": m.n_windup, "sse": m.sse,
                })
            meta["scopes"].append(entry)  # type: ignore[union-attr]
        buf = io.BytesIO()
        np.savez_compressed(
            buf, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays
        )
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())

    @classmethod
    def load(cls, path: str) -> "RLSForecaster":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode())
            scopes: dict[tuple[str, float], ScopeSpec] = {}
            for e in meta["scopes"]:
                spec = ScopeSpec(
                    signal=e["signal"], horizon_s=float(e["horizon_s"]),
                    mask=z[f"mask::{e['i']}"].copy(),
                    baseline_names=tuple(e["baseline_names"]),
                    vol_floor=float(e["vol_floor"]), lo=e["lo"], hi=e["hi"],
                )
                scopes[spec.key] = spec
            f = cls(
                feature_names=meta["feature_names"], scopes=scopes,
                ridge_lambda=meta["ridge_lambda"], forgetting=meta["forgetting"],
                per_regime=meta["per_regime"], frozen=meta["frozen"],
                vol_span_s=meta["vol_span_s"],
                standardize_warmup=meta.get("standardize_warmup", 120),
            )
            for e in meta["scopes"]:
                sm = f.scopes[(e["signal"], float(e["horizon_s"]))]
                sm.version_n = int(e["version_n"])
                sm.standardize = bool(e["standardize"])
                sm.sigma_floor = float(e["sigma_floor"])
                sm.clip_z = float(e["clip_z"])
                sm.target_clip = float(e["target_clip"])
                sm.trust_k = float(e["trust_k"])
                sm.trust_region_scope = e.get("trust_region_scope", "correction")
                if e["has_std"]:
                    sm.freeze_standardizer(z[f"mu::{e['i']}"], z[f"sg::{e['i']}"])
                sm._unc.extend(e.get("unc", []))
                for md in e["models"]:
                    w = z[f"w::{e['i']}::{md['j']}"].copy()
                    P = z[f"P::{e['i']}::{md['j']}"].copy()
                    if w.size != md["d"] or P.shape != (md["d"], md["d"]):
                        raise ValueError(
                            f"corrupt model file {path}: scope {e['signal']} expected "
                            f"d={md['d']}, got w={w.shape} P={P.shape}"
                        )
                    m = RLSRegressor(
                        d=int(md["d"]), ridge_lambda=meta["ridge_lambda"],
                        forgetting=meta["forgetting"], w=w, P=P,
                    )
                    m.n_updates = int(md["n_updates"])
                    m.n_skipped = int(md.get("n_skipped", 0))
                    m.n_windup = int(md.get("n_windup", 0))
                    m.sse = float(md.get("sse", 0.0))
                    sm.models[md["regime"]] = m
        return f


def build_scope_specs(
    signals: Sequence[str],
    horizons: Sequence[float],
    masks: dict[str, np.ndarray],
    baseline_names: Sequence[str],
    ranges: dict[str, tuple[float | None, float | None]],
    vol_floor_frac: float = 0.005,
) -> dict[tuple[str, float], ScopeSpec]:
    out: dict[tuple[str, float], ScopeSpec] = {}
    for sig in signals:
        mask = masks.get(sig)
        if mask is None:
            continue
        lo, hi = ranges.get(sig, (None, None))
        span = (hi - lo) if (lo is not None and hi is not None and hi > lo) else 100.0
        for hor in horizons:
            spec = ScopeSpec(
                signal=sig, horizon_s=float(hor), mask=np.asarray(mask, dtype=np.int64),
                baseline_names=tuple(baseline_names),
                vol_floor=vol_floor_frac * span, lo=lo, hi=hi,
            )
            out[spec.key] = spec
    return out
