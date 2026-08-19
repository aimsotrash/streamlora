"""Adaptation controller: decide, train a candidate, gate it, promote or reject.

This is where continual learning either earns its keep or is caught not doing
so. The loop per scope (one signal x one horizon):

    resolved outcome -> buffer
                     -> policy: is it time?
                     -> candidate = copy(active); apply incremental updates
                     -> gate: score candidate and active on held-out newest data
                     -> promote (new version, activate) | reject (discard)

Three design points that make the difference between this and a naive online
learner:

1. **The gate is chronologically held out.** The newest ``gate_window``
   outcomes are never trained on by the candidate; the candidate trains on
   older data and is judged on newer. Same orientation as production.
2. **Both models are scored on the same rows.** The comparison re-runs the
   active model over the identical gate features, so the difference is the
   update, not a different sample of reality.
3. **Rejection is cheap and normal.** A rejected candidate is discarded and the
   active version continues; the event is recorded either way. A pipeline that
   only ever promotes is not a safety mechanism, it is an unconditional update
   with extra logging.

Post-promotion regression watch
-------------------------------
Passing the gate does not prove a version is good -- the gate is 25-80 samples.
So after promoting, the controller keeps scoring the live version on the next
outcomes and, if its error is materially worse than the version it replaced over
a full window, rolls back. That is the difference between a gate and a guarantee.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..config import AdaptConfig
from ..forecast.base import TrainingExample
from ..forecast.linear import RLSForecaster, RLSRegressor
from ..store.repo import Repos
from ..util.ids import config_hash
from ..util.logging import get_logger
from .buffer import TrainingBuffer
from .policy import AdaptationPolicy, ScopeState, Trigger
from .registry import ModelVersionRegistry

log = get_logger("adapt.controller")


def stable_seed_offset(key: str, modulo: int = 10_000) -> int:
    """Process-independent integer derived from a string.

    ``hash(str)`` is salted per interpreter, so anything seeded from it is
    reproducible within a run and not across runs.
    """
    return int.from_bytes(hashlib.blake2s(key.encode(), digest_size=4).digest(), "big") % modulo


@dataclass(slots=True)
class GateResult:
    n: int
    active_mae: float | None
    candidate_mae: float | None
    promote: bool
    reason: str
    active_rmse: float | None = None
    candidate_rmse: float | None = None
    #: Best baseline MAE on the same gate rows. Recorded for observability: it is
    #: the number that says whether the learned model is worth serving at all.
    baseline_mae: float | None = None
    baseline_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "active_mae": None if self.active_mae is None else round(self.active_mae, 4),
            "candidate_mae": None if self.candidate_mae is None else round(self.candidate_mae, 4),
            "active_rmse": None if self.active_rmse is None else round(self.active_rmse, 4),
            "candidate_rmse": None if self.candidate_rmse is None else round(self.candidate_rmse, 4),
            "baseline_mae": None if self.baseline_mae is None else round(self.baseline_mae, 4),
            "baseline_name": self.baseline_name,
            "promote": self.promote,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AdaptationOutcome:
    scope: str
    signal: str
    horizon_s: float
    trigger: str
    decision: str
    gate: GateResult
    active_version: str | None
    candidate_version: str | None
    n_train: int
    duration_ms: float
    detail: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _Watch:
    """Post-promotion regression watch for one scope."""

    version: str
    predecessor: str | None
    predecessor_mae: float | None
    #: Absorbed high-water mark *before* this promotion. Restored on rollback so
    #: the reinstated model can still learn from the examples the rejected
    #: version consumed -- otherwise every rollback permanently discards a
    #: batch of training data.
    absorbed_before: int = 0
    errors: list[float] = field(default_factory=list)

    def observe(self, abs_error: float) -> None:
        self.errors.append(abs_error)


class AdaptationController:
    """Owns training buffers, policies, gating and version transitions."""

    def __init__(
        self,
        forecaster: RLSForecaster,
        registry: ModelVersionRegistry,
        repos: Repos,
        config: AdaptConfig | None = None,
        run_id: str | None = None,
        seed: int = 1337,
        config_fingerprint: str = "",
    ) -> None:
        self.forecaster = forecaster
        self.registry = registry
        self.repos = repos
        self.cfg = config or AdaptConfig()
        self.policy = AdaptationPolicy(self.cfg)
        self.run_id = run_id
        self.seed = int(seed)
        self.config_fingerprint = config_fingerprint
        self._buffers: dict[str, TrainingBuffer] = {}
        self._states: dict[str, ScopeState] = {}
        self._watch: dict[str, _Watch] = {}
        self.outcomes: list[AdaptationOutcome] = []
        self.n_promoted = 0
        self.n_rejected = 0
        self.n_rolled_back = 0

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def scope_of(signal: str, horizon_s: float) -> str:
        return f"{signal}@{int(horizon_s)}"

    def buffer(self, scope: str) -> TrainingBuffer:
        b = self._buffers.get(scope)
        if b is None:
            b = TrainingBuffer(
                recency=self.cfg.recency_buffer, reservoir=self.cfg.reservoir_buffer,
                # Per-scope seed offset so the scopes do not draw correlated
                # reservoir samples. Derived with a *stable* hash: Python's
                # built-in hash() is randomised per process (PYTHONHASHSEED), so
                # using it here made two runs of the same experiment in separate
                # processes produce different models. The in-process determinism
                # test did not catch it, because one process has one hash seed.
                seed=self.seed + stable_seed_offset(scope),
            )
            self._buffers[scope] = b
        return b

    def state(self, scope: str) -> ScopeState:
        s = self._states.get(scope)
        if s is None:
            s = ScopeState(scope=scope)
            self._states[scope] = s
        return s

    # -- ingestion ---------------------------------------------------------
    def observe_outcome(self, ex: TrainingExample, abs_error: float) -> None:
        scope = self.scope_of(ex.signal, ex.horizon_s)
        self.buffer(scope).add(ex)
        self.state(scope).note_outcome(abs_error)
        w = self._watch.get(scope)
        if w is not None:
            w.observe(abs_error)

    def note_drift(
        self, signal: str | None, horizon_s: float | None, detector: str,
        feature: str | None = None,
    ) -> None:
        """Route a drift event to the scopes it should accelerate.

        An error-based alarm names its own scope. A feature-distribution alarm
        does not, but it reports *which feature* moved -- and feature names are
        ``<signal>|<family>``, so the implicated signal is recoverable. Routing by
        it matters: fanning every feature alarm out to all scopes turned 55 drift
        events into 330 adaptation triggers, and each adaptation is a selection
        event against a finite gate window.
        """
        if signal is not None and horizon_s is not None:
            self.state(self.scope_of(signal, horizon_s)).note_drift(detector)
            return
        implicated: str | None = None
        if feature and "|" in feature:
            head = feature.split("|", 1)[0]
            if head not in ("time", "regime", "meta"):
                implicated = head
        for scope, st in self._states.items():
            if implicated is not None and not scope.startswith(implicated + "@"):
                continue
            st.note_drift(detector)

    # -- the loop ----------------------------------------------------------
    def maybe_adapt(self, now: float, scopes: Sequence[str] | None = None) -> list[AdaptationOutcome]:
        if not self.cfg.enabled:
            return []
        out: list[AdaptationOutcome] = []
        for scope in list(scopes if scopes is not None else self._states):
            st = self.state(scope)
            trig = self.policy.should_adapt(st, now)
            if trig is None:
                continue
            res = self._adapt_scope(scope, trig, now)
            if res is not None:
                out.append(res)
        return out

    def _split_scope(self, scope: str) -> tuple[str, float]:
        sig, _, hor = scope.rpartition("@")
        return sig, float(hor)

    def _adapt_scope(self, scope: str, trig: Trigger, now: float) -> AdaptationOutcome | None:
        t0 = time.perf_counter()
        signal, horizon_s = self._split_scope(scope)
        st = self.state(scope)
        buf = self.buffer(scope)

        train_pool, gate = buf.split_gate(self.cfg.gate_window)
        if self.cfg.gate_enabled and len(gate) < self.cfg.gate_min_samples:
            # Not enough held-out data to judge anything. Skipping is recorded:
            # a silent skip looks identical to "no trigger fired" when
            # debugging why a model stopped improving.
            st.pending_drift.clear()
            return self._record(
                now, scope, signal, horizon_s, trig, "skipped",
                GateResult(len(gate), None, None, False, "insufficient_gate_samples"),
                None, None, 0, (time.perf_counter() - t0) * 1000.0,
                {"gate_needed": self.cfg.gate_min_samples},
            )
        if not train_pool:
            st.pending_drift.clear()
            return self._record(
                now, scope, signal, horizon_s, trig, "skipped",
                GateResult(len(gate), None, None, False, "no_training_examples"),
                None, None, 0, (time.perf_counter() - t0) * 1000.0, {},
            )

        # With the gate disabled (the online_no_gate ablation) the held-out set is
        # legitimately empty, so there is no boundary to exclude training rows
        # against. min() over an empty gate raised here.
        gate_boundary = min((e.ts_target for e in gate), default=None)
        batch, batch_seq = buf.draw_batch(
            n=max(self.cfg.gate_window, self.cfg.every_n_samples),
            reservoir_fraction=self.cfg.reservoir_fraction,
            exclude_after_ts=gate_boundary,
        )
        if not batch:
            st.pending_drift.clear()
            return self._record(
                now, scope, signal, horizon_s, trig, "skipped",
                GateResult(len(gate), None, None, False, "empty_batch"),
                None, None, 0, (time.perf_counter() - t0) * 1000.0, {},
            )

        active_info = self.registry.active(scope)
        active_version = active_info.version if active_info else None
        active_units = self.forecaster.clone_unit(signal, horizon_s)

        # Candidate = deep copy of the live models, then incremental updates.
        # Copying rather than refitting from scratch is the point: the update
        # cost is O(batch x d^2), independent of total history size.
        candidate_units = {k: v.copy() for k, v in active_units.items()}
        applied = self._apply_updates(candidate_units, batch, signal, horizon_s)

        gate_res = self._gate(signal, horizon_s, active_units, candidate_units, gate)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        if gate_res.promote:
            # Only a promoted candidate's examples count as absorbed: a rejected
            # candidate is discarded, so its batch must remain available. The
            # pre-promotion mark is retained so a rollback can un-absorb it.
            absorbed_before = buf.absorbed_seq
            buf.mark_absorbed(batch_seq)
            buf.stats.absorbed = batch_seq
            # Close the prediction lifecycle: mark exactly which predictions fed
            # this promoted update, so the stored record answers "was this
            # prediction used for an adaptation update?".
            used = [e.prediction_id for e in batch if e.prediction_id]
            if used:
                self.repos.predictions.mark_used_for_adapt(used)
            version = self.registry.next_version(scope)
            self.forecaster.install_unit(signal, horizon_s, candidate_units)
            n = int(version.rsplit("v", 1)[-1])
            self.forecaster.set_version(signal, horizon_s, n)
            self.registry.register(
                scope=scope, version=version, ts=now, models=candidate_units,
                parent=active_version, n_train=applied,
                metrics={
                    "gate": gate_res.as_dict(), "trigger": trig.name,
                    "mae_before": gate_res.active_mae, "mae_after": gate_res.candidate_mae,
                },
                config_hash=self.config_fingerprint,
                train_start=min(e.ts_target for e in batch),
                train_end=max(e.ts_target for e in batch),
                run_id=self.run_id, activate=True,
                meta={"trigger": trig.name, "batch": len(batch)},
            )
            self._watch[scope] = _Watch(
                version=version, predecessor=active_version,
                predecessor_mae=gate_res.active_mae, absorbed_before=absorbed_before,
            )
            st.promotions += 1
            self.n_promoted += 1
            decision = "promoted"
            new_version = version
        else:
            st.rejections += 1
            self.n_rejected += 1
            decision = "rejected"
            new_version = None

        st.reset_after_adapt(now)
        return self._record(
            now, scope, signal, horizon_s, trig, decision, gate_res, active_version,
            new_version, applied, duration_ms,
            {"batch": len(batch), "gate_boundary": gate_boundary},
        )

    def _apply_updates(
        self, units: dict[str, RLSRegressor], batch: Sequence[TrainingExample],
        signal: str, horizon_s: float,
    ) -> int:
        sm = self.forecaster.model(signal, horizon_s)
        if sm is None or not sm.standardizer_ready:
            return 0
        applied = 0
        for ex in batch:
            if ex.extras is None:
                continue
            v = sm.prepare(sm.raw_input(ex.x, ex.extras))
            if v is None:
                continue
            key = ex.regime if self.forecaster.per_regime else "*"
            m = units.get(key)
            if m is None:
                # First time this regime is seen: start it from the shared model
                # if one exists, otherwise fresh. Starting fresh for a rare
                # regime would throw away everything general we know.
                base = units.get("*")
                m = base.copy() if base is not None else RLSRegressor(
                    d=sm.spec.d, ridge_lambda=self.forecaster.ridge_lambda,
                    forgetting=self.forecaster.forgetting,
                )
                units[key] = m
            y = sm.target(ex.actual, ex.anchor, ex.vol)
            # Feed the scope's trust-region pool. The controller trains *cloned*
            # regressors, so it must do this explicitly -- routing updates
            # straight to RLSRegressor.update bypasses ScopeModel.update and
            # left the pool permanently empty, which silently disabled the trust
            # region for the entire online path while leaving it active in
            # offline fits. That single omission was worth ~40% MAE on the
            # regime-change scenario.
            sm.note_uncertainty(m.uncertainty(v))
            if m.update(v, float(y), weight=ex.weight):
                applied += 1
        return applied

    def _score(
        self, units: dict[str, RLSRegressor], gate: Sequence[TrainingExample],
        signal: str, horizon_s: float,
    ) -> tuple[float | None, float | None, int]:
        errs: list[float] = []
        for ex in gate:
            if ex.extras is None:
                continue
            pred = self.forecaster.predict_from_parts(
                signal, horizon_s, ex.x, ex.extras, ex.anchor, ex.vol, ex.regime,
                models=units,
            )
            if pred is None:
                continue
            if not math.isfinite(pred):
                # A non-finite prediction is a hard failure, scored as such
                # rather than dropped -- otherwise a broken candidate looks
                # better than a working one by having fewer scored rows.
                errs.append(float("inf"))
                continue
            errs.append(abs(pred - ex.actual))
        if not errs:
            return None, None, 0
        arr = np.asarray(errs)
        if not np.all(np.isfinite(arr)):
            return float("inf"), float("inf"), int(arr.size)
        return float(arr.mean()), float(np.sqrt((arr**2).mean())), int(arr.size)

    def _gate(
        self, signal: str, horizon_s: float, active: dict[str, RLSRegressor],
        candidate: dict[str, RLSRegressor], gate: Sequence[TrainingExample],
    ) -> GateResult:
        a_mae, a_rmse, n_a = self._score(active, gate, signal, horizon_s)
        c_mae, c_rmse, n_c = self._score(candidate, gate, signal, horizon_s)
        n = min(n_a, n_c) if (n_a and n_c) else max(n_a, n_c)
        b_name, b_mae = self._baseline_floor(signal, horizon_s, gate)

        def result(promote: bool, reason: str) -> GateResult:
            return GateResult(n, a_mae, c_mae, promote, reason, a_rmse, c_rmse,
                              baseline_mae=b_mae, baseline_name=b_name)

        if not self.cfg.gate_enabled:
            # Ablation arm: promote unconditionally but still measure and record
            # what the gate would have seen, so the two arms are comparable.
            return result(True, "gate_disabled")
        if c_mae is None:
            return result(False, "candidate_unscorable")
        if a_mae is None:
            # No active model yet: the first fitted candidate is always an
            # improvement over having no forecast at all.
            return result(True, "no_active_model")
        if not math.isfinite(c_mae):
            return result(False, "candidate_non_finite")
        if n < self.cfg.gate_min_samples:
            return result(False, "insufficient_scored_rows")
        if c_mae > a_mae * self.cfg.gate_reject_ratio:
            return result(False, "hard_reject_worse")
        if c_mae <= a_mae * (1.0 + self.cfg.gate_tolerance):
            return result(True, "within_tolerance")
        return result(False, "worse_than_tolerance")

    def _baseline_floor(
        self, signal: str, horizon_s: float, gate: Sequence[TrainingExample]
    ) -> tuple[str | None, float | None]:
        """Best baseline MAE on the gate rows, reconstructed from stored inputs.

        Each example's ``extras`` holds every baseline's forecast as a normalised
        offset from the anchor, so the baselines' predictions on exactly these
        rows are recoverable without recomputing them:
        ``pred = anchor + extra * vol``.
        """
        if not self.cfg.gate_report_baselines:
            return None, None
        sm = self.forecaster.model(signal, horizon_s)
        if sm is None or not sm.spec.baseline_names:
            return None, None
        names = sm.spec.baseline_names
        sums = [0.0] * len(names)
        counts = [0] * len(names)
        for ex in gate:
            if ex.extras is None or len(ex.extras) != len(names):
                continue
            for i in range(len(names)):
                pred = ex.anchor + float(ex.extras[i]) * ex.vol
                if sm.spec.lo is not None:
                    pred = max(pred, sm.spec.lo)
                if sm.spec.hi is not None:
                    pred = min(pred, sm.spec.hi)
                sums[i] += abs(pred - ex.actual)
                counts[i] += 1
        best_i, best = -1, None
        for i in range(len(names)):
            if counts[i] == 0:
                continue
            mae = sums[i] / counts[i]
            if best is None or mae < best:
                best_i, best = i, mae
        return (names[best_i] if best_i >= 0 else None), best

    def _record(
        self, now: float, scope: str, signal: str, horizon_s: float, trig: Trigger,
        decision: str, gate: GateResult, active_version: str | None,
        candidate_version: str | None, n_train: int, duration_ms: float,
        detail: dict[str, object],
    ) -> AdaptationOutcome:
        out = AdaptationOutcome(
            scope=scope, signal=signal, horizon_s=horizon_s, trigger=trig.name,
            decision=decision, gate=gate, active_version=active_version,
            candidate_version=candidate_version, n_train=n_train, duration_ms=duration_ms,
            detail={**detail, **trig.detail, "gate": gate.as_dict()},
        )
        self.outcomes.append(out)
        if len(self.outcomes) > 2000:
            del self.outcomes[:1000]
        self.repos.events.add_adapt(
            # Simulated time under replay, wall clock when live. Using
            # time.time() here would put replay events on today's date while
            # their telemetry sits in the past, breaking every joined view.
            ts=now,
            scope=scope, kind="forecast", trigger=trig.name, decision=decision,
            active_version=active_version, candidate_version=candidate_version,
            metric_name="mae", metric_before=gate.active_mae, metric_after=gate.candidate_mae,
            gate_n=gate.n, n_train=n_train, duration_ms=duration_ms,
            detail=out.detail, run_id=self.run_id,
        )
        log.info(
            "adaptation decision", scope=scope, trigger=trig.name, decision=decision,
            mae_before=None if gate.active_mae is None else round(gate.active_mae, 3),
            mae_after=None if gate.candidate_mae is None else round(gate.candidate_mae, 3),
            gate_n=gate.n, reason=gate.reason, n_train=n_train, ms=round(duration_ms, 1),
        )
        return out

    # -- regression watch --------------------------------------------------
    def check_regressions(self, now: float, min_samples: int | None = None,
                          worse_ratio: float = 1.25) -> list[str]:
        """Roll back promoted versions that turned out worse in production.

        The gate uses tens of samples; this uses a full window of live outcomes.
        A version that passed the gate but is materially worse over that window
        is rolled back to its predecessor, which is the safeguard the spec asks
        for: continual learning is allowed to be wrong, not allowed to stay wrong.
        """
        need = min_samples if min_samples is not None else self.cfg.gate_window
        rolled: list[str] = []
        for scope, w in list(self._watch.items()):
            if len(w.errors) < need:
                continue
            live_mae = float(np.mean(w.errors[:need]))
            base = w.predecessor_mae
            del self._watch[scope]
            if base is None or w.predecessor is None or not math.isfinite(base) or base <= 0:
                continue
            if live_mae > base * worse_ratio:
                prev = self.registry.rollback(scope, now)
                if prev is None:
                    continue
                signal, horizon_s = self._split_scope(scope)
                try:
                    units = self.registry.load_unit(scope, prev.version)
                except (OSError, ValueError, KeyError) as exc:
                    log.error(
                        "rollback failed to load previous version", scope=scope,
                        version=prev.version, error=str(exc),
                    )
                    self.repos.events.add_ops(
                        now, "rollback_failed", "error",
                        {"scope": scope, "version": prev.version, "error": str(exc)},
                        self.run_id,
                    )
                    continue
                self.forecaster.install_unit(signal, horizon_s, units)
                self.forecaster.set_version(
                    signal, horizon_s, int(prev.version.rsplit("v", 1)[-1])
                )
                self.buffer(scope).rewind_absorbed(w.absorbed_before)
                st = self.state(scope)
                st.rollbacks += 1
                self.n_rolled_back += 1
                rolled.append(scope)
                self.repos.events.add_adapt(
                    ts=now, scope=scope, kind="forecast", trigger="regression_watch",
                    decision="rolled_back", active_version=w.version,
                    candidate_version=prev.version, metric_name="mae",
                    metric_before=base, metric_after=live_mae, gate_n=need, n_train=0,
                    duration_ms=0.0,
                    detail={
                        "reason": "live_mae_worse_than_predecessor",
                        "worse_ratio": worse_ratio, "live_mae": round(live_mae, 4),
                        "predecessor_mae": round(base, 4),
                    },
                    run_id=self.run_id,
                )
                log.warning(
                    "rolled back regressed version", scope=scope, version=w.version,
                    to=prev.version, live_mae=round(live_mae, 3),
                    predecessor_mae=round(base, 3),
                )
        return rolled

    # -- bootstrap ---------------------------------------------------------
    def register_initial(self, signal: str, horizon_s: float, now: float,
                         n_train: int, metrics: dict[str, object] | None = None) -> str:
        """Record v001 for a scope so later versions have a rollback target."""
        scope = self.scope_of(signal, horizon_s)
        version = self.registry.next_version(scope)
        units = self.forecaster.clone_unit(signal, horizon_s)
        self.registry.register(
            scope=scope, version=version, ts=now, models=units, parent=None,
            n_train=n_train, metrics=metrics or {}, config_hash=self.config_fingerprint,
            run_id=self.run_id, activate=True, meta={"trigger": "bootstrap"},
        )
        self.forecaster.set_version(signal, horizon_s, int(version.rsplit("v", 1)[-1]))
        return version

    def report(self) -> dict[str, object]:
        return {
            "policy": self.policy.describe(),
            "promoted": self.n_promoted,
            "rejected": self.n_rejected,
            "rolled_back": self.n_rolled_back,
            "scopes": {k: v.as_dict() for k, v in sorted(self._states.items())},
            "buffers": {k: v.stats.as_dict() for k, v in sorted(self._buffers.items())},
        }
