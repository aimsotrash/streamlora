"""Reproducible experiment runner.

An experiment is: one dataset, several *arms*, one metrics table. An arm is a set
of dotted config overrides applied to a base config, so the only thing that
differs between arms is stated explicitly in the arm definition -- never in
uncommitted code.

Three properties make the results trustworthy:

* **Identical data.** Every arm replays the same in-memory sample list, in the
  same order, through a fresh pipeline. Baseline forecasts come out
  bit-identical across arms, which the runner asserts.
* **Held-out chronological tail.** Metrics are computed only over targets after
  a training boundary. Without this, the learned arm is scored partly on its own
  warm-up while baselines -- which need none -- are not, and the comparison is
  biased against learning. This was a real bug in an earlier version of this
  harness, and it made the learned model look ~30% worse than it is.
* **Config stored with results.** The full config and its hash go into ``runs``,
  and every metric row carries the experiment and arm name, so a number in a
  document can be traced back to the configuration that produced it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from ..config import Config
from ..evaluate import report as R
from ..replay.player import ReplayPlayer, ReplayResult
from ..replay.recorder import read_dataset
from ..replay.synthetic import build_scenario, generate, synthetic_registry
from ..store.repo import Repos
from ..telemetry.schema import SignalRegistry, TelemetrySample
from ..util.ids import config_hash, short_uid
from ..util.logging import get_logger

log = get_logger("experiments")


@dataclass(slots=True)
class Arm:
    name: str
    description: str = ""
    #: Dotted config overrides, e.g. {"adapt.policies": "drift"}.
    overrides: dict[str, str] = field(default_factory=dict)
    #: Freeze learning at the training boundary (the "static model" arm).
    freeze_after_train: bool = False

    def apply(self, base: Config) -> Config:
        cfg = Config.from_dict(base.to_dict())
        for k, v in self.overrides.items():
            cfg.override(k, v)
        return cfg

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "description": self.description,
            "overrides": dict(self.overrides),
            "freeze_after_train": self.freeze_after_train,
        }


@dataclass(slots=True)
class DatasetRef:
    """Where an experiment's telemetry comes from."""

    kind: str          # "scenario" | "file" | "db"
    name: str
    minutes: float | None = None
    seed: int | None = None
    run_id: str | None = None

    @classmethod
    def parse(cls, spec: str, minutes: float | None = None, seed: int | None = None) -> "DatasetRef":
        """``scenario:idle_to_build`` / ``file:data/x.jsonl`` / ``db:live-2026...``"""
        if ":" not in spec:
            raise ValueError(
                f"dataset must be 'scenario:<name>', 'file:<path>' or 'db:<run_id>', got {spec!r}"
            )
        kind, _, name = spec.partition(":")
        if kind not in ("scenario", "file", "db"):
            raise ValueError(f"unknown dataset kind {kind!r}")
        return cls(kind=kind, name=name, minutes=minutes, seed=seed,
                   run_id=name if kind == "db" else None)

    def label(self) -> str:
        base = f"{self.kind}:{self.name}"
        if self.kind == "scenario" and self.minutes:
            base += f"@{int(self.minutes)}min"
        return base

    def load(self, repos: Repos | None) -> tuple[list[TelemetrySample], SignalRegistry]:
        if self.kind == "scenario":
            spec = build_scenario(self.name, minutes=self.minutes, seed=self.seed)
            samples, reg = generate(spec)
            return samples, reg
        if self.kind == "file":
            it, reg, _hdr = read_dataset(self.name)
            return list(it), reg
        if repos is None:
            raise ValueError("db dataset requires an open database")
        samples = list(repos.telemetry.iter_samples(run_id=self.run_id or None))
        return samples, repos.signals.registry()


@dataclass(slots=True)
class ArmResult:
    arm: Arm
    run_id: str
    replay: ReplayResult
    rows: list[R.Row]
    rows_by_regime: list[R.Row] = field(default_factory=list)
    rows_over_time: list[R.Row] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm.as_dict(),
            "run_id": self.run_id,
            "replay": self.replay.as_dict(),
            "adapt": (self.replay.engine_report or {}).get("adapt"),
            "drift": (self.replay.engine_report or {}).get("drift"),
            "metrics": [r.as_dict() for r in self.rows],
            "metrics_by_regime": [r.as_dict() for r in self.rows_by_regime],
            "metrics_over_time": [r.as_dict() for r in self.rows_over_time],
        }


@dataclass(slots=True)
class ExperimentResult:
    name: str
    experiment_id: str
    dataset: str
    train_frac: float
    boundary_ts: float
    arms: list[ArmResult]
    started_ts: float
    ended_ts: float
    base_config: dict[str, object]
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name, "experiment_id": self.experiment_id, "dataset": self.dataset,
            "train_frac": self.train_frac, "boundary_ts": self.boundary_ts,
            "started_ts": self.started_ts, "ended_ts": self.ended_ts,
            "duration_s": round(self.ended_ts - self.started_ts, 1),
            "notes": self.notes, "base_config": self.base_config,
            "arms": [a.as_dict() for a in self.arms],
        }

    def comparison_table(self) -> str:
        """MAE per scope, one column per arm, learned arms only plus baselines."""
        merged: list[R.Row] = []
        seen_baselines = False
        for ar in self.arms:
            for r in ar.rows:
                if r.arm == "rls":
                    merged.append(
                        R.Row(arm=ar.arm.name, signal=r.signal, horizon_s=r.horizon_s,
                              regime=r.regime, window=r.window, m=r.m,
                              model_versions=r.model_versions, dm_stat=r.dm_stat, dm_p=r.dm_p)
                    )
                elif not seen_baselines:
                    merged.append(r)
            seen_baselines = True
        arms = [r for r in dict.fromkeys(x.arm for x in merged)]
        return R.to_table(merged, arms=arms)


class ExperimentRunner:
    def __init__(self, base_config: Config, repos: Repos | None = None) -> None:
        self.base = base_config
        self.base.ensure_dirs()
        self.repos = repos or Repos.open(base_config.db_file)
        self._owns_repos = repos is None

    def close(self) -> None:
        if self._owns_repos:
            self.repos.close()

    def run(
        self,
        name: str,
        dataset: DatasetRef,
        arms: Sequence[Arm],
        train_frac: float = 0.5,
        notes: str = "",
        progress: Callable[[str], None] | None = None,
    ) -> ExperimentResult:
        started = time.time()
        exp_id = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}-{short_uid()}"
        samples, registry = dataset.load(self.repos)
        if len(samples) < 200:
            raise ValueError(
                f"dataset {dataset.label()} has only {len(samples)} samples; "
                "need at least 200 for a chronological split"
            )
        self.repos.signals.register(registry.as_mapping().values())
        ts_all = [s.ts for s in samples]
        boundary = float(ts_all[int(len(ts_all) * train_frac)])
        log.info(
            "experiment starting", experiment=name, dataset=dataset.label(),
            samples=len(samples), arms=[a.name for a in arms],
            train_frac=train_frac, boundary_ts=boundary,
        )

        results: list[ArmResult] = []
        baseline_signature: dict[str, float] | None = None
        for arm in arms:
            cfg = arm.apply(self.base)
            run_id = f"{exp_id}:{arm.name}"
            if progress:
                progress(f"arm {arm.name}")
            player = ReplayPlayer(
                cfg, self.repos, registry, run_id=run_id, store_telemetry=False,
                record_predictions=True, renormalize=True,
                enable_drift=cfg.drift.enabled, enable_adapt=True,
                freeze_after_ts=boundary if arm.freeze_after_train else None,
            )
            self.repos.runs.start(
                run_id, "experiment", started, cfg.to_dict(), config_hash(cfg.to_dict()),
                notes=f"experiment={name} arm={arm.name} dataset={dataset.label()}",
            )
            replay = player.run(samples)

            # Held-out tail only. This is the whole point of train_frac.
            recs = [
                r for r in self.repos.predictions.resolved(run_id=run_id)
                if r.ts_target >= boundary
            ]
            rows = R.summarize(recs, reference=R.DEFAULT_REFERENCE)
            rows_reg = R.summarize(recs, reference=R.DEFAULT_REFERENCE, by_regime=True, dm_test=False)
            rows_time = R.summarize_over_time(recs, n_windows=4)
            self._store_metrics(exp_id, name, arm.name, run_id, rows, "experiment")
            self._store_metrics(exp_id, name, arm.name, run_id, rows_reg, "experiment_regime")

            # Sanity check: baselines are deterministic, so their metrics must be
            # identical across arms. If they are not, the arms did not see the
            # same data and no comparison between them is valid.
            sig = {
                f"{r.signal}@{int(r.horizon_s)}": round(r.m.mae or -1.0, 9)
                for r in rows if r.arm == "persistence"
            }
            if baseline_signature is None:
                baseline_signature = sig
            elif sig != baseline_signature:
                diffs = {k: (baseline_signature.get(k), v) for k, v in sig.items()
                         if baseline_signature.get(k) != v}
                raise AssertionError(
                    f"arm {arm.name!r} saw different data: persistence MAE changed {diffs}"
                )
            self.repos.runs.finish(
                run_id, time.time(), "complete",
                summary={"replay": replay.as_dict(), "n_metric_rows": len(rows)},
            )
            results.append(
                ArmResult(arm=arm, run_id=run_id, replay=replay, rows=rows,
                          rows_by_regime=rows_reg, rows_over_time=rows_time)
            )
            log.info(
                "arm complete", experiment=name, arm=arm.name,
                learned_mae={
                    f"{r.signal}@{int(r.horizon_s)}": round(r.m.mae, 4)
                    for r in rows if r.arm == "rls" and r.m.mae is not None
                },
            )

        result = ExperimentResult(
            name=name, experiment_id=exp_id, dataset=dataset.label(), train_frac=train_frac,
            boundary_ts=boundary, arms=results, started_ts=started, ended_ts=time.time(),
            base_config=self.base.to_dict(), notes=notes,
        )
        self._write_report(result)
        return result

    def _store_metrics(
        self, exp_id: str, name: str, arm: str, run_id: str, rows: Sequence[R.Row], scope: str
    ) -> None:
        payload = []
        now = time.time()
        for r in rows:
            payload.append({
                "ts": now, "scope": scope, "experiment": exp_id, "arm": arm,
                "signal": r.signal, "horizon_s": r.horizon_s, "model_kind": r.arm,
                "model_version": (r.model_versions or [None])[0], "regime": r.regime,
                "window_start": None, "window_end": None, "n": r.m.n, "mae": r.m.mae,
                "rmse": r.m.rmse, "smape": r.m.smape, "bias": r.m.bias,
                "p90_abs_error": r.m.p90_abs_error, "coverage": r.m.coverage,
                "interval_width": r.m.mean_interval_width, "skill": r.m.skill,
                "detail": {
                    "experiment_name": name, "dm_stat": r.dm_stat, "dm_p": r.dm_p,
                    "window": r.window, "model_versions": r.model_versions[:8],
                },
                "run_id": run_id,
            })
        self.repos.events.add_metrics(payload)

    def _write_report(self, result: ExperimentResult) -> None:
        os.makedirs(self.base.runs_dir, exist_ok=True)
        path = os.path.join(self.base.runs_dir, f"{result.experiment_id}.json")
        with open(path, "w") as fh:
            json.dump(result.as_dict(), fh, indent=2, default=str)
        log.info("experiment report written", path=path)
        result_path = os.path.join(self.base.runs_dir, "latest.json")
        try:
            with open(result_path, "w") as fh:
                json.dump(result.as_dict(), fh, indent=2, default=str)
        except OSError:  # pragma: no cover
            pass
