"""Version registry for forecast models.

Requirements this satisfies: every adaptation produces an identifiable version;
the user can inspect current model, update time, training window and
before/after metrics; and a bad version can be rolled back.

Granularity is per *scope* -- one (signal, horizon) pair -- not per process.
That is a deliberate choice: adaptation quality differs sharply between
``battery.percent@1800`` and ``cpu.util_pct@300``, and a single global version
would force promoting or rejecting all nine models together on evidence that
only applies to one. Per-scope versioning means a bad CPU update never rolls
back a good battery model.

Storage is a compressed ``.npz`` per version plus a row in ``model_versions``.
No pickle: model files are loaded with ``allow_pickle=False``, so a corrupted or
tampered file fails as a parse error rather than executing code.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..forecast.linear import RLSRegressor
from ..store.repo import Repos
from ..util.ids import version_label
from ..util.logging import get_logger

log = get_logger("adapt.registry")

KIND = "forecast-model"


def scope_path_part(scope: str) -> str:
    """Filesystem-safe form of ``cpu.util_pct@300``."""
    return scope.replace("/", "_").replace("@", "_at_")


@dataclass(slots=True)
class VersionInfo:
    kind: str
    scope: str
    version: str
    path: str | None
    created_ts: float
    parent: str | None
    n_train: int
    metrics: dict[str, Any]
    active: bool
    train_start: float | None = None
    train_end: float | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "VersionInfo":
        try:
            metrics = json.loads(row.get("metrics") or "{}")
        except json.JSONDecodeError:
            metrics = {}
        return cls(
            kind=row["kind"], scope=row["scope"], version=row["version"], path=row.get("path"),
            created_ts=float(row["created_ts"]), parent=row.get("parent"),
            n_train=int(row.get("n_train") or 0), metrics=metrics,
            active=bool(row.get("active")), train_start=row.get("train_start"),
            train_end=row.get("train_end"),
        )


class ModelVersionRegistry:
    """Persists RLS units and tracks which version is live."""

    def __init__(self, repos: Repos, models_dir: str, keep: int = 10) -> None:
        self.repos = repos
        self.root = os.path.join(models_dir, "forecast")
        self.keep = int(keep)
        os.makedirs(self.root, exist_ok=True)

    # -- files -------------------------------------------------------------
    def _dir(self, scope: str) -> str:
        d = os.path.join(self.root, scope_path_part(scope))
        os.makedirs(d, exist_ok=True)
        return d

    def path_for(self, scope: str, version: str) -> str:
        return os.path.join(self._dir(scope), f"{version}.npz")

    def save_unit(
        self, scope: str, version: str, models: dict[str, RLSRegressor],
        meta: dict[str, Any] | None = None,
    ) -> str:
        path = self.path_for(scope, version)
        arrays: dict[str, np.ndarray] = {}
        index: list[dict[str, Any]] = []
        for i, (regime, m) in enumerate(sorted(models.items())):
            arrays[f"w::{i}"] = m.w
            arrays[f"P::{i}"] = m.P
            index.append({
                "regime": regime, "i": i, "d": m.d, "ridge_lambda": m.ridge_lambda,
                "forgetting": m.forgetting, "n_updates": m.n_updates,
                "n_skipped": m.n_skipped, "n_windup": m.n_windup, "sse": m.sse,
            })
        blob = json.dumps({"scope": scope, "version": version, "index": index,
                           "meta": meta or {}}).encode()
        buf = io.BytesIO()
        np.savez_compressed(buf, manifest=np.frombuffer(blob, dtype=np.uint8), **arrays)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(buf.getvalue())
        # Atomic replace: a crash mid-write must not leave a half-written model
        # that the registry believes is loadable.
        os.replace(tmp, path)
        return path

    def load_unit(self, scope: str, version: str) -> dict[str, RLSRegressor]:
        path = self.path_for(scope, version)
        with np.load(path, allow_pickle=False) as z:
            manifest = json.loads(bytes(z["manifest"]).decode())
            out: dict[str, RLSRegressor] = {}
            for entry in manifest["index"]:
                i = entry["i"]
                w = z[f"w::{i}"].copy()
                P = z[f"P::{i}"].copy()
                if w.size != entry["d"] or P.shape != (entry["d"], entry["d"]):
                    raise ValueError(
                        f"corrupt model {scope}/{version}: expected d={entry['d']}, "
                        f"got w={w.shape} P={P.shape}"
                    )
                m = RLSRegressor(
                    d=int(entry["d"]), ridge_lambda=float(entry["ridge_lambda"]),
                    forgetting=float(entry["forgetting"]), w=w, P=P,
                )
                m.n_updates = int(entry.get("n_updates", 0))
                m.n_skipped = int(entry.get("n_skipped", 0))
                m.n_windup = int(entry.get("n_windup", 0))
                m.sse = float(entry.get("sse", 0.0))
                out[entry["regime"]] = m
        return out

    # -- registry ----------------------------------------------------------
    def next_version(self, scope: str) -> str:
        return version_label(KIND, self.repos.models.next_version_number(KIND, scope))

    def register(
        self, scope: str, version: str, ts: float, models: dict[str, RLSRegressor],
        parent: str | None, n_train: int, metrics: dict[str, Any],
        config_hash: str = "", train_start: float | None = None,
        train_end: float | None = None, run_id: str | None = None,
        activate: bool = False, meta: dict[str, Any] | None = None,
    ) -> VersionInfo:
        path = self.save_unit(scope, version, models, meta)
        self.repos.models.register(
            KIND, scope, version, ts, parent=parent, path=path, config_hash=config_hash,
            train_start=train_start, train_end=train_end, n_train=n_train,
            metrics=metrics, active=activate, run_id=run_id,
        )
        if activate:
            self.repos.models.activate(KIND, scope, version, ts)
        pruned = self.repos.models.prune(KIND, scope, self.keep)
        for v in pruned:
            p = self.path_for(scope, v)
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as exc:  # pragma: no cover
                log.warning("could not remove pruned model", version=v, error=str(exc))
        return VersionInfo(
            kind=KIND, scope=scope, version=version, path=path, created_ts=ts, parent=parent,
            n_train=n_train, metrics=metrics, active=activate,
            train_start=train_start, train_end=train_end,
        )

    def activate(self, scope: str, version: str, ts: float) -> None:
        self.repos.models.activate(KIND, scope, version, ts)

    def active(self, scope: str) -> VersionInfo | None:
        row = self.repos.models.active(KIND, scope)
        return VersionInfo.from_row(row) if row else None

    def history(self, scope: str | None = None, limit: int = 200) -> list[VersionInfo]:
        return [
            VersionInfo.from_row(r)
            for r in self.repos.models.history(KIND, scope, limit=limit)
        ]

    def previous_good(self, scope: str, before_version: str) -> VersionInfo | None:
        """The version to roll back to.

        Prefers the *recorded parent* of ``before_version`` -- the exact state
        that was live before it -- falling back to "most recent other version"
        only if the parent's file is gone. Ordering alone is not enough: pruning
        can remove intermediate versions, and an earlier implementation that
        walked creation order rolled back 24 versions at once because everything
        in between had been pruned.

        Rollback restores a known-good state rather than the best-scoring one:
        the last version promoted through the gate is by definition the last one
        that passed it.
        """
        hist = self.history(scope, limit=500)
        by_version = {v.version: v for v in hist}
        cur = by_version.get(before_version)
        if cur is not None and cur.parent:
            parent = by_version.get(cur.parent)
            if parent is not None and os.path.exists(self.path_for(scope, parent.version)):
                return parent
        for v in hist:  # newest first
            if v.version == before_version:
                continue
            if v.created_ts <= (cur.created_ts if cur else float("inf")) and os.path.exists(
                self.path_for(scope, v.version)
            ):
                return v
        return None

    def rollback(self, scope: str, ts: float) -> VersionInfo | None:
        """Activate the previous version. Returns it, or None if none exists."""
        cur = self.active(scope)
        if cur is None:
            return None
        prev = self.previous_good(scope, cur.version)
        if prev is None:
            log.warning("rollback requested but no earlier version exists", scope=scope)
            return None
        self.activate(scope, prev.version, ts)
        log.info("rolled back", scope=scope, from_version=cur.version, to_version=prev.version)
        return prev
