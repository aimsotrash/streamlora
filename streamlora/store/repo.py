"""Typed repositories over the SQLite schema.

Every SQL statement in StreamLoRA lives here. Components above this layer deal
in dataclasses and plain dicts, which keeps the forecasting and adaptation code
testable against an in-memory database and keeps query changes in one place.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from ..telemetry.schema import Quality, Reading, SignalRegistry, SignalSpec, TelemetrySample
from .db import Database


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))


class SignalRepo:
    """Interned signal names plus their specs."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._id_cache: dict[str, int] = {}
        self._name_cache: dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        for row in self.db.conn.execute("SELECT id, name FROM signals"):
            self._id_cache[row["name"]] = row["id"]
            self._name_cache[row["id"]] = row["name"]

    def register(self, specs: Iterable[SignalSpec]) -> None:
        with self.db.write_lock, self.db.conn:
            for s in specs:
                self.db.conn.execute(
                    """INSERT INTO signals(name, unit, kind, lo, hi, description)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(name) DO UPDATE SET
                         unit=excluded.unit, kind=excluded.kind, lo=excluded.lo,
                         hi=excluded.hi, description=excluded.description""",
                    (s.name, s.unit, str(s.kind), s.lo, s.hi, s.description),
                )
        self._id_cache.clear()
        self._name_cache.clear()
        self._load()

    def id_for(self, name: str) -> int:
        sid = self._id_cache.get(name)
        if sid is not None:
            return sid
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                "INSERT OR IGNORE INTO signals(name) VALUES(?)", (name,)
            )
        row = self.db.conn.execute("SELECT id FROM signals WHERE name=?", (name,)).fetchone()
        sid = int(row["id"])
        self._id_cache[name] = sid
        self._name_cache[sid] = name
        return sid

    def name_for(self, sid: int) -> str:
        if sid not in self._name_cache:
            self._load()
        return self._name_cache.get(sid, f"signal#{sid}")

    def all_specs(self) -> list[SignalSpec]:
        out = []
        for row in self.db.conn.execute(
            "SELECT name, unit, kind, lo, hi, description FROM signals ORDER BY name"
        ):
            out.append(
                SignalSpec(
                    row["name"], row["unit"], row["kind"], row["lo"], row["hi"],
                    description=row["description"],
                )
            )
        return out

    def registry(self) -> SignalRegistry:
        return SignalRegistry(self.all_specs())


@dataclass(slots=True)
class SampleWindow:
    """A pivoted block of telemetry: aligned timestamps x signals.

    ``values`` is (n_samples, n_signals) float64 with NaN for unusable
    readings; ``quality`` is the parallel integer array. NaN is used rather
    than a masked array because every consumer either drops or imputes, and
    ``np.isnan`` is the cheapest way to express both.
    """

    ts: np.ndarray
    signals: list[str]
    values: np.ndarray
    quality: np.ndarray

    def column(self, signal: str) -> np.ndarray | None:
        try:
            i = self.signals.index(signal)
        except ValueError:
            return None
        return self.values[:, i]

    def __len__(self) -> int:
        return int(self.ts.shape[0])


class TelemetryRepo:
    def __init__(self, db: Database, signals: SignalRepo) -> None:
        self.db = db
        self.signals = signals

    def insert_samples(self, samples: Sequence[TelemetrySample], run_id: str | None = None) -> list[int]:
        """Insert a batch of samples in one transaction. Returns sample ids."""
        if not samples:
            return []
        ids: list[int] = []
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.cursor()
            for s in samples:
                cur.execute(
                    """INSERT INTO samples(ts, origin, collect_ms, gap_s, worst_quality, run_id)
                       VALUES(?,?,?,?,?,?)""",
                    (s.ts, s.origin, s.collect_ms, s.gap_s, int(s.worst_quality), run_id),
                )
                sid = int(cur.lastrowid or 0)
                ids.append(sid)
                rows = [
                    (sid, self.signals.id_for(name), r.value, int(r.quality))
                    for name, r in s.readings.items()
                ]
                if rows:
                    cur.executemany(
                        "INSERT OR REPLACE INTO readings(sample_id, signal_id, value, quality) VALUES(?,?,?,?)",
                        rows,
                    )
            cur.close()
        return ids

    def count(self, run_id: str | None = None) -> int:
        if run_id is None:
            row = self.db.conn.execute("SELECT COUNT(*) AS n FROM samples").fetchone()
        else:
            row = self.db.conn.execute(
                "SELECT COUNT(*) AS n FROM samples WHERE run_id=?", (run_id,)
            ).fetchone()
        return int(row["n"])

    def time_range(self, run_id: str | None = None) -> tuple[float, float] | None:
        q = "SELECT MIN(ts) AS a, MAX(ts) AS b FROM samples"
        args: tuple[Any, ...] = ()
        if run_id is not None:
            q += " WHERE run_id=?"
            args = (run_id,)
        row = self.db.conn.execute(q, args).fetchone()
        if row is None or row["a"] is None:
            return None
        return float(row["a"]), float(row["b"])

    def iter_samples(
        self,
        ts_from: float | None = None,
        ts_to: float | None = None,
        run_id: str | None = None,
        origin: str | None = None,
        limit: int | None = None,
    ) -> Iterator[TelemetrySample]:
        """Stream samples in chronological order, reconstructing readings.

        Uses one query joined on readings and groups in Python, which is far
        cheaper than N+1 queries and keeps memory bounded to one sample.
        """
        where = []
        args: list[Any] = []
        if ts_from is not None:
            where.append("s.ts >= ?")
            args.append(ts_from)
        if ts_to is not None:
            where.append("s.ts <= ?")
            args.append(ts_to)
        if run_id is not None:
            where.append("s.run_id = ?")
            args.append(run_id)
        if origin is not None:
            where.append("s.origin = ?")
            args.append(origin)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        sub = f"SELECT id, ts, origin, collect_ms, gap_s FROM samples s {clause} ORDER BY s.ts, s.id"
        if limit is not None:
            sub += f" LIMIT {int(limit)}"
        q = f"""
            SELECT p.id, p.ts, p.origin, p.collect_ms, p.gap_s,
                   r.signal_id, r.value, r.quality
            FROM ({sub}) p
            LEFT JOIN readings r ON r.sample_id = p.id
            ORDER BY p.ts, p.id
        """
        cur = self.db.conn.execute(q, args)
        current: TelemetrySample | None = None
        current_id: int | None = None
        for row in cur:
            if row["id"] != current_id:
                if current is not None:
                    yield current
                current_id = row["id"]
                current = TelemetrySample(
                    ts=float(row["ts"]),
                    readings={},
                    collect_ms=float(row["collect_ms"] or 0.0),
                    origin=row["origin"],
                    gap_s=row["gap_s"],
                )
            if row["signal_id"] is not None and current is not None:
                current.readings[self.signals.name_for(int(row["signal_id"]))] = Reading(
                    row["value"], Quality(int(row["quality"]))
                )
        if current is not None:
            yield current
        cur.close()

    def window(
        self,
        signals: Sequence[str],
        ts_from: float | None = None,
        ts_to: float | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> SampleWindow:
        """Pivot a time range into aligned arrays for the requested signals."""
        names = list(signals)
        index = {n: i for i, n in enumerate(names)}
        ts_list: list[float] = []
        val_rows: list[np.ndarray] = []
        qual_rows: list[np.ndarray] = []
        for s in self.iter_samples(ts_from, ts_to, run_id, limit=limit):
            v = np.full(len(names), np.nan)
            q = np.full(len(names), int(Quality.MISSING), dtype=np.int16)
            for name, r in s.readings.items():
                i = index.get(name)
                if i is None:
                    continue
                q[i] = int(r.quality)
                if r.usable and r.value is not None:
                    v[i] = float(r.value)
            ts_list.append(s.ts)
            val_rows.append(v)
            qual_rows.append(q)
        if not ts_list:
            return SampleWindow(
                np.empty(0), names, np.empty((0, len(names))), np.empty((0, len(names)), dtype=np.int16)
            )
        return SampleWindow(
            np.asarray(ts_list), names, np.vstack(val_rows), np.vstack(qual_rows)
        )

    def latest(self, n: int = 1, run_id: str | None = None) -> list[TelemetrySample]:
        q = "SELECT id FROM samples"
        args: tuple[Any, ...] = ()
        if run_id is not None:
            q += " WHERE run_id=?"
            args = (run_id,)
        q += " ORDER BY ts DESC, id DESC LIMIT ?"
        rows = self.db.conn.execute(q, (*args, int(n))).fetchall()
        if not rows:
            return []
        ids = [int(r["id"]) for r in rows]
        marks = ",".join("?" * len(ids))
        out: dict[int, TelemetrySample] = {}
        cur = self.db.conn.execute(
            f"""SELECT s.id, s.ts, s.origin, s.collect_ms, s.gap_s, r.signal_id, r.value, r.quality
                FROM samples s LEFT JOIN readings r ON r.sample_id=s.id
                WHERE s.id IN ({marks}) ORDER BY s.ts""",
            ids,
        )
        for row in cur:
            sid = int(row["id"])
            samp = out.get(sid)
            if samp is None:
                samp = TelemetrySample(
                    ts=float(row["ts"]), readings={}, collect_ms=float(row["collect_ms"] or 0.0),
                    origin=row["origin"], gap_s=row["gap_s"],
                )
                out[sid] = samp
            if row["signal_id"] is not None:
                samp.readings[self.signals.name_for(int(row["signal_id"]))] = Reading(
                    row["value"], Quality(int(row["quality"]))
                )
        return [out[i] for i in sorted(out, key=lambda k: out[k].ts)]


class FeatureRepo:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._known_schemas: set[str] = set()

    def register_schema(self, schema_hash: str, names: Sequence[str], ts: float) -> None:
        if schema_hash in self._known_schemas:
            return
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                "INSERT OR IGNORE INTO feature_schemas(hash, names_json, created_ts) VALUES(?,?,?)",
                (schema_hash, _json(list(names)), ts),
            )
        self._known_schemas.add(schema_hash)

    def schema_names(self, schema_hash: str) -> list[str]:
        row = self.db.conn.execute(
            "SELECT names_json FROM feature_schemas WHERE hash=?", (schema_hash,)
        ).fetchone()
        return json.loads(row["names_json"]) if row else []

    def insert(
        self, ts: float, schema_hash: str, vec: np.ndarray, regime: str,
        coverage: float, run_id: str | None,
    ) -> int:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO feature_windows(ts, schema_hash, regime, coverage, vec, run_id)
                   VALUES(?,?,?,?,?,?)""",
                (ts, schema_hash, regime, coverage, blob, run_id),
            )
            return int(cur.lastrowid or 0)

    def get(self, feature_id: int) -> tuple[np.ndarray, str, str] | None:
        row = self.db.conn.execute(
            "SELECT vec, schema_hash, regime FROM feature_windows WHERE id=?", (feature_id,)
        ).fetchone()
        if row is None:
            return None
        return (
            np.frombuffer(row["vec"], dtype=np.float32).astype(np.float64),
            row["schema_hash"],
            row["regime"],
        )

    def get_many(self, ids: Sequence[int]) -> dict[int, np.ndarray]:
        if not ids:
            return {}
        out: dict[int, np.ndarray] = {}
        CHUNK = 500
        for i in range(0, len(ids), CHUNK):
            chunk = list(ids[i : i + CHUNK])
            marks = ",".join("?" * len(chunk))
            for row in self.db.conn.execute(
                f"SELECT id, vec FROM feature_windows WHERE id IN ({marks})", chunk
            ):
                out[int(row["id"])] = np.frombuffer(row["vec"], dtype=np.float32).astype(np.float64)
        return out


@dataclass(slots=True)
class PredictionRecord:
    id: int
    ts_made: float
    ts_target: float
    horizon_s: float
    signal: str
    value: float
    lo: float | None
    hi: float | None
    model_kind: str
    model_version: str
    regime: str
    anchor: float | None
    feature_id: int | None
    resolved: bool = False
    actual: float | None = None
    error: float | None = None
    abs_error: float | None = None
    in_interval: int | None = None
    used_for_adapt: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "PredictionRecord":
        return cls(
            id=int(row["id"]), ts_made=float(row["ts_made"]), ts_target=float(row["ts_target"]),
            horizon_s=float(row["horizon_s"]), signal=row["signal"], value=float(row["value"]),
            lo=row["lo"], hi=row["hi"], model_kind=row["model_kind"],
            model_version=row["model_version"], regime=row["regime"], anchor=row["anchor"],
            feature_id=row["feature_id"], resolved=bool(row["resolved"]), actual=row["actual"],
            error=row["error"], abs_error=row["abs_error"], in_interval=row["in_interval"],
            used_for_adapt=bool(row["used_for_adapt"]),
        )


class PredictionRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_many(self, rows: Sequence[dict[str, Any]]) -> list[int]:
        if not rows:
            return []
        cols = (
            "ts_made", "ts_target", "horizon_s", "signal", "value", "lo", "hi", "alpha",
            "model_kind", "model_version", "regime", "anchor", "feature_id", "infer_ms", "run_id",
        )
        marks = ",".join("?" * len(cols))
        ids: list[int] = []
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.cursor()
            for r in rows:
                cur.execute(
                    f"INSERT INTO predictions({','.join(cols)}) VALUES({marks})",
                    tuple(r.get(c) for c in cols),
                )
                ids.append(int(cur.lastrowid or 0))
            cur.close()
        return ids

    def due(self, now: float, signal: str | None = None, run_id: str | None = None,
            limit: int = 5000) -> list[PredictionRecord]:
        """Unresolved predictions whose target time has arrived."""
        q = "SELECT * FROM predictions WHERE resolved=0 AND ts_target <= ?"
        args: list[Any] = [now]
        if signal is not None:
            q += " AND signal=?"
            args.append(signal)
        if run_id is not None:
            q += " AND run_id=?"
            args.append(run_id)
        q += " ORDER BY ts_target LIMIT ?"
        args.append(limit)
        return [PredictionRecord.from_row(r) for r in self.db.conn.execute(q, args)]

    def resolve_many(self, updates: Sequence[dict[str, Any]]) -> None:
        if not updates:
            return
        with self.db.write_lock, self.db.conn:
            self.db.conn.executemany(
                """UPDATE predictions SET resolved=1, ts_resolved=:ts_resolved, actual=:actual,
                       error=:error, abs_error=:abs_error, in_interval=:in_interval,
                       resolve_quality=:resolve_quality
                   WHERE id=:id""",
                updates,
            )

    def mark_used_for_adapt(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        with self.db.write_lock, self.db.conn:
            self.db.conn.executemany(
                "UPDATE predictions SET used_for_adapt=1 WHERE id=?", [(int(i),) for i in ids]
            )

    def resolved(
        self, signal: str | None = None, horizon_s: float | None = None,
        model_kind: str | None = None, model_version: str | None = None,
        ts_from: float | None = None, ts_to: float | None = None,
        run_id: str | None = None, regime: str | None = None,
        limit: int | None = None, newest_first: bool = False,
    ) -> list[PredictionRecord]:
        where = ["resolved=1"]
        args: list[Any] = []
        for col, val in (
            ("signal", signal), ("horizon_s", horizon_s), ("model_kind", model_kind),
            ("model_version", model_version), ("run_id", run_id), ("regime", regime),
        ):
            if val is not None:
                where.append(f"{col}=?")
                args.append(val)
        if ts_from is not None:
            where.append("ts_target >= ?")
            args.append(ts_from)
        if ts_to is not None:
            where.append("ts_target <= ?")
            args.append(ts_to)
        q = f"SELECT * FROM predictions WHERE {' AND '.join(where)} ORDER BY ts_target"
        if newest_first:
            q += " DESC"
        if limit is not None:
            q += f" LIMIT {int(limit)}"
        return [PredictionRecord.from_row(r) for r in self.db.conn.execute(q, args)]

    def latest_unresolved(self, run_id: str | None = None, limit: int = 200) -> list[PredictionRecord]:
        q = "SELECT * FROM predictions WHERE resolved=0"
        args: list[Any] = []
        if run_id is not None:
            q += " AND run_id=?"
            args.append(run_id)
        q += " ORDER BY ts_made DESC, horizon_s LIMIT ?"
        args.append(limit)
        return [PredictionRecord.from_row(r) for r in self.db.conn.execute(q, args)]

    def get(self, pid: int) -> PredictionRecord | None:
        row = self.db.conn.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
        return PredictionRecord.from_row(row) if row else None


class EventRepo:
    """Drift, adaptation and operational events plus the metrics table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def add_drift(self, **kw: Any) -> int:
        kw.setdefault("detail", {})
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO drift_events(ts, detector, scope, signal, horizon_s,
                       statistic, threshold, severity, detail, run_id)
                   VALUES(:ts,:detector,:scope,:signal,:horizon_s,:statistic,:threshold,
                          :severity,:detail,:run_id)""",
                {**kw, "detail": _json(kw["detail"])},
            )
            return int(cur.lastrowid or 0)

    def drift_events(self, ts_from: float | None = None, run_id: str | None = None,
                     limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM drift_events WHERE 1=1"
        args: list[Any] = []
        if ts_from is not None:
            q += " AND ts >= ?"
            args.append(ts_from)
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def add_adapt(self, **kw: Any) -> int:
        kw.setdefault("detail", {})
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO adapt_events(ts, scope, kind, trigger, decision, active_version,
                       candidate_version, metric_name, metric_before, metric_after, gate_n,
                       n_train, duration_ms, detail, run_id)
                   VALUES(:ts,:scope,:kind,:trigger,:decision,:active_version,:candidate_version,
                          :metric_name,:metric_before,:metric_after,:gate_n,:n_train,
                          :duration_ms,:detail,:run_id)""",
                {**kw, "detail": _json(kw["detail"])},
            )
            return int(cur.lastrowid or 0)

    def adapt_events(self, ts_from: float | None = None, run_id: str | None = None,
                     kind: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM adapt_events WHERE 1=1"
        args: list[Any] = []
        if ts_from is not None:
            q += " AND ts >= ?"
            args.append(ts_from)
        if run_id is not None:
            q += " AND run_id = ?"
            args.append(run_id)
        if kind is not None:
            q += " AND kind = ?"
            args.append(kind)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def add_ops(self, ts: float, kind: str, level: str = "info",
                detail: dict[str, Any] | None = None, run_id: str | None = None) -> None:
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                "INSERT INTO ops_events(ts, kind, level, detail, run_id) VALUES(?,?,?,?,?)",
                (ts, kind, level, _json(detail or {}), run_id),
            )

    def ops_events(self, ts_from: float | None = None, limit: int = 300) -> list[dict[str, Any]]:
        q = "SELECT * FROM ops_events WHERE 1=1"
        args: list[Any] = []
        if ts_from is not None:
            q += " AND ts >= ?"
            args.append(ts_from)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def add_metrics(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = (
            "ts", "scope", "experiment", "arm", "signal", "horizon_s", "model_kind",
            "model_version", "regime", "window_start", "window_end", "n", "mae", "rmse",
            "smape", "bias", "p90_abs_error", "coverage", "interval_width", "skill",
            "detail", "run_id",
        )
        marks = ",".join("?" * len(cols))
        payload = [
            tuple(_json(r[c]) if c == "detail" else r.get(c) for c in cols) for r in rows
        ]
        with self.db.write_lock, self.db.conn:
            self.db.conn.executemany(
                f"INSERT INTO metrics({','.join(cols)}) VALUES({marks})", payload
            )

    def metrics(self, experiment: str | None = None, scope: str | None = None,
                run_id: str | None = None, limit: int = 2000) -> list[dict[str, Any]]:
        q = "SELECT * FROM metrics WHERE 1=1"
        args: list[Any] = []
        for col, val in (("experiment", experiment), ("scope", scope), ("run_id", run_id)):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        q += " ORDER BY signal, horizon_s, arm LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]


class ModelRepo:
    """Version registry for forecast models and language adapters."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def next_version_number(self, kind: str, scope: str) -> int:
        """Monotonically increasing version number for a (kind, scope).

        Held in ``meta`` rather than derived from ``COUNT(*)`` on
        ``model_versions``. Counting rows looks equivalent until pruning deletes
        old versions, at which point the counter stops advancing and every
        promotion reissues the same label -- overwriting the previous version's
        file and leaving rollback ping-ponging between two identical numbers.
        That failure was observed before this was changed.
        """
        key = f"version_counter:{kind}:{scope}"
        with self.db.write_lock, self.db.conn:
            row = self.db.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            n = (int(row["value"]) if row else 0) + 1
            self.db.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(n)),
            )
        return n

    def register(self, kind: str, scope: str, version: str, created_ts: float,
                 parent: str | None = None, path: str | None = None, config_hash: str = "",
                 train_start: float | None = None, train_end: float | None = None,
                 n_train: int = 0, metrics: dict[str, Any] | None = None,
                 active: bool = False, run_id: str | None = None) -> int:
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO model_versions(kind, version, scope, created_ts, parent, path,
                       config_hash, train_start, train_end, n_train, metrics, active, run_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(kind, scope, version) DO UPDATE SET
                       metrics=excluded.metrics, active=excluded.active, path=excluded.path""",
                (kind, version, scope, created_ts, parent, path, config_hash, train_start,
                 train_end, n_train, _json(metrics or {}), int(active), run_id),
            )
            return int(cur.lastrowid or 0)

    def activate(self, kind: str, scope: str, version: str, ts: float) -> None:
        """Make ``version`` the only active one for its (kind, scope)."""
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                "UPDATE model_versions SET active=0, retired_ts=? WHERE kind=? AND scope=? AND active=1",
                (ts, kind, scope),
            )
            self.db.conn.execute(
                "UPDATE model_versions SET active=1, retired_ts=NULL WHERE kind=? AND scope=? AND version=?",
                (kind, scope, version),
            )

    def active(self, kind: str, scope: str) -> dict[str, Any] | None:
        row = self.db.conn.execute(
            "SELECT * FROM model_versions WHERE kind=? AND scope=? AND active=1", (kind, scope)
        ).fetchone()
        return dict(row) if row else None

    def history(self, kind: str | None = None, scope: str | None = None,
                limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM model_versions WHERE 1=1"
        args: list[Any] = []
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        if scope is not None:
            q += " AND scope=?"
            args.append(scope)
        q += " ORDER BY created_ts DESC, id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def prune(self, kind: str, scope: str, keep: int) -> list[str]:
        """Drop registry rows for superseded versions beyond ``keep``.

        Returns the versions removed so the caller can delete their files. Neither
        the active version nor its recorded parent is ever pruned.
        """
        rows = self.db.conn.execute(
            """SELECT version, parent, active FROM model_versions
               WHERE kind=? AND scope=? ORDER BY created_ts DESC""",
            (kind, scope),
        ).fetchall()
        inactive = [r["version"] for r in rows if not r["active"]]
        # The active version's parent is the rollback target, so it must survive
        # pruning regardless of age. Without this, a long run with frequent
        # promotions prunes the immediate predecessor and rollback lands many
        # versions back.
        protected = {r["parent"] for r in rows if r["active"] and r["parent"]}
        doomed = [v for v in inactive[keep:] if v not in protected]
        if doomed:
            marks = ",".join("?" * len(doomed))
            with self.db.write_lock, self.db.conn:
                self.db.conn.execute(
                    f"DELETE FROM model_versions WHERE kind=? AND scope=? AND version IN ({marks})",
                    (kind, scope, *doomed),
                )
        return doomed


class FeedbackRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, ts: float, kind: str, text: str = "", label: str | None = None,
            prediction_id: int | None = None, signal: str | None = None,
            regime: str | None = None, ts_from: float | None = None, ts_to: float | None = None,
            payload: dict[str, Any] | None = None, run_id: str | None = None) -> int:
        with self.db.write_lock, self.db.conn:
            cur = self.db.conn.execute(
                """INSERT INTO feedback(ts, kind, prediction_id, signal, label, regime,
                       ts_from, ts_to, text, payload, run_id)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, kind, prediction_id, signal, label, regime, ts_from, ts_to, text,
                 _json(payload or {}), run_id),
            )
            return int(cur.lastrowid or 0)

    def list(self, ts_from: float | None = None, kind: str | None = None,
             unconsumed_only: bool = False, limit: int = 500) -> list[dict[str, Any]]:
        q = "SELECT * FROM feedback WHERE 1=1"
        args: list[Any] = []
        if ts_from is not None:
            q += " AND ts >= ?"
            args.append(ts_from)
        if kind is not None:
            q += " AND kind = ?"
            args.append(kind)
        if unconsumed_only:
            q += " AND consumed_by IS NULL"
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def mark_consumed(self, ids: Sequence[int], adapter_version: str) -> None:
        if not ids:
            return
        with self.db.write_lock, self.db.conn:
            self.db.conn.executemany(
                "UPDATE feedback SET consumed_by=? WHERE id=?",
                [(adapter_version, int(i)) for i in ids],
            )


class RunRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def start(self, run_id: str, kind: str, started_ts: float, config: dict[str, Any],
              config_hash: str, notes: str = "") -> None:
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                """INSERT OR REPLACE INTO runs(id, kind, started_ts, config, config_hash, notes, status)
                   VALUES(?,?,?,?,?,?,'running')""",
                (run_id, kind, started_ts, _json(config), config_hash, notes),
            )

    def finish(self, run_id: str, ended_ts: float, status: str = "complete",
               summary: dict[str, Any] | None = None) -> None:
        with self.db.write_lock, self.db.conn:
            self.db.conn.execute(
                "UPDATE runs SET ended_ts=?, status=?, summary=? WHERE id=?",
                (ended_ts, status, _json(summary or {}), run_id),
            )

    def list(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q = "SELECT * FROM runs WHERE 1=1"
        args: list[Any] = []
        if kind is not None:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY started_ts DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.db.conn.execute(q, args)]

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.db.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


@dataclass(slots=True)
class Repos:
    """Bundle passed to components that need storage."""

    db: Database
    signals: SignalRepo
    telemetry: TelemetryRepo
    features: FeatureRepo
    predictions: PredictionRepo
    events: EventRepo
    models: ModelRepo
    feedback: FeedbackRepo
    runs: RunRepo

    @classmethod
    def open(cls, path: str) -> "Repos":
        db = Database(path)
        sig = SignalRepo(db)
        return cls(
            db=db, signals=sig, telemetry=TelemetryRepo(db, sig), features=FeatureRepo(db),
            predictions=PredictionRepo(db), events=EventRepo(db), models=ModelRepo(db),
            feedback=FeedbackRepo(db), runs=RunRepo(db),
        )

    def close(self) -> None:
        self.db.close()
