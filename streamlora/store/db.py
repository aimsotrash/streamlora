"""SQLite schema and connection management.

Why SQLite and not something bigger: the workload is a single writer at 0.2 Hz
with analytical reads over at most a few million rows, on one machine, holding
data the user considers private. That is squarely inside SQLite's competence.
Anything distributed would add operational surface without answering a single
research question.

Two pragmas do the real work:

* ``journal_mode=WAL`` lets the API read while the collector writes.
* ``synchronous=NORMAL`` avoids an fsync per transaction. The exposure is
  losing the last transaction on a power cut, which for telemetry batched at
  6 samples (~30 s) is an acceptable trade for not spinning the disk 0.2 times
  a second on a laptop.

Storage layout note: readings are stored **long-format** (one row per
signal per tick) with an interned signal id. That is what makes "add GPU
temperature later" a zero-migration change, which the spec requires. The cost
is a pivot on read; at these volumes it is microseconds.
"""

from __future__ import annotations

import os
import sqlite3
import threading

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Interned signal names, so readings rows stay small and typos are visible.
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    unit        TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'gauge',
    lo          REAL,
    hi          REAL,
    description TEXT NOT NULL DEFAULT ''
);

-- One row per collection tick.
CREATE TABLE IF NOT EXISTS samples (
    id            INTEGER PRIMARY KEY,
    ts            REAL NOT NULL,
    origin        TEXT NOT NULL DEFAULT 'live',
    collect_ms    REAL NOT NULL DEFAULT 0,
    gap_s         REAL,
    worst_quality INTEGER NOT NULL DEFAULT 0,
    run_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);
CREATE INDEX IF NOT EXISTS idx_samples_run ON samples(run_id, ts);

CREATE TABLE IF NOT EXISTS readings (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    signal_id INTEGER NOT NULL REFERENCES signals(id),
    value     REAL,
    quality   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sample_id, signal_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_readings_signal ON readings(signal_id, sample_id);

-- Feature vectors are stored once per tick and referenced by every prediction
-- made from them. Storing them is what makes gated promotion possible: a
-- candidate model can be scored on exactly the inputs the active model saw.
CREATE TABLE IF NOT EXISTS feature_schemas (
    hash       TEXT PRIMARY KEY,
    names_json TEXT NOT NULL,
    created_ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feature_windows (
    id          INTEGER PRIMARY KEY,
    ts          REAL NOT NULL,
    schema_hash TEXT NOT NULL REFERENCES feature_schemas(hash),
    regime      TEXT NOT NULL DEFAULT 'unknown',
    coverage    REAL NOT NULL DEFAULT 1.0,
    vec         BLOB NOT NULL,
    run_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_fw_ts ON feature_windows(ts);
CREATE INDEX IF NOT EXISTS idx_fw_run ON feature_windows(run_id, ts);

-- Full prediction lifecycle: made -> resolved -> (optionally) used to adapt.
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY,
    ts_made        REAL NOT NULL,
    ts_target      REAL NOT NULL,
    horizon_s      REAL NOT NULL,
    signal         TEXT NOT NULL,
    value          REAL NOT NULL,
    lo             REAL,
    hi             REAL,
    alpha          REAL,
    model_kind     TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    regime         TEXT NOT NULL DEFAULT 'unknown',
    anchor         REAL,
    feature_id     INTEGER REFERENCES feature_windows(id),
    infer_ms       REAL NOT NULL DEFAULT 0,
    run_id         TEXT,
    -- resolution
    resolved       INTEGER NOT NULL DEFAULT 0,
    ts_resolved    REAL,
    actual         REAL,
    error          REAL,
    abs_error      REAL,
    in_interval    INTEGER,
    resolve_quality INTEGER,
    used_for_adapt INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pred_target ON predictions(resolved, ts_target);
CREATE INDEX IF NOT EXISTS idx_pred_lookup ON predictions(signal, horizon_s, model_kind, ts_made);
CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_id, ts_made);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version, ts_made);

CREATE TABLE IF NOT EXISTS drift_events (
    id         INTEGER PRIMARY KEY,
    ts         REAL NOT NULL,
    detector   TEXT NOT NULL,
    scope      TEXT NOT NULL,           -- 'error' | 'feature'
    signal     TEXT,
    horizon_s  REAL,
    statistic  REAL NOT NULL,
    threshold  REAL NOT NULL,
    severity   REAL NOT NULL DEFAULT 0,
    detail     TEXT NOT NULL DEFAULT '{}',
    run_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_drift_ts ON drift_events(ts);
CREATE INDEX IF NOT EXISTS idx_drift_run ON drift_events(run_id, ts);

CREATE TABLE IF NOT EXISTS model_versions (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,        -- 'forecast-model' | 'adapter'
    version       TEXT NOT NULL,
    scope         TEXT NOT NULL DEFAULT '',  -- e.g. 'cpu.util_pct@300'
    created_ts    REAL NOT NULL,
    parent        TEXT,
    path          TEXT,
    config_hash   TEXT NOT NULL DEFAULT '',
    train_start   REAL,
    train_end     REAL,
    n_train       INTEGER NOT NULL DEFAULT 0,
    metrics       TEXT NOT NULL DEFAULT '{}',
    active        INTEGER NOT NULL DEFAULT 0,
    retired_ts    REAL,
    run_id        TEXT,
    UNIQUE (kind, scope, version)
);
CREATE INDEX IF NOT EXISTS idx_mv_active ON model_versions(kind, scope, active);

CREATE TABLE IF NOT EXISTS adapt_events (
    id                INTEGER PRIMARY KEY,
    ts                REAL NOT NULL,
    scope             TEXT NOT NULL,
    kind              TEXT NOT NULL,    -- 'forecast' | 'language'
    trigger           TEXT NOT NULL,    -- which policy fired
    decision          TEXT NOT NULL,    -- promoted | rejected | rolled_back | skipped | failed
    active_version    TEXT,
    candidate_version TEXT,
    metric_name       TEXT NOT NULL DEFAULT 'mae',
    metric_before     REAL,
    metric_after      REAL,
    gate_n            INTEGER NOT NULL DEFAULT 0,
    n_train           INTEGER NOT NULL DEFAULT 0,
    duration_ms       REAL NOT NULL DEFAULT 0,
    detail            TEXT NOT NULL DEFAULT '{}',
    run_id            TEXT
);
CREATE INDEX IF NOT EXISTS idx_adapt_ts ON adapt_events(ts);
CREATE INDEX IF NOT EXISTS idx_adapt_run ON adapt_events(run_id, ts);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY,
    ts            REAL NOT NULL,
    kind          TEXT NOT NULL,        -- forecast_useful | event_happened | label | note
    prediction_id INTEGER REFERENCES predictions(id),
    signal        TEXT,
    label         TEXT,                 -- structured value where one exists
    regime        TEXT,
    ts_from       REAL,
    ts_to         REAL,
    text          TEXT NOT NULL DEFAULT '',
    payload       TEXT NOT NULL DEFAULT '{}',
    consumed_by   TEXT,                 -- adapter version that trained on it
    run_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_ts ON feedback(ts);

CREATE TABLE IF NOT EXISTS metrics (
    id            INTEGER PRIMARY KEY,
    ts            REAL NOT NULL,
    scope         TEXT NOT NULL,        -- 'live' | 'experiment' | 'gate'
    experiment    TEXT,
    arm           TEXT,
    signal        TEXT NOT NULL,
    horizon_s     REAL NOT NULL,
    model_kind    TEXT NOT NULL,
    model_version TEXT,
    regime        TEXT,
    window_start  REAL,
    window_end    REAL,
    n             INTEGER NOT NULL,
    mae           REAL,
    rmse          REAL,
    smape         REAL,
    bias          REAL,
    p90_abs_error REAL,
    coverage      REAL,
    interval_width REAL,
    skill         REAL,                 -- 1 - MAE/MAE_persistence
    detail        TEXT NOT NULL DEFAULT '{}',
    run_id        TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(experiment, arm, signal, horizon_s);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id, ts);

CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,          -- collect | replay | experiment | serve
    started_ts  REAL NOT NULL,
    ended_ts    REAL,
    config      TEXT NOT NULL DEFAULT '{}',
    config_hash TEXT NOT NULL DEFAULT '',
    notes       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'running',
    summary     TEXT NOT NULL DEFAULT '{}'
);

-- Operational events: collection health, latency outliers, failures, rollbacks.
CREATE TABLE IF NOT EXISTS ops_events (
    id     INTEGER PRIMARY KEY,
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    level  TEXT NOT NULL DEFAULT 'info',
    detail TEXT NOT NULL DEFAULT '{}',
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_ops_ts ON ops_events(ts);
"""


def connect(path: str, *, read_only: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    """Open a connection with the pragmas this workload needs."""
    if path != ":memory:":
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    if read_only and path != ":memory:":
        uri = f"file:{os.path.abspath(path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False)
    else:
        conn = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    if not read_only:
        # :memory: does not support WAL; skip rather than fail.
        if path != ":memory:":
            cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(_SCHEMA)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
        elif int(row["value"]) != SCHEMA_VERSION:
            # There is exactly one schema version so far. When a second one
            # exists this is where the migration ladder goes; failing loudly is
            # better than silently reading a schema we do not understand.
            raise RuntimeError(
                f"database schema version {row['value']} != expected {SCHEMA_VERSION}"
            )


class Database:
    """Owns one connection and serialises writes from multiple threads.

    SQLite handles concurrent readers fine under WAL, but a single Python
    connection is not safe for concurrent use, and the collector, adaptation
    controller and API all write. One lock around writes is simpler and fast
    enough here than a connection pool.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = connect(path)
        init_schema(self.conn)
        self.write_lock = threading.RLock()

    def close(self) -> None:
        try:
            with self.write_lock:
                self.conn.commit()
        finally:
            self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def vacuum_older_than(self, cutoff_ts: float) -> int:
        """Delete telemetry older than ``cutoff_ts``. Returns rows removed.

        Retention is a user-facing privacy control, so it is an explicit
        operation rather than a background job that quietly discards data.
        """
        with self.write_lock, self.conn:
            cur = self.conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff_ts,))
            n = cur.rowcount or 0
            self.conn.execute("DELETE FROM feature_windows WHERE ts < ?", (cutoff_ts,))
        return n
