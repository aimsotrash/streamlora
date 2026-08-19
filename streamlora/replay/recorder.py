"""Dataset export/import.

Datasets are newline-delimited JSON: one object per sample, with a header line
carrying the signal specs. Chosen over a binary format because a telemetry
dataset is the primary artefact an experiment is judged on, and being able to
``head`` it, diff it, and read it in five years without this codebase is worth
more than the space saving. Files compress ~10x with gzip if that matters.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

from ..store.repo import Repos
from ..telemetry.schema import (
    Quality,
    Reading,
    SignalKind,
    SignalRegistry,
    SignalSpec,
    TelemetrySample,
)

FORMAT_VERSION = 1


@dataclass(slots=True)
class DatasetInfo:
    path: str
    n_samples: int
    ts_start: float
    ts_end: float
    signals: list[str]
    origin: str = ""

    @property
    def duration_s(self) -> float:
        return self.ts_end - self.ts_start

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path, "n_samples": self.n_samples, "ts_start": self.ts_start,
            "ts_end": self.ts_end, "duration_s": round(self.duration_s, 1),
            "signals": self.signals, "origin": self.origin,
        }


def _open(path: str, mode: str):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def write_dataset(
    path: str, samples: Iterable[TelemetrySample], registry: SignalRegistry,
    notes: str = "",
) -> DatasetInfo:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    ts_start = ts_end = 0.0
    names: list[str] = []
    with _open(path, "wt") as fh:
        header = {
            "format": FORMAT_VERSION,
            "notes": notes,
            "signals": [
                {
                    "name": s.name, "unit": s.unit, "kind": str(s.kind), "lo": s.lo, "hi": s.hi,
                    "max_rate_per_s": s.max_rate_per_s, "max_hold_s": s.max_hold_s,
                    "description": s.description,
                }
                for s in (registry.get(x) for x in registry.names())
                if s is not None
            ],
        }
        fh.write(json.dumps(header) + "\n")
        for s in samples:
            if n == 0:
                ts_start = s.ts
                names = sorted(s.readings)
            ts_end = s.ts
            row = {
                "ts": s.ts, "origin": s.origin, "gap_s": s.gap_s,
                "collect_ms": round(s.collect_ms, 3),
                "r": {
                    k: ([None, int(v.quality)] if v.value is None else
                        ([v.value] if v.quality == Quality.OK else [v.value, int(v.quality)]))
                    for k, v in s.readings.items()
                },
            }
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return DatasetInfo(path, n, ts_start, ts_end, names, notes)


def read_dataset(path: str) -> tuple[Iterator[TelemetrySample], SignalRegistry, dict]:
    """Return a lazy sample iterator plus the dataset's signal registry."""
    fh = _open(path, "rt")
    header = json.loads(fh.readline())
    specs = []
    for s in header.get("signals", []):
        specs.append(
            SignalSpec(
                name=s["name"], unit=s.get("unit", ""),
                kind=SignalKind(s.get("kind", "gauge")), lo=s.get("lo"), hi=s.get("hi"),
                max_rate_per_s=s.get("max_rate_per_s"), max_hold_s=s.get("max_hold_s", 30.0),
                description=s.get("description", ""),
            )
        )
    registry = SignalRegistry(specs)

    def gen() -> Iterator[TelemetrySample]:
        try:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                readings = {}
                for k, v in row["r"].items():
                    val = v[0]
                    q = Quality(v[1]) if len(v) > 1 else Quality.OK
                    readings[k] = Reading(val, q)
                yield TelemetrySample(
                    ts=row["ts"], readings=readings, origin=row.get("origin", "replay"),
                    collect_ms=row.get("collect_ms", 0.0), gap_s=row.get("gap_s"),
                )
        finally:
            fh.close()

    return gen(), registry, header


def export_from_db(
    repos: Repos, path: str, run_id: str | None = None, ts_from: float | None = None,
    ts_to: float | None = None, notes: str = "",
) -> DatasetInfo:
    """Snapshot recorded telemetry into a portable dataset file."""
    registry = repos.signals.registry()
    samples = repos.telemetry.iter_samples(ts_from=ts_from, ts_to=ts_to, run_id=run_id)
    return write_dataset(path, samples, registry, notes=notes or f"export run_id={run_id}")


def import_to_db(repos: Repos, path: str, run_id: str) -> int:
    it, registry, _header = read_dataset(path)
    repos.signals.register(registry.as_mapping().values())
    batch: list[TelemetrySample] = []
    n = 0
    for s in it:
        batch.append(s)
        if len(batch) >= 500:
            repos.telemetry.insert_samples(batch, run_id=run_id)
            n += len(batch)
            batch = []
    if batch:
        repos.telemetry.insert_samples(batch, run_id=run_id)
        n += len(batch)
    return n
