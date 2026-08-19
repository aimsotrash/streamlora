"""Aggregation of stored predictions into grouped, comparable metrics.

The spec is explicit that a single aggregate number is meaningless, and it is
right: a model can look fine overall while being useless at the horizon anyone
cares about, or good only because the machine was idle for most of the window.
So every report is broken down by

    arm (model kind)  x  signal  x  horizon  x  [regime]  x  [time window]

and every row carries a skill score against a named reference arm, plus the
sample count so a "great" row backed by nine predictions is visibly not a result.

Reads exclusively from the ``predictions`` table, which is written before any
outcome exists -- so nothing reported here can have been computed with hindsight.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..store.repo import PredictionRecord, Repos
from . import metrics as M

#: The arm every other arm is scored against.
DEFAULT_REFERENCE = "persistence"


@dataclass(slots=True)
class Row:
    arm: str
    signal: str
    horizon_s: float
    regime: str | None
    window: str | None
    m: M.ErrorMetrics
    model_versions: list[str] = field(default_factory=list)
    dm_stat: float | None = None
    dm_p: float | None = None

    @property
    def scope(self) -> str:
        return f"{self.signal}@{int(self.horizon_s)}"

    def as_dict(self) -> dict[str, object]:
        d = {
            "arm": self.arm, "signal": self.signal, "horizon_s": self.horizon_s,
            "scope": self.scope, "regime": self.regime, "window": self.window,
            "model_versions": self.model_versions[:8],
            "dm_stat": None if self.dm_stat is None else round(self.dm_stat, 3),
            "dm_p": None if self.dm_p is None else round(self.dm_p, 5),
        }
        d.update(self.m.as_dict())
        return d


def _group_key(r: PredictionRecord, by_regime: bool) -> tuple:
    return (r.model_kind, r.signal, r.horizon_s, r.regime if by_regime else None)


def summarize(
    records: Iterable[PredictionRecord],
    reference: str = DEFAULT_REFERENCE,
    by_regime: bool = False,
    window: str | None = None,
    min_n: int = 1,
    dm_test: bool = True,
) -> list[Row]:
    """Group resolved predictions and compute metrics per group."""
    groups: dict[tuple, list[PredictionRecord]] = defaultdict(list)
    for r in records:
        if not r.resolved or r.actual is None:
            continue
        groups[_group_key(r, by_regime)].append(r)

    # Reference MAE per (signal, horizon, regime) so skill is comparable.
    ref_mae: dict[tuple, float] = {}
    ref_err: dict[tuple, dict[float, float]] = {}
    for (kind, sig, hor, reg), rs in groups.items():
        if kind != reference:
            continue
        errs = [r.error for r in rs if r.error is not None]
        if errs:
            ref_mae[(sig, hor, reg)] = float(np.mean(np.abs(errs)))
            # Keyed by target time so the DM test compares the same instants.
            ref_err[(sig, hor, reg)] = {
                r.ts_target: float(r.error) for r in rs if r.error is not None
            }

    out: list[Row] = []
    for (kind, sig, hor, reg), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(rs) < min_n:
            continue
        rs.sort(key=lambda r: r.ts_target)
        actual = [r.actual for r in rs]
        pred = [r.value for r in rs]
        lo = [r.lo for r in rs]
        hi = [r.hi for r in rs]
        m = M.compute(
            actual, pred, lo, hi,
            reference_mae=ref_mae.get((sig, hor, reg)),
            # The reference arm keeps its own name and scores 0.0 against
            # itself. Leaving it None would make the table's skill column look
            # like a missing value rather than the definitional zero it is.
            reference_name=reference,
        )
        row = Row(
            arm=kind, signal=sig, horizon_s=hor, regime=reg, window=window, m=m,
            model_versions=sorted({r.model_version for r in rs}),
        )
        if dm_test and kind != reference:
            ref = ref_err.get((sig, hor, reg))
            if ref:
                # Pair on target timestamp: comparing unaligned error series
                # would test two different sets of instants.
                pairs = [(r.error, ref[r.ts_target]) for r in rs
                         if r.error is not None and r.ts_target in ref]
                if len(pairs) >= 30:
                    res = M.diebold_mariano([p[0] for p in pairs], [p[1] for p in pairs])
                    if res is not None:
                        row.dm_stat, row.dm_p = res
        out.append(row)
    return out


def summarize_from_db(
    repos: Repos, run_id: str | None = None, ts_from: float | None = None,
    ts_to: float | None = None, reference: str = DEFAULT_REFERENCE,
    by_regime: bool = False, signals: Sequence[str] | None = None,
) -> list[Row]:
    recs = repos.predictions.resolved(run_id=run_id, ts_from=ts_from, ts_to=ts_to)
    if signals:
        keep = set(signals)
        recs = [r for r in recs if r.signal in keep]
    return summarize(recs, reference=reference, by_regime=by_regime)


def time_windows(
    records: Sequence[PredictionRecord], n: int = 4
) -> list[tuple[str, float, float]]:
    """Split the record span into ``n`` equal windows.

    Used to answer "is it getting better over time?", which a single aggregate
    cannot: a continually adapting model is *supposed* to improve across the run,
    and that shows up only as a trend across windows.
    """
    if not records:
        return []
    ts = sorted(r.ts_target for r in records)
    lo, hi = ts[0], ts[-1]
    if hi <= lo:
        return [("all", lo, hi)]
    step = (hi - lo) / n
    return [(f"w{i + 1}", lo + i * step, lo + (i + 1) * step + (1e-6 if i == n - 1 else 0.0))
            for i in range(n)]


def summarize_over_time(
    records: Sequence[PredictionRecord], n_windows: int = 4,
    reference: str = DEFAULT_REFERENCE,
) -> list[Row]:
    out: list[Row] = []
    for name, lo, hi in time_windows(records, n_windows):
        sub = [r for r in records if lo <= r.ts_target < hi]
        if len(sub) < 20:
            continue
        out.extend(summarize(sub, reference=reference, window=name, dm_test=False))
    return out


#: Arms whose sample counts differ by more than this factor are not directly
#: comparable, and the table says so instead of implying they are.
COMPARABILITY_RATIO = 2.0


def to_table(rows: Sequence[Row], arms: Sequence[str] | None = None) -> str:
    """Fixed-width MAE comparison table, one row per scope, one column per arm.

    The ``n`` column shows ``min-max`` across arms, not a single number, and a
    row whose counts differ by more than ``COMPARABILITY_RATIO`` is flagged with
    ``!``. This is not cosmetic: on real telemetry one arm had 7 resolved
    predictions at battery@1800 against ~1198 for the others, all 7 of them
    landing on a charger unplug, and its MAE of 44.0 next to everyone else's 6.5
    read as a catastrophic model failure. It was a sample-size artefact.
    """
    if not rows:
        return "(no resolved predictions)"
    order = list(arms) if arms else sorted({r.arm for r in rows})
    scopes: dict[tuple[str, float, str | None, str | None], dict[str, Row]] = defaultdict(dict)
    for r in rows:
        scopes[(r.signal, r.horizon_s, r.regime, r.window)][r.arm] = r
    w = 13
    head = (f"{'scope':<28}" + "".join(f"{a[:w - 1]:>{w}}" for a in order)
            + f"{'best':>14}{'n':>14}")
    lines = [head, "-" * len(head)]
    flagged = False
    for key in sorted(scopes, key=lambda k: (k[0], k[1], str(k[2]), str(k[3]))):
        by_arm = scopes[key]
        sig, hor, reg, win = key
        label = f"{sig}@{int(hor)}"
        if reg:
            label += f" [{reg}]"
        if win:
            label += f" ({win})"
        cells = []
        best_arm, best_mae = None, None
        counts: list[int] = []
        for a in order:
            r = by_arm.get(a)
            if r is None or r.m.mae is None:
                cells.append(f"{'-':>{w}}")
                continue
            cells.append(f"{r.m.mae:>{w}.4f}")
            counts.append(r.m.n)
            if best_mae is None or r.m.mae < best_mae:
                best_arm, best_mae = a, r.m.mae
        if counts:
            lo, hi = min(counts), max(counts)
            uneven = lo > 0 and hi / lo > COMPARABILITY_RATIO
            flagged |= uneven
            n_text = f"{lo}-{hi}{'!' if uneven else ''}" if lo != hi else str(lo)
        else:
            n_text = "-"
        lines.append(f"{label:<28}" + "".join(cells) + f"{(best_arm or '-'):>14}{n_text:>14}")
    if flagged:
        lines.append("")
        lines.append("! sample counts differ by more than "
                     f"{COMPARABILITY_RATIO:g}x across arms; those rows are not "
                     "a like-for-like comparison")
    return "\n".join(lines)
