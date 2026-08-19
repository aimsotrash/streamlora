"""Chronological evaluation splits.

The single most important rule in this project: **never shuffle**. A random
split on telemetry leaks the future into the past twice over -- once because
consecutive samples are seconds apart and nearly identical, and once because a
feature window at time t overlaps the window at t+5s. A shuffled split on this
data reports an MAE two to five times better than the system can ever achieve
live, and every conclusion drawn from it is wrong.

So the only splits available here are chronological, and the boundary
additionally *embargoes* a gap of at least one forecast horizon: a training
example whose target time falls after the test set begins has already observed
test-period telemetry. Without the embargo, the last few hundred training rows
leak, which is a subtle enough bug to survive review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(slots=True)
class Split:
    """Index ranges into a chronologically ordered array."""

    name: str
    train: np.ndarray
    test: np.ndarray
    validation: np.ndarray | None = None
    #: Timestamps bounding each part, for reporting.
    train_span: tuple[float, float] | None = None
    test_span: tuple[float, float] | None = None
    embargoed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_train": int(self.train.size),
            "n_validation": None if self.validation is None else int(self.validation.size),
            "n_test": int(self.test.size),
            "train_span": self.train_span,
            "test_span": self.test_span,
            "embargoed": self.embargoed,
        }


def chronological_split(
    ts: Sequence[float],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    horizon_s: float = 0.0,
    name: str = "holdout",
) -> Split:
    """Split by time into train / validation / test with a horizon embargo.

    ``horizon_s`` removes training rows whose *target* time would land inside
    the following segment. Passing 0 disables the embargo and is only correct
    when the rows are not horizon-based.
    """
    t = np.asarray(ts, dtype=np.float64)
    n = t.size
    if n < 10:
        raise ValueError(f"need at least 10 rows to split, got {n}")
    if not (0 < train_frac < 1) or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid fractions: train={train_frac} val={val_frac}")
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    train = np.arange(0, i_tr)
    val = np.arange(i_tr, i_va)
    test = np.arange(i_va, n)
    embargoed = 0
    if horizon_s > 0:
        # Drop rows at the end of each earlier segment whose forecast target
        # falls in the next segment.
        if val.size:
            cut = t[val[0]] - horizon_s
            keep = train[t[train] < cut]
            embargoed += int(train.size - keep.size)
            train = keep
        if test.size and val.size:
            cut = t[test[0]] - horizon_s
            keep = val[t[val] < cut]
            embargoed += int(val.size - keep.size)
            val = keep
        elif test.size:
            cut = t[test[0]] - horizon_s
            keep = train[t[train] < cut]
            embargoed += int(train.size - keep.size)
            train = keep
    return Split(
        name=name, train=train, test=test, validation=val if val.size else None,
        train_span=(float(t[train[0]]), float(t[train[-1]])) if train.size else None,
        test_span=(float(t[test[0]]), float(t[test[-1]])) if test.size else None,
        embargoed=embargoed,
    )


def rolling_origin_splits(
    ts: Sequence[float],
    n_folds: int = 4,
    initial_frac: float = 0.4,
    horizon_s: float = 0.0,
    expanding: bool = True,
) -> Iterator[Split]:
    """Rolling- or expanding-origin evaluation.

    This is the honest version of cross-validation for time series: fit on
    everything up to an origin, predict the next block, advance the origin.
    ``expanding`` grows the training set (the usual choice, since a real system
    never discards history); setting it False keeps a fixed-width window, which
    is the right comparison when studying how much old data helps.
    """
    t = np.asarray(ts, dtype=np.float64)
    n = t.size
    if n < 40 or n_folds < 1:
        return
    start = int(n * initial_frac)
    if start < 10:
        start = min(10, n // 2)
    block = max(1, (n - start) // n_folds)
    for f in range(n_folds):
        te_lo = start + f * block
        te_hi = min(n, te_lo + block)
        if te_hi - te_lo < 5 or te_lo >= n:
            return
        tr_lo = 0 if expanding else max(0, te_lo - start)
        train = np.arange(tr_lo, te_lo)
        test = np.arange(te_lo, te_hi)
        embargoed = 0
        if horizon_s > 0 and train.size:
            cut = t[test[0]] - horizon_s
            keep = train[t[train] < cut]
            embargoed = int(train.size - keep.size)
            train = keep
        if train.size < 10:
            continue
        yield Split(
            name=f"fold{f + 1}", train=train, test=test, embargoed=embargoed,
            train_span=(float(t[train[0]]), float(t[train[-1]])),
            test_span=(float(t[test[0]]), float(t[test[-1]])),
        )
