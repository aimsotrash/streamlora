"""Training buffer for incremental updates.

Two pools, because a single one fails in a specific way:

* **Recency deque.** Recent examples are what makes the model track current
  behaviour. Training on those alone is catastrophic forgetting by design: an
  hour of idle telemetry overwrites everything the model knew about builds, and
  the first build after that is mispredicted as badly as on day one.
* **Reservoir.** A uniform random sample of *all* history via Algorithm R, so
  rare-but-important regimes keep a foothold in every update batch. Uniform is
  the right choice over "keep the interesting ones": deciding what is
  interesting requires knowing what the model will need, and getting that wrong
  silently biases every subsequent update.

Batches mix the two by ``reservoir_fraction``. The RNG is seeded, so a replay
draws the identical batch -- without that, two runs of the same experiment
produce different models and no ablation is interpretable.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from ..forecast.base import TrainingExample


@dataclass(slots=True)
class BufferStats:
    seen: int = 0
    recency: int = 0
    reservoir: int = 0
    drawn: int = 0
    absorbed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen, "recency": self.recency,
            "reservoir": self.reservoir, "drawn": self.drawn,
            "absorbed": self.absorbed,
        }


class TrainingBuffer:
    """Recency window plus reservoir sample, for one scope."""

    def __init__(
        self, recency: int = 600, reservoir: int = 1200, seed: int = 1337,
    ) -> None:
        self.recency_cap = max(1, int(recency))
        self.reservoir_cap = max(1, int(reservoir))
        self._recent: deque[tuple[int, TrainingExample]] = deque(maxlen=self.recency_cap)
        self._reservoir: list[TrainingExample] = []
        self._rng = random.Random(seed)
        self.stats = BufferStats()
        #: Monotonic sequence number per added example, and a high-water mark of
        #: what has already been absorbed into the live model.
        self._seq = 0
        self._absorbed_seq = 0

    def add(self, ex: TrainingExample) -> None:
        self.stats.seen += 1
        self._seq += 1
        self._recent.append((self._seq, ex))
        if len(self._reservoir) < self.reservoir_cap:
            self._reservoir.append(ex)
        else:
            # Algorithm R: replace with probability cap/seen, giving every
            # example an equal chance of being in the reservoir.
            j = self._rng.randrange(self.stats.seen)
            if j < self.reservoir_cap:
                self._reservoir[j] = ex
        self.stats.recency = len(self._recent)
        self.stats.reservoir = len(self._reservoir)

    def __len__(self) -> int:
        return len(self._recent)

    @property
    def absorbed_seq(self) -> int:
        return self._absorbed_seq

    def rewind_absorbed(self, seq: int) -> None:
        """Un-absorb back to ``seq`` after a rollback."""
        self._absorbed_seq = max(0, int(seq))

    @property
    def n_unabsorbed(self) -> int:
        return sum(1 for seq, _ in self._recent if seq > self._absorbed_seq)

    def mark_absorbed(self, upto_seq: int | None = None) -> None:
        """Record that everything up to ``upto_seq`` is now in the live model."""
        self._absorbed_seq = self._seq if upto_seq is None else max(
            self._absorbed_seq, int(upto_seq)
        )

    def recent(self, n: int) -> list[TrainingExample]:
        """The ``n`` newest examples, oldest first."""
        if n <= 0:
            return []
        items = [ex for _, ex in self._recent]
        return items[-n:]

    def split_gate(self, gate_n: int) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """Chronological train/gate split of the recency window.

        The gate is the *newest* ``gate_n`` examples and the training candidates
        are everything older. This is the only correct orientation: gating a
        candidate on data it trained on measures memorisation, and gating on
        data older than its training set measures nothing about the future.
        """
        items = sorted((ex for _, ex in self._recent), key=lambda e: e.ts_target)
        if gate_n <= 0 or len(items) <= gate_n:
            return items, []
        return items[:-gate_n], items[-gate_n:]

    def draw_batch(
        self, n: int, reservoir_fraction: float, exclude_after_ts: float | None = None,
        only_unabsorbed: bool = True,
    ) -> tuple[list[TrainingExample], int]:
        """Draw an update batch, mixing recency and reservoir.

        ``exclude_after_ts`` removes anything resolving at or after the gate
        boundary, which is what keeps the gate genuinely held out.

        ``only_unabsorbed`` restricts the recency part to examples the live model
        has not already been updated on. This matters more than it looks: with
        overlapping batches, RLS absorbs the same observation once per
        adaptation cycle, which shrinks the covariance as if it had seen far
        more independent data than it has, over-weights whatever happened to sit
        in the overlap, and produced measurably worse models than absorbing each
        observation exactly once. The reservoir is intentionally exempt -- its
        whole job is deliberate rehearsal of old examples.

        Returns ``(batch, max_seq_absorbed)`` so the caller can advance the
        high-water mark only if the candidate is actually promoted.
        """
        if n <= 0:
            return [], self._absorbed_seq
        items = list(self._recent)
        if only_unabsorbed:
            items = [(seq, e) for seq, e in items if seq > self._absorbed_seq]
        if exclude_after_ts is not None:
            items = [(seq, e) for seq, e in items if e.ts_target < exclude_after_ts]
        pool = list(self._reservoir)
        if exclude_after_ts is not None:
            pool = [e for e in pool if e.ts_target < exclude_after_ts]
        n_res = min(len(pool), int(round(n * max(0.0, min(1.0, reservoir_fraction)))))
        n_rec = min(len(items), max(0, n - n_res))
        chosen = items[-n_rec:] if n_rec else []
        max_seq = max((seq for seq, _ in chosen), default=self._absorbed_seq)
        batch = [e for _, e in chosen]
        if n_res:
            batch = batch + self._rng.sample(pool, n_res)
        # Chronological order within the batch: RLS with a forgetting factor is
        # order-dependent, and "oldest first" means the most recent example has
        # the largest influence, which is the intent.
        batch.sort(key=lambda e: e.ts_target)
        self.stats.drawn += len(batch)
        return batch, max_seq

    def clear(self) -> None:
        self._recent.clear()
        self._reservoir.clear()
        self._seq = 0
        self._absorbed_seq = 0
        self.stats = BufferStats()
