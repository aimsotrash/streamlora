"""Structured feedback: the input side of personalisation.

The brief is explicit that saving chat logs and calling it learning does not
count. So feedback is stored as typed rows with the fields that make it usable as
a training or configuration signal, and each kind has a defined consumer:

* ``forecast_useful``  -> adjusts persona verbosity; links to a prediction id so
  "which forecasts were useful" is answerable per scope.
* ``event_happened``   -> a human label on whether a predicted event occurred.
  Recorded against the prediction, giving a second accuracy measure that is not
  derivable from telemetry (a spike can be correctly predicted and still be the
  wrong thing to have flagged).
* ``label``            -> a semantic rule ("compiling is normal for me"). Parsed
  into a ``SemanticRule`` and enforced by the style verifier.
* ``note``             -> free context, injected into every evidence pack.

Every row also carries ``consumed_by``, set to the adapter version that trained
on it, so it is always possible to say which feedback has actually influenced the
model and which is still pending.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from ..store.repo import Repos
from ..util.logging import get_logger
from .persona import PersonaSpec, build_persona, infer_rule_from_text

log = get_logger("language.feedback")

VALID_KINDS = ("forecast_useful", "event_happened", "label", "note")


@dataclass(slots=True)
class FeedbackSummary:
    total: int = 0
    by_kind: dict[str, int] = None  # type: ignore[assignment]
    unconsumed: int = 0
    useful_yes: int = 0
    useful_no: int = 0
    rules_inferred: int = 0
    persona_revision: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total, "by_kind": self.by_kind or {},
            "unconsumed": self.unconsumed, "useful_yes": self.useful_yes,
            "useful_no": self.useful_no, "rules_inferred": self.rules_inferred,
            "persona_revision": self.persona_revision,
        }


class FeedbackStore:
    def __init__(self, repos: Repos) -> None:
        self.repos = repos

    def add(
        self, kind: str, text: str = "", label: str | None = None,
        prediction_id: int | None = None, signal: str | None = None,
        regime: str | None = None, ts_from: float | None = None, ts_to: float | None = None,
        now: float | None = None, run_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        if kind not in VALID_KINDS:
            raise ValueError(f"unknown feedback kind {kind!r}; expected one of {VALID_KINDS}")
        now = time.time() if now is None else now
        # Derive the signal from the referenced prediction when not supplied, so
        # per-scope usefulness is answerable without asking the user twice.
        if prediction_id is not None and signal is None:
            rec = self.repos.predictions.get(prediction_id)
            if rec is not None:
                signal = rec.signal
                regime = regime or rec.regime
        fid = self.repos.feedback.add(
            ts=now, kind=kind, text=text, label=label, prediction_id=prediction_id,
            signal=signal, regime=regime, ts_from=ts_from, ts_to=ts_to,
            payload=payload or {}, run_id=run_id,
        )
        rule = infer_rule_from_text(text) if kind in ("label", "note") and text else None
        log.info(
            "feedback recorded", id=fid, kind=kind, signal=signal,
            prediction_id=prediction_id,
            inferred_rule=(f"{rule.regime}/{rule.verdict}" if rule else None),
        )
        return fid

    def persona(self, base: PersonaSpec | None = None) -> PersonaSpec:
        return build_persona(self.repos, base=base)

    def pending(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.repos.feedback.list(unconsumed_only=True, limit=limit)

    def mark_consumed(self, ids: Sequence[int], adapter_version: str) -> None:
        self.repos.feedback.mark_consumed(ids, adapter_version)

    def summary(self, limit: int = 500) -> FeedbackSummary:
        rows = self.repos.feedback.list(limit=limit)
        s = FeedbackSummary(by_kind={})
        for r in rows:
            s.total += 1
            s.by_kind[r["kind"]] = s.by_kind.get(r["kind"], 0) + 1
            if r.get("consumed_by") is None:
                s.unconsumed += 1
            if r["kind"] == "forecast_useful":
                if (r.get("label") or "").lower() in ("yes", "true", "1", "useful"):
                    s.useful_yes += 1
                else:
                    s.useful_no += 1
            if r["kind"] in ("label", "note") and (r.get("text") or "").strip():
                if infer_rule_from_text(r["text"]) is not None:
                    s.rules_inferred += 1
        s.persona_revision = self.persona().revision
        return s

    def accuracy_by_user(self, limit: int = 500) -> dict[str, dict[str, int]]:
        """Human-judged outcomes per scope, from ``event_happened`` rows.

        This is deliberately separate from MAE: a forecast can be numerically
        close and still be judged unhelpful, and a spike can be predicted
        correctly at the wrong time. Only the user can report that.
        """
        out: dict[str, dict[str, int]] = {}
        for r in self.repos.feedback.list(kind="event_happened", limit=limit):
            sig = r.get("signal") or "unknown"
            d = out.setdefault(sig, {"yes": 0, "no": 0})
            if (r.get("label") or "").lower() in ("yes", "true", "1", "happened"):
                d["yes"] += 1
            else:
                d["no"] += 1
        return out
