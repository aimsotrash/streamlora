"""Deterministic answer renderer. No model, always available.

Three jobs, and it is worth being explicit that they are three:

1. **The no-LLM fallback.** The spec requires the application to work when no
   language model is available. This is not a stub: it answers the same
   questions from the same evidence pack, and by construction it is 100%
   grounded and 100% style-compliant, because it only ever emits numbers that
   are in the pack and it applies the persona rules directly.

2. **The reference for the language experiment.** It is the floor every model
   arm is compared against.

3. **The target generator for LoRA training data.** This is the honest
   limitation of the whole language experiment and it is stated plainly here and
   in docs/experiments.md: there is no corpus of human-written explanations of
   this laptop's telemetry, so the training targets are rendered by this module
   from an explicit persona. The experiment therefore measures whether a
   low-rank update can *acquire a specified style-and-grounding function* from a
   modest number of examples more cheaply than full fine-tuning. It does not
   measure whether real users prefer the result -- that needs human labels this
   project does not have.

   To keep the task from degenerating into memorising one string, the renderer
   draws from several paraphrase templates per clause, selected by a hash of the
   content, so the same *style* is expressed with varied surface form.
"""

from __future__ import annotations

import hashlib
import time
from typing import Sequence

from ..grounding import (
    EvidencePack,
    Fact,
    UNIT_SUFFIX,
    fmt,
    fmt_duration,
    label_for,
)
from ..persona import PersonaSpec
from .base import GenerationResult, LanguageBackend


def _pick(options: Sequence[str], seed_text: str) -> str:
    """Deterministic choice among paraphrases, keyed by content."""
    h = int(hashlib.blake2s(seed_text.encode(), digest_size=4).hexdigest(), 16)
    return options[h % len(options)]


def _find(facts: Sequence[Fact], key: str) -> Fact | None:
    for f in facts:
        if f.key == key:
            return f
    return None


def _find_prefix(facts: Sequence[Fact], prefix: str) -> list[Fact]:
    return [f for f in facts if f.key.startswith(prefix)]


class TemplateRenderer:
    """Builds a grounded answer from an evidence pack and a persona."""

    def __init__(self, persona: PersonaSpec | None = None) -> None:
        self.persona = persona or PersonaSpec()

    # -- clause builders ---------------------------------------------------
    @staticmethod
    def _primary_signal(pack: EvidencePack) -> str | None:
        """The one signal an answer is about.

        Every clause must agree on this. Letting each clause pick its own
        produced answers like "CPU utilisation is 62.9%. In 5 min it is predicted
        at 99.5%" -- where 99.5% was the *battery* forecast. Both numbers were in
        the evidence, so the groundedness check passed; the sentence was simply
        about two different signals. Attribution is not something a numeric
        containment check can verify, so it is enforced structurally instead.
        """
        if pack.focus_signal:
            return pack.focus_signal
        return pack.current[0].key if pack.current else None

    def _now_clause(self, pack: EvidencePack) -> str | None:
        focus = self._primary_signal(pack)
        f = _find(pack.current, focus) if focus else (pack.current[0] if pack.current else None)
        if f is None or f.value is None:
            return None
        lab = label_for(f.key)
        v = fmt(f.value, f.unit)
        return _pick(
            (f"{lab} is {v} right now.",
             f"{v} {lab} at the moment.",
             f"Right now {lab} is {v}."),
            f"now:{f.key}:{v}",
        )

    def _change_clause(self, pack: EvidencePack) -> str | None:
        focus = self._primary_signal(pack)
        cands = _find_prefix(pack.window, f"{focus}.change") if focus else []
        if not cands:
            cands = [f for f in pack.window if f.key.endswith(".change")]
        if not cands:
            return None
        f = cands[0]
        if f.value is None:
            return None
        lab = label_for(f.key.rsplit(".", 1)[0])
        mag = abs(f.value)
        unit = UNIT_SUFFIX.get(f.unit, "")
        direction = "up" if f.value > 0 else "down"
        dur = fmt_duration(pack.window_s)
        return _pick(
            (f"It is {direction} {mag:.1f}{unit} over the last {dur}.",
             f"That is {mag:.1f}{unit} {direction} across {dur}.",
             f"{lab} moved {direction} {mag:.1f}{unit} in {dur}."),
            f"chg:{f.key}:{mag:.1f}",
        )

    def _battery_clause(self, pack: EvidencePack) -> str | None:
        rate = _find(pack.window, "battery.rate_pct_per_min")
        plugged = _find(pack.window, "battery.plugged")
        if plugged is not None and plugged.value and plugged.value > 0.5:
            return "It is on AC power, so it is not discharging."
        ttl = None
        for th in (20, 10, 0):
            ttl = _find(pack.window, f"battery.time_to_{th}")
            if ttl is not None:
                break
        if ttl is None or ttl.value is None:
            if rate is not None and rate.value is not None:
                return (f"The discharge rate is {abs(rate.value):.3f}%/min, "
                        "not yet enough history for a reliable projection.")
            return None
        mins = ttl.value
        th = ttl.key.rsplit("_", 1)[-1]
        base = _pick(
            (f"At the current rate that is about {mins:.0f} minutes to {th}%.",
             f"That projects to roughly {mins:.0f} minutes before it reaches {th}%.",
             f"About {mins:.0f} minutes remain until {th}% at this rate."),
            f"batt:{th}:{mins:.0f}",
        )
        return base

    def _driver_clause(self, pack: EvidencePack) -> str | None:
        if not pack.drivers:
            return None
        d = pack.drivers[0]
        lab = label_for(d.signal)
        unit = UNIT_SUFFIX.get(d.unit, "")
        lag = ""
        if d.lag_s >= 15.0:
            lag = f", leading it by about {fmt_duration(d.lag_s)}"
        # Never "caused by": this is a correlation over a short window.
        return _pick(
            (f"The most strongly correlated signal is {lab} (r={d.r:+.2f}), "
             f"which changed {d.change:+.1f}{unit}{lag}.",
             f"{lab} moved with it most closely (r={d.r:+.2f}, {d.change:+.1f}{unit}{lag}).",
             f"Correlated with {lab} at r={d.r:+.2f} ({d.change:+.1f}{unit}{lag})."),
            f"drv:{d.signal}:{d.r:.2f}",
        )

    def _forecast_clause(self, pack: EvidencePack) -> str | None:
        if not pack.forecasts:
            return None
        focus = self._primary_signal(pack)
        fc = next((f for f in pack.forecasts if focus and f.signal == focus), None)
        if fc is None:
            # No forecast for the signal this answer is about. Say nothing rather
            # than quote a different signal's number as if it were this one.
            return None
        lab = label_for(fc.signal)
        val = fmt(fc.value, fc.unit)
        dur = fmt_duration(fc.horizon_s)
        if fc.lo is not None and fc.hi is not None:
            rng = f", likely {fmt(fc.lo, fc.unit)} to {fmt(fc.hi, fc.unit)}"
        else:
            rng = " (no calibrated range yet)"
        # The signal is always named: "it" across a sentence boundary is exactly
        # how the misattribution above happened.
        return _pick(
            (f"The {dur} forecast for {lab} is {val}{rng}.",
             f"In {dur}, {lab} is predicted at {val}{rng}.",
             f"{lab} is expected to be {val} in {dur}{rng}."),
            f"fc:{fc.signal}:{fc.horizon_s}:{val}",
        )

    def _accuracy_clause(self, pack: EvidencePack) -> str | None:
        rls = [f for f in pack.accuracy if f.key.startswith("mae.rls.")]
        if not rls:
            rls = [f for f in pack.accuracy if f.key.startswith("mae.")]
        if not rls:
            return None
        f = rls[0]
        return f.text + "."

    def _event_clause(self, pack: EvidencePack) -> str | None:
        if not pack.events:
            return None
        drift = [f for f in pack.events if f.key == "drift"]
        adapt = [f for f in pack.events if f.key.startswith("adapt.")]
        pick = (drift or adapt)[0]
        return pick.text + "."

    def _history_clause(self, pack: EvidencePack) -> str | None:
        if not pack.history:
            return None
        return pack.history[0].text + "."

    def _rule_clause(self, pack: EvidencePack) -> str | None:
        rules = self.persona.rules_for(pack.regime, pack.focus_signal)
        if not rules:
            return None
        r = rules[0]
        word = r.required_word or r.verdict
        if r.note:
            return _pick(
                (f"You have told me this is {word} for you: \"{r.note}\".",
                 f"Per your note (\"{r.note}\"), treating this as {word}.",
                 f"Marked {word} for your setup: \"{r.note}\"."),
                f"rule:{r.regime}:{word}",
            )
        return f"You have marked this pattern as {word}."

    # -- assembly ----------------------------------------------------------
    def render(self, pack: EvidencePack) -> str:
        if not pack.current:
            return (
                "I have no telemetry yet, so there is nothing I can tell you about this "
                "machine. Start collection with `streamlora collect` and ask again."
            )
        intent = pack.intent
        order: list[str]
        if intent == "battery":
            order = ["now", "battery", "forecast", "driver", "rule"]
        elif intent == "why_change":
            order = ["now", "change", "driver", "history", "rule", "forecast"]
        elif intent == "forecast":
            order = ["now", "forecast", "change", "rule"]
        elif intent == "forecast_error":
            order = ["accuracy", "now", "event"]
        elif intent == "anomaly":
            order = ["event", "now", "change", "rule"]
        elif intent == "adaptation":
            order = ["event", "accuracy"]
        elif intent == "trend":
            order = ["now", "change", "driver", "history", "forecast"]
        else:
            order = ["now", "change", "forecast", "rule"]

        builders = {
            "now": self._now_clause, "change": self._change_clause,
            "battery": self._battery_clause, "driver": self._driver_clause,
            "forecast": self._forecast_clause, "accuracy": self._accuracy_clause,
            "event": self._event_clause, "rule": self._rule_clause,
            "history": self._history_clause,
        }
        clauses: list[str] = []
        # A persona rule is a promise to the user, so it is never dropped by the
        # sentence budget: it is emitted even if that means displacing a clause.
        budget = max(1, self.persona.style.max_sentences)
        rule_text = self._rule_clause(pack) if "rule" in order else None
        for name in order:
            if name == "rule":
                continue
            fn = builders[name]
            txt = fn(pack)
            if txt:
                clauses.append(txt)
            if len(clauses) >= (budget - (1 if rule_text else 0)):
                break
        if rule_text:
            clauses.append(rule_text)
        if not clauses:
            clauses.append(
                f"{label_for(pack.current[0].key)} is "
                f"{fmt(pack.current[0].value, pack.current[0].unit)}."
            )
        if pack.notes and len(clauses) < budget:
            clauses.append(pack.notes[0])
        text = " ".join(clauses[:budget])
        for bad, good in self.persona.style.vocabulary.items():
            text = text.replace(bad, good)
        return text


class TemplateBackend(LanguageBackend):
    name = "template"
    available = True
    supports_adapters = False

    def __init__(self, persona: PersonaSpec | None = None) -> None:
        self.renderer = TemplateRenderer(persona)

    def set_persona(self, persona: PersonaSpec) -> None:
        self.renderer.persona = persona

    def render_pack(self, pack: EvidencePack) -> str:
        return self.renderer.render(pack)

    def generate(self, prompt: str, system: str = "", max_new_tokens: int = 200,
                 temperature: float = 0.3) -> GenerationResult:
        # The template backend does not consume prompts; the service calls
        # render_pack directly. This path exists so the backend satisfies the
        # interface and can be swapped in anywhere.
        t0 = time.perf_counter()
        return GenerationResult(
            text=prompt, backend=self.name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            detail={"note": "template backend renders from evidence, not from prompts"},
        )
