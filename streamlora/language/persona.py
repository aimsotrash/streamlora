"""The user's explanation preferences, as a checkable specification.

Why this exists as an explicit artefact rather than as free text in a prompt:
personalisation has to be *measurable* or the LoRA question cannot be answered.
Every field here has a deterministic checker in ``verify.py``, so "did the
adaptation layer become more useful?" reduces to a compliance rate on held-out
examples rather than to an opinion about tone.

The persona is built from two sources and both are real:

* defaults, and
* the user's own structured feedback -- ``label`` rows become semantic rules
  ("treat sustained CPU during compiles as normal for me"), ``note`` rows become
  context, and ``forecast_useful`` rows adjust verbosity.

So the spec is not a static config file; it is the accumulated state of what the
user has told the system.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..store.repo import Repos


@dataclass
class StyleSpec:
    """Deterministically checkable presentation preferences."""

    #: Hard ceiling on sentences. Terse answers are the common request.
    max_sentences: int = 3
    #: Answer must open with a number, not a preamble.
    lead_with_number: bool = True
    #: Any forecast mentioned must carry a range or an explicit hedge.
    state_uncertainty: bool = True
    #: Battery answers must give a duration, not only a percentage.
    battery_in_hours: bool = True
    #: Never assert causation from a correlation.
    hedge_causation: bool = True
    #: Preferred words. Maps a disfavoured term to the user's term.
    vocabulary: dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticRule:
    """A user-taught interpretation, e.g. "compiling is normal for me"."""

    #: Regime this applies to ("build", "gpu_heavy", ...) or "*".
    regime: str = "*"
    #: Signal this applies to, or None for any.
    signal: str | None = None
    #: "normal" | "expected" | "concerning" | "ignore"
    verdict: str = "normal"
    #: The user's own words, quoted back in explanations.
    note: str = ""
    #: Word the answer must contain when this rule fires.
    required_word: str = "normal"
    source_feedback_id: int | None = None

    def matches(self, regime: str, signal: str | None) -> bool:
        if self.regime != "*" and self.regime != regime:
            return False
        if self.signal is not None and signal is not None and self.signal != signal:
            return False
        return True


@dataclass
class PersonaSpec:
    name: str = "default"
    style: StyleSpec = field(default_factory=StyleSpec)
    rules: list[SemanticRule] = field(default_factory=list)
    #: Free-form context lines the user has supplied.
    context: list[str] = field(default_factory=list)
    #: Monotonic revision, bumped whenever feedback changes the persona. Used to
    #: decide whether an adapter is stale relative to what the user has taught.
    revision: int = 0

    def rules_for(self, regime: str, signal: str | None = None) -> list[SemanticRule]:
        return [r for r in self.rules if r.matches(regime, signal)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersonaSpec":
        style = StyleSpec(**d.get("style", {})) if d.get("style") else StyleSpec()
        rules = [SemanticRule(**r) for r in d.get("rules", [])]
        return cls(
            name=d.get("name", "default"), style=style, rules=rules,
            context=list(d.get("context", [])), revision=int(d.get("revision", 0)),
        )

    @classmethod
    def load(cls, path: str) -> "PersonaSpec":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json())

    def prompt_block(self) -> str:
        """The persona as prompt text, for the prompt-only comparison arm."""
        s = self.style
        lines = ["USER PREFERENCES:"]
        lines.append(f"- Answer in at most {s.max_sentences} sentences.")
        if s.lead_with_number:
            lines.append("- Begin with the key number, not a preamble.")
        if s.state_uncertainty:
            lines.append("- When you state a forecast, give its likely range.")
        if s.battery_in_hours:
            lines.append("- For battery, give the time remaining, not just a percentage.")
        if s.hedge_causation:
            lines.append('- Say "correlated with", never "caused by".')
        for bad, good in s.vocabulary.items():
            lines.append(f'- Say "{good}" rather than "{bad}".')
        for r in self.rules:
            scope = "" if r.regime == "*" else f" during {r.regime}"
            lines.append(
                f'- Treat this behaviour{scope} as {r.verdict}'
                + (f' ("{r.note}").' if r.note else ".")
            )
        for c in self.context[:6]:
            lines.append(f"- Context: {c}")
        return "\n".join(lines)


#: Regimes a "that's normal for me" note plausibly refers to, by keyword.
_REGIME_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("build", ("compil", "build", "cargo", "make", "gradle", "webpack", "rustc", "linking")),
    ("gpu_heavy", ("game", "gaming", "render", "cuda", "train", "inference", "gpu")),
    ("io_heavy", ("backup", "sync", "index", "copy", "download large")),
    ("battery_constrained", ("unplugged", "on battery", "low battery")),
    ("charging", ("charging", "plugged in")),
    ("media", ("video", "stream", "movie", "call", "zoom", "meet")),
)

#: Phrases that mean "this is expected, stop flagging it".
_NORMAL_PHRASES = (
    "that's normal", "thats normal", "is normal", "normal for me", "normal for my",
    "treat that as normal", "expected", "always happens", "happens when i",
    "on purpose", "deliberate", "that's fine", "thats fine", "no problem",
)
_CONCERN_PHRASES = ("shouldn't", "should not", "not normal", "unexpected", "problem", "worrying")


def infer_rule_from_text(text: str, feedback_id: int | None = None) -> SemanticRule | None:
    """Turn a free-text note into a structured rule where possible.

    Deliberately conservative: it returns None rather than guessing, because a
    wrong rule silently changes every future explanation. Text that cannot be
    parsed is still kept as context, so nothing the user says is discarded --
    it just does not become a hard rule.
    """
    t = text.lower().strip()
    if not t:
        return None
    verdict: str | None = None
    if any(p in t for p in _NORMAL_PHRASES):
        verdict = "normal"
    if any(p in t for p in _CONCERN_PHRASES):
        verdict = "concerning"
    if verdict is None:
        return None
    regime = "*"
    for name, keys in _REGIME_HINTS:
        if any(k in t for k in keys):
            regime = name
            break
    return SemanticRule(
        regime=regime, verdict=verdict, note=text.strip()[:160],
        required_word="normal" if verdict == "normal" else "unusual",
        source_feedback_id=feedback_id,
    )


def build_persona(
    repos: Repos, base: PersonaSpec | None = None, limit: int = 200
) -> PersonaSpec:
    """Assemble the live persona from stored feedback."""
    p = base or PersonaSpec()
    rows = repos.feedback.list(limit=limit)
    seen_rules: set[tuple[str, str, str]] = {(r.regime, r.verdict, r.note) for r in p.rules}
    n_useful = n_not_useful = 0
    for r in reversed(rows):  # oldest first, so later feedback wins
        kind = r["kind"]
        text = (r.get("text") or "").strip()
        if kind in ("label", "note") and text:
            rule = infer_rule_from_text(text, feedback_id=r["id"])
            if rule is not None:
                key = (rule.regime, rule.verdict, rule.note)
                if key not in seen_rules:
                    p.rules.append(rule)
                    seen_rules.add(key)
                    p.revision += 1
            elif text not in p.context:
                p.context.append(text)
                p.revision += 1
        elif kind == "forecast_useful":
            if (r.get("label") or "").lower() in ("yes", "true", "1", "useful"):
                n_useful += 1
            else:
                n_not_useful += 1
    # Verbosity responds to explicit usefulness feedback: persistent "not
    # useful" means the answers are not carrying enough, so allow more room.
    if n_not_useful > n_useful + 2:
        p.style.max_sentences = min(6, p.style.max_sentences + 1)
        p.revision += 1
    elif n_useful > n_not_useful + 4:
        p.style.max_sentences = max(2, p.style.max_sentences - 1)
        p.revision += 1
    return p


def sentence_count(text: str) -> int:
    """Count sentences, tolerant of decimals like "48.2%"."""
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text.strip())
    parts = [p for p in re.split(r"[.!?]+(?:\s|$)", t) if p.strip()]
    return len(parts)
