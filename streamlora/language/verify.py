"""Automatic checks on a generated answer.

Two things get verified, and both are the reason the language layer is
measurable at all:

**Groundedness.** Every number in the answer must appear in the evidence pack
(within a tolerance that allows rounding). A number that does not is a
hallucinated measurement. This is a strictly mechanical check -- it cannot
detect a fluent sentence that misinterprets a correct number -- but it catches
the failure that actually matters here, which is a model inventing telemetry.

**Style compliance.** Each persona field has a checker, so "did personalisation
improve?" becomes a compliance rate on held-out examples instead of an opinion.

Both are reported per answer and aggregated per experiment arm.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .grounding import EvidencePack
from .persona import PersonaSpec, sentence_count

#: Numbers, optionally signed, with optional thousands separators and decimals.
_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

#: Tokens that are structural rather than measurements, so they are exempt.
#: Kept deliberately small -- a large exemption list would hollow out the check.
_EXEMPT_CONTEXT = (
    "top 1", "top 3", "1st", "2nd", "3rd", "24/7", "v001", "v002",
)


@dataclass
class GroundingReport:
    n_numbers: int = 0
    n_grounded: int = 0
    ungrounded: list[float] = field(default_factory=list)
    style_violations: list[str] = field(default_factory=list)
    style_checks: int = 0

    @property
    def groundedness(self) -> float:
        """Fraction of numbers found in the evidence. 1.0 when no numbers."""
        return 1.0 if self.n_numbers == 0 else self.n_grounded / self.n_numbers

    @property
    def fully_grounded(self) -> bool:
        return not self.ungrounded

    @property
    def style_compliance(self) -> float:
        if self.style_checks == 0:
            return 1.0
        return max(0.0, 1.0 - len(self.style_violations) / self.style_checks)

    @property
    def style_ok(self) -> bool:
        return not self.style_violations

    def as_dict(self) -> dict[str, object]:
        return {
            "n_numbers": self.n_numbers, "n_grounded": self.n_grounded,
            "groundedness": round(self.groundedness, 4),
            "ungrounded": self.ungrounded[:10],
            "style_violations": self.style_violations,
            "style_checks": self.style_checks,
            "style_compliance": round(self.style_compliance, 4),
        }


def extract_numbers(text: str) -> list[float]:
    lowered = text.lower()
    spans_to_skip: list[tuple[int, int]] = []
    for token in _EXEMPT_CONTEXT:
        start = 0
        while True:
            i = lowered.find(token, start)
            if i < 0:
                break
            spans_to_skip.append((i, i + len(token)))
            start = i + 1
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        if any(a <= m.start() < b for a, b in spans_to_skip):
            continue
        raw = m.group(0).replace(",", "")
        try:
            out.append(float(raw))
        except ValueError:
            continue
    return out


def is_grounded(value: float, surface: list[float], rel_tol: float = 0.06,
                abs_tol: float = 0.15) -> bool:
    """True if ``value`` matches any evidence number within tolerance.

    Tolerance is generous on purpose: an answer that rounds 48.23 to 48 is not
    hallucinating. Both a relative and an absolute tolerance are needed because
    the values range from 0.002 (%/min discharge) to 20000 (MB/s).
    """
    for s in surface:
        if not math.isfinite(s):
            continue
        if abs(value - s) <= max(abs_tol, rel_tol * abs(s)):
            return True
        # Magnitude-only match: "dropped 11%" against a change fact of -11.
        if abs(abs(value) - abs(s)) <= max(abs_tol, rel_tol * abs(s)):
            return True
    return False


def check_grounding(text: str, pack: EvidencePack) -> GroundingReport:
    rep = GroundingReport()
    surface = pack.numeric_values()
    for v in extract_numbers(text):
        rep.n_numbers += 1
        if is_grounded(v, surface):
            rep.n_grounded += 1
        else:
            rep.ungrounded.append(v)
    return rep


_CAUSAL_PHRASES = ("caused by", "because of", "due to", "is causing", "causes the")
_HEDGE_PHRASES = ("correlated", "coincid", "associated", "alongside", "tracks", "moved with",
                  "strongest correlat", "lines up with")
_UNCERTAINTY_PHRASES = ("likely", "range", "between", "roughly", "about", "approximately",
                        "around", "estimate", "to ", "+/-", "±")
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours|min|mins|minute|minutes)\b")


def _states_a_forecast(text: str, pack: EvidencePack) -> bool:
    """True if the answer quotes one of the pack's forecast values."""
    nums = extract_numbers(text)
    if not nums:
        return False
    for fc in pack.forecasts:
        for v in nums:
            if abs(v - fc.value) <= max(0.15, 0.06 * abs(fc.value)):
                return True
    return False


def check_style(text: str, pack: EvidencePack, persona: PersonaSpec) -> GroundingReport:
    """Style compliance. Returns a report carrying only style fields."""
    rep = GroundingReport()
    s = persona.style
    low = text.lower()

    rep.style_checks += 1
    if sentence_count(text) > s.max_sentences:
        rep.style_violations.append(
            f"too_long({sentence_count(text)}>{s.max_sentences})"
        )

    if s.lead_with_number:
        rep.style_checks += 1
        first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0] if text.strip() else ""
        if not re.search(r"\d", first):
            rep.style_violations.append("no_number_in_first_sentence")

    if s.state_uncertainty and pack.forecasts and _states_a_forecast(text, pack):
        # The rule is "when you state a forecast, give its range" -- so it is
        # conditioned on the *answer* mentioning one, not on the evidence pack
        # merely containing one. Keying it off the pack penalised a correct
        # answer that ran out of sentence budget and left the forecast out.
        rep.style_checks += 1
        if not any(p in low for p in _UNCERTAINTY_PHRASES):
            rep.style_violations.append("forecast_without_uncertainty")

    if s.battery_in_hours and pack.focus_signal == "battery.percent":
        # Only meaningful when there is a runtime to state. On AC power there is
        # no time-to-empty, so demanding a duration would penalise the correct
        # answer ("it is charging").
        has_projection = any(f.key.startswith("battery.time_to_") for f in pack.window)
        if has_projection:
            rep.style_checks += 1
            if not _DURATION_RE.search(low):
                rep.style_violations.append("battery_without_duration")

    if s.hedge_causation and pack.drivers:
        rep.style_checks += 1
        if any(p in low for p in _CAUSAL_PHRASES) and not any(
            p in low for p in _HEDGE_PHRASES
        ):
            rep.style_violations.append("asserted_causation")

    for bad, good in s.vocabulary.items():
        rep.style_checks += 1
        if bad.lower() in low and good.lower() not in low:
            rep.style_violations.append(f"vocabulary({bad}->{good})")

    for rule in persona.rules_for(pack.regime, pack.focus_signal):
        rep.style_checks += 1
        if rule.required_word and rule.required_word.lower() not in low:
            rep.style_violations.append(f"missing_rule_word({rule.required_word})")
    return rep


def verify(text: str, pack: EvidencePack, persona: PersonaSpec) -> GroundingReport:
    g = check_grounding(text, pack)
    st = check_style(text, pack, persona)
    g.style_violations = st.style_violations
    g.style_checks = st.style_checks
    return g
