"""Prompt construction for the model-backed language arms.

Kept in one place so every arm (base, few-shot, LoRA, full fine-tune) sees an
identical prompt shape and the only difference between them is the adaptation
mechanism. If the arms differed in prompt as well as in weights, the comparison
would measure nothing.
"""

from __future__ import annotations

from typing import Sequence

from .grounding import EvidencePack
from .persona import PersonaSpec

SYSTEM_PROMPT = (
    "You are the explanation layer of a laptop telemetry forecasting system. "
    "You are given measured evidence and a question. "
    "Answer using only numbers that appear in the evidence; never invent a measurement. "
    "Describe relationships as correlations, not causes. Be concise and specific."
)


def render_prompt(
    pack: EvidencePack,
    persona: PersonaSpec | None = None,
    include_persona: bool = False,
    shots: Sequence[tuple[str, str]] = (),
) -> str:
    """Build the user-turn text.

    ``include_persona`` is the prompt-only personalisation arm: the persona is
    spelled out in the prompt instead of being baked into weights. That arm is
    the honest comparison for LoRA -- "is a low-rank update better than simply
    telling the model the preferences?" -- and it costs tokens on every single
    request, which is the trade the experiment quantifies.
    """
    parts: list[str] = []
    if include_persona and persona is not None:
        parts.append(persona.prompt_block())
        parts.append("")
    for ev, ans in shots:
        parts.append("EVIDENCE:")
        parts.append(ev)
        parts.append("ANSWER: " + ans)
        parts.append("")
    parts.append("EVIDENCE:")
    parts.append(pack.render())
    parts.append("")
    parts.append(f"QUESTION: {pack.question or 'What is my machine doing, and what happens next?'}")
    parts.append("ANSWER:")
    return "\n".join(parts)


def render_training_pair(
    pack: EvidencePack, target: str, persona: PersonaSpec | None = None,
    include_persona: bool = False,
) -> tuple[str, str]:
    """(prompt, target) pair for supervised fine-tuning."""
    return render_prompt(pack, persona, include_persona=include_persona), target.strip()
