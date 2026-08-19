"""Chat orchestration: question -> evidence -> answer -> verification.

The order is the important part. Evidence is assembled *first*, from the
database, with no model involved. Only then is a model asked to phrase it, and
the result is verified against the evidence before it is returned. If the model
is unavailable, or its answer fails the groundedness check, the deterministic
renderer's answer is served instead.

That last clause is the design decision that makes the feature honest rather than
decorative: a fluent answer containing an invented number is worse than a plain
one containing only real ones, so the system prefers the plain one and says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import Config
from ..store.repo import Repos
from ..util.logging import get_logger
from .backends.base import LanguageBackend
from .backends.template import TemplateBackend, TemplateRenderer
from .grounding import EvidencePack, GroundingEngine
from .persona import PersonaSpec, build_persona
from .prompt import SYSTEM_PROMPT, render_prompt
from .verify import GroundingReport, verify

log = get_logger("language.service")


@dataclass
class Answer:
    text: str
    backend: str
    intent: str
    grounded: bool = True
    groundedness: float = 1.0
    style_compliance: float = 1.0
    style_violations: list[str] = field(default_factory=list)
    adapter: str | None = None
    latency_ms: float = 0.0
    evidence: list[str] = field(default_factory=list)
    evidence_text: str = ""
    fallback_reason: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    persona_revision: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text, "backend": self.backend, "intent": self.intent,
            "grounded": self.grounded, "groundedness": round(self.groundedness, 4),
            "style_compliance": round(self.style_compliance, 4),
            "style_violations": self.style_violations, "adapter": self.adapter,
            "latency_ms": round(self.latency_ms, 1), "evidence": self.evidence,
            "evidence_text": self.evidence_text, "fallback_reason": self.fallback_reason,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "persona_revision": self.persona_revision,
        }


class LanguageService:
    def __init__(
        self, config: Config, repos: Repos, backend: LanguageBackend | None = None,
        persona: PersonaSpec | None = None,
    ) -> None:
        self.cfg = config
        self.repos = repos
        self.grounding = GroundingEngine(config, repos)
        self.persona = persona or build_persona(repos)
        self.template = TemplateBackend(self.persona)
        self.renderer = TemplateRenderer(self.persona)
        self.backend = backend
        self._adapter_path: str | None = None
        if backend is None:
            self.backend = self._make_backend()

    # -- setup -------------------------------------------------------------
    def _make_backend(self) -> LanguageBackend:
        want = self.cfg.language.backend
        if want in ("template", "none"):
            return self.template
        if want == "hf":
            from .backends.hf_local import HFLocalBackend

            b = HFLocalBackend(
                model_id=self.cfg.language.model_id, device=self.cfg.language.device,
                max_seq_len=self.cfg.language.max_seq_len,
            )
            if not b.load():
                log.warning(
                    "language model unavailable; using the deterministic renderer",
                    error=b.load_error, model_id=self.cfg.language.model_id,
                )
                return self.template
            self._attach_active_adapter(b)
            return b
        log.warning("unknown language backend; using the deterministic renderer", backend=want)
        return self.template

    def _attach_active_adapter(self, backend: LanguageBackend) -> None:
        if not backend.supports_adapters:
            return
        row = self.repos.models.active("adapter", "language")
        path = (row or {}).get("path")
        if path and backend.load_adapter(path):
            self._adapter_path = row["version"]
            log.info("language adapter attached", version=row["version"], path=path)

    def refresh_persona(self) -> PersonaSpec:
        self.persona = build_persona(self.repos)
        self.template.set_persona(self.persona)
        self.renderer.persona = self.persona
        return self.persona

    # -- the main call -----------------------------------------------------
    def ask(
        self, question: str, now: float | None = None, window_s: float | None = None,
        force_backend: str | None = None,
    ) -> Answer:
        t0 = time.perf_counter()
        now = time.time() if now is None else now
        self.refresh_persona()
        pack = self.grounding.build(question, now=now, window_s=window_s)
        deterministic = self.renderer.render(pack)

        backend = self.backend
        if force_backend == "template":
            backend = self.template

        if backend is self.template or not getattr(backend, "available", False):
            rep = verify(deterministic, pack, self.persona)
            return self._answer(
                deterministic, "template", pack, rep, t0,
                fallback_reason=None if backend is self.template else "backend_unavailable",
            )

        prompt = render_prompt(
            pack, self.persona,
            # With an adapter attached, the persona is in the weights; repeating
            # it in the prompt would confound the two mechanisms and waste
            # context on every request.
            include_persona=(self._adapter_path is None),
        )
        try:
            gen = backend.generate(
                prompt, system=SYSTEM_PROMPT,
                max_new_tokens=self.cfg.language.max_new_tokens,
                temperature=self.cfg.language.temperature,
            )
        except Exception:
            log.exception("language backend raised; falling back to renderer")
            rep = verify(deterministic, pack, self.persona)
            return self._answer(deterministic, "template", pack, rep, t0,
                                fallback_reason="backend_exception")
        text = (gen.text or "").strip()
        if not text:
            rep = verify(deterministic, pack, self.persona)
            return self._answer(deterministic, "template", pack, rep, t0,
                                fallback_reason="empty_generation")
        rep = verify(text, pack, self.persona)
        if not rep.fully_grounded:
            # Refuse to serve invented measurements. The generation is still
            # logged so the failure rate is measurable.
            log.warning(
                "generation contained ungrounded numbers; serving the renderer instead",
                ungrounded=rep.ungrounded[:5], backend=gen.backend,
            )
            det_rep = verify(deterministic, pack, self.persona)
            ans = self._answer(deterministic, "template", pack, det_rep, t0,
                               fallback_reason="ungrounded_numbers")
            ans.groundedness = rep.groundedness
            return ans
        ans = self._answer(text, gen.backend, pack, rep, t0)
        ans.adapter = self._adapter_path
        ans.tokens_in = gen.tokens_in
        ans.tokens_out = gen.tokens_out
        return ans

    def _answer(
        self, text: str, backend: str, pack: EvidencePack, rep: GroundingReport,
        t0: float, fallback_reason: str | None = None,
    ) -> Answer:
        evidence = [f.text for f in (*pack.current, *pack.window)][:8]
        evidence += [f.describe() for f in pack.forecasts][:4]
        evidence += [
            f"{c.signal}: r={c.r:+.2f} over the window" for c in pack.drivers
        ][:3]
        evidence += [f.text for f in pack.history][:2]
        evidence += [f.text for f in pack.events][:3]
        return Answer(
            text=text, backend=backend, intent=pack.intent,
            grounded=rep.fully_grounded, groundedness=rep.groundedness,
            style_compliance=rep.style_compliance, style_violations=rep.style_violations,
            latency_ms=(time.perf_counter() - t0) * 1000.0, evidence=evidence,
            evidence_text=pack.render(), fallback_reason=fallback_reason,
            persona_revision=self.persona.revision,
        )

    def describe(self) -> dict[str, object]:
        b = self.backend
        return {
            "backend": getattr(b, "name", "?"),
            "available": getattr(b, "available", False),
            "adapter": self._adapter_path,
            "persona": self.persona.to_dict(),
            "detail": b.describe() if b is not None else {},
        }
