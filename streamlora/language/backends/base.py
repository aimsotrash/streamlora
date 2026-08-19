"""Language backend interface.

The forecasting layer must remain usable with no language model at all, so the
backend is an interface with a deterministic implementation that is always
available. ``TemplateBackend`` is not a stub: it produces the same grounded
answers, from the same evidence pack, without a model. The HF backend adds
fluency and personalisation on top.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field


@dataclass(slots=True)
class GenerationResult:
    text: str
    backend: str
    adapter: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    detail: dict[str, object] = field(default_factory=dict)


class LanguageBackend(abc.ABC):
    name: str = "abstract"
    #: True when the backend can generate text right now.
    available: bool = False
    #: True when the backend supports loading a LoRA adapter.
    supports_adapters: bool = False

    @abc.abstractmethod
    def generate(self, prompt: str, system: str = "", max_new_tokens: int = 200,
                 temperature: float = 0.3) -> GenerationResult:
        ...

    def load_adapter(self, path: str | None) -> bool:  # pragma: no cover - overridden
        return False

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name, "available": self.available,
            "supports_adapters": self.supports_adapters,
        }

    def close(self) -> None:  # pragma: no cover
        pass
