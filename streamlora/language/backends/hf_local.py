"""Local transformers backend, with optional LoRA adapters.

Local by default and by design: telemetry is private data, and the spec requires
the core pipeline to work without any cloud service. Nothing here makes a network
call except the one-time model download, and that is a deliberate, logged action.

The model is small on purpose. A 135M-360M instruct model fits comfortably in a
small laptop GPU alongside a LoRA optimiser state, trains an adapter in seconds
rather than hours, and can therefore be adapted *continuously* -- which is the
whole point. A 7B model would give better prose and make the continual-adaptation
loop impossible on this hardware, which would be the wrong trade for this project.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import GenerationResult, LanguageBackend


class HFLocalBackend(LanguageBackend):
    name = "hf_local"
    supports_adapters = True

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: str = "auto",
        max_seq_len: int = 640,
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.max_seq_len = int(max_seq_len)
        self.available = False
        self.adapter: str | None = None
        self._tok: Any = None
        self._model: Any = None
        self._base_model: Any = None
        self._peft: Any = None
        self._torch: Any = None
        self.device = "cpu"
        self.load_error: str | None = None
        self._requested_device = device
        self._dtype = dtype

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> bool:
        if self.available:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            self.load_error = f"torch/transformers unavailable: {exc}"
            return False
        try:
            if self._requested_device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = self._requested_device
            dtype = None
            if self._dtype == "auto":
                dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
            if self._tok.pad_token is None:
                # Causal LMs frequently ship without a pad token; reusing EOS is
                # the standard fix and is safe because attention masks are used.
                self._tok.pad_token = self._tok.eos_token
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype
            ).to(self.device)
            self._base_model.eval()
            self._model = self._base_model
            self._torch = torch
            self.available = True
            return True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return False

    def close(self) -> None:
        self._peft = None
        self._model = None
        self._base_model = None
        self._tok = None
        self.available = False
        if self._torch is not None:
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass

    # -- adapters ----------------------------------------------------------
    def load_adapter(self, path: str | None) -> bool:
        """Attach a LoRA adapter, or detach with ``None``.

        Detaching is exact: ``PeftModel.unload()`` removes the injected LoRA
        modules and leaves the base weights bitwise identical (asserted in the
        test suite). That is what makes adapter rollback exact rather than
        approximate -- there is no merged state to invert.

        The unwrap step is not optional. ``PeftModel.from_pretrained`` injects
        its modules into the base model *in place*, so simply repointing at the
        base leaves it wrapped, and the next attach stacks a second adapter on
        top of the first.
        """
        if not self.available and not self.load():
            return False
        if path is None:
            self._detach()
            return True
        # Detach any current adapter first, so adapters never stack.
        self._detach()
        if not os.path.isdir(path):
            self.load_error = f"adapter path does not exist: {path}"
            return False
        try:
            from peft import PeftModel
        except Exception as exc:
            self.load_error = f"peft unavailable: {exc}"
            return False
        try:
            peft_model = PeftModel.from_pretrained(self._base_model, path)
            peft_model.eval()
            self._peft = peft_model
            self._model = peft_model
            self.adapter = path
            return True
        except Exception as exc:
            self.load_error = f"adapter load failed: {type(exc).__name__}: {exc}"
            self._detach()
            return False

    def _detach(self) -> None:
        """Remove any attached adapter and restore the base model exactly."""
        if self._peft is not None:
            try:
                self._peft.unload()
            except Exception as exc:  # pragma: no cover - defensive
                self.load_error = f"adapter unload failed: {type(exc).__name__}: {exc}"
            self._peft = None
        self._model = self._base_model
        self.adapter = None

    # -- generation --------------------------------------------------------
    def _chat(self, prompt: str, system: str) -> str:
        if getattr(self._tok, "chat_template", None):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return (system + "\n\n" if system else "") + prompt

    def generate(self, prompt: str, system: str = "", max_new_tokens: int = 200,
                 temperature: float = 0.3) -> GenerationResult:
        t0 = time.perf_counter()
        if not self.available and not self.load():
            return GenerationResult(
                text="", backend=self.name, detail={"error": self.load_error or "unavailable"}
            )
        torch = self._torch
        text = self._chat(prompt, system)
        enc = self._tok(
            text, return_tensors="pt", truncation=True, max_length=self.max_seq_len
        ).to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self._tok.pad_token_id,
        }
        if temperature and temperature > 0.01:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kwargs)
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        answer = self._tok.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            text=answer, backend=self.name, adapter=self.adapter,
            tokens_in=int(enc["input_ids"].shape[1]), tokens_out=int(new_tokens.shape[0]),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            detail={"model_id": self.model_id, "device": self.device},
        )

    def describe(self) -> dict[str, object]:
        d = super().describe()
        d.update({
            "model_id": self.model_id, "device": self.device, "adapter": self.adapter,
            "load_error": self.load_error,
        })
        return d
