"""LoRA adaptation of the language layer: training, versioning, gating, rollback.

Why LoRA is here at all
-----------------------
Not because it is interesting technology. It is here because the language layer
has to keep changing as the user tells the system things, and the alternatives
are worse on this hardware:

* **Prompt-only personalisation** puts the persona in every request. It works --
  it is one of the comparison arms -- but it costs tokens and latency on every
  single answer, and it cannot encode anything that does not fit in a prompt.
* **Full fine-tuning** of even a 135M model updates 100% of parameters, needs
  optimiser state for all of them, and produces a fresh multi-hundred-megabyte
  checkpoint per update. On a small laptop GPU that is possible but not something
  you would do every hour.
* **LoRA** trains ~1% of parameters, produces adapters measured in megabytes,
  trains in seconds on this hardware, and can be attached and detached exactly --
  which makes adapter rollback trivially correct rather than approximate.

The explicit hypothesis, tested in ``language/experiment.py``: *small continuous
low-rank updates personalise the interaction layer more efficiently than repeated
full fine-tuning*. "More efficiently" is measured, not asserted: held-out loss,
groundedness, style compliance, trainable parameters, wall-clock, peak memory.

Training details that matter
----------------------------
Loss is computed on the **answer tokens only**; prompt tokens are masked to
-100. Without that, the model spends most of its capacity learning to reproduce
the evidence block, which is the input, and the thing we actually want it to
learn -- the answer style -- is a small fraction of the loss.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Config
from ..store.repo import Repos
from ..util.ids import version_label
from ..util.logging import get_logger
from .backends.hf_local import HFLocalBackend
from .dataset import LanguageExample
from .persona import PersonaSpec
from .prompt import SYSTEM_PROMPT, render_prompt

log = get_logger("language.lora")

ADAPTER_KIND = "adapter"
ADAPTER_SCOPE = "language"


@dataclass
class TrainStats:
    trainable_params: int = 0
    total_params: int = 0
    steps: int = 0
    epochs: int = 0
    train_loss: float = float("nan")
    eval_loss_before: float = float("nan")
    eval_loss_after: float = float("nan")
    wall_s: float = 0.0
    peak_mem_mb: float = 0.0
    n_train: int = 0
    n_eval: int = 0
    method: str = "lora"
    artifact_mb: float = 0.0
    device: str = "cpu"
    batch_size: int = 0
    error: str | None = None

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_fraction": round(self.trainable_fraction, 6),
            "steps": self.steps, "epochs": self.epochs,
            "train_loss": None if math.isnan(self.train_loss) else round(self.train_loss, 4),
            "eval_loss_before": (
                None if math.isnan(self.eval_loss_before) else round(self.eval_loss_before, 4)
            ),
            "eval_loss_after": (
                None if math.isnan(self.eval_loss_after) else round(self.eval_loss_after, 4)
            ),
            "eval_ppl_after": (
                None if math.isnan(self.eval_loss_after)
                else round(math.exp(min(self.eval_loss_after, 20.0)), 3)
            ),
            "device": self.device, "batch_size": self.batch_size,
            "wall_s": round(self.wall_s, 2),
            "peak_mem_mb": round(self.peak_mem_mb, 1),
            "artifact_mb": round(self.artifact_mb, 2),
            "n_train": self.n_train, "n_eval": self.n_eval,
            "error": self.error,
        }


@dataclass
class AdapterInfo:
    version: str
    path: str
    created_ts: float
    parent: str | None
    stats: TrainStats
    persona_revision: int = 0
    promoted: bool = False
    gate_reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version, "path": self.path, "created_ts": self.created_ts,
            "parent": self.parent, "persona_revision": self.persona_revision,
            "promoted": self.promoted, "gate_reason": self.gate_reason,
            "stats": self.stats.as_dict(),
        }


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total / (1 << 20)


class LanguageTrainer:
    """Trains and versions language adapters. Requires the ``lora`` extra."""

    def __init__(self, config: Config, repos: Repos, backend: HFLocalBackend | None = None) -> None:
        self.cfg = config
        self.repos = repos
        self.backend = backend or HFLocalBackend(
            model_id=config.language.model_id, device=config.language.device,
            max_seq_len=config.language.max_seq_len,
        )
        self.adapters_dir = os.path.join(config.models_dir, "adapters")
        os.makedirs(self.adapters_dir, exist_ok=True)

    # -- availability ------------------------------------------------------
    @property
    def available(self) -> bool:
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    def _require(self) -> None:
        if not self.available:
            raise RuntimeError(
                "the language adaptation layer needs the optional extra: "
                "pip install 'streamlora[lora]'"
            )
        if not self.backend.load():
            raise RuntimeError(f"could not load base model: {self.backend.load_error}")

    # -- tokenisation ------------------------------------------------------
    def _encode(self, ex: LanguageExample, persona: PersonaSpec, include_persona: bool):
        """Return (input_ids, labels) with the prompt masked out of the loss."""
        import torch

        tok = self.backend._tok
        from .dataset import pack_from_example

        pack = pack_from_example(ex)
        pack.question = ex.question
        prompt = render_prompt(pack, persona, include_persona=include_persona)
        # The stored evidence text is authoritative; pack_from_example only
        # reconstructs the numeric surface for scoring.
        prompt = prompt.replace(pack.render(), ex.evidence)
        if getattr(tok, "chat_template", None):
            text_prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            text_prompt = SYSTEM_PROMPT + "\n\n" + prompt
        p_ids = tok(text_prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(ex.target + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        max_len = self.cfg.language.max_seq_len
        if len(p_ids) + len(a_ids) > max_len:
            # Truncate the *front* of the prompt. The tail holds the question and
            # the most recent evidence, which is what the answer depends on;
            # truncating the tail would remove the target's justification.
            keep = max_len - len(a_ids)
            p_ids = p_ids[-max(keep, 8):]
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + list(a_ids)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def _batches(self, examples: Sequence[LanguageExample], persona: PersonaSpec,
                 include_persona: bool, batch_size: int, shuffle: bool = False,
                 seed: int = 0):
        import torch

        idx = list(range(len(examples)))
        if shuffle:
            import random as _r

            _r.Random(seed).shuffle(idx)
        tok = self.backend._tok
        pad = tok.pad_token_id or 0
        for i in range(0, len(idx), batch_size):
            chunk = [examples[j] for j in idx[i : i + batch_size]]
            enc = [self._encode(e, persona, include_persona) for e in chunk]
            n = max(x[0].numel() for x in enc)
            ids = torch.full((len(enc), n), pad, dtype=torch.long)
            labels = torch.full((len(enc), n), -100, dtype=torch.long)
            mask = torch.zeros((len(enc), n), dtype=torch.long)
            for k, (a, b) in enumerate(enc):
                ids[k, : a.numel()] = a
                labels[k, : b.numel()] = b
                mask[k, : a.numel()] = 1
            yield (ids.to(self.backend.device), labels.to(self.backend.device),
                   mask.to(self.backend.device))

    # -- evaluation --------------------------------------------------------
    def eval_loss(self, model: Any, examples: Sequence[LanguageExample],
                  persona: PersonaSpec, include_persona: bool = False,
                  batch_size: int | None = None) -> float:
        """Mean cross-entropy per answer token on held-out examples."""
        import torch

        if not examples:
            return float("nan")
        # Batch 1 by default. The memory here is dominated by the logits tensor,
        # (B x T x vocab) -- at 640 tokens and a 49k vocab that is ~125 MB per
        # sequence in fp32, so a batch of 4 is half a gigabyte for a number we
        # only need summed.
        bs = batch_size or 1
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for ids, labels, mask in self._batches(examples, persona, include_persona, bs):
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits[:, :-1, :]
                tgt = labels[:, 1:]
                keep = tgt != -100
                if keep.sum() == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(
                    logits[keep], tgt[keep], reduction="sum"
                )
                total += float(loss.item())
                count += int(keep.sum().item())
        return total / count if count else float("nan")

    # -- training ----------------------------------------------------------
    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        name = type(exc).__name__
        return "OutOfMemory" in name or "out of memory" in str(exc).lower()

    #: Substrings identifying a failure to build a GPU kernel rather than a
    #: modelling problem. On Linux the usual cause is missing CPython headers:
    #: torch's Triton path shells out to gcc and needs Python.h, which ships in
    #: python3-devel / python3-dev. That is a packaging gap on the host, not
    #: something the user should have to diagnose from a compiler traceback.
    _GPU_BUILD_FAILURE_HINTS = (
        "python.h", "cuda_utils", "triton", "returned non-zero exit status",
        "no such file or directory", "ptxas", "nvcc",
    )

    def _is_gpu_build_failure(self, exc: BaseException) -> bool:
        blob = f"{type(exc).__name__} {exc}".lower()
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            blob += f" {type(cause).__name__} {cause}".lower()
        cmd = getattr(exc, "cmd", None)
        if cmd:
            blob += f" {cmd}".lower()
        return any(h in blob for h in self._GPU_BUILD_FAILURE_HINTS)

    def train(
        self,
        train_examples: Sequence[LanguageExample],
        eval_examples: Sequence[LanguageExample],
        persona: PersonaSpec,
        method: str = "lora",
        include_persona: bool = False,
        out_dir: str | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        batch_size: int | None = None,
        _allow_cpu_fallback: bool = True,
        _oom_retries: int = 2,
    ) -> tuple[TrainStats, str | None]:
        """Train an adapter (``method='lora'``) or all weights (``'full'``).

        Returns ``(stats, artifact_path)``. The full-fine-tune path exists purely
        as the efficiency comparison; it is never used to serve.

        Failure handling is the interesting part, because the target is a laptop
        rather than a cluster. On CUDA out-of-memory the batch is halved and
        retried; if that is exhausted, or the GPU cannot build its kernels, the
        whole arm falls back to CPU. A small card genuinely cannot hold a full
        fine-tune of a 135M model alongside Adam state and the serving copy, and
        the right response is to use less of the machine rather than to report a
        failure. The device actually used is recorded in the stats, so a
        comparison never silently mixes devices.
        """
        self._require()
        import torch

        stats = TrainStats(method=method)
        if len(train_examples) < 2:
            stats.error = f"need at least 2 training examples, got {len(train_examples)}"
            return stats, None

        lc = self.cfg.language
        epochs = int(epochs if epochs is not None else lc.epochs)
        lr = float(lr if lr is not None else lc.lr)
        bs = int(batch_size if batch_size is not None else lc.batch_size)
        device = self.backend.device
        base = self.backend._base_model
        if device == "cuda":
            # Start from a clean allocator: several arms run in one process and
            # fragmentation left by a previous arm's generation cache is enough
            # to OOM the next one's optimiser.
            gc.collect()
            torch.cuda.empty_cache()

        try:
            return self._train_inner(
                stats, train_examples, eval_examples, persona, method,
                include_persona, out_dir, epochs, lr, bs, device, base, torch,
            )
        except Exception as exc:
            stats.error = f"{type(exc).__name__}: {exc}"
            if device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
            if self._is_oom(exc) and bs > 1 and _oom_retries > 0:
                log.warning(
                    "CUDA out of memory; retrying with a smaller batch",
                    method=method, batch_size=bs, retry_batch_size=max(1, bs // 2),
                )
                return self.train(
                    train_examples, eval_examples, persona, method=method,
                    include_persona=include_persona, out_dir=out_dir, epochs=epochs,
                    lr=lr, batch_size=max(1, bs // 2),
                    _allow_cpu_fallback=_allow_cpu_fallback,
                    _oom_retries=_oom_retries - 1,
                )
            if (
                _allow_cpu_fallback
                and device == "cuda"
                and (self._is_oom(exc) or self._is_gpu_build_failure(exc))
            ):
                reason = "out of memory" if self._is_oom(exc) else "kernel build failed"
                log.warning(
                    f"GPU {reason}; retrying this arm on CPU. For GPU training, "
                    "install the CPython development headers (python3-devel / "
                    "python3-dev) and/or use a smaller model or batch.",
                    method=method, error=str(exc)[:200],
                )
                previous_pref = self.backend._requested_device
                self.backend.close()
                self.backend._requested_device = "cpu"
                if not self.backend.load():
                    stats.error = f"CPU fallback failed: {self.backend.load_error}"
                    return stats, None
                try:
                    return self.train(
                        train_examples, eval_examples, persona, method=method,
                        include_persona=include_persona, out_dir=out_dir, epochs=epochs,
                        lr=lr, batch_size=batch_size, _allow_cpu_fallback=False,
                        _oom_retries=0,
                    )
                finally:
                    # Restore the device preference. Leaving it on CPU is sticky
                    # across *arms*, so one arm that could not fit silently drags
                    # every later arm onto CPU too -- and a cost comparison that
                    # mixes devices measures nothing.
                    self.backend.close()
                    self.backend._requested_device = previous_pref
                    self.backend.load()
            log.exception("language training failed", method=method)
            return stats, None

    def _train_inner(
        self,
        stats: TrainStats,
        train_examples: Sequence[LanguageExample],
        eval_examples: Sequence[LanguageExample],
        persona: PersonaSpec,
        method: str,
        include_persona: bool,
        out_dir: str | None,
        epochs: int,
        lr: float,
        bs: int,
        device: str,
        base: Any,
        torch: Any,
    ) -> tuple[TrainStats, str | None]:
        lc = self.cfg.language
        if method == "lora":
            from peft import LoraConfig, get_peft_model

            peft_cfg = LoraConfig(
                r=lc.lora_r, lora_alpha=lc.lora_alpha, lora_dropout=lc.lora_dropout,
                target_modules=list(lc.lora_targets), bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(base, peft_cfg)
        elif method == "full":
            import copy

            # Deep-copied so the shared base the serving backend holds is never
            # mutated by the comparison arm, and copied *on CPU* first: deep
            # copying a CUDA module allocates a second full model on the device
            # before the first is released, which is exactly the peak a small card
            # cannot absorb alongside Adam state.
            if device == "cuda":
                base.to("cpu")
                model = copy.deepcopy(base)
                base.to(device)
                torch.cuda.empty_cache()
            else:
                model = copy.deepcopy(base)
            for p in model.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(f"unknown method {method!r}")

        model.to(device)
        stats.total_params = sum(p.numel() for p in model.parameters())
        stats.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats.n_train = len(train_examples)
        stats.n_eval = len(eval_examples)
        stats.device = device
        stats.batch_size = bs

        # fp16 weights and an fp32-only optimiser do not mix; train in fp32 and
        # cast back afterwards. At 135M this costs little and avoids silent NaNs.
        orig_dtype = next(model.parameters()).dtype
        if orig_dtype == torch.float16:
            model.float()

        stats.eval_loss_before = self.eval_loss(model, eval_examples, persona, include_persona)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
        )
        t0 = time.perf_counter()
        model.train()
        losses: list[float] = []
        for ep in range(epochs):
            for ids, labels, mask in self._batches(
                train_examples, persona, include_persona, max(1, bs),
                shuffle=True, seed=self.cfg.general.seed + ep,
            ):
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits[:, :-1, :]
                tgt = labels[:, 1:]
                keep = tgt != -100
                if keep.sum() == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(logits[keep], tgt[keep])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                losses.append(float(loss.item()))
                stats.steps += 1
            stats.epochs = ep + 1
        stats.wall_s = time.perf_counter() - t0
        if losses:
            stats.train_loss = sum(losses[-20:]) / len(losses[-20:])
        if device == "cuda":
            stats.peak_mem_mb = torch.cuda.max_memory_allocated() / (1 << 20)
        stats.eval_loss_after = self.eval_loss(model, eval_examples, persona, include_persona)

        path: str | None = None
        if method == "lora":
            path = out_dir or os.path.join(self.adapters_dir, f"tmp-{int(time.time())}")
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            stats.artifact_mb = _dir_size_mb(path)
            with open(os.path.join(path, "streamlora_meta.json"), "w") as fh:
                json.dump({
                    "persona": persona.to_dict(), "stats": stats.as_dict(),
                    "base_model": self.cfg.language.model_id,
                    "include_persona_in_prompt": include_persona,
                }, fh, indent=2)
        else:
            # Measured, not written: the point of this arm is its cost, and
            # persisting a full checkpoint per run would consume disk for an
            # artefact that is never served.
            stats.artifact_mb = stats.total_params * 4 / (1 << 20)

        # Leave the serving backend exactly as it was found.
        if method == "lora":
            try:
                model.unload()
            except Exception:  # pragma: no cover - defensive
                log.warning("could not unload the training adapter")
        del model
        if orig_dtype == torch.float16:
            base.half()
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        return stats, path

    def adapt(
        self,
        train_examples: Sequence[LanguageExample],
        eval_examples: Sequence[LanguageExample],
        persona: PersonaSpec,
        now: float | None = None,
        run_id: str | None = None,
    ) -> AdapterInfo:
        """Train a candidate adapter and promote it only if it is not worse.

        Same safety shape as the forecaster: a candidate is measured on held-out
        data against the incumbent, and a regression is discarded rather than
        served. The metric is held-out cross-entropy on the answer tokens.
        """
        self._require()
        now = time.time() if now is None else now
        lc = self.cfg.language
        if len(train_examples) < lc.min_examples:
            return AdapterInfo(
                version="(none)", path="", created_ts=now, parent=None,
                stats=TrainStats(
                    method="lora", n_train=len(train_examples),
                    error=(f"only {len(train_examples)} examples; "
                           f"language.min_examples={lc.min_examples}"),
                ),
                persona_revision=persona.revision, promoted=False,
                gate_reason="insufficient_examples",
            )

        active = self.repos.models.active(ADAPTER_KIND, ADAPTER_SCOPE)
        parent = active["version"] if active else None
        n = self.repos.models.next_version_number(ADAPTER_KIND, ADAPTER_SCOPE)
        version = version_label(ADAPTER_KIND, n)
        out_dir = os.path.join(self.adapters_dir, version)
        stats, path = self.train(
            train_examples, eval_examples, persona, method="lora", out_dir=out_dir
        )
        info = AdapterInfo(
            version=version, path=path or "", created_ts=now, parent=parent, stats=stats,
            persona_revision=persona.revision,
        )
        if stats.error is not None or path is None:
            info.gate_reason = f"training_failed: {stats.error}"
            self._record(info, "failed", run_id, now)
            return info

        # Gate: the incumbent's held-out loss is the reference. On the first
        # adapter the reference is the base model, which is what
        # eval_loss_before measures.
        ref = stats.eval_loss_before
        cand = stats.eval_loss_after
        if math.isnan(cand):
            info.gate_reason = "candidate_unscorable"
        elif math.isnan(ref):
            info.promoted = True
            info.gate_reason = "no_reference"
        elif cand <= ref * (1.0 + lc.gate_tolerance):
            info.promoted = True
            info.gate_reason = "improved_or_equal"
        else:
            info.gate_reason = "worse_than_reference"

        self.repos.models.register(
            ADAPTER_KIND, ADAPTER_SCOPE, version, now, parent=parent, path=path,
            n_train=len(train_examples), metrics=stats.as_dict(),
            active=info.promoted, run_id=run_id,
        )
        if info.promoted:
            self.repos.models.activate(ADAPTER_KIND, ADAPTER_SCOPE, version, now)
        else:
            # Keep the rejected artefact: it is the evidence for why it was
            # rejected, and it is a few megabytes.
            pass
        self._record(info, "promoted" if info.promoted else "rejected", run_id, now)
        self._prune()
        return info

    def _record(self, info: AdapterInfo, decision: str, run_id: str | None, now: float) -> None:
        self.repos.events.add_adapt(
            ts=now, scope=ADAPTER_SCOPE, kind="language", trigger="feedback",
            decision=decision, active_version=info.parent, candidate_version=info.version,
            metric_name="eval_loss", metric_before=(
                None if math.isnan(info.stats.eval_loss_before) else info.stats.eval_loss_before
            ),
            metric_after=(
                None if math.isnan(info.stats.eval_loss_after) else info.stats.eval_loss_after
            ),
            gate_n=info.stats.n_eval, n_train=info.stats.n_train,
            duration_ms=info.stats.wall_s * 1000.0,
            detail={"gate_reason": info.gate_reason, **info.stats.as_dict()},
            run_id=run_id,
        )
        log.info(
            "language adapter decision", version=info.version, decision=decision,
            loss_before=info.stats.eval_loss_before, loss_after=info.stats.eval_loss_after,
            trainable=info.stats.trainable_params, wall_s=round(info.stats.wall_s, 2),
            reason=info.gate_reason,
        )

    def _prune(self) -> None:
        for v in self.repos.models.prune(
            ADAPTER_KIND, ADAPTER_SCOPE, self.cfg.adapt.keep_versions
        ):
            p = os.path.join(self.adapters_dir, v)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)

    def rollback(self, now: float | None = None, run_id: str | None = None) -> dict[str, Any] | None:
        """Reactivate the previous adapter.

        Correct by construction with LoRA: adapters are attached to an untouched
        base, so reverting is detaching one directory and attaching another --
        there is no merged state to invert.
        """
        now = time.time() if now is None else now
        hist = self.repos.models.history(ADAPTER_KIND, ADAPTER_SCOPE, limit=100)
        active = next((h for h in hist if h["active"]), None)
        if active is None:
            return None
        target = None
        if active.get("parent"):
            target = next((h for h in hist if h["version"] == active["parent"]), None)
        if target is None:
            target = next(
                (h for h in hist if h["version"] != active["version"] and h["path"]
                 and os.path.isdir(h["path"])), None
            )
        if target is None:
            return None
        self.repos.models.activate(ADAPTER_KIND, ADAPTER_SCOPE, target["version"], now)
        self.repos.events.add_adapt(
            ts=now, scope=ADAPTER_SCOPE, kind="language", trigger="manual",
            decision="rolled_back", active_version=active["version"],
            candidate_version=target["version"], metric_name="eval_loss",
            metric_before=None, metric_after=None, gate_n=0, n_train=0, duration_ms=0.0,
            detail={"reason": "adapter rollback"}, run_id=run_id,
        )
        return target

    def active_adapter_path(self) -> str | None:
        row = self.repos.models.active(ADAPTER_KIND, ADAPTER_SCOPE)
        if not row or not row.get("path"):
            return None
        return row["path"] if os.path.isdir(row["path"]) else None
