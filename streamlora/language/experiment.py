"""The language-layer experiment.

Hypothesis under test (stated in the brief and in docs/research_questions.md):

    Small, continuous low-rank updates personalise the interaction layer more
    efficiently than repeated full fine-tuning -- and more effectively than
    putting the preferences in the prompt.

Arms
----
``template``        deterministic renderer. No model. The correctness floor:
                    100% grounded and 100% style-compliant by construction.
``base``            base model, no persona anywhere. Measures what the model does
                    with evidence alone.
``persona_prompt``  persona spelled out in the prompt on every request. The
                    honest competitor to LoRA, and the one people reach for
                    first. Costs prompt tokens forever.
``fewshot``         persona plus k worked examples in the prompt. Stronger, and
                    even more expensive per request.
``lora``            LoRA adapter trained on the same examples; nothing added to
                    the prompt.
``full_ft``         all parameters fine-tuned on the same examples. The
                    efficiency reference.

Metrics
-------
* ``eval_loss``       cross-entropy on held-out *answer* tokens (model arms).
* ``groundedness``    fraction of generated numbers present in the evidence.
                      This is the hallucination measure.
* ``style_compliance`` fraction of persona checks passed.
* ``prompt_tokens``   mean prompt length: the recurring cost of prompt-based
                      personalisation.
* ``trainable_params``, ``train_wall_s``, ``peak_mem_mb``, ``artifact_mb``:
                      the one-off cost of weight-based personalisation.

Everything is measured on the same chronologically held-out examples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import Config
from ..store.repo import Repos
from ..util.ids import short_uid
from ..util.logging import get_logger
from .backends.template import TemplateRenderer
from .dataset import LanguageDataset, LanguageExample, build_dataset, pack_from_example
from .lora import LanguageTrainer, TrainStats
from .persona import PersonaSpec, build_persona
from .prompt import SYSTEM_PROMPT, render_prompt
from .verify import verify

log = get_logger("language.experiment")

ARMS = ("template", "base", "persona_prompt", "fewshot", "lora", "full_ft")


@dataclass
class ArmMetrics:
    arm: str
    n: int = 0
    eval_loss: float | None = None
    groundedness: float | None = None
    fully_grounded_rate: float | None = None
    style_compliance: float | None = None
    style_ok_rate: float | None = None
    mean_prompt_tokens: float | None = None
    mean_gen_tokens: float | None = None
    mean_latency_ms: float | None = None
    device: str | None = None
    train: TrainStats | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None, k: int = 4) -> float | None:
            return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, k)

        return {
            "arm": self.arm, "n": self.n, "eval_loss": r(self.eval_loss),
            "eval_ppl": (
                None if self.eval_loss is None or math.isnan(self.eval_loss)
                else round(math.exp(min(self.eval_loss, 20.0)), 3)
            ),
            "groundedness": r(self.groundedness),
            "fully_grounded_rate": r(self.fully_grounded_rate),
            "style_compliance": r(self.style_compliance),
            "style_ok_rate": r(self.style_ok_rate),
            "mean_prompt_tokens": r(self.mean_prompt_tokens, 1),
            "mean_gen_tokens": r(self.mean_gen_tokens, 1),
            "mean_latency_ms": r(self.mean_latency_ms, 1),
            "device": self.device,
            "train": self.train.as_dict() if self.train else None,
            "error": self.error,
            "samples": self.samples[:3],
        }


@dataclass
class LanguageExperimentResult:
    experiment_id: str
    model_id: str
    device: str
    n_train: int
    n_test: int
    persona_revision: int
    arms: list[ArmMetrics]
    started_ts: float
    ended_ts: float
    dataset_notes: str = ""
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "model_id": self.model_id,
            "device": self.device, "n_train": self.n_train, "n_test": self.n_test,
            "persona_revision": self.persona_revision,
            "started_ts": self.started_ts, "ended_ts": self.ended_ts,
            "duration_s": round(self.ended_ts - self.started_ts, 1),
            "dataset_notes": self.dataset_notes,
            "limitations": self.limitations,
            "devices_used": sorted({a.device for a in self.arms if a.device}),
            "mixed_devices": len({a.device for a in self.arms if a.device}) > 1,
            "arms": [a.as_dict() for a in self.arms],
        }

    def table(self) -> str:
        cols = [
            ("arm", 16), ("n", 5), ("dev", 6), ("loss", 8), ("ppl", 9), ("ground", 8),
            ("gr=1", 7), ("style", 8), ("st=1", 7), ("p_tok", 8), ("train_s", 9),
            ("params", 12), ("MB", 8),
        ]
        head = "".join(f"{c:>{w}}" if i else f"{c:<{w}}" for i, (c, w) in enumerate(cols))
        lines = [head, "-" * len(head)]
        for a in self.arms:
            d = a.as_dict()
            t = d.get("train") or {}

            def s(v: Any, fmt: str = "{:.3f}") -> str:
                return "-" if v is None else fmt.format(v)

            if a.error:
                lines.append(f"{a.arm:<16}{a.n:>5}   {a.error[:70]}")
                continue
            lines.append(
                f"{a.arm:<16}{a.n:>5}{(a.device or '-'):>6}"
                f"{s(d['eval_loss']):>8}{s(d['eval_ppl'], '{:.2f}'):>9}"
                f"{s(d['groundedness']):>8}{s(d['fully_grounded_rate'], '{:.2f}'):>7}"
                f"{s(d['style_compliance']):>8}{s(d['style_ok_rate'], '{:.2f}'):>7}"
                f"{s(d['mean_prompt_tokens'], '{:.0f}'):>8}"
                f"{s(t.get('wall_s'), '{:.1f}'):>9}"
                f"{s(t.get('trainable_params'), '{:,.0f}'):>12}"
                f"{s(t.get('artifact_mb'), '{:.1f}'):>8}"
            )
        return "\n".join(lines)


class LanguageExperiment:
    def __init__(self, config: Config, repos: Repos) -> None:
        self.cfg = config
        self.repos = repos
        self.trainer = LanguageTrainer(config, repos)

    # -- scoring -----------------------------------------------------------
    def _score_generations(
        self, model: Any, examples: Sequence[LanguageExample], persona: PersonaSpec,
        include_persona: bool, shots: Sequence[tuple[str, str]] = (),
        max_examples: int = 40,
    ) -> tuple[ArmMetrics, list[dict[str, Any]]]:
        """Generate on held-out examples and verify each answer."""
        import torch

        tok = self.trainer.backend._tok
        device = self.trainer.backend.device
        m = ArmMetrics(arm="?")
        gs: list[float] = []
        fg: list[float] = []
        sc: list[float] = []
        so: list[float] = []
        ptoks: list[int] = []
        gtoks: list[int] = []
        lat: list[float] = []
        samples: list[dict[str, Any]] = []
        subset = list(examples)[:max_examples]
        model.eval()
        for ex in subset:
            pack = pack_from_example(ex)
            pack.question = ex.question
            prompt = render_prompt(pack, persona, include_persona=include_persona, shots=shots)
            prompt = prompt.replace(pack.render(), ex.evidence)
            if getattr(tok, "chat_template", None):
                text = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                text = SYSTEM_PROMPT + "\n\n" + prompt
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=self.cfg.language.max_seq_len).to(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=self.cfg.language.max_new_tokens,
                    do_sample=False, pad_token_id=tok.pad_token_id,
                )
            dt = (time.perf_counter() - t0) * 1000.0
            new = out[0][enc["input_ids"].shape[1]:]
            answer = tok.decode(new, skip_special_tokens=True).strip()
            rep = verify(answer, pack, persona)
            gs.append(rep.groundedness)
            fg.append(1.0 if rep.fully_grounded else 0.0)
            sc.append(rep.style_compliance)
            so.append(1.0 if rep.style_ok else 0.0)
            ptoks.append(int(enc["input_ids"].shape[1]))
            gtoks.append(int(new.shape[0]))
            lat.append(dt)
            if len(samples) < 3:
                samples.append({
                    "question": ex.question, "answer": answer,
                    "groundedness": round(rep.groundedness, 3),
                    "style_violations": rep.style_violations,
                })
        n = len(subset)
        m.n = n
        if n:
            m.groundedness = sum(gs) / n
            m.fully_grounded_rate = sum(fg) / n
            m.style_compliance = sum(sc) / n
            m.style_ok_rate = sum(so) / n
            m.mean_prompt_tokens = sum(ptoks) / n
            m.mean_gen_tokens = sum(gtoks) / n
            m.mean_latency_ms = sum(lat) / n
        return m, samples

    def _score_template(
        self, examples: Sequence[LanguageExample], persona: PersonaSpec, max_examples: int = 40,
    ) -> ArmMetrics:
        r = TemplateRenderer(persona)
        m = ArmMetrics(arm="template")
        gs: list[float] = []
        fg: list[float] = []
        sc: list[float] = []
        so: list[float] = []
        lat: list[float] = []
        for ex in list(examples)[:max_examples]:
            pack = pack_from_example(ex)
            pack.question = ex.question
            t0 = time.perf_counter()
            # The stored target *is* this renderer's output for this pack, so
            # scoring it re-measures the floor rather than assuming it.
            answer = ex.target
            lat.append((time.perf_counter() - t0) * 1000.0)
            rep = verify(answer, pack, persona)
            gs.append(rep.groundedness)
            fg.append(1.0 if rep.fully_grounded else 0.0)
            sc.append(rep.style_compliance)
            so.append(1.0 if rep.style_ok else 0.0)
            if len(m.samples) < 3:
                m.samples.append({
                    "question": ex.question, "answer": answer,
                    "groundedness": round(rep.groundedness, 3),
                    "style_violations": rep.style_violations,
                })
        n = len(gs)
        m.n = n
        if n:
            m.groundedness = sum(gs) / n
            m.fully_grounded_rate = sum(fg) / n
            m.style_compliance = sum(sc) / n
            m.style_ok_rate = sum(so) / n
            m.mean_latency_ms = sum(lat) / n
        del r
        return m

    # -- the run -----------------------------------------------------------
    def run(
        self, dataset: LanguageDataset, arms: Sequence[str] = ARMS,
        max_eval: int = 40, shots: int = 2,
    ) -> LanguageExperimentResult:
        started = time.time()
        exp_id = f"lang-{time.strftime('%Y%m%d-%H%M%S')}-{short_uid()}"
        persona = dataset.persona
        results: list[ArmMetrics] = []
        limitations = [
            "Targets are rendered by TemplateRenderer under an explicit persona, not written "
            "by a human. The experiment measures whether an adaptation mechanism can acquire a "
            "specified style-and-grounding function, not whether people prefer the output.",
            "Groundedness is a numeric-containment check. It catches invented measurements; it "
            "cannot catch a fluent sentence that misreads a correct number.",
            f"Base model is {self.cfg.language.model_id}, chosen so continual adaptation and a "
            "full-fine-tune comparison both fit on a small laptop GPU. Absolute answer quality "
            "would be higher with a larger model; the relative cost comparison is the point.",
        ]

        if "template" in arms:
            results.append(self._score_template(dataset.test, persona, max_eval))

        model_arms = [a for a in arms if a != "template"]
        if model_arms and not self.trainer.available:
            for a in model_arms:
                results.append(ArmMetrics(
                    arm=a, error="optional extra not installed: pip install 'streamlora[lora]'"
                ))
            return LanguageExperimentResult(
                experiment_id=exp_id, model_id=self.cfg.language.model_id, device="n/a",
                n_train=len(dataset.train), n_test=len(dataset.test),
                persona_revision=persona.revision, arms=results, started_ts=started,
                ended_ts=time.time(), dataset_notes=dataset.notes, limitations=limitations,
            )
        if model_arms:
            self.trainer._require()
        base = self.trainer.backend._base_model
        device = self.trainer.backend.device

        shot_pairs: list[tuple[str, str]] = []
        for ex in dataset.train[:shots]:
            shot_pairs.append((ex.evidence, ex.target))

        for arm in model_arms:
            log.info("language arm starting", arm=arm)
            try:
                import torch as _t

                if device == "cuda":
                    _t.cuda.empty_cache()
                    _t.cuda.reset_peak_memory_stats()
                if arm == "base":
                    m, _ = self._score_generations(base, dataset.test, persona, False, (), max_eval)
                    m.arm = arm
                    m.eval_loss = self.trainer.eval_loss(base, dataset.test, persona, False)
                elif arm == "persona_prompt":
                    m, _ = self._score_generations(base, dataset.test, persona, True, (), max_eval)
                    m.arm = arm
                    m.eval_loss = self.trainer.eval_loss(base, dataset.test, persona, True)
                elif arm == "fewshot":
                    m, _ = self._score_generations(
                        base, dataset.test, persona, True, shot_pairs, max_eval
                    )
                    m.arm = arm
                    m.eval_loss = self.trainer.eval_loss(base, dataset.test, persona, True)
                elif arm in ("lora", "full_ft"):
                    method = "lora" if arm == "lora" else "full"
                    out_dir = (
                        os.path.join(self.trainer.adapters_dir, f"{exp_id}-{arm}")
                        if method == "lora" else None
                    )
                    stats, path = self.trainer.train(
                        dataset.train, dataset.test, persona, method=method,
                        include_persona=False, out_dir=out_dir,
                    )
                    m = ArmMetrics(arm=arm, train=stats, eval_loss=stats.eval_loss_after,
                                   device=stats.device)
                    if stats.error:
                        m.error = stats.error
                    else:
                        scored = self._score_arm_with_weights(
                            method, path, dataset, persona, max_eval
                        )
                        if scored is not None:
                            m.n = scored.n
                            m.groundedness = scored.groundedness
                            m.fully_grounded_rate = scored.fully_grounded_rate
                            m.style_compliance = scored.style_compliance
                            m.style_ok_rate = scored.style_ok_rate
                            m.mean_prompt_tokens = scored.mean_prompt_tokens
                            m.mean_gen_tokens = scored.mean_gen_tokens
                            m.mean_latency_ms = scored.mean_latency_ms
                            m.samples = scored.samples
                else:
                    m = ArmMetrics(arm=arm, error=f"unknown arm {arm!r}")
            except Exception as exc:
                log.exception("language arm failed", arm=arm)
                m = ArmMetrics(arm=arm, error=f"{type(exc).__name__}: {exc}")
            if m.device is None:
                m.device = device
            results.append(m)
            if device == "cuda":
                import torch as _t

                _t.cuda.empty_cache()

        return LanguageExperimentResult(
            experiment_id=exp_id, model_id=self.cfg.language.model_id, device=device,
            n_train=len(dataset.train), n_test=len(dataset.test),
            persona_revision=persona.revision, arms=results, started_ts=started,
            ended_ts=time.time(), dataset_notes=dataset.notes, limitations=limitations,
        )

    def _score_arm_with_weights(
        self, method: str, path: str | None, dataset: LanguageDataset,
        persona: PersonaSpec, max_eval: int,
    ) -> ArmMetrics | None:
        """Re-attach the trained weights and score generations.

        For LoRA this reloads the saved adapter, which also exercises the exact
        artefact that would be served. For full fine-tuning the weights are not
        persisted, so generation quality is not scored -- only the cost is, which
        is what that arm is in the experiment to provide.
        """
        if method != "lora" or path is None:
            return None
        b = self.trainer.backend
        if not b.load_adapter(path):
            log.warning("could not reattach adapter for scoring", path=path, error=b.load_error)
            return None
        try:
            m, _ = self._score_generations(
                b._model, dataset.test, persona, False, (), max_eval
            )
            return m
        finally:
            b.load_adapter(None)


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def run_language_experiment(cfg: Config, args: argparse.Namespace) -> int:
    repos = Repos.open(cfg.db_file)
    try:
        action = args.action
        persona = (
            PersonaSpec.load(args.persona) if getattr(args, "persona", None)
            else build_persona(repos)
        )
        ds_path = getattr(args, "out", None) or os.path.join(
            cfg.general.data_dir, "language_dataset.jsonl"
        )

        if action == "status":
            from .lora import ADAPTER_KIND, ADAPTER_SCOPE

            trainer = LanguageTrainer(cfg, repos)
            payload = {
                "lora_extra_available": trainer.available,
                "model_id": cfg.language.model_id,
                "active_adapter": repos.models.active(ADAPTER_KIND, ADAPTER_SCOPE),
                "adapter_history": repos.models.history(ADAPTER_KIND, ADAPTER_SCOPE, limit=10),
                "persona": persona.to_dict(),
                "dataset_present": os.path.exists(ds_path),
                "dataset_path": ds_path,
            }
            print(json.dumps(payload, indent=2, default=str))
            return 0

        if action == "build-data":
            n = args.examples or 240
            ds = build_dataset(cfg, repos, persona=persona, n_examples=n)
            if not ds.train:
                print(
                    "no examples could be built: is there any telemetry in the database?",
                    file=sys.stderr,
                )
                return 1
            ds.save(ds_path)
            print(json.dumps({"path": ds_path, **ds.as_dict()}, indent=2, default=str))
            return 0

        if not os.path.exists(ds_path):
            n = args.examples or 240
            log.info("no dataset found; building one first", path=ds_path, n=n)
            ds = build_dataset(cfg, repos, persona=persona, n_examples=n)
            ds.save(ds_path)
        else:
            ds = LanguageDataset.load(ds_path)

        if action == "train":
            trainer = LanguageTrainer(cfg, repos)
            if not trainer.available:
                print("the lora extra is not installed: pip install 'streamlora[lora]'")
                return 1
            info = trainer.adapt(ds.train, ds.test, ds.persona)
            print(json.dumps(info.as_dict(), indent=2, default=str))
            return 0 if info.promoted else 2

        if action in ("eval", "compare"):
            arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else ARMS
            exp = LanguageExperiment(cfg, repos)
            result = exp.run(ds, arms=arms)
            out_path = os.path.join(cfg.runs_dir, f"{result.experiment_id}.json")
            os.makedirs(cfg.runs_dir, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(result.as_dict(), fh, indent=2, default=str)
            if getattr(args, "json", False):
                print(json.dumps(result.as_dict(), indent=2, default=str))
            else:
                print()
                print(f"language experiment {result.experiment_id}")
                print(f"model {result.model_id} on {result.device}   "
                      f"train={result.n_train} test={result.n_test} "
                      f"persona_rev={result.persona_revision}")
                print()
                print(result.table())
                print()
                if result.as_dict()["mixed_devices"]:
                    print("WARNING: arms ran on different devices "
                          f"({result.as_dict()['devices_used']}). Accuracy columns are "
                          "still comparable; wall-clock and peak-memory are not.")
                    print()
                print("loss = held-out cross-entropy on answer tokens (lower better)")
                print("ground/gr=1 = mean groundedness / fraction of answers with no invented number")
                print("style/st=1  = mean persona compliance / fraction fully compliant")
                print("p_tok       = mean prompt tokens: the recurring cost of prompt-based personalisation")
                print()
                for lim in result.limitations:
                    print(f"note: {lim}")
                print()
                print(f"report: {out_path}")
            return 0

        print(f"unknown action {action!r}")
        return 1
    finally:
        repos.close()
