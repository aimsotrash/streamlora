"""LoRA adaptation layer.

Everything here needs the optional ``lora`` extra and a model download, so the
whole module skips cleanly when either is absent -- which is itself part of the
contract: the forecasting system must run without any of this.
"""

from __future__ import annotations

import os

import pytest

from streamlora.language.dataset import LanguageExample, build_dataset
from streamlora.language.lora import ADAPTER_KIND, ADAPTER_SCOPE, LanguageTrainer
from streamlora.language.persona import PersonaSpec, infer_rule_from_text
from streamlora.replay.player import ReplayPlayer
from streamlora.replay.synthetic import build_scenario, generate

pytestmark = pytest.mark.lora


def _extra_available() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


requires_extra = pytest.mark.skipif(
    not _extra_available(), reason="optional extra not installed: pip install 'streamlora[lora]'"
)
requires_model = pytest.mark.skipif(
    os.environ.get("STREAMLORA_SKIP_MODEL_TESTS") == "1",
    reason="model download disabled by STREAMLORA_SKIP_MODEL_TESTS",
)


def test_trainer_reports_unavailability_instead_of_crashing(config, repos):
    """The contract when the extra is missing: a clear message, not a traceback."""
    t = LanguageTrainer(config, repos)
    if t.available:
        pytest.skip("extra is installed; this path is only reachable without it")
    with pytest.raises(RuntimeError, match="lora"):
        t._require()


@pytest.fixture
def dataset(config, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=120))
    repos.signals.register(reg.as_mapping().values())
    repos.telemetry.insert_samples(samples, run_id="lora")
    ReplayPlayer(config, repos, reg, run_id="lora").run(samples)
    persona = PersonaSpec()
    persona.rules.append(infer_rule_from_text("compiling is normal for me"))
    ds = build_dataset(config, repos, persona=persona, n_examples=60, train_frac=0.7)
    assert len(ds.train) >= 20
    return ds


@requires_extra
@requires_model
def test_lora_trains_and_lowers_held_out_loss(config, repos, dataset, tmp_path):
    config.language.epochs = 2
    config.language.batch_size = 2
    t = LanguageTrainer(config, repos)
    stats, path = t.train(dataset.train, dataset.test, dataset.persona,
                          method="lora", out_dir=str(tmp_path / "ad"))
    assert stats.error is None, stats.error
    assert path and os.path.isdir(path)
    assert stats.steps > 0
    # The headline efficiency claim: a small fraction of parameters is trained.
    assert 0.0 < stats.trainable_fraction < 0.10
    assert stats.eval_loss_after < stats.eval_loss_before, (
        f"loss did not improve: {stats.eval_loss_before} -> {stats.eval_loss_after}"
    )
    assert stats.artifact_mb < 60.0


@requires_extra
@requires_model
def test_adapter_is_versioned_gated_and_rollback_restores_it(config, repos, dataset):
    config.language.epochs = 1
    config.language.batch_size = 2
    config.language.min_examples = 5
    t = LanguageTrainer(config, repos)

    first = t.adapt(dataset.train, dataset.test, dataset.persona)
    assert first.version.endswith("v001")
    assert first.promoted, first.gate_reason
    assert repos.models.active(ADAPTER_KIND, ADAPTER_SCOPE)["version"] == first.version

    second = t.adapt(dataset.train, dataset.test, dataset.persona)
    assert second.version.endswith("v002")
    assert second.parent == first.version

    if second.promoted:
        back = t.rollback()
        assert back is not None and back["version"] == first.version
        assert repos.models.active(ADAPTER_KIND, ADAPTER_SCOPE)["version"] == first.version

    events = repos.events.adapt_events(kind="language")
    assert events
    assert all(e["metric_name"] == "eval_loss" for e in events)


@requires_extra
@requires_model
def test_insufficient_examples_is_reported_not_attempted(config, repos, dataset):
    config.language.min_examples = 10_000
    t = LanguageTrainer(config, repos)
    info = t.adapt(dataset.train, dataset.test, dataset.persona)
    assert not info.promoted
    assert info.gate_reason == "insufficient_examples"


@requires_extra
@requires_model
def test_attaching_and_detaching_an_adapter_is_exact(config, repos, dataset, tmp_path):
    """LoRA rollback is exact because detaching restores the base bitwise.

    Asserted on the weights, not on a reproduced loss: a CUDA reduction is not
    bit-reproducible, so a loss comparison would be a weaker claim than the one
    the design actually makes.
    """
    import torch

    config.language.epochs = 1
    config.language.batch_size = 2
    t = LanguageTrainer(config, repos)
    assert t.backend.load()
    before = {k: v.detach().clone() for k, v in t.backend._base_model.state_dict().items()}

    stats, path = t.train(dataset.train, dataset.test, dataset.persona,
                          method="lora", out_dir=str(tmp_path / "ad2"))
    assert stats.error is None
    assert t.backend.load_adapter(path)
    assert t.backend.adapter == path
    assert t.backend._model is not t.backend._base_model

    assert t.backend.load_adapter(None)
    assert t.backend.adapter is None
    assert t.backend._model is t.backend._base_model
    after = t.backend._base_model.state_dict()
    assert set(after) == set(before), "detach left the base model wrapped"
    for k in before:
        assert torch.equal(before[k], after[k]), f"detach changed {k}"


@requires_extra
@requires_model
def test_adapters_do_not_stack_across_attaches(config, repos, dataset, tmp_path):
    import torch

    config.language.epochs = 1
    config.language.batch_size = 2
    t = LanguageTrainer(config, repos)
    assert t.backend.load()
    stats, path = t.train(dataset.train, dataset.test, dataset.persona,
                          method="lora", out_dir=str(tmp_path / "ad3"))
    assert stats.error is None
    assert t.backend.load_adapter(path)
    once = t.eval_loss(t.backend._model, dataset.test, dataset.persona)
    assert t.backend.load_adapter(path)          # attach again without detaching
    twice = t.eval_loss(t.backend._model, dataset.test, dataset.persona)
    assert twice == pytest.approx(once, rel=1e-3), "adapter was applied twice"


@requires_extra
@requires_model
def test_prompt_tokens_are_masked_out_of_the_loss(config, repos, dataset):
    """Without masking, most of the loss is the evidence block, not the answer."""
    import torch

    t = LanguageTrainer(config, repos)
    t._require()
    ids, labels = t._encode(dataset.train[0], dataset.persona, include_persona=False)
    assert ids.numel() == labels.numel()
    masked = int((labels == -100).sum())
    assert masked > 0
    assert masked < labels.numel(), "everything was masked; nothing to learn from"
    # The answer is a small tail of a long prompt.
    assert masked / labels.numel() > 0.5


@requires_extra
@requires_model
def test_generation_is_verified_against_the_evidence(config, repos, dataset):
    from streamlora.language.backends.hf_local import HFLocalBackend
    from streamlora.language.service import LanguageService

    b = HFLocalBackend(model_id=config.language.model_id, max_seq_len=512)
    if not b.load():
        pytest.skip(f"model unavailable: {b.load_error}")
    config.language.backend = "hf"
    svc = LanguageService(config, repos, backend=b)
    now = max(e.ts for e in dataset.test)
    a = svc.ask("What is my machine doing right now?", now=now)
    assert a.text
    # Whatever the model said, the served answer must be grounded: an ungrounded
    # generation is replaced by the deterministic renderer.
    assert a.grounded
    if a.fallback_reason == "ungrounded_numbers":
        assert a.backend == "template"
    b.close()


@requires_extra
@requires_model
def test_cpu_fallback_does_not_stick_across_arms(config, repos, dataset, monkeypatch):
    """One arm that cannot fit must not drag every later arm onto CPU.

    A cost comparison that silently mixes devices measures nothing, so the
    device preference is restored after a fallback.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; the stickiness only matters when a GPU exists")
    config.language.epochs = 1
    config.language.batch_size = 1
    t = LanguageTrainer(config, repos)
    assert t.backend.load()
    assert t.backend.device == "cuda"

    calls = {"n": 0}
    real_inner = t._train_inner

    def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real_inner(*args, **kwargs)

    monkeypatch.setattr(t, "_train_inner", fail_once)
    stats, _path = t.train(dataset.train[:8], dataset.test[:4], dataset.persona,
                           method="lora")
    assert stats.device == "cpu", "the fallback did not actually move to CPU"
    # The preference must be restored so the next arm tries the GPU again.
    assert t.backend._requested_device != "cpu"
    assert t.backend.device == "cuda"
