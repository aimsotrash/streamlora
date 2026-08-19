"""Language layer: grounding, verification, persona, deterministic rendering.

None of this needs a model. The forecasting pipeline and the interaction layer's
correctness guarantees both hold with the optional extra absent.
"""

from __future__ import annotations

import pytest

from streamlora.language.backends.template import TemplateBackend, TemplateRenderer
from streamlora.language.dataset import QUESTION_BANK, build_dataset
from streamlora.language.feedback import FeedbackStore
from streamlora.language.grounding import (
    Correlation,
    EvidencePack,
    Fact,
    ForecastFact,
    GroundingEngine,
    detect_focus_signal,
    detect_intent,
    fmt_duration,
)
from streamlora.language.persona import PersonaSpec, build_persona, infer_rule_from_text, sentence_count
from streamlora.language.service import LanguageService
from streamlora.language.verify import check_grounding, extract_numbers, verify
from streamlora.replay.player import ReplayPlayer
from streamlora.replay.synthetic import build_scenario, generate

from .conftest import T0


# --------------------------------------------------------------------------
# intent + focus routing
# --------------------------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("Why is my CPU usage rising?", "why_change"),
    ("Will my battery last another hour?", "battery"),
    ("Why did my battery drop so quickly?", "battery"),
    ("Why was the last forecast wrong?", "forecast_error"),
    ("How accurate have the forecasts been?", "forecast_error"),
    ("Was today's behaviour unusual?", "anomaly"),
    ("Has the model been adapting?", "adaptation"),
    ("What is my machine doing right now?", "status"),
    ("Is memory trending up?", "trend"),
])
def test_intent_routing(q, expected):
    assert detect_intent(q) == expected


def test_focus_signal_prefers_the_most_specific_match():
    avail = ["gpu.temp_c", "gpu.util_pct", "cpu.util_pct", "thermal.cpu_c"]
    assert detect_focus_signal("why is gpu temperature so high", avail) == "gpu.temp_c"
    assert detect_focus_signal("what is the cpu doing", avail) == "cpu.util_pct"
    assert detect_focus_signal("tell me about the weather", avail) is None


def test_duration_formatting_reads_naturally():
    assert fmt_duration(45) == "45 s"
    assert fmt_duration(300) == "5 min"
    assert fmt_duration(9000) == "2.5 h"


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def _pack():
    p = EvidencePack(now=T0, regime="build", focus_signal="battery.percent",
                     window_s=2520.0)
    p.current = [Fact("battery.percent", "battery 71.0%", 71.0, "percent")]
    p.window = [
        Fact("battery.percent.change", "battery changed -11.0%", -11.0, "percent"),
        Fact("battery.rate_pct_per_min", "-0.262 %/min", -0.262, "percent"),
        Fact("battery.time_to_20", "reaches 20% in 195 min", 195.0),
    ]
    p.forecasts = [ForecastFact("battery.percent", 1800.0, 66.0, 63.0, 69.0, "rls",
                                "forecast-model-v004", T0, T0 + 1800, "percent")]
    p.drivers = [Correlation("cpu.util_pct", 0.81, 60.0, 34.0, "percent")]
    return p


def test_number_extraction_handles_signs_decimals_and_percent():
    got = extract_numbers("Battery 71.0% now, down 11% in 42 min; likely 63.0-69.0%.")
    assert 71.0 in got and 42.0 in got and 63.0 in got


def test_grounded_answer_scores_one():
    p = _pack()
    text = ("Battery is 71.0% and falling 0.262%/min. Likely 63.0-69.0% in 30 minutes, "
            "correlated with CPU at 34.0%.")
    r = check_grounding(text, p)
    assert r.groundedness == 1.0 and r.fully_grounded


def test_invented_measurements_are_caught():
    p = _pack()
    r = check_grounding("Battery is 55.5% and will die in 12 minutes.", p)
    assert not r.fully_grounded
    assert 55.5 in r.ungrounded


def test_rounding_is_not_treated_as_hallucination():
    p = _pack()
    r = check_grounding("Battery is 71%.", p)
    assert r.fully_grounded


def test_style_checks_catch_each_violation():
    p = _pack()
    persona = PersonaSpec()
    persona.rules.append(infer_rule_from_text("compiling is normal for me"))
    bad = ("The battery situation is being monitored closely at this time. "
           "It is 71.0%. This is caused by the CPU. There is more to say. "
           "And more. And still more.")
    r = verify(bad, p, persona)
    v = set(r.style_violations)
    assert any(x.startswith("too_long") for x in v)
    assert "no_number_in_first_sentence" in v
    assert "asserted_causation" in v
    assert "missing_rule_word(normal)" in v
    assert r.style_compliance < 1.0


def test_correlation_hedging_is_accepted():
    p = _pack()
    persona = PersonaSpec()
    good = "71.0% battery, correlated with CPU at 34.0%, likely 63.0-69.0% soon in 30 minutes."
    r = verify(good, p, persona)
    assert "asserted_causation" not in r.style_violations


def test_battery_duration_rule_only_applies_when_there_is_a_projection():
    persona = PersonaSpec()
    p = EvidencePack(now=T0, focus_signal="battery.percent")
    p.current = [Fact("battery.percent", "battery 100.0%", 100.0, "percent")]
    r = verify("100.0% and charging.", p, persona)
    assert "battery_without_duration" not in r.style_violations


def test_sentence_count_survives_decimals():
    assert sentence_count("CPU is 48.2% now. It rose 11.4 points.") == 2


# --------------------------------------------------------------------------
# persona from feedback
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,regime,verdict", [
    ("That happens when I'm compiling projects. Treat that as normal for me.", "build", "normal"),
    ("This is normal, I game in the evenings", "gpu_heavy", "normal"),
    ("That shouldn't happen when unplugged", "battery_constrained", "concerning"),
])
def test_rules_are_inferred_from_natural_language(text, regime, verdict):
    r = infer_rule_from_text(text)
    assert r is not None and r.regime == regime and r.verdict == verdict


def test_unparseable_text_is_kept_as_context_not_guessed_into_a_rule(repos):
    assert infer_rule_from_text("I use this laptop for research") is None
    store = FeedbackStore(repos)
    store.add("note", text="I use this laptop for research", now=T0)
    p = store.persona()
    assert p.rules == []
    assert "I use this laptop for research" in p.context


def test_persona_revision_advances_with_feedback(repos):
    store = FeedbackStore(repos)
    base = store.persona().revision
    store.add("label", text="compiling is normal for me", now=T0)
    assert store.persona().revision > base


def test_usefulness_feedback_adjusts_verbosity(repos):
    store = FeedbackStore(repos)
    before = store.persona().style.max_sentences
    for i in range(6):
        store.add("forecast_useful", label="no", now=T0 + i)
    assert store.persona().style.max_sentences > before


def test_feedback_rejects_unknown_kinds(repos):
    store = FeedbackStore(repos)
    with pytest.raises(ValueError):
        store.add("nonsense", text="x")


def test_persona_round_trips(tmp_path):
    p = PersonaSpec()
    p.rules.append(infer_rule_from_text("compiling is normal for me"))
    p.context.append("ml research")
    path = str(tmp_path / "p.json")
    p.save(path)
    back = PersonaSpec.load(path)
    assert back.rules[0].regime == "build"
    assert back.context == ["ml research"]
    assert "compil" in back.prompt_block() or "normal" in back.prompt_block()


# --------------------------------------------------------------------------
# deterministic renderer
# --------------------------------------------------------------------------

def test_renderer_is_fully_grounded_and_compliant_by_construction():
    p = _pack()
    persona = PersonaSpec()
    persona.rules.append(infer_rule_from_text("compiling is normal for me"))
    text = TemplateRenderer(persona).render(p)
    r = verify(text, p, persona)
    assert r.fully_grounded, f"renderer emitted an ungrounded number: {r.ungrounded}"
    assert r.style_ok, f"renderer violated its own persona: {r.style_violations}"


def test_renderer_respects_the_sentence_budget():
    p = _pack()
    persona = PersonaSpec()
    persona.style.max_sentences = 2
    text = TemplateRenderer(persona).render(p)
    assert sentence_count(text) <= 2


def test_renderer_always_honours_a_user_rule():
    p = _pack()
    persona = PersonaSpec()
    persona.style.max_sentences = 2
    persona.rules.append(infer_rule_from_text("compiling is normal for me"))
    text = TemplateRenderer(persona).render(p)
    assert "normal" in text.lower()


def test_renderer_paraphrases_across_different_content():
    persona = PersonaSpec()
    r = TemplateRenderer(persona)
    outs = set()
    for v in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        p = EvidencePack(now=T0, intent="status")
        p.current = [Fact("cpu.util_pct", f"cpu {v}", v, "percent")]
        outs.add(r.render(p).split(".")[0].replace(str(v), "X"))
    assert len(outs) > 1, "renderer produced a single memorisable template"


def test_renderer_says_so_when_there_is_no_telemetry():
    text = TemplateRenderer(PersonaSpec()).render(EvidencePack(now=T0))
    assert "no telemetry" in text.lower()


# --------------------------------------------------------------------------
# end to end against a real pipeline
# --------------------------------------------------------------------------

@pytest.fixture
def populated(config, repos):
    samples, reg = generate(build_scenario("idle_to_build", minutes=90))
    repos.signals.register(reg.as_mapping().values())
    repos.telemetry.insert_samples(samples, run_id="lang")
    player = ReplayPlayer(config, repos, reg, run_id="lang")
    player.run(samples)
    return config, repos, samples[-1].ts


def test_evidence_pack_is_built_from_real_records(populated):
    config, repos, now = populated
    g = GroundingEngine(config, repos)
    pack = g.build("Why is my CPU usage rising?", now=now, window_s=1800.0)
    assert pack.current, "no current facts from a populated database"
    assert pack.intent == "why_change"
    assert pack.focus_signal == "cpu.util_pct"
    assert pack.numeric_values()
    rendered = pack.render()
    assert "NOW:" in rendered


def test_service_answers_are_grounded_without_any_model(populated):
    config, repos, now = populated
    svc = LanguageService(config, repos)
    for q in list(QUESTION_BANK)[:8]:
        a = svc.ask(q, now=now)
        assert a.text
        assert a.backend == "template"
        assert a.grounded, f"{q!r} produced ungrounded output: {a.text}"


def test_dataset_split_is_chronological(populated):
    config, repos, _now = populated
    ds = build_dataset(config, repos, n_examples=60, train_frac=0.7)
    assert ds.train and ds.test
    assert max(e.ts for e in ds.train) <= min(e.ts for e in ds.test)
    assert all(len(e.target) > 20 for e in ds.train)


def test_dataset_file_round_trip(populated, tmp_path):
    config, repos, _now = populated
    from streamlora.language.dataset import LanguageDataset

    ds = build_dataset(config, repos, n_examples=40)
    path = str(tmp_path / "ds.jsonl")
    ds.save(path)
    back = LanguageDataset.load(path)
    assert len(back.train) == len(ds.train)
    assert back.train[0].target == ds.train[0].target
    assert back.persona.revision == ds.persona.revision


def test_historical_comparison_reports_comparable_periods(populated):
    config, repos, now = populated
    g = GroundingEngine(config, repos)
    hor = config.forecast.horizons_s[0]
    # Find a regime that actually has enough resolved persistence rows.
    recs = repos.predictions.resolved(
        signal="cpu.util_pct", horizon_s=hor, model_kind="persistence"
    )
    assert recs
    from collections import Counter

    regime = Counter(r.regime for r in recs).most_common(1)[0][0]
    facts = g.historical_comparison("cpu.util_pct", regime, now, hor)
    assert facts, f"no comparable periods for regime {regime!r}"
    assert "comparable periods" in facts[0].text
    assert facts[0].value is not None


def test_historical_comparison_stays_silent_without_enough_history(populated):
    config, repos, now = populated
    g = GroundingEngine(config, repos)
    # A regime the machine was never in has no comparable periods, and an
    # authoritative claim built on zero samples is worse than saying nothing.
    assert g.historical_comparison(
        "cpu.util_pct", "battery_constrained", now, config.forecast.horizons_s[0]
    ) == []


def test_history_facts_are_grounded_when_rendered(populated):
    config, repos, now = populated
    g = GroundingEngine(config, repos)
    persona = PersonaSpec()
    r = TemplateRenderer(persona)
    for q in ("Why is my CPU usage rising?", "Is memory trending up?"):
        pack = g.build(q, now=now, window_s=1800.0)
        text = r.render(pack)
        rep = verify(text, pack, persona)
        assert rep.fully_grounded, f"{q}: {rep.ungrounded} in {text!r}"


def test_every_clause_talks_about_the_same_signal(populated):
    """A grounded answer can still misattribute a number to the wrong signal.

    The failure this guards: "CPU utilisation is 62.9%. In 5 min it is predicted
    at 99.5%" -- where 99.5% was the battery forecast. Both numbers are in the
    evidence, so groundedness passes; only structure catches it.
    """
    config, repos, now = populated
    g = GroundingEngine(config, repos)
    r = TemplateRenderer(PersonaSpec())
    for q in ("What is my machine doing right now?", "Give me a status summary.",
              "Why is my CPU usage rising?", "Is memory trending up?"):
        pack = g.build(q, now=now, window_s=1800.0)
        if not pack.forecasts or not pack.current:
            continue
        primary = r._primary_signal(pack)
        text = r.render(pack)
        for fc in pack.forecasts:
            if fc.signal == primary:
                continue
            # A forecast for a different signal must not be quoted as this
            # answer's forecast.
            other = f"{fc.value:.1f}"
            if other in text and fc.signal.split(".")[0] not in text.lower():
                raise AssertionError(
                    f"{q!r}: quoted {fc.signal}'s forecast {other} while talking "
                    f"about {primary}: {text!r}"
                )


def test_forecast_clause_is_omitted_when_the_signal_has_none(populated):
    config, repos, now = populated
    from streamlora.language.grounding import Fact, ForecastFact

    pack = EvidencePack(now=now, intent="status", focus_signal="cpu.util_pct")
    pack.current = [Fact("cpu.util_pct", "CPU utilisation: 40.0%", 40.0, "percent")]
    pack.forecasts = [ForecastFact("battery.percent", 300.0, 99.5, 99.4, 99.6,
                                   "rls", "v1", now, now + 300, "percent")]
    text = TemplateRenderer(PersonaSpec()).render(pack)
    assert "99.5" not in text, f"quoted another signal's forecast: {text!r}"
