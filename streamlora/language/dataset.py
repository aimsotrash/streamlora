"""Building the language layer's supervised dataset.

An example is (evidence pack, question) -> target explanation. Evidence comes
from real stored records; the target is rendered by ``TemplateRenderer`` under an
explicit persona.

**Stated limitation.** There is no corpus of human-written explanations of this
laptop's telemetry, so targets are machine-rendered from a persona spec. The
language experiment therefore measures whether a low-rank update can acquire a
*specified* style-and-grounding function from a modest number of examples, and at
what cost relative to full fine-tuning. It does not measure human preference.
That would need human labels this project does not have, and claiming otherwise
would be exactly the kind of unfounded result the brief rules out.

What keeps the task non-trivial:

* Evidence packs are drawn from many different times, regimes and questions, so
  the model sees genuinely varied inputs.
* The renderer paraphrases, so there is no single string to memorise.
* The split is **chronological**, matching how the system would actually acquire
  examples over time.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

from ..config import Config
from ..store.repo import Repos
from .backends.template import TemplateRenderer
from .grounding import EvidencePack, GroundingEngine
from .persona import PersonaSpec

#: The question bank. Drawn from the interaction examples in the brief plus
#: paraphrases, so the model is not trained on one phrasing per intent.
QUESTION_BANK: tuple[str, ...] = (
    "Why is my CPU usage rising?",
    "Why did my CPU usage change?",
    "What is driving CPU load right now?",
    "Will my battery last another hour?",
    "How long will the battery last?",
    "Why did my battery drop so quickly?",
    "What is my machine doing right now?",
    "Give me a status summary.",
    "Was today's behaviour unusual?",
    "Is anything abnormal right now?",
    "Why was the last forecast wrong?",
    "How accurate have the forecasts been?",
    "What does my machine usually do when I start compiling?",
    "What changed compared with earlier?",
    "Is memory usage trending up?",
    "What is the memory forecast?",
    "Is the machine about to get hot?",
    "Has the model been adapting?",
    "What did the model learn recently?",
    "Is the GPU busy?",
)


@dataclass
class LanguageExample:
    ts: float
    question: str
    intent: str
    regime: str
    evidence: str
    target: str
    #: Numeric surface of the pack, kept so groundedness can be scored offline
    #: without rebuilding the pack from a database that may have moved on.
    surface: list[float] = field(default_factory=list)
    focus_signal: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "LanguageExample":
        return cls(
            ts=float(d["ts"]), question=str(d["question"]), intent=str(d["intent"]),
            regime=str(d["regime"]), evidence=str(d["evidence"]), target=str(d["target"]),
            surface=[float(x) for x in (d.get("surface") or [])],
            focus_signal=d.get("focus_signal"),  # type: ignore[arg-type]
        )


@dataclass
class LanguageDataset:
    persona: PersonaSpec
    train: list[LanguageExample] = field(default_factory=list)
    test: list[LanguageExample] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "persona": self.persona.to_dict(), "notes": self.notes,
            "n_train": len(self.train), "n_test": len(self.test),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(json.dumps({
                "format": 1, "persona": self.persona.to_dict(), "notes": self.notes,
            }) + "\n")
            for split, rows in (("train", self.train), ("test", self.test)):
                for r in rows:
                    fh.write(json.dumps({"split": split, **r.as_dict()}) + "\n")

    @classmethod
    def load(cls, path: str) -> "LanguageDataset":
        with open(path) as fh:
            header = json.loads(fh.readline())
            ds = cls(
                persona=PersonaSpec.from_dict(header.get("persona", {})),
                notes=header.get("notes", ""),
            )
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                split = row.pop("split", "train")
                ex = LanguageExample.from_dict(row)
                (ds.train if split == "train" else ds.test).append(ex)
        return ds


def sample_timestamps(
    repos: Repos, n: int, run_id: str | None = None, min_gap_s: float = 60.0
) -> list[float]:
    """Timestamps with enough history behind them to build a real pack.

    Spaced by ``min_gap_s`` so consecutive examples are not near-duplicates:
    two packs five seconds apart contain the same numbers, and training on both
    inflates the dataset without adding information.
    """
    rng = repos.telemetry.time_range(run_id=run_id)
    if rng is None:
        return []
    lo, hi = rng
    # Skip the first 20 minutes: packs built there have no window to summarise.
    lo = min(lo + 1200.0, lo + (hi - lo) * 0.25)
    if hi <= lo:
        return []
    span = hi - lo
    step = max(min_gap_s, span / max(n, 1))
    out: list[float] = []
    t = lo
    while t <= hi and len(out) < n:
        out.append(t)
        t += step
    return out


def build_dataset(
    config: Config,
    repos: Repos,
    persona: PersonaSpec | None = None,
    n_examples: int = 240,
    run_id: str | None = None,
    train_frac: float = 0.7,
    window_s: float = 1800.0,
    seed: int = 1337,
    questions: Sequence[str] = QUESTION_BANK,
) -> LanguageDataset:
    persona = persona or PersonaSpec()
    engine = GroundingEngine(config, repos)
    renderer = TemplateRenderer(persona)
    rng = random.Random(seed)
    stamps = sample_timestamps(repos, n_examples, run_id=run_id)
    rows: list[LanguageExample] = []
    for ts in stamps:
        q = questions[rng.randrange(len(questions))]
        pack = engine.build(q, now=ts, window_s=window_s)
        if not pack.current:
            continue
        target = renderer.render(pack)
        if not target or len(target) < 20:
            continue
        rows.append(LanguageExample(
            ts=ts, question=q, intent=pack.intent, regime=pack.regime,
            evidence=pack.render(), target=target, surface=pack.numeric_values(),
            focus_signal=pack.focus_signal,
        ))
    rows.sort(key=lambda r: r.ts)
    cut = int(len(rows) * train_frac)
    return LanguageDataset(
        persona=persona, train=rows[:cut], test=rows[cut:],
        notes=(f"chronological split at {train_frac:.0%}; targets rendered by "
               f"TemplateRenderer under persona '{persona.name}' rev {persona.revision}"),
    )


def pack_from_example(ex: LanguageExample) -> EvidencePack:
    """Rebuild a minimal pack for verification, from the stored surface.

    Only the fields the verifier reads are populated; this is a scoring shim, not
    a real pack.
    """
    from .grounding import Fact

    pack = EvidencePack(
        now=ex.ts, question=ex.question, intent=ex.intent, regime=ex.regime,
        focus_signal=ex.focus_signal,
    )
    pack.current = [Fact(key="surface", text="", value=v) for v in ex.surface]
    # The verifier's style checks look at whether forecasts/drivers were present.
    if "FORECASTS:" in ex.evidence:
        from .grounding import ForecastFact

        pack.forecasts = [ForecastFact("", 0.0, 0.0, None, None, "", "", 0.0, 0.0)]
    if "CORRELATED ACTIVITY:" in ex.evidence:
        from .grounding import Correlation

        pack.drivers = [Correlation("", 0.0, 0.0, 0.0)]
    if "battery.time_to_" in ex.evidence or "until" in ex.evidence:
        from .grounding import Fact as F

        pack.window = [F(key="battery.time_to_20", text="", value=0.0)]
    return pack
