"""Configuration tree.

Plain dataclasses, loadable from TOML (stdlib ``tomllib``) or JSON, with
dotted-path overrides from the CLI. Two properties matter:

* **Serialisable.** ``to_dict()`` round-trips, so the exact configuration of
  every experiment is stored next to its results and hashed into a run id.
  Reproducibility is a storage problem before it is a modelling problem.
* **No hidden defaults.** Everything a model or policy depends on lives here,
  not as a literal buried in a function.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]


@dataclass
class GeneralConfig:
    data_dir: str = "data"
    #: SQLite file, relative to data_dir unless absolute.
    db_path: str = "streamlora.sqlite"
    log_level: str = "info"
    log_format: str = "human"
    log_file: str | None = None
    seed: int = 1337


@dataclass
class CollectConfig:
    #: Sampling cadence. 5 s is a deliberate compromise: fast enough that a
    #: build ramp is several samples wide, slow enough that per-tick cost
    #: (~45 ms dominated by process enumeration) stays under 1% of a core.
    interval_s: float = 5.0
    #: None means "probe everything available".
    sources: list[str] | None = None
    #: Gap larger than this multiple of interval_s is a discontinuity.
    gap_factor: float = 3.0
    #: Samples buffered before a single DB transaction. Batching keeps SQLite
    #: off the hot path without risking more than this many samples on crash.
    write_batch: int = 6
    #: Include process-category attribution (aggregate only, never names).
    process_attribution: bool = True


@dataclass
class FeatureConfig:
    #: Length of history summarised into one feature vector.
    window_s: float = 300.0
    #: Lag offsets, in seconds back from now, sampled into the vector.
    lags_s: list[float] = field(default_factory=lambda: [0.0, 15.0, 30.0, 60.0, 120.0, 300.0])
    #: Rolling-aggregate spans, in seconds.
    agg_spans_s: list[float] = field(default_factory=lambda: [30.0, 120.0, 300.0])
    #: Signals used as model inputs. None means "every available signal".
    inputs: list[str] | None = None
    #: Include hour-of-day / day-of-week cyclical encodings.
    time_of_day: bool = True
    #: Include the one-hot regime encoding and regime x lag interactions.
    regime_features: bool = True
    #: Minimum fraction of the window that must contain usable samples.
    min_coverage: float = 0.5


@dataclass
class ForecastConfig:
    #: Signals to forecast.
    targets: list[str] = field(
        default_factory=lambda: ["cpu.util_pct", "mem.used_pct", "battery.percent"]
    )
    #: Prediction horizons in seconds.
    horizons_s: list[float] = field(default_factory=lambda: [300.0, 900.0, 1800.0])
    #: "rls" (online recursive least squares) or "static" (frozen after fit).
    model: str = "rls"
    #: Ridge regularisation on the RLS information matrix. Calibrated for
    #: standardised inputs, where the information matrix diagonal grows like the
    #: sample count, so ~10 is a light-but-real prior over ~85 columns.
    ridge_lambda: float = 10.0
    #: Inputs observed per scope before its standardiser is frozen. Ten minutes
    #: at the default cadence. Nothing predicts or trains before this.
    standardize_warmup: int = 120
    #: Standardise inputs using frozen warm-up statistics.
    standardize: bool = True
    #: Floor on the per-input standard deviation, in scaled feature units.
    #: Without it, a feature that happened to be constant during an idle warm-up
    #: gets divided by numerical dust and reaches tens of sigma the moment the
    #: machine does anything, which is what made an early version diverge.
    sigma_floor: float = 0.02
    #: Standardised inputs are clipped to +/- this many sigma.
    clip_z: float = 5.0
    #: Exponential forgetting factor. 1.0 = never forget (pure least squares);
    #: below 1.0 down-weights old data with an effective memory of
    #: 1/(1-lambda) samples. 0.999 at 5 s cadence is ~83 minutes.
    forgetting: float = 0.999
    #: Clip predictions to the target signal's declared range.
    clip_to_range: bool = True
    #: Span used for the volatility scale the target is expressed in.
    vol_span_s: float = 300.0
    #: Floor on that volatility, as a fraction of the signal's declared range.
    vol_floor_frac: float = 0.005
    #: The normalised correction is clipped to +/- this many volatility units,
    #: bounding how far a prediction can depart from the anchor.
    target_clip: float = 8.0
    #: Trust-region strength. The correction is shrunk by
    #: 1 / (1 + (s / (trust_k * s_typical))^2) where s = sqrt(x' P x). Smaller
    #: values shrink harder on unfamiliar inputs.
    trust_k: float = 3.0
    #: Which part of the output the trust region shrinks.
    #:
    #: "all" (default) shrinks the whole normalised correction, so an unfamiliar
    #: input backs the forecast off toward persistence.
    #:
    #: "correction" shrinks only the telemetry-derived block, leaving the
    #: intercept and the baseline forecasts un-shrunk, so the fallback is the
    #: learned baseline blend rather than persistence. That *sounds* strictly
    #: better -- persistence is often the worst baseline -- and it was the
    #: default until it was measured. It is worse: on the regime-change scenario
    #: it gave battery@300 MAE 0.298 against 0.0013 for "all", and memory@300
    #: 1.673 against 1.369. The reason is that the baseline inputs are
    #: themselves normalised by the current volatility, so on an
    #: out-of-distribution input they are large too, and leaving their
    #: coefficients un-shrunk lets the "safe" block extrapolate just as badly.
    #: "correction" is kept as an ablation arm; see experiments/ablations.py.
    trust_region_scope: str = "all"
    #: Include each baseline's forecast as a model input. This is what makes the
    #: learned model a correction *on top of* the baselines rather than a
    #: competitor to them: with all weights at zero it reproduces persistence,
    #: and a single unit coefficient reproduces any one baseline exactly.
    baseline_features: bool = True
    #: Separate model per regime instead of regime features in one model.
    per_regime_models: bool = False
    #: Baseline arms evaluated alongside the learned model. Recording all of
    #: them live is what makes the dashboard's comparison table real rather
    #: than a number from an offline run pasted into the UI.
    baselines: list[str] = field(
        default_factory=lambda: ["persistence", "moving_average", "ewma", "linear_trend"]
    )
    #: Warm-up: resolved outcomes required per scope before the learned model is
    #: allowed to emit predictions at all.
    min_train_before_predict: int = 20


@dataclass
class UncertaintyConfig:
    #: "conformal" (adaptive conformal intervals) or "none".
    method: str = "conformal"
    #: Target miscoverage. 0.1 -> nominal 90% intervals.
    alpha: float = 0.1
    #: Residual pool size per (signal, horizon).
    window: int = 400
    #: Adaptive-conformal step size; 0 disables online recalibration.
    step: float = 0.02
    #: Minimum residuals before intervals are emitted at all.
    min_residuals: int = 30


@dataclass
class DriftConfig:
    enabled: bool = True
    #: Page-Hinkley: allowed drift magnitude before accumulation starts.
    ph_delta: float = 0.5
    #: Page-Hinkley alarm threshold on the cumulative statistic.
    ph_threshold: float = 25.0
    #: ADWIN-style windowed error comparison.
    adwin_enabled: bool = True
    adwin_min_window: int = 40
    adwin_delta: float = 0.002
    #: Feature-distribution shift: reference and recent window sizes.
    feature_window: int = 120
    feature_threshold: float = 4.0
    #: Cooldown after an alarm, in seconds, to avoid alarm storms.
    cooldown_s: float = 300.0


@dataclass
class AdaptConfig:
    enabled: bool = True
    #: Any of: "periodic_samples", "periodic_time", "error", "drift".
    #: Multiple policies OR together; the controller records which fired.
    policies: list[str] = field(default_factory=lambda: ["periodic_samples", "drift"])
    #: periodic_samples: adapt every N resolved outcomes. At the default cadence
    #: this is roughly one adaptation per scope every 20 minutes. Much smaller
    #: values were tried and are actively harmful: each adaptation is a selection
    #: event against a finite gate window, so hundreds of them overfit the gate.
    every_n_samples: int = 240
    #: periodic_time: adapt at most every N seconds.
    every_n_seconds: float = 900.0
    #: error: adapt when accumulated MAE excess over the recent baseline
    #: exceeds this many units of the target signal.
    error_budget: float = 400.0
    #: Recency buffer size (most recent examples) and reservoir size (uniform
    #: sample of all history, guarding against catastrophic forgetting).
    recency_buffer: int = 600
    reservoir_buffer: int = 1200
    #: Fraction of the update batch drawn from the reservoir.
    reservoir_fraction: float = 0.25
    #: Master switch for the promotion gate. Disabling it turns the controller
    #: into a plain online learner that promotes every candidate, which is the
    #: ablation arm that shows what the safety machinery is actually buying.
    gate_enabled: bool = True
    #: Held-out recent outcomes used to gate promotion. Never trained on.
    #: Larger is better: the gate is a statistical test, and at 40 samples a
    #: neutral candidate passes roughly half the time by chance.
    gate_window: int = 120
    #: Candidate is promoted if its gate MAE <= active MAE x (1 + tolerance).
    #:
    #: This is 0.0, not a small positive number, and that matters. An earlier
    #: version used 0.02, reasoning that admitting statistically neutral updates
    #: lets the model track slow drift. It does -- and it also lets the model
    #: ratchet *downwards*: every adaptation is permitted to be 2% worse than the
    #: last, and over the ~200 adaptations a multi-hour run produced that
    #: compounds to (1.02)^200, about 50x. Measured skill was negative on every
    #: scope until this was set to zero. Tracking drift is the forgetting
    #: factor's job, not the gate's.
    gate_tolerance: float = 0.0
    #: Minimum gate samples required to make any promotion decision.
    gate_min_samples: int = 60
    #: Record the best baseline's MAE on the gate window.
    #:
    #: Reported, not enforced. Enforcing it would deadlock: a model that starts
    #: worse than the baselines could never be promoted, so it could never
    #: improve. What the number is for is visibility -- it lands in the adapt
    #: event, so "is the learned model worth serving for this scope at all?" is
    #: answerable from stored data. Serving safety comes from the gate (no
    #: promotion that is worse than the incumbent) and the regression watch (roll
    #: back what the gate let through).
    #:
    #: Automatically *serving* the best arm per scope is deliberately not
    #: implemented: it would need its own hysteresis and evaluation to avoid
    #: flapping, and the honest interim answer is to report the per-scope
    #: comparison and let the user see that, for example, linear extrapolation
    #: beats the learned model on battery.
    gate_report_baselines: bool = True
    #: Hard ceiling: reject a candidate this much worse regardless of tolerance.
    gate_reject_ratio: float = 1.15
    #: Keep this many superseded versions on disk for rollback.
    keep_versions: int = 10


@dataclass
class RegimeConfig:
    enabled: bool = True
    #: Labels are derived from thresholds on interpretable features. They are
    #: seeds, not immutable truth: see docs/architecture.md.
    idle_cpu_pct: float = 8.0
    interactive_cpu_pct: float = 35.0
    heavy_cpu_pct: float = 65.0
    gpu_active_pct: float = 25.0
    build_share: float = 0.30
    low_battery_pct: float = 25.0
    #: Consecutive samples a candidate label must hold before the regime
    #: switches. Prevents single-tick flapping between labels.
    hysteresis: int = 3


@dataclass
class LanguageConfig:
    #: "template" (deterministic, no model), "hf" (local transformers), "none".
    backend: str = "template"
    #: 135M is the default deliberately. The point of the LoRA layer is that
    #: adaptation happens *continuously* on the user's own laptop, which means an
    #: update has to take seconds, not hours, and has to leave room for the
    #: full-fine-tune comparison arm to fit in a small GPU's memory alongside
    #: Adam state. A 7B model would write nicer prose and make the continual
    #: loop impossible on laptop-class hardware. 360M also works: set
    #: language.model_id.
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "auto"
    max_new_tokens: int = 220
    temperature: float = 0.3
    #: LoRA hyperparameters for the adapter trained on grounded explanations.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lr: float = 1e-4
    epochs: int = 2
    batch_size: int = 4
    max_seq_len: int = 640
    #: Minimum new feedback/explanation examples before an adapter update.
    min_examples: int = 24
    #: Adapter promotion gate: candidate must not be worse than active by more
    #: than this fraction on held-out loss.
    gate_tolerance: float = 0.0
    #: Never send telemetry to a remote service unless explicitly enabled.
    allow_remote: bool = False


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    #: Seconds of history the dashboard requests by default.
    default_window_s: float = 3600.0


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    collect: CollectConfig = field(default_factory=CollectConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    # -- paths -------------------------------------------------------------
    @property
    def db_file(self) -> str:
        p = self.general.db_path
        return p if os.path.isabs(p) else os.path.join(self.general.data_dir, p)

    @property
    def models_dir(self) -> str:
        return os.path.join(self.general.data_dir, "models")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.general.data_dir, "runs")

    def ensure_dirs(self) -> None:
        for d in (self.general.data_dir, self.models_dir, self.runs_dir):
            os.makedirs(d, exist_ok=True)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | None) -> "Config":
        if not path:
            return cls()
        with open(path, "rb") as fh:
            raw = fh.read()
        if path.endswith(".json"):
            data = json.loads(raw.decode())
        else:
            if tomllib is None:  # pragma: no cover
                raise RuntimeError("TOML config requires Python 3.11+; use JSON instead")
            data = tomllib.loads(raw.decode())
        return cls.from_dict(data)

    def override(self, dotted: str, value: str) -> None:
        """Apply ``--set forecast.horizons_s=60,300`` style overrides.

        Values are coerced to the declared field type so a config sourced from
        the CLI is indistinguishable from one loaded from a file.
        """
        parts = dotted.split(".")
        obj: Any = self
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise KeyError(f"unknown config section: {p!r} in {dotted!r}")
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"unknown config key: {dotted!r}")
        current_type = _field_type(obj, leaf)
        setattr(obj, leaf, _coerce(value, current_type, getattr(obj, leaf)))


def _field_type(obj: Any, name: str) -> str:
    for f in fields(obj):
        if f.name == name:
            return str(f.type)
    return "str"


def _coerce(value: str, type_str: str, current: Any) -> Any:
    t = type_str.replace(" ", "")
    if "list[str]" in t:
        return [] if value == "" else [s.strip() for s in value.split(",")]
    if "list[float]" in t:
        return [] if value == "" else [float(s) for s in value.split(",")]
    if "list[int]" in t:
        return [] if value == "" else [int(s) for s in value.split(",")]
    if t.startswith("bool") or isinstance(current, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if "int" in t and "float" not in t:
        return int(value)
    if "float" in t:
        return float(value)
    if value.lower() in ("none", "null", ""):
        return None
    return value


def _build(cls: Any, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, val in data.items():
        if key not in known:
            raise KeyError(f"unknown config key: {key!r} for {cls.__name__}")
        f = known[key]
        ftype = f.type
        if isinstance(val, dict) and is_dataclass(_resolve(ftype)):
            kwargs[key] = _build(_resolve(ftype), val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


_SECTIONS = {
    "GeneralConfig": GeneralConfig, "CollectConfig": CollectConfig,
    "FeatureConfig": FeatureConfig, "ForecastConfig": ForecastConfig,
    "UncertaintyConfig": UncertaintyConfig, "DriftConfig": DriftConfig,
    "AdaptConfig": AdaptConfig, "RegimeConfig": RegimeConfig,
    "LanguageConfig": LanguageConfig, "ApiConfig": ApiConfig,
}


def _resolve(ftype: Any) -> Any:
    if is_dataclass(ftype):
        return ftype
    return _SECTIONS.get(str(ftype).strip("'\""), str)
