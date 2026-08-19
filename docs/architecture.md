# Architecture

## The two layers, and why they are separate

```
                       LAYER A — forecasting (no language model anywhere)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  sources ─► registry ─► normaliser ─► feature window ─► regime       │
  │                                            │                          │
  │                                            ▼                          │
  │                              ┌──── forecast engine ────┐              │
  │                              │  baselines (4)          │              │
  │                              │  learned model (RLS)    │              │
  │                              │  conformal intervals    │              │
  │                              └───────────┬─────────────┘              │
  │                                          ▼                            │
  │            prediction store ─► resolution ─► error ─► drift           │
  │                                          │                            │
  │                        training buffer ◄─┘                            │
  │                                │                                      │
  │                    adaptation controller ─► gate ─► version registry  │
  │                                                  └─► rollback         │
  └──────────────────────────────────────────────────────────────────────┘
                                    │  facts only, never opinions
                                    ▼
                       LAYER B — language / interaction
  ┌──────────────────────────────────────────────────────────────────────┐
  │  evidence pack ─► prompt ─► backend (template | HF+LoRA) ─► verifier  │
  │        ▲                                                     │        │
  │        └──────── persona ◄─── structured feedback ◄──────────┘        │
  └──────────────────────────────────────────────────────────────────────┘
```

Layer A never imports Layer B. `pytest tests/test_forecast.py tests/test_adapt.py`
passes with `torch` uninstalled, and the dashboard and CLI work fully without it.
That is not incidental: it is what makes "did the forecasting model improve?" and
"did the interaction layer become more useful?" separately answerable, which is
the separation the project is built around.

---

## Layer A

### Telemetry (`streamlora/telemetry/`)

A **source** is a small, independently-failing unit that emits named signals in
canonical units. The registry probes each one, records why unavailable ones are
unavailable, and quarantines sources that start failing with exponential backoff.
A source that throws on every read is skipped rather than re-failing every tick;
one that dies never affects another. *A missing battery must not stop CPU
forecasting* is a test, not an aspiration.

Sources declare a native cadence. NVML costs 0.02 ms so the GPU source is read
every tick; the `nvidia-smi` fallback costs 45 ms, so it declares a 10 s cadence
and the collector decimates it.

The **normaliser** is where real-world misbehaviour is absorbed, and every rule
exists because of a specific failure:

| problem | handling | quality |
|---|---|---|
| NaN / inf from a driver | rejected before it can reach a model | `MISSING` |
| 100.4% CPU (psutil rounding) | clipped silently | `OK` |
| 6000% battery (driver bug) | clipped and logged | `CLAMPED` |
| battery jumps 40% in 5 s | kept but flagged | `SUSPECT` |
| sensor misses one tick | last value held, bounded by `max_hold_s` | `STALE` |
| suspend/resume, 4 h gap | hold state discarded — holding across it would be a lie | `MISSING` |

One NaN in a least-squares update corrupts every subsequent coefficient
permanently, which is why the boundary is absolute.

The **collector** ticks on a fixed grid derived from the start time, not
`sleep(interval)` after work. The latter accumulates per-tick cost into the
cadence: 5.000 s becomes 5.045 s, which loses ~13 minutes of samples a day and
makes every counter-derived rate subtly wrong.

### Time (`util/clock.py`)

Nothing outside this module calls `time.time()` or `time.sleep()`. Every
component takes a `Clock`. A replay swaps `RealClock` for `SimulatedClock` and
the *identical* collector, feature extractor, forecaster, drift detector and
adaptation controller run over historical timestamps at ~450× real time. A bug
that only appears in production is a bug the replay can reproduce.

### Storage (`store/`)

SQLite, WAL, one file. The workload is a single writer at 0.2 Hz with analytical
reads over a few million rows on one machine, holding data the user considers
private — squarely inside SQLite's competence, and anything distributed would add
operational surface without answering a research question.

Readings are **long-format** (`sample_id, signal_id, value, quality`). Adding GPU
temperature later is an insert, not a migration. The cost is a pivot on read,
which at these volumes is microseconds.

Feature vectors are stored **once per tick** and referenced by every prediction
made from them. That is what makes gated promotion possible: a candidate can be
scored on exactly the inputs the active model saw.

### Features (`forecast/features.py`)

One vector per tick, shared; each model selects columns with a fixed mask.

- Own signal: lags (0/15/30/60/120/300 s), rolling means, std, slopes, and the
  change over the longest lag.
- Every other signal: current value, one medium mean, one long slope.
- Cyclical hour-of-day and day-of-week; the regime one-hot; a coverage indicator.

Sharing the vector saves storage; masking keeps each model at ~80 columns instead
of ~250, which is the difference between a workable parameter-to-data ratio and
guaranteed overfitting under a bounded online memory.

Signals absent on this machine are **dropped from the feature space**, not
imputed forever, and the schema hash changes accordingly so a model can never be
loaded against a mismatched feature space.

### Regimes (`forecast/regime.py`)

70% CPU during a compile and 70% during a video call have different futures.
Labels are threshold rules over interpretable features, with **hysteresis** —
without it, CPU oscillating around a threshold relabels every tick, turning the
one-hot into noise and generating a stream of spurious drift alarms.

The labels are seeds, not ground truth: they are data rather than control flow,
every metric is sliced by them so a useless label is visible, and user feedback
can override them.

### The learned forecaster (`forecast/linear.py`)

**Recursive least squares**, chosen deliberately over the alternatives:

- *SGD* is truly incremental but dominated by a learning-rate schedule, and two
  runs over the same data in a different order give different models — which
  makes an accuracy gap hard to attribute.
- *GBMs / neural nets* have better asymptotics but neither updates incrementally
  without replay-based retraining (the thing being avoided) or catastrophic
  forgetting.
- *RLS* is closed-form and deterministic: no learning rate, no epochs, and
  replaying a stream twice gives bit-identical coefficients. With forgetting
  factor 1 it **equals batch ridge exactly** (asserted to 1e-15), so the static
  and continually-adapted arms are the same estimator differing only in the
  forgetting factor and update schedule.

The target took three iterations, and the failures are documented in
`docs/experiments.md`:

```
y = clip( (actual(t+h) − value(t)) / vol(t),  ±8 )
```

where `vol(t)` is the signal's standard deviation over the last 300 s, floored.
Two properties follow. The target is **stationary across regimes**, so one ridge
penalty is correct for an idle laptop and a compiling one. And **zero weights
mean persistence**, so shrinkage degrades the model gracefully.

Each baseline's forecast enters as an input, `(baseline_pred − value(t)) / vol(t)`.
The learned model is therefore a **stack plus a correction**: it can learn "trust
EWMA at five minutes, trust linear-trend for battery at thirty" and correct from
there. This changes how the headline number should be read, and the
documentation says so.

**Trust region.** Out-of-distribution inputs are where linear extrapolation does
real damage — eighty small coefficients pushed the same way sum to a huge
correction. RLS already tracks the parameter covariance, so `s = sqrt(x'Px)` is
free and is exactly the "how unfamiliar is this input" signal. The correction is
shrunk by `1 / (1 + (s / (k·s_typical))²)`. On idle→build this cut the static
model's worst-case MAE from 28.3 to 11.2.

Two numerical guards, both for documented RLS failure modes: `P` is symmetrised
every update (asymmetry accumulates until definiteness is lost), and its trace is
bounded (with forgetting below 1 and uninformative inputs, `P` grows like
`λ^-n` until one informative sample causes an enormous jump).

### Uncertainty (`forecast/uncertainty.py`)

**Adaptive conformal inference.** The alternatives were considered and rejected:
a Gaussian from the parameter covariance captures coefficient uncertainty only,
not observation noise, and telemetry noise is wildly heteroscedastic; split
conformal assumes exchangeability, which continual learning breaks by
construction, since the model that produced old residuals no longer exists.

ACI keeps a pool of recent absolute residuals and adjusts the working quantile
online, `α_{t+1} = α_t + step·(α_target − miss_t)`. Coverage converges to the
nominal level regardless of distribution shift or model change. Its guarantee is
long-run *average* coverage, not conditional — intervals are too wide when calm
and too narrow just after a change, and the recorded coverage metric shows it.

Intervals are clipped to the signal's physical range. "CPU will be between −12%
and 38%" is not a statement about a CPU; clipping is conservative for coverage
because the actual value cannot fall outside the range either.

### Drift (`drift/`)

Two complementary questions, because they fail differently.

*Is the model getting worse?* — Page-Hinkley and ADWIN on the error stream.
Directly actionable, but blind to a shift the model handles well, and only fires
after accuracy has already degraded.

*Do the inputs look different?* — a per-feature standardised-shift detector.
Fires at the moment behaviour changes, before errors accumulate, and reports
*which* feature moved and by how many sigma, which is what makes the alarm usable
in the UI and in an explanation.

Everything consumes a **standardised** stream, so thresholds are scale-free and a
config that works on one signal works on another. After an alarm each detector
re-establishes its reference: the point of detecting a regime change is that the
new regime is now normal.

The ADWIN bound is the variance-aware one from ADWIN2, not plain Hoeffding.
Hoeffding requires bounded variables; on unbounded standardised errors it
under-estimates by ~3× and produced a false alarm every ~180 samples on pure
noise.

Errors from **persistence**, not from the learned model, drive the detectors:
they measure how unpredictable the world is independent of which model is live,
so a drift alarm can never be caused by our own adaptation.

### Adaptation (`adapt/`)

```
resolved outcome ─► buffer ─► policy fires? ─► candidate = copy(active) + updates
                                                     │
                        gate: score both on held-out newest rows
                                                     │
                          promote (new version, activate)  |  reject (discard)
                                                     │
                          regression watch ─► rollback if materially worse
```

**The buffer has two pools.** A recency deque keeps the model current; training
on that alone is catastrophic forgetting by design, since an hour of idle
telemetry overwrites everything known about builds. A reservoir (Algorithm R)
holds a uniform sample of all history so rare regimes keep a foothold. Uniform is
deliberate over "keep the interesting ones", which requires knowing what the
model will need and silently biases every update when wrong.

**Each observation is absorbed exactly once.** Overlapping batches let RLS absorb
the same row once per cycle, shrinking the covariance as if it had seen far more
independent data than it has. A rejected candidate's batch stays available; a
rollback rewinds the mark so the restored model can still learn from it.

**The gate is chronologically held out**: the candidate trains on older rows and
is judged on newer ones, the same orientation as production. Both models are
scored on the identical rows, so the difference is the update rather than a
different sample of reality. Tolerance is **zero** — see the ratchet failure in
the experiments document.

**Versions are per scope** — one `(signal, horizon)` pair — because adaptation
quality differs sharply between `battery.percent@1800` and `cpu.util_pct@300`,
and a single global version would force promoting or rejecting all nine on
evidence that applies to one.

**The regression watch** is the guarantee behind the gate. Passing a 120-sample
test does not prove a version is good, so the controller keeps scoring the live
version and reverts it if it is materially worse than its predecessor over a full
window. Rollback targets the recorded *parent*, and pruning never removes it.

Model files are `.npz` loaded with `allow_pickle=False`, written atomically. A
corrupted file fails as a parse error rather than executing code.

---

## Layer B

### Grounding (`language/grounding.py`)

Contains no language model and never will. It reads telemetry, predictions, drift
and adaptation history out of SQLite and produces an **evidence pack**: typed
facts with values, units and timestamps, plus derived analyses — discharge rate
and time-to-threshold, driver correlations at multiple lags, recent accuracy per
scope.

Numbers exist only here. Everything downstream is phrasing.

### Verification (`language/verify.py`)

Every number in an answer must appear in the evidence within a rounding
tolerance. One that does not is a hallucinated measurement, and the service
**discards the generation and serves the deterministic answer instead** — a
fluent answer with an invented number is worse than a plain one with real ones.

Style compliance is checked the same way, against a persona whose every field has
a deterministic checker, so "did personalisation improve?" is a rate on held-out
examples rather than an opinion.

### Persona (`language/persona.py`)

Built from the user's own structured feedback. `label` and `note` rows are parsed
into semantic rules ("compiling is normal for me" → regime `build`, verdict
`normal`); text that cannot be parsed is kept as context rather than guessed into
a rule, because a wrong rule silently changes every future explanation.

### LoRA (`language/lora.py`)

LoRA is here for a reason, not because it is interesting. The interaction layer
has to keep changing as the user teaches it, and on this hardware:

- **prompt-only** works but costs tokens and latency on *every* request forever;
- **full fine-tuning** updates 100% of parameters and produces a
  hundreds-of-megabytes checkpoint per update;
- **LoRA** trains ~1% of parameters, produces megabyte adapters, trains in
  seconds, and attaches and detaches **exactly** — which makes rollback correct
  by construction rather than approximate.

Loss is computed on answer tokens only; the prompt is masked to −100. Without
that, most of the capacity goes into reproducing the evidence block, which is the
input.

Detaching calls `PeftModel.unload()`, restoring the base bitwise (asserted in the
tests). Skipping that step leaves the base wrapped and stacks the next adapter on
top of the first — a bug the test suite caught.

The base model is small on purpose. A 135M instruct model fits alongside Adam
state on a small laptop GPU *and* leaves room for the full-fine-tune comparison
arm. A 7B model would write nicer prose and make the continual loop impossible,
which is the wrong trade for this project.

---

## Repository layout

```
streamlora/
  config.py            one dataclass tree; stored with every experiment
  pipeline.py          the standard assembly
  cli.py               doctor, collect, replay, eval, experiment, lora, serve …
  util/                clock (the replay hinge), structured logging, math
  telemetry/           schema, sources/, registry, normaliser, collector
  store/               SQLite schema and typed repositories
  forecast/            features, scaling, regimes, baselines, RLS, conformal, engine
  drift/               detectors and the alarm controller
  adapt/               buffer, policies, gate controller, version registry
  evaluate/            metrics, chronological splits, grouped reporting
  replay/              synthetic scenarios, dataset recorder, player
  experiments/         runner and the ablation arms
  language/            grounding, persona, verify, prompt, backends/, lora, service
  api/                 FastAPI app and shared state
web/                   dashboard (hand-rolled SVG, no build step)
tests/                 241 tests
docs/                  this file, experiments, research questions, telemetry
```
