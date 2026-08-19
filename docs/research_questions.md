# Research questions

Each question is stated, then answered with a measurement and the arm that
produced it. Where the answer is "no" or "it depends", it says so.

Reproduce everything with the commands in [experiments.md](experiments.md).

---

## 1. Does continual adaptation outperform a static model?

**Yes, on non-stationary data — and the exception is informative.**

*Arms:* `static` (learns on the chronological prefix, then frozen) vs
`online_periodic` / `online_drift` / `online_periodic_drift`. Same estimator,
same data, same held-out tail; the only difference is whether learning continues.

**On the synthetic drift scenarios, online beat static on 34 of 45 scopes and was
worse on 5. On 3.3 h of real telemetry it was 3–3.**
Largest gains where behaviour genuinely changes:

| scenario | scope | static | best online | gain |
|---|---|---|---|---|
| idle → build | cpu@5m | 10.522 | 5.870 | +44.2% |
| idle → build | mem@15m | 4.322 | 2.454 | +43.2% |
| discharge → charge | battery@5m | 3.323 | 1.954 | +41.2% |
| idle → build | cpu@15m | 22.702 | 14.432 | +36.4% |
| gappy idle | cpu@30m | 3.349 | 2.454 | +26.7% |
| permanent change | cpu@5m | 11.372 | 9.282 | +18.4% |

The exception is `spike_burst`, a **stationary** process — the same spike pattern
for five hours. There the static model wins (cpu@5m 3.698 vs 4.351, −17.7%). When
the world is not changing, adaptation only adds variance.

On **real telemetry** the picture is much weaker: 3 scopes better, 3 worse
(mem@5m +24.9%, mem@30m −15.1%). The mechanism demonstrably works when behaviour
shifts; 3.3 hours with a single regime change near the split is not enough data
to demonstrate it, and saying otherwise would overstate what was measured.
When the world is not changing, adaptation only adds variance. This is the
correct answer and it is the reason drift-triggered policies exist.

---

## 2. Does the learned model beat simple baselines?

**Sometimes. It depends entirely on the signal, and reporting only the wins would
have been dishonest.**

**Where it wins.** On 3.3 h of real laptop telemetry it takes 4 of 8 comparable
scopes against the *best* baseline for each:

| scope | best baseline | learned | skill |
|---|---|---|---|
| mem@30m | 10.813 (moving avg) | **10.112** | +6.5% |
| mem@15m | 9.139 (persistence) | **8.721** | +4.6% |
| cpu@15m | 14.950 (EWMA) | **14.637** | +2.1% |
| cpu@30m | 20.864 (EWMA) | **20.710** | +0.7% |

On `spike_burst`, where structure genuinely exists and repeats, it beats *every*
baseline by a wide margin: cpu@5m 3.698 vs 4.513 persistence (+18%) and 22.6
EWMA (+84%). On `idle_to_build` it beats EWMA at the short horizon (5.870 vs
6.139, +4.4%).

**Where it loses.**

- **EWMA and moving average are very strong** on smooth signals and win outright
  at the short horizon (real cpu@5m: EWMA 9.387 vs learned 9.658). A project
  without them would have claimed a false success.
- **Linear extrapolation dominates battery**, now confirmed on a real 100% → 55%
  discharge and not only in simulation: 0.482 vs 1.974 at five minutes, a 4×
  gap that no configuration closed.
- **Memory at the short horizon** loses to persistence (5.142 vs 5.269). Memory
  drifts slowly and near-monotonically; over five minutes there is little for a
  model to add.

The honest summary: the learned layer earns its place at the **longer horizons**
and under **regime change**, is roughly a wash at five minutes, and is the wrong
tool for battery. The margins on real data (+2% to +6.5%) are far smaller than on
the synthetic scenarios, which is what 3.3 hours of data buys.

---

## 3. Does drift-triggered adaptation improve efficiency?

**It changes the trade rather than dominating it.**

On `idle_to_build`, per arm: `online_periodic` 83 promotions / 40 rejections,
`online_drift` 105 / 73, `online_periodic_drift` 123 / 105. Drift-triggered
adaptation does *more* work, not less, because a regime change trips several
detectors and each one is a trigger.

Accuracy differences between the three policies are small and scenario-dependent
(`online_periodic` best on 4 of the 9 idle→build scopes, `online_drift` on 3).
What drift buys is **latency**: on the permanent-change scenario the drift arm
reaches its post-change accuracy sooner, because it does not wait for the sample
counter. The default is both — periodic as a floor, drift as a fast path.

A concrete efficiency fix came out of this: routing every feature-shift alarm to
all nine scopes turned 55 drift events into 330 triggers. Routing by the
implicated signal (recoverable from the feature name) cut that without changing
accuracy.

---

## 4. When does adaptation hurt?

**Three measured ways, each now guarded.**

1. **A tolerant gate ratchets downward.** With a 2% tolerance, every adaptation
   may be 2% worse than the last; over ~200 adaptations that compounds to ≈50×.
   Skill was negative on every scope until the tolerance was set to zero.
2. **Adapting too often overfits the gate.** Every adaptation is a selection
   event against a finite window. At `every_n_samples=60` the version counter
   reached v130 in five hours and accuracy degraded; 240 is the default.
3. **Stationary data.** See `spike_burst` above — adaptation is a loss there.

---

## 5. What is the safety machinery worth?

**Measurable, and it fires on real data.**

The `safety` suite compares `online_periodic_drift` against `online_no_gate`
(promotes every candidate). On `permanent_regime_change` the gate wins on **all
six scopes** (cpu@5m 9.579 vs 10.723; cpu@15m 12.331 vs 13.828):

| | promoted | rejected | rolled back in production |
|---|---|---|---|
| gated (default) | 124 | 106 (46%) | **9** |
| no gate | 231 | 0 | **20** |

The gate rejects nearly half of all candidates and halves production rollbacks.
Without it, bad versions reach production and the regression watch cleans up
afterwards — which means the user sees the bad forecasts in the meantime.

On real telemetry (3.3 h, four arms) the gate rejected **132 of 438 candidates
(30%)** — 84 as `hard_reject_worse`, 48 as `worse_than_tolerance` — and the
regression watch rolled back **35** versions that had passed it, including:

```
rolled back: battery.percent@900 (regression_watch)  MAE 6.088 → 18.487
```

A version that looked fine on 120 held-out samples was 3× worse in production and
was reverted automatically. Without the watch, that version stays live.

---

## 6. Does the trust region matter?

**Substantially — it is the single largest correctness fix in the project.**

Ablation `no_trust_region` (shrinkage effectively disabled), on
`permanent_regime_change`:

| scope | default | no trust region |
|---|---|---|
| cpu@5m | **9.579** | 16.871 |
| cpu@15m | **12.331** | 19.319 |
| cpu@30m | **16.446** | 33.206 |
| mem@5m | **1.362** | 4.038 |
| mem@15m | **3.567** | 6.779 |
| mem@30m | **6.848** | 10.466 |

Worse on **every** scope, by 1.8–2.9×. It also promoted far more candidates
(170 vs 124) and rolled back nearly twice as many (16 vs 9) — an unshrunk model
keeps producing versions that look acceptable on the gate and fail in production.

And the *variant I expected to win, lost.* I predicted that shrinking only the
telemetry-derived block — leaving the intercept and baseline forecasts un-shrunk,
so the fallback is the learned baseline blend rather than persistence — would be
strictly better. Measured:

| scope | shrink everything (default) | shrink correction only |
|---|---|---|
| cpu@5m | **9.579** | 9.984 |
| cpu@15m | 12.331 | **11.724** |
| cpu@30m | 16.446 | **15.595** |
| mem@5m | **1.362** | 1.671 |
| mem@15m | **3.567** | 3.849 |
| mem@30m | **6.848** | 7.270 |

It takes 2 of 6 — both long-horizon CPU — and loses the rest, decisively on
memory. The reason is that the baseline inputs are volatility-normalised too, so
on an out-of-distribution input they are large as well, and leaving their
coefficients un-shrunk lets the "safe" block extrapolate just as badly. The
default follows the measurement; the variant is retained as the
`trust_region_correction` arm because the long-horizon CPU result suggests the
right answer may be per-signal rather than global.

**Baselines as inputs** (`no_baseline_features`) matter far less than the trust
region: the default wins 5 of 6 scopes but almost always by under 1% (12.331 vs
12.345 at cpu@15m), and loses at cpu@30m (16.446 vs 15.991). The stack is a sound
design property — zero weights reproduce persistence exactly — but on this data
it is not what makes the model work.

---

## 6b. Is regime-conditioned modelling worth it?

**No — regime as a *feature* beats a model per regime.**

| scope | regime feature | model per regime |
|---|---|---|
| cpu@5m | **9.579** | 9.662 |
| mem@15m | **3.567** | 4.149 |
| mem@30m | **6.848** | 9.111 |

Worse on **all six scopes**, with the margin widening at long horizons (33%
worse at mem@30m). Splitting a bounded online memory across seven regimes leaves
each model too few examples to estimate ~80 coefficients. The regime label still
earns its place — it slices every metric, drives explanations and routes drift
alarms — but as an input, not a partition.

---

## 7. Are the prediction intervals calibrated?

**Yes in the long run, with a stated caveat.**

Adaptive conformal reaches **90.1% coverage against a nominal 90%** on stationary
noise, and after a 6× variance shift dips to 89.0% and recovers to 89.9%.
Coverage tracks the requested α (5% → 95.4%, 20% → 79.8%).

On 3.3 h of real telemetry, CPU coverage is **0.84 at +5 min, 0.77 at +15 min and
0.58 at +30 min** against a nominal 0.90. It degrades with horizon because the
residual pool is small there and the adaptive quantile has fewer resolved
outcomes to converge on. The dashboard reports coverage next to interval width,
so the tell is visible: the +30 min intervals are *narrower* than the +5 min ones
while covering far less.

The guarantee is long-run *average* coverage, not conditional. Intervals are too
wide in calm periods and too narrow in the first minutes of a new regime.

---

## 8. Does user feedback improve personalisation?

**It changes behaviour deterministically and measurably; whether users prefer the
result is not measured.**

Free-text feedback is parsed into structured rules — "That happens when I'm
compiling projects. Treat that as normal for me." becomes regime `build`, verdict
`normal` — which the style verifier then *enforces*: an answer during a build
regime that omits the user's framing is a recorded violation. Text that cannot be
parsed is kept as context rather than guessed into a rule.

The measurable claim is therefore compliance, not preference. Preference needs
human labels this project does not have.

---

## 9. Does LoRA beat prompt-only personalisation, and is it cheaper than full fine-tuning?

**Yes on efficiency and effectiveness for the style objective; no on grounding.**

| arm | eval loss | prompt tok/req | trainable params | train s | artifact |
|---|---|---|---|---|---|
| persona_prompt | 2.651 | 636 | – | – | – |
| **lora** | **0.635** | 606 | 1,843,200 (1.4%) | 482 | 7.1 MB |
| full_ft | 0.177 | 606 | 134,515,008 (100%) | 1103 | 513 MB |

- Putting the persona in the prompt moved held-out loss **0.5%**. The adapter
  moved it **76%** — with the persona removed from the prompt, so it is also
  cheaper per request, permanently.
- LoRA reaches most of full fine-tuning's benefit for **1.4% of the parameters,
  a 72× smaller artefact and ~10× less compute per step**. Full fine-tuning does
  reach a lower loss (0.177 vs 0.635); the honest claim is "most of the
  adaptation at a fraction of the cost", not parity.
- **But LoRA slightly *hurt* groundedness** (0.819 → 0.742): it learned the style
  and became more fluent with numbers without learning to keep them to the
  evidence. And **few-shot prompting collapsed groundedness to 0.528** — 3% of
  answers free of invented numbers — because the model copies figures out of the
  worked examples.

The deterministic renderer remains the only arm at 0.95 groundedness and 1.00
style compliance. Personalisation is a weights problem; hallucination is not, and
the verifier stays.

**The honest limitation, stated up front:** targets are rendered by the
deterministic template renderer under an explicit persona, not written by a
human. The experiment measures whether an adaptation mechanism can acquire a
*specified* style-and-grounding function and at what cost. It does not measure
whether people prefer the output.

---

## 10. What would move the needle next?

Ranked by expected value, with the reasoning:

1. **More real telemetry.** 3.3 h from one machine is the binding constraint on
   every accuracy claim here, and it is why online-vs-static is 3–3 on real data
   against 34–5 on synthetic drift scenarios. Diurnal structure — the thing a
   linear model with time features should exploit — is invisible in three hours.
2. **A non-linear forecaster with a genuine incremental update.** The linear
   model cannot express "CPU spikes when the build reaches linking". Hoeffding
   trees or a small online GBM would test whether the ceiling is the model class
   or the data.
3. **Learned regimes.** The labels are threshold rules. Clustering the feature
   space and validating against per-regime accuracy would test whether the
   hand-written categories are the right ones.
4. **Human evaluation of the language layer**, which is the only thing that
   converts question 9 from a mechanism study into a usefulness result.
