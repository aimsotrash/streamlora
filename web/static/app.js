/* StreamLoRA dashboard.
 *
 * Six views, each answering one question:
 *   Overview     what is this machine doing, and what happens next?
 *   Forecasts    is the forecast tracking reality, and where does it fail?
 *   Telemetry    what has the machine been doing? (small multiples, one axis each)
 *   Adaptation   what did the system learn, and did it help?
 *   Experiments  does the learned model beat the baselines?
 *   Chat         grounded questions about the telemetry
 *
 * No build step and no third-party script: the page is served from the same
 * local process that holds the telemetry.
 */

import {
  COLORS, SERIES, timeSeries, groupedBars, eventLane, sparkline, legend, tableView,
  withTableToggle, fmtNum, fmtClock, fmtDuration, h,
} from "./chart.js";

const state = {
  view: "overview",
  windowS: 3600,
  overview: null,
  settings: null,
  forecastSignal: null,
  forecastHorizon: null,
  chat: [],
  timer: null,
};

const SIGNAL_LABELS = {
  "cpu.util_pct": "CPU utilisation",
  "cpu.util_max_core_pct": "Busiest core",
  "cpu.iowait_pct": "CPU I/O wait",
  "cpu.load1_per_core": "Load per core",
  "cpu.freq_mhz": "CPU frequency",
  "mem.used_pct": "Memory used",
  "mem.available_gb": "Memory available",
  "mem.cached_gb": "Page cache",
  "mem.swap_used_pct": "Swap used",
  "battery.percent": "Battery",
  "battery.power_w": "Battery power",
  "battery.plugged": "On AC",
  "battery.charging": "Charging",
  "gpu.util_pct": "GPU utilisation",
  "gpu.power_w": "GPU power",
  "gpu.temp_c": "GPU temperature",
  "gpu.mem_used_pct": "GPU memory",
  "thermal.cpu_c": "CPU temperature",
  "thermal.gpu_c": "GPU temp (sensor)",
  "thermal.max_c": "Hottest sensor",
  "disk.busy_pct": "Disk busy",
  "disk.read_mbps": "Disk read",
  "disk.write_mbps": "Disk write",
  "net.recv_mbps": "Network down",
  "net.sent_mbps": "Network up",
  "proc.cpu_build_pct": "Build activity",
  "proc.cpu_browser_pct": "Browser activity",
  "proc.cpu_ml_pct": "ML/Python activity",
  "proc.top1_cpu_pct": "Busiest process",
  "proc.concentration": "Workload concentration",
  "proc.count": "Processes",
};
const label = (s) => SIGNAL_LABELS[s] || s;

/* Units come from the signal specs the API reports, not from guessing at name
 * suffixes -- "battery.percent" has no "_pct" suffix and was rendering unitless. */
const UNIT_SUFFIX = {
  percent: "%", celsius: " °C", watt: " W", gigabyte: " GB", megahertz: " MHz",
  megabyte_per_second: " MB/s", ratio: "", count: "", bool: "",
};
const unitBySignal = new Map();
function noteUnits(signals) {
  for (const s of signals || []) unitBySignal.set(s.name, UNIT_SUFFIX[s.unit] ?? "");
}
function unitOf(sig) {
  return unitBySignal.get(sig) ?? "";
}

/** Sort "cpu.util_pct@1800" style scopes by signal then numeric horizon. */
function scopeSort(a, b) {
  const [sa, ha] = a.split("@");
  const [sb, hb] = b.split("@");
  return sa.localeCompare(sb) || Number(ha) - Number(hb);
}

/** True when every bar in a group is ~0, e.g. battery while on AC. */
function degenerate(bars) {
  return bars.every((b) => !isFinite(b.value) || Math.abs(b.value) < 1e-6);
}

/* ---------- fetch ---------- */

async function ensureSettings() {
  if (!state.settings) {
    state.settings = await api("/api/settings");
    noteUnits(state.settings.signals);
  }
  return state.settings;
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch { /* not json */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json();
}

function card(parent, title, sub) {
  const c = h("div", "card", parent);
  const head = h("div", "card-head", c);
  const box = h("div", null, head);
  const t = h("h3", null, box);
  t.textContent = title;
  if (sub) { const p = h("p", null, box); p.textContent = sub; }
  return c;
}

function badge(parent, text, kind, icon) {
  const b = h("span", `badge${kind ? " " + kind : ""}`, parent);
  if (icon) { const i = h("span", "ic", b); i.textContent = icon; }
  const s = h("span", null, b);
  s.textContent = text;
  return b;
}

/* status colour + icon + label: a status colour never carries meaning alone */
const DECISION_STYLE = {
  promoted: { color: COLORS.good, kind: "good", icon: "▲", label: "promoted" },
  rejected: { color: COLORS.serious, kind: "serious", icon: "■", label: "rejected" },
  rolled_back: { color: COLORS.critical, kind: "critical", icon: "▼", label: "rolled back" },
  skipped: { color: "var(--text-muted)", kind: "", icon: "·", label: "skipped" },
  failed: { color: COLORS.critical, kind: "critical", icon: "✕", label: "failed" },
};

/* ---------- OVERVIEW ---------- */

async function renderOverview(root) {
  root.textContent = "";
  await ensureSettings();
  const ov = await api(`/api/overview?window_s=${state.windowS}`);
  state.overview = ov;

  if (!ov.sample_ts) {
    const c = card(root, "No telemetry yet", "Nothing has been collected into this database.");
    const p = h("p", "empty", c);
    p.textContent = "Start collection with `streamlora collect --forecast`, or run `streamlora serve` (which collects by default).";
    return;
  }

  // ---- stat tiles
  const tiles = h("div", "grid tiles", root);
  const tileSpecs = [
    { sig: "cpu.util_pct", digits: 1 },
    { sig: "mem.used_pct", digits: 1 },
    { sig: "battery.percent", digits: 1 },
    { sig: "thermal.cpu_c", digits: 1 },
  ].filter((t) => ov.current[t.sig]);

  const hist = await api(
    `/api/telemetry?signals=${tileSpecs.map((t) => t.sig).join(",")}&window_s=${state.windowS}&max_points=120`
  );
  for (const spec of tileSpecs) {
    const t = h("div", "tile", tiles);
    const lab = h("div", "label", t);
    lab.textContent = label(spec.sig);
    const v = h("div", "value", t);
    const cur = ov.current[spec.sig].value;
    v.textContent = fmtNum(cur, spec.digits);
    const u = h("small", null, v);
    u.textContent = unitOf(spec.sig);
    const series = (hist.series[spec.sig] || []).filter((x) => x !== null);
    if (series.length > 3) {
      const first = series[0];
      const d = h("div", `delta${cur >= first ? "" : " good"}`, t);
      d.textContent = `${cur - first >= 0 ? "+" : ""}${fmtNum(cur - first, 1)}${unitOf(spec.sig)} over ${fmtDuration(state.windowS)}`;
      const sp = h("div", "spark", t);
      requestAnimationFrame(() => sparkline(sp, series));
    }
  }

  // ---- regime + next forecast
  const grid = h("div", "grid c2", root);
  const rc = card(grid, "Current regime", "Workload label, with hysteresis to stop it flapping");
  const hero = h("div", "hero", rc);
  hero.textContent = ov.regime || "unknown";
  const dist = Object.entries(ov.regime_distribution || {}).slice(0, 5);
  if (dist.length) {
    const p = h("p", "small muted", rc);
    p.textContent = "recent mix: " + dist.map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(" · ");
  }
  if (ov.sample_age_s !== null) {
    const p = h("p", "small muted", rc);
    p.textContent = `last sample ${fmtDuration(ov.sample_age_s)} ago`;
  }

  // next forecasts: learned model where available, else persistence
  const fc = card(grid, "What happens next", "Most recent forecast per signal and horizon");
  const byScope = new Map();
  for (const f of ov.forecasts) {
    const key = `${f.signal}@${f.horizon_s}`;
    const cur = byScope.get(key);
    const better = f.model_kind === "rls";
    if (!cur || (better && cur.model_kind !== "rls") ||
        (better === (cur.model_kind === "rls") && f.ts_made > cur.ts_made)) {
      byScope.set(key, f);
    }
  }
  const rows = [...byScope.values()].sort(
    (a, b) => a.signal.localeCompare(b.signal) || a.horizon_s - b.horizon_s
  );
  if (!rows.length) {
    const p = h("p", "empty", fc);
    p.textContent = "No forecasts yet — the model needs more history before it will predict.";
  } else {
    tableView(fc,
      ["Signal", "Horizon", "Now", "Forecast", "Likely range", "Model"],
      rows.map((f) => [
        label(f.signal),
        fmtDuration(f.horizon_s),
        f.anchor === null ? "–" : fmtNum(f.anchor, 1) + unitOf(f.signal),
        fmtNum(f.value, 1) + unitOf(f.signal),
        f.lo === null ? "not calibrated yet"
          : `${fmtNum(f.lo, 1)} – ${fmtNum(f.hi, 1)}${unitOf(f.signal)}`,
        f.model_kind === "rls" ? f.model_version : f.model_kind,
      ])
    );
  }

  // ---- recent accuracy vs baselines
  const acc = card(root, "Recent forecast accuracy",
    "Mean absolute error over the selected window, learned model against every baseline. Lower is better.");
  const learned = ov.accuracy.filter((r) => r.arm === "rls");
  if (!learned.length) {
    const p = h("p", "empty", acc);
    p.textContent = "No resolved forecasts in this window yet.";
  } else {
    const scopes = [...new Set(ov.accuracy.map((r) => `${r.signal}@${r.horizon_s}`))].sort(scopeSort);
    const arms = ["persistence", "moving_average", "ewma", "linear_trend", "rls"];
    const armColor = {
      persistence: "var(--text-muted)", moving_average: "var(--text-muted)",
      ewma: COLORS.baseline, linear_trend: COLORS.baseline, rls: COLORS.forecast,
    };
    legend(acc, [
      { name: "learned model", color: COLORS.forecast, rect: true },
      { name: "best baseline (EWMA / linear trend)", color: COLORS.baseline, rect: true },
      { name: "other baselines", color: "var(--text-muted)", rect: true },
    ]);
    withTableToggle(acc,
      (box) => {
        const groups = scopes.map((sc) => {
          const [sig, hor] = sc.split("@");
          return {
            label: label(sig).replace(" utilisation", ""),
            sublabel: fmtDuration(Number(hor)),
            bars: arms.map((a) => {
              const row = ov.accuracy.find(
                (r) => r.arm === a && `${r.signal}@${r.horizon_s}` === sc
              );
              return { name: a, value: row ? row.mae : NaN, color: armColor[a] };
            }).filter((b) => isFinite(b.value)),
          };
        }).filter((g) => g.bars.length && !degenerate(g.bars));
        if (!groups.length) {
          const p = h("p", "empty", box);
          p.textContent = "Every scope has zero error in this window (a constant signal, e.g. battery while on AC). Use the table view for the raw values.";
          return;
        }
        requestAnimationFrame(() => groupedBars(box, { groups, yLabel: "MAE", height: 250 }));
      },
      (box) => {
        tableView(box,
          ["Scope", ...arms, "n"],
          scopes.map((sc) => {
            const cells = arms.map((a) => {
              const row = ov.accuracy.find(
                (r) => r.arm === a && `${r.signal}@${r.horizon_s}` === sc
              );
              return row ? row.mae : null;
            });
            const best = Math.min(...cells.filter((c) => c !== null));
            const n = ov.accuracy.find((r) => `${r.signal}@${r.horizon_s}` === sc)?.n || 0;
            return [sc, ...cells.map((c) => c === null
              ? "–" : { text: fmtNum(c, 3), best: c === best }), n];
          })
        );
      }
    );
  }

  // ---- events
  const ev = card(root, "Recent drift and adaptation",
    "Behaviour changes detected, and what the system did about them");
  const all = [
    ...ov.drift_events.map((d) => ({
      ts: d.ts, lane: "drift", color: COLORS.warning,
      label: `drift: ${d.detector} (${d.scope}${d.signal ? " · " + d.signal : ""})`,
      value: `stat ${fmtNum(d.statistic, 2)}`,
    })),
    ...ov.adapt_events.filter((a) => a.decision !== "skipped").map((a) => {
      const st = DECISION_STYLE[a.decision] || DECISION_STYLE.skipped;
      return {
        ts: a.ts, lane: a.decision === "rolled_back" ? "rollback" : "adapt",
        color: st.color,
        label: `${st.label}: ${a.scope} (${a.trigger})`,
        value: a.metric_before !== null && a.metric_after !== null
          ? `MAE ${fmtNum(a.metric_before, 3)} → ${fmtNum(a.metric_after, 3)}` : "",
      };
    }),
  ];
  if (!all.length) {
    const p = h("p", "empty", ev);
    p.textContent = "No drift or adaptation events in the last 24 hours.";
  } else {
    legend(ev, [
      { name: "▲ promoted", color: COLORS.good, rect: true },
      { name: "■ rejected", color: COLORS.serious, rect: true },
      { name: "▼ rolled back", color: COLORS.critical, rect: true },
      { name: "drift detected", color: COLORS.warning, rect: true },
    ]);
    const box = h("div", null, ev);
    requestAnimationFrame(() => eventLane(box, {
      events: all, height: 118,
      lanes: [
        { key: "drift", label: "drift" },
        { key: "adapt", label: "adaptation" },
        { key: "rollback", label: "rollback" },
      ],
    }));
    const recent = all.sort((a, b) => b.ts - a.ts).slice(0, 6);
    const ul = h("div", "small", ev);
    ul.style.marginTop = "10px";
    for (const e of recent) {
      const line = h("div", null, ul);
      line.style.padding = "2px 0";
      const t = h("span", "pill", line);
      t.textContent = fmtClock(e.ts);
      const s = h("span", null, line);
      s.textContent = "  " + e.label + (e.value ? "  " + e.value : "");
      s.style.color = "var(--text-secondary)";
    }
  }
}

/* ---------- FORECASTS ---------- */

async function renderForecasts(root) {
  root.textContent = "";
  const settings = await ensureSettings();
  const targets = settings.targets.filter((t) =>
    settings.signals.some((s) => s.name === t));
  if (!targets.length) {
    const c = card(root, "No forecast targets available", "");
    const p = h("p", "empty", c);
    p.textContent = "None of the configured forecast targets exist on this machine.";
    return;
  }
  state.forecastSignal = state.forecastSignal && targets.includes(state.forecastSignal)
    ? state.forecastSignal : targets[0];
  const horizons = settings.horizons_s;
  state.forecastHorizon = state.forecastHorizon ?? horizons[0];

  // filters in one row above the charts
  const row = h("div", "row", root);
  const mk = (labelText, options, value, onChange) => {
    const f = h("label", "field", row);
    const sp = h("span", null, f);
    sp.textContent = labelText;
    const sel = h("select", null, f);
    for (const o of options) {
      const opt = h("option", null, sel);
      opt.value = String(o.value);
      opt.textContent = o.label;
      if (String(o.value) === String(value)) opt.selected = true;
    }
    sel.addEventListener("change", () => onChange(sel.value));
    return sel;
  };
  mk("Signal", targets.map((t) => ({ value: t, label: label(t) })), state.forecastSignal,
    (v) => { state.forecastSignal = v; renderForecasts(root); });
  mk("Horizon", horizons.map((hz) => ({ value: hz, label: fmtDuration(hz) })),
    state.forecastHorizon, (v) => { state.forecastHorizon = Number(v); renderForecasts(root); });
  mk("Window", [
    { value: 1800, label: "30 min" }, { value: 3600, label: "1 hour" },
    { value: 10800, label: "3 hours" }, { value: 43200, label: "12 hours" },
    { value: 86400, label: "24 hours" },
  ], state.windowS, (v) => { state.windowS = Number(v); renderForecasts(root); });

  const sig = state.forecastSignal;
  const hz = state.forecastHorizon;
  const [fc, ev] = await Promise.all([
    api(`/api/forecasts?signal=${encodeURIComponent(sig)}&horizon_s=${hz}&window_s=${state.windowS}&model_kind=rls`),
    api(`/api/events?window_s=${state.windowS}`),
  ]);

  const overlay = card(root,
    `${label(sig)} — measured against forecast at ${fmtDuration(hz)}`,
    "Each forecast is plotted at the time it was for, so the dashed line is what the model said would happen at that instant.");
  legend(overlay, [
    { name: "measured", color: COLORS.actual },
    { name: `forecast (+${fmtDuration(hz)})`, color: COLORS.forecast, dash: true },
    { name: "likely range (90%)", color: COLORS.forecast, rect: true },
  ]);
  const actualPts = fc.actual.ts.map((t, i) => [t, fc.actual.values[i]]);
  const fcPts = fc.resolved.map((r) => [r.ts_target, r.value]);
  const pending = fc.pending.map((r) => [r.ts_target, r.value]);
  const band = fc.resolved.filter((r) => r.lo !== null).map((r) => [r.ts_target, r.lo, r.hi]);
  const bandP = fc.pending.filter((r) => r.lo !== null).map((r) => [r.ts_target, r.lo, r.hi]);
  const driftEvents = ev.drift
    .filter((d) => !d.signal || d.signal === sig)
    .map((d) => ({ ts: d.ts, color: COLORS.warning }));

  withTableToggle(overlay,
    (box) => requestAnimationFrame(() => timeSeries(box, {
      series: [
        { name: "measured", color: COLORS.actual, points: actualPts },
        { name: "forecast", color: COLORS.forecast, points: fcPts, dash: true },
        { name: "forecast (pending)", color: COLORS.forecast, points: pending, dash: true, width: 1.5 },
      ].filter((s) => s.points.length),
      bands: [
        { name: "range", color: COLORS.forecast, points: band },
        { name: "range", color: COLORS.forecast, points: bandP },
      ].filter((b) => b.points.length),
      events: driftEvents,
      now: Date.now() / 1000,
      height: 300,
      yLabel: label(sig) + unitOf(sig),
      valueFmt: (v) => fmtNum(v, 1) + unitOf(sig),
    })),
    (box) => tableView(box,
      ["Target time", "Forecast", "Actual", "Error", "In range", "Model version"],
      fc.resolved.slice(-60).reverse().map((r) => [
        fmtClock(r.ts_target), fmtNum(r.value, 2), fmtNum(r.actual, 2),
        fmtNum(r.error, 2), r.in_interval === null ? "–" : (r.in_interval ? "yes" : "no"),
        r.model_version,
      ]),
      { caption: "most recent 60 resolved forecasts" })
  );

  // error over time, with drift/adaptation markers
  const errCard = card(root, "Absolute error over time",
    "Rising error is what the drift detectors watch. Vertical marks are drift alarms; the adaptation lane below shows the response.");
  const errPts = fc.resolved.map((r) => [r.ts_target, Math.abs(r.error)]);
  legend(errCard, [
    { name: "|error| of the learned model", color: COLORS.forecast },
    { name: "drift alarm", color: COLORS.warning, dash: true },
  ]);
  withTableToggle(errCard,
    (box) => requestAnimationFrame(() => timeSeries(box, {
      series: [{ name: "|error|", color: COLORS.forecast, points: errPts }],
      events: driftEvents, height: 190, yZero: true,
      yLabel: "absolute error" + unitOf(sig),
      valueFmt: (v) => fmtNum(v, 2),
    })),
    (box) => tableView(box, ["Target time", "|error|"],
      errPts.slice(-60).reverse().map(([t, v]) => [fmtClock(t), fmtNum(v, 3)]))
  );

  // Per-forecast feedback: "was this forecast useful?" and "did the predicted
  // event happen?", stored against the prediction id rather than as chat text.
  const fbCard = card(root, "Was this forecast useful?",
    "Feedback is stored against the specific prediction, so usefulness is answerable per scope — not inferred from chat logs.");
  const latest = fc.pending[fc.pending.length - 1] || fc.resolved[fc.resolved.length - 1];
  if (!latest) {
    const p = h("p", "empty", fbCard);
    p.textContent = "No forecast to rate yet.";
  } else {
    const info = h("p", "small muted", fbCard);
    info.textContent =
      `${label(sig)} at ${fmtClock(latest.ts_target)}: ${fmtNum(latest.value, 1)}${unitOf(sig)}` +
      (latest.actual !== undefined && latest.actual !== null
        ? ` (actual ${fmtNum(latest.actual, 1)}${unitOf(sig)})` : "");
    const row2 = h("div", "row", fbCard);
    const out = h("span", "small muted", row2);
    const send = async (kind, lbl, btn) => {
      for (const b of row2.querySelectorAll("button")) b.disabled = true;
      try {
        const r = await api("/api/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            kind, label: lbl, signal: sig,
            prediction_id: latest.id ?? null,
            text: `${sig}@${hz}`,
          }),
        });
        out.textContent = `recorded (${r.summary.total} feedback items)`;
      } catch (e) { out.textContent = "failed: " + e.message; }
    };
    for (const [labelText, kind, lbl] of [
      ["Useful", "forecast_useful", "yes"],
      ["Not useful", "forecast_useful", "no"],
      ["The predicted change happened", "event_happened", "yes"],
      ["It did not happen", "event_happened", "no"],
    ]) {
      const b = h("button", "ghost", row2);
      b.type = "button"; b.textContent = labelText;
      b.addEventListener("click", () => send(kind, lbl, b));
    }
    row2.appendChild(out);
  }

  const adaptEvents = ev.adaptation
    .filter((a) => a.scope === `${sig}@${Math.round(hz)}` && a.decision !== "skipped")
    .map((a) => {
      const st = DECISION_STYLE[a.decision] || DECISION_STYLE.skipped;
      return {
        ts: a.ts, lane: a.decision === "rolled_back" ? "rollback" : "adapt", color: st.color,
        label: `${st.label} · ${a.trigger}`,
        value: a.metric_before !== null ? `MAE ${fmtNum(a.metric_before, 3)} → ${fmtNum(a.metric_after, 3)}` : "",
      };
    });
  const laneCard = card(root, "Adaptation for this scope",
    "Every gate decision for this signal and horizon");
  if (!adaptEvents.length) {
    const p = h("p", "empty", laneCard);
    p.textContent = "No adaptation decisions for this scope in the window.";
  } else {
    legend(laneCard, [
      { name: "▲ promoted", color: COLORS.good, rect: true },
      { name: "■ rejected", color: COLORS.serious, rect: true },
      { name: "▼ rolled back", color: COLORS.critical, rect: true },
    ]);
    const box = h("div", null, laneCard);
    requestAnimationFrame(() => eventLane(box, {
      events: adaptEvents, height: 96,
      lanes: [{ key: "adapt", label: "gate decision" }, { key: "rollback", label: "rollback" }],
    }));
  }
}

/* ---------- TELEMETRY ---------- */

async function renderTelemetry(root) {
  root.textContent = "";
  const settings = await ensureSettings();
  const groups = {
    CPU: ["cpu.util_pct", "cpu.util_max_core_pct", "cpu.iowait_pct", "cpu.freq_mhz"],
    Memory: ["mem.used_pct", "mem.available_gb", "mem.cached_gb", "mem.swap_used_pct"],
    Power: ["battery.percent", "battery.power_w", "battery.plugged"],
    Thermal: ["thermal.cpu_c", "thermal.gpu_c", "thermal.disk_c", "thermal.max_c"],
    GPU: ["gpu.util_pct", "gpu.power_w", "gpu.temp_c", "gpu.mem_used_pct"],
    "I/O": ["disk.busy_pct", "disk.read_mbps", "disk.write_mbps", "net.recv_mbps", "net.sent_mbps"],
    Workload: ["proc.cpu_build_pct", "proc.cpu_browser_pct", "proc.cpu_ml_pct",
               "proc.top1_cpu_pct", "proc.concentration"],
  };
  const available = new Set(settings.signals.map((s) => s.name));

  const row = h("div", "row", root);
  const f = h("label", "field", row);
  const sp = h("span", null, f);
  sp.textContent = "Window";
  const sel = h("select", null, f);
  for (const o of [
    { v: 1800, l: "30 min" }, { v: 3600, l: "1 hour" }, { v: 10800, l: "3 hours" },
    { v: 43200, l: "12 hours" }, { v: 86400, l: "24 hours" },
  ]) {
    const opt = h("option", null, sel);
    opt.value = String(o.v); opt.textContent = o.l;
    if (o.v === state.windowS) opt.selected = true;
  }
  sel.addEventListener("change", () => {
    state.windowS = Number(sel.value); renderTelemetry(root);
  });
  const note = h("span", "small muted", row);
  note.textContent = "Signals are shown as small multiples with one axis each — never two scales on one chart.";

  const wanted = Object.values(groups).flat().filter((s) => available.has(s));
  const data = await api(
    `/api/telemetry?signals=${wanted.join(",")}&window_s=${state.windowS}&max_points=900`
  );
  for (const [group, sigs] of Object.entries(groups)) {
    const present = sigs.filter((s) => available.has(s) && (data.series[s] || []).some((v) => v !== null));
    if (!present.length) continue;
    const hdr = h("h2", null, root);
    hdr.textContent = group;
    hdr.style.cssText = "font-size:13px;margin:18px 0 8px;color:var(--text-secondary);font-weight:650";
    const grid = h("div", "grid c3", root);
    for (const sig of present) {
      const c = card(grid, label(sig), sig);
      const pts = data.ts.map((t, i) => [t, data.series[sig][i]]);
      withTableToggle(c,
        (box) => requestAnimationFrame(() => timeSeries(box, {
          series: [{ name: label(sig), color: COLORS.actual, points: pts }],
          height: 150, yLabel: unitOf(sig).trim(),
          valueFmt: (v) => fmtNum(v, 1) + unitOf(sig),
        })),
        (box) => tableView(box, ["Time", label(sig)],
          pts.filter((p) => p[1] !== null).slice(-40).reverse()
            .map(([t, v]) => [fmtClock(t), fmtNum(v, 2)]))
      );
    }
  }
  const meta = h("p", "small muted", root);
  meta.textContent = `${data.n_total} samples in window` +
    (data.decimation > 1 ? `, showing every ${data.decimation}${data.decimation === 2 ? "nd" : "th"}` : "");
}

/* ---------- ADAPTATION ---------- */

async function renderAdaptation(root) {
  root.textContent = "";
  const [models, ev] = await Promise.all([
    api("/api/models?kind=forecast-model&limit=400"),
    api("/api/events?window_s=604800"),
  ]);

  const summary = card(root, "Adaptation outcomes",
    "Every candidate is scored on held-out recent data against the version it would replace. Rejection is the normal case, not a failure.");
  const counts = {};
  for (const a of ev.adaptation) counts[a.decision] = (counts[a.decision] || 0) + 1;
  const tiles = h("div", "grid tiles", summary);
  for (const key of ["promoted", "rejected", "rolled_back", "skipped"]) {
    const st = DECISION_STYLE[key];
    const t = h("div", "tile", tiles);
    const l = h("div", "label", t);
    l.textContent = st.label;
    const v = h("div", "value", t);
    v.textContent = String(counts[key] || 0);
    v.style.color = key === "skipped" ? "var(--text-primary)" : st.color;
  }
  const p = h("p", "small muted", summary);
  const promoted = counts.promoted || 0, rejected = counts.rejected || 0;
  p.textContent = promoted + rejected > 0
    ? `The gate rejected ${((rejected / (promoted + rejected)) * 100).toFixed(0)}% of candidates.`
    : "No adaptation decisions recorded yet.";

  // active versions
  const active = models.versions.filter((v) => v.active);
  const av = card(root, "Active model versions",
    "One version per (signal, horizon), so a bad update to one scope never rolls back another");
  if (!active.length) {
    const e = h("p", "empty", av);
    e.textContent = "No model versions registered yet.";
  } else {
    const wrap = h("div", "tablewrap", av);
    const t = h("table", "data", wrap);
    const thead = h("thead", null, t);
    const tr = h("tr", null, thead);
    for (const c of ["Scope", "Version", "Trained on", "Gate MAE before → after", "Updated", ""]) {
      const th = h("th", null, tr); th.textContent = c;
    }
    const tb = h("tbody", null, t);
    for (const v of active.sort((a, b) => a.scope.localeCompare(b.scope))) {
      const row = h("tr", null, tb);
      const g = (v.metrics && v.metrics.gate) || {};
      const cells = [
        v.scope, v.version, String(v.n_train),
        g.active_mae !== undefined && g.active_mae !== null
          ? `${fmtNum(g.active_mae, 3)} → ${fmtNum(g.candidate_mae, 3)}` : "–",
        fmtClock(v.created_ts),
      ];
      for (const c of cells) { const td = h("td", null, row); td.textContent = c; }
      const td = h("td", null, row);
      const btn = h("button", "ghost", td);
      btn.type = "button"; btn.textContent = "Roll back";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          const r = await api("/api/models/rollback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scope: v.scope, kind: "forecast-model" }),
          });
          btn.textContent = `→ ${r.to}`;
          setTimeout(() => renderAdaptation(root), 700);
        } catch (err) {
          btn.textContent = "failed";
          const e = h("div", "err", td); e.textContent = String(err.message);
        }
      });
    }
  }

  // timeline
  const tl = card(root, "Adaptation and drift timeline", "The last seven days");
  const events = [
    ...ev.drift.map((d) => ({
      ts: d.ts, lane: "drift", color: COLORS.warning,
      label: `drift · ${d.detector} · ${d.scope}`, value: `stat ${fmtNum(d.statistic, 2)}`,
    })),
    ...ev.adaptation.filter((a) => a.decision !== "skipped").map((a) => {
      const st = DECISION_STYLE[a.decision] || DECISION_STYLE.skipped;
      return {
        ts: a.ts, lane: a.kind === "language" ? "language" :
          (a.decision === "rolled_back" ? "rollback" : "adapt"),
        color: st.color, label: `${st.label} · ${a.scope}`,
        value: a.metric_before !== null
          ? `${a.metric_name} ${fmtNum(a.metric_before, 3)} → ${fmtNum(a.metric_after, 3)}` : "",
      };
    }),
  ];
  if (!events.length) {
    const e = h("p", "empty", tl);
    e.textContent = "No events yet.";
  } else {
    legend(tl, [
      { name: "▲ promoted", color: COLORS.good, rect: true },
      { name: "■ rejected", color: COLORS.serious, rect: true },
      { name: "▼ rolled back", color: COLORS.critical, rect: true },
      { name: "drift", color: COLORS.warning, rect: true },
    ]);
    const box = h("div", null, tl);
    requestAnimationFrame(() => eventLane(box, {
      events, height: 150,
      lanes: [
        { key: "drift", label: "drift" }, { key: "adapt", label: "forecast gate" },
        { key: "rollback", label: "rollback" }, { key: "language", label: "language adapter" },
      ],
    }));
  }

  // full history table
  const hist = card(root, "Decision log", "Newest first");
  tableView(hist,
    ["Time", "Scope", "Kind", "Trigger", "Decision", "MAE before", "MAE after", "Gate n", "Trained"],
    ev.adaptation.slice(0, 200).map((a) => [
      fmtClock(a.ts), a.scope, a.kind, a.trigger,
      (DECISION_STYLE[a.decision] || { label: a.decision }).label,
      a.metric_before === null ? "–" : fmtNum(a.metric_before, 4),
      a.metric_after === null ? "–" : fmtNum(a.metric_after, 4),
      a.gate_n, a.n_train,
    ])
  );
}

/* ---------- EXPERIMENTS ---------- */

async function renderExperiments(root) {
  root.textContent = "";
  const list = await api("/api/experiments");
  if (!list.experiments.length) {
    const c = card(root, "No experiments yet", "");
    const p = h("p", "empty", c);
    p.textContent = "Run `streamlora experiment --dataset scenario:permanent_regime_change --suite core` and reload.";
    return;
  }
  const row = h("div", "row", root);
  const f = h("label", "field", row);
  const sp = h("span", null, f); sp.textContent = "Experiment";
  const sel = h("select", null, f);
  for (const e of list.experiments) {
    const o = h("option", null, sel);
    o.value = e.id; o.textContent = `${e.name} · ${e.dataset || ""}`.slice(0, 90);
  }
  const body = h("div", null, root);
  const load = async () => {
    body.textContent = "";
    const data = await api(`/api/experiments/${encodeURIComponent(sel.value)}`);
    if (data.model_id) renderLanguageExperiment(body, data);
    else renderForecastExperiment(body, data);
  };
  sel.addEventListener("change", load);
  await load();
}

function renderForecastExperiment(root, data) {
  const meta = card(root, data.name || data.experiment_id,
    `${data.dataset} · train_frac ${data.train_frac} · metrics on the held-out tail only`);
  const armNames = data.arms.map((a) => a.arm.name);
  const first = data.arms[0];
  if (!first) return;
  const scopes = [...new Set(first.metrics.map((m) => `${m.signal}@${m.horizon_s}`))]
    .sort(scopeSort);
  const baselineArms = ["persistence", "ewma", "moving_average", "linear_trend"];

  // Fixed slot order, never cycled: a 9th arm would fold into "other" rather
  // than reuse slot 1 and silently impersonate the first arm.
  const shown = data.arms.slice(0, SERIES.length);
  const armColor = (name) => {
    const i = shown.findIndex((a) => a.arm.name === name);
    return i < 0 ? COLORS.muted : SERIES[i];
  };

  const chart = card(root, "MAE by scope",
    "Baselines shared across arms (identical data); one learned-model bar per arm. Lower is better.");
  legend(chart, [
    { name: "best baseline", color: "var(--text-muted)", rect: true },
    ...shown.map((a) => ({ name: a.arm.name, color: armColor(a.arm.name), rect: true })),
  ]);
  if (data.arms.length > shown.length) {
    const p = h("p", "small muted", chart);
    p.textContent = `showing the first ${shown.length} arms; re-run with fewer arms to compare the rest`;
  }
  withTableToggle(chart,
    (box) => {
      const groups = scopes.map((sc) => {
        const bars = [];
        const bl = first.metrics.filter(
          (m) => `${m.signal}@${m.horizon_s}` === sc && baselineArms.includes(m.arm)
        );
        if (bl.length) {
          const best = bl.reduce((a, b) => (a.mae <= b.mae ? a : b));
          bars.push({ name: `best baseline (${best.arm})`, value: best.mae, color: "var(--text-muted)" });
        }
        shown.forEach((a) => {
          const m = a.metrics.find((x) => x.arm === "rls" && `${x.signal}@${x.horizon_s}` === sc);
          if (m && m.mae !== null) {
            bars.push({ name: a.arm.name, value: m.mae, color: armColor(a.arm.name) });
          }
        });
        const [sig, hor] = sc.split("@");
        return { label: label(sig).replace(" utilisation", ""), sublabel: fmtDuration(Number(hor)), bars };
      }).filter((g) => g.bars.length > 1 && !degenerate(g.bars));
      if (!groups.length) {
        const p = h("p", "empty", box);
        p.textContent = "No scope in this experiment has non-zero error. Use the table view.";
        return;
      }
      requestAnimationFrame(() => groupedBars(box, { groups, yLabel: "MAE", height: 290 }));
    },
    (box) => {
      const cols = ["Scope", "best baseline", ...armNames];
      const rows = scopes.map((sc) => {
        const bl = first.metrics.filter(
          (m) => `${m.signal}@${m.horizon_s}` === sc && baselineArms.includes(m.arm)
        );
        const best = bl.length ? bl.reduce((a, b) => (a.mae <= b.mae ? a : b)) : null;
        const vals = data.arms.map((a) => {
          const m = a.metrics.find((x) => x.arm === "rls" && `${x.signal}@${x.horizon_s}` === sc);
          return m ? m.mae : null;
        });
        const all = [best ? best.mae : null, ...vals].filter((v) => v !== null);
        const bestVal = all.length ? Math.min(...all) : null;
        return [sc,
          best ? { text: `${fmtNum(best.mae, 4)} (${best.arm})`, best: best.mae === bestVal } : "–",
          ...vals.map((v) => v === null ? "–" : { text: fmtNum(v, 4), best: v === bestVal })];
      });
      tableView(box, cols, rows,
        { caption: "MAE on the held-out tail. Bold is the best value in the row." });
    }
  );

  const ops = card(root, "Adaptation cost per arm",
    "How much work each policy did to get that accuracy");
  tableView(ops,
    ["Arm", "Description", "Promoted", "Rejected", "Rolled back", "Drift events", "Speedup"],
    data.arms.map((a) => [
      a.arm.name, a.arm.description,
      (a.adapt || {}).promoted ?? "–", (a.adapt || {}).rejected ?? "–",
      (a.adapt || {}).rolled_back ?? "–", (a.drift || {}).events ?? "–",
      a.replay ? `${fmtNum(a.replay.speedup, 0)}×` : "–",
    ])
  );
}

function renderLanguageExperiment(root, data) {
  const meta = card(root, `Language layer · ${data.model_id}`,
    `${data.device} · train ${data.n_train} / test ${data.n_test} · persona rev ${data.persona_revision}`);
  const arms = data.arms.filter((a) => !a.error);
  const failed = data.arms.filter((a) => a.error);

  const q = card(root, "Grounding and style compliance",
    "Groundedness = share of generated numbers that appear in the evidence. Style = share of persona checks passed. Both higher is better.");
  legend(q, [
    { name: "groundedness", color: COLORS.actual, rect: true },
    { name: "style compliance", color: COLORS.baseline, rect: true },
  ]);
  withTableToggle(q,
    (box) => requestAnimationFrame(() => groupedBars(box, {
      groups: arms.map((a) => ({
        label: a.arm,
        bars: [
          { name: "groundedness", value: a.groundedness ?? NaN, color: COLORS.actual },
          { name: "style", value: a.style_compliance ?? NaN, color: COLORS.baseline },
        ].filter((b) => isFinite(b.value)),
      })).filter((g) => g.bars.length),
      yLabel: "rate", height: 240, lowerIsBetter: false,
      valueFmt: (v) => v.toFixed(2),
    })),
    (box) => tableView(box,
      ["Arm", "n", "eval loss", "ppl", "grounded", "fully grounded", "style", "fully compliant"],
      arms.map((a) => [a.arm, a.n, a.eval_loss ?? "–", a.eval_ppl ?? "–",
        a.groundedness ?? "–", a.fully_grounded_rate ?? "–",
        a.style_compliance ?? "–", a.style_ok_rate ?? "–"])
    )
  );

  const cost = card(root, "Cost of personalisation",
    "Prompt tokens are paid on every request; training cost is paid once.");
  tableView(cost,
    ["Arm", "Prompt tokens/req", "Trainable params", "Train wall (s)", "Peak GPU (MB)", "Artifact (MB)"],
    data.arms.map((a) => {
      const t = a.train || {};
      return [a.arm,
        a.mean_prompt_tokens === null || a.mean_prompt_tokens === undefined
          ? "–" : fmtNum(a.mean_prompt_tokens, 0),
        t.trainable_params ? t.trainable_params.toLocaleString() : "–",
        t.wall_s ?? "–", t.peak_mem_mb ?? "–", t.artifact_mb ?? "–"];
    })
  );

  const samples = card(root, "Sample answers", "First held-out example per arm");
  for (const a of arms) {
    if (!a.samples || !a.samples.length) continue;
    const s = a.samples[0];
    const b = h("div", null, samples);
    b.style.cssText = "margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid var(--grid)";
    const t = h("div", null, b);
    t.style.cssText = "font-weight:650;font-size:12px;margin-bottom:3px";
    t.textContent = a.arm;
    const qq = h("div", "small muted", b); qq.textContent = s.question;
    const aa = h("div", null, b); aa.textContent = s.answer;
    aa.style.marginTop = "4px";
    const m = h("div", "meta", b);
    badge(m, `grounded ${s.groundedness}`, s.groundedness === 1 ? "good" : "critical",
      s.groundedness === 1 ? "✓" : "!");
    if (s.style_violations && s.style_violations.length) {
      badge(m, s.style_violations.join(", "), "serious", "■");
    }
  }
  if (failed.length) {
    const f = card(root, "Arms that did not run", "");
    tableView(f, ["Arm", "Reason"], failed.map((a) => [a.arm, a.error]));
  }
  if (data.limitations && data.limitations.length) {
    const lim = card(root, "Stated limitations", "");
    for (const l of data.limitations) {
      const p = h("p", "small muted", lim);
      p.textContent = "• " + l;
    }
  }
}

/* ---------- CHAT ---------- */

async function renderChat(root) {
  root.textContent = "";
  const c = card(root, "Ask about your machine",
    "Answers are built from stored telemetry and forecasts. Any number that is not in the evidence is refused, and the deterministic answer is served instead.");
  const suggest = h("div", "suggest", c);
  for (const q of [
    "What is my machine doing right now?",
    "Why is my CPU usage rising?",
    "Will my battery last another hour?",
    "Was today's behaviour unusual?",
    "Why was the last forecast wrong?",
    "Has the model been adapting?",
  ]) {
    const b = h("button", "ghost", suggest);
    b.type = "button"; b.textContent = q;
    b.addEventListener("click", () => { input.value = q; send(); });
  }
  const log = h("div", "chatlog", c);
  const bar = h("div", "chatbar", c);
  const input = h("input", null, bar);
  input.type = "text";
  input.placeholder = "Ask about CPU, memory, battery, forecasts, drift…";
  input.setAttribute("aria-label", "Question");
  const btn = h("button", "ghost", bar);
  btn.type = "button"; btn.textContent = "Ask";

  function draw() {
    log.textContent = "";
    for (const m of state.chat) {
      if (m.role === "user") {
        const d = h("div", "msg user", log);
        d.textContent = m.text;
        continue;
      }
      const d = h("div", "msg bot", log);
      const body = h("div", "body", d);
      body.textContent = m.answer.text;
      const meta = h("div", "meta", d);
      badge(meta, m.answer.backend, "", "◆");
      if (m.answer.adapter) badge(meta, `adapter ${m.answer.adapter}`, "good", "▲");
      badge(meta, m.answer.grounded ? "grounded" : "ungrounded — served deterministic answer",
        m.answer.grounded ? "good" : "critical", m.answer.grounded ? "✓" : "!");
      if (m.answer.style_violations && m.answer.style_violations.length) {
        badge(meta, `style: ${m.answer.style_violations.join(", ")}`, "serious", "■");
      }
      const ms = h("span", "pill", meta);
      ms.textContent = `${Math.round(m.answer.latency_ms)} ms`;
      // feedback -> structured rows, not chat logs
      const fbYes = h("button", "ghost", meta);
      fbYes.type = "button"; fbYes.textContent = "Useful";
      const fbNo = h("button", "ghost", meta);
      fbNo.type = "button"; fbNo.textContent = "Not useful";
      const sendFb = async (label) => {
        fbYes.disabled = fbNo.disabled = true;
        try {
          const r = await api("/api/feedback", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ kind: "forecast_useful", label, text: m.question }),
          });
          badge(meta, `recorded (persona rev ${r.persona_revision})`, "good", "✓");
        } catch (e) { badge(meta, "failed", "critical", "✕"); }
      };
      fbYes.addEventListener("click", () => sendFb("yes"));
      fbNo.addEventListener("click", () => sendFb("no"));

      if (m.answer.evidence_text) {
        const det = h("details", "evidence", d);
        const sum = h("summary", null, det);
        sum.textContent = "Show the evidence this answer used";
        const pre = h("pre", null, det);
        pre.textContent = m.answer.evidence_text;
      }
    }
    log.scrollTop = log.scrollHeight;
  }

  async function send() {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    state.chat.push({ role: "user", text: q });
    draw();
    try {
      const a = await api("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, window_s: state.windowS }),
      });
      state.chat.push({ role: "bot", question: q, answer: a });
    } catch (e) {
      state.chat.push({
        role: "bot", question: q,
        answer: { text: `Error: ${e.message}`, backend: "error", grounded: false,
                  latency_ms: 0, evidence: [], style_violations: [] },
      });
    }
    draw();
  }
  btn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  draw();

  // teach the system something
  const teach = card(root, "Teach the system",
    "Free-text notes become structured rules where they can be parsed, and are quoted back in future explanations. Everything else is kept as context.");
  const ta = h("textarea", null, teach);
  ta.rows = 2; ta.style.width = "100%";
  ta.placeholder = 'e.g. "That happens when I\'m compiling projects. Treat that as normal for me."';
  const tb = h("div", "row", teach);
  const save = h("button", "ghost", tb);
  save.type = "button"; save.textContent = "Save as label";
  const out = h("span", "small muted", tb);
  save.addEventListener("click", async () => {
    const text = ta.value.trim();
    if (!text) return;
    save.disabled = true;
    try {
      const r = await api("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "label", text }),
      });
      out.textContent = `saved · persona revision ${r.persona_revision} · ` +
        `${r.summary.rules_inferred} rule(s) inferred from ${r.summary.total} feedback items`;
      ta.value = "";
    } catch (e) { out.textContent = "failed: " + e.message; }
    save.disabled = false;
  });

  try {
    const fb = await api("/api/feedback?limit=50");
    const pc = card(root, "What the system has learned about you",
      `Persona revision ${fb.persona.revision}`);
    const rules = fb.persona.rules || [];
    if (!rules.length && !(fb.persona.context || []).length) {
      const p = h("p", "empty", pc);
      p.textContent = "Nothing yet. Tell it something above.";
    } else {
      tableView(pc, ["Regime", "Verdict", "Your words"],
        rules.map((r) => [r.regime, r.verdict, r.note]));
      for (const ctx of fb.persona.context || []) {
        const p = h("p", "small muted", pc);
        p.textContent = "context: " + ctx;
      }
    }
  } catch { /* feedback view is optional */ }
}

/* ---------- shell ---------- */

const VIEWS = {
  overview: renderOverview, forecasts: renderForecasts, telemetry: renderTelemetry,
  adaptation: renderAdaptation, experiments: renderExperiments, chat: renderChat,
};

async function show(name) {
  state.view = name;
  for (const btn of document.querySelectorAll("#tabs button")) {
    if (btn.dataset.view === name) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  }
  for (const sec of document.querySelectorAll("section.view")) {
    sec.hidden = sec.id !== `view-${name}`;
  }
  const root = document.getElementById(`view-${name}`);
  try {
    await VIEWS[name](root);
  } catch (e) {
    root.textContent = "";
    const c = card(root, "Could not load this view", "");
    const p = h("p", "err", c);
    p.textContent = String(e.message || e);
  }
}

async function poll() {
  try {
    const hlth = await api("/api/health");
    const dot = document.getElementById("livedot");
    const txt = document.getElementById("livetext");
    const age = hlth.telemetry?.collector?.last_ts
      ? (Date.now() / 1000 - hlth.telemetry.collector.last_ts) : null;
    if (!hlth.collecting) {
      dot.className = "livedot off";
      const upTo = hlth.telemetry_span?.to;
      txt.textContent = upTo
        ? `stored data to ${fmtClock(upTo)} · ${hlth.samples_total.toLocaleString()} samples`
        : `stored data · ${hlth.samples_total.toLocaleString()} samples`;
    } else if (age !== null && age > 30) {
      dot.className = "livedot stale";
      txt.textContent = `stale ${fmtDuration(age)}`;
    } else {
      dot.className = "livedot";
      txt.textContent = `live · ${hlth.samples_total.toLocaleString()} samples`;
    }
  } catch {
    const dot = document.getElementById("livedot");
    dot.className = "livedot off";
    document.getElementById("livetext").textContent = "disconnected";
  }
  if (state.view === "overview") {
    try { await renderOverview(document.getElementById("view-overview")); } catch { /* keep polling */ }
  }
}

document.getElementById("tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) show(b.dataset.view);
});
document.getElementById("themebtn").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : cur === "light" ? null : "dark";
  if (next) document.documentElement.setAttribute("data-theme", next);
  else document.documentElement.removeAttribute("data-theme");
  if (state.view !== "chat") show(state.view);
});
window.addEventListener("resize", () => {
  clearTimeout(state.resizeTimer);
  state.resizeTimer = setTimeout(() => { if (state.view !== "chat") show(state.view); }, 220);
});

show("overview");
poll();
setInterval(poll, 15000);
