/* Hand-rolled SVG chart primitives.
 *
 * No charting library: the dashboard must work with no network access and no
 * build step, and the four forms needed here (time series with a forecast
 * overlay, grouped bars, an event lane, a sparkline) are a few hundred lines.
 * A CDN dependency would also mean telemetry pages fetching from a third party,
 * which is the opposite of the privacy stance in the rest of the system.
 *
 * Conventions held throughout, per the project's visualisation rules:
 *  - one y-axis per chart, never two;
 *  - 2px lines with round caps, >=8px markers with a 2px surface ring;
 *  - hairline solid gridlines, recessive;
 *  - a legend whenever two or more series are drawn, plus sparing direct
 *    end-labels; text never wears the series colour;
 *  - crosshair + tooltip on time series, per-mark tooltip on bars;
 *  - every chart is paired with a table view, so no value is reachable only by
 *    telling two colours apart.
 */

const SVGNS = "http://www.w3.org/2000/svg";

/** Fixed categorical order. Index into this; never generate or cycle a hue. */
export const SERIES = [
  "var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)",
];

export const COLORS = {
  actual: "var(--series-1)",
  forecast: "var(--series-2)",
  baseline: "var(--series-3)",
  muted: "var(--text-muted)",
  good: "var(--status-good)",
  warning: "var(--status-warning)",
  serious: "var(--status-serious)",
  critical: "var(--status-critical)",
};

function el(name, attrs = {}, parent = null) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    n.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(n);
  return n;
}

function html(tag, cls, parent) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (parent) parent.appendChild(n);
  return n;
}

/* ---- scales & ticks ---- */

function niceTicks(lo, hi, count = 5) {
  if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
  if (hi === lo) { hi = lo + 1; lo = lo - 1; }
  const span = hi - lo;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    out.push(Math.round(v / step) * step);
  }
  return out.length >= 2 ? out : [lo, hi];
}

export function fmtNum(v, digits = 1) {
  if (v === null || v === undefined || !isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(Math.min(digits, 1));
  if (a < 0.01 && a > 0) return v.toExponential(1);
  return v.toFixed(digits);
}

export function fmtClock(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function fmtDuration(sec) {
  const s = Math.abs(sec);
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

/* ---- shared tooltip ---- */

let tipEl = null;
function tooltip() {
  if (!tipEl) {
    tipEl = html("div", "tooltip", document.body);
    tipEl.setAttribute("role", "status");
  }
  return tipEl;
}

function showTip(x, y, head, rows) {
  const t = tooltip();
  t.textContent = "";
  const h = html("div", "tt-head", t);
  h.textContent = head;
  for (const r of rows) {
    const row = html("div", "tt-row", t);
    const key = html("span", "tt-key", row);
    key.style.color = r.color;
    if (r.dash) key.style.borderTopStyle = "dashed";
    const name = html("span", "tt-name", row);
    name.textContent = r.name;            // untrusted: textContent, never innerHTML
    const val = html("span", "tt-val", row);
    val.textContent = r.value;
  }
  t.classList.add("on");
  const pad = 14;
  const r = t.getBoundingClientRect();
  let left = x + pad;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
  let top = y - r.height / 2;
  top = Math.max(8, Math.min(top, window.innerHeight - r.height - 8));
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}

function hideTip() {
  if (tipEl) tipEl.classList.remove("on");
}

/* ---- legend + table helpers ---- */

export function legend(container, items) {
  const ul = html("ul", "legend", container);
  for (const it of items) {
    const li = html("li", null, ul);
    const key = html("span", `key${it.dash ? " dash" : ""}${it.rect ? " rect" : ""}`, li);
    if (it.rect) key.style.background = it.color;
    else key.style.borderTopColor = it.color;
    const label = html("span", null, li);
    label.textContent = it.name;
  }
  return ul;
}

export function tableView(container, columns, rows, opts = {}) {
  const wrap = html("div", "tablewrap", container);
  const t = html("table", "data", wrap);
  const thead = html("thead", null, t);
  const tr = html("tr", null, thead);
  for (const c of columns) {
    const th = html("th", null, tr);
    th.textContent = c;
  }
  const tb = html("tbody", null, t);
  for (const r of rows) {
    const row = html("tr", null, tb);
    r.forEach((cell, i) => {
      const td = html("td", null, row);
      if (cell && typeof cell === "object") {
        td.textContent = cell.text;
        if (cell.best) td.className = "best";
      } else {
        td.textContent = cell === null || cell === undefined ? "–" : String(cell);
      }
    });
  }
  if (opts.caption) {
    const cap = html("p", "small muted", wrap);
    cap.textContent = opts.caption;
  }
  return wrap;
}

/**
 * Attach a "chart / table" toggle. The table is the relief required because one
 * light-mode series colour sits below 3:1 against the surface: every value must
 * be reachable without distinguishing colours.
 */
export function withTableToggle(card, drawChart, buildTable) {
  const head = card.querySelector(".card-head") || card;
  const btn = html("button", "ghost", head);
  btn.type = "button";
  btn.textContent = "Table";
  btn.setAttribute("aria-pressed", "false");
  const chartBox = html("div", null, card);
  const tableBox = html("div", null, card);
  tableBox.hidden = true;
  drawChart(chartBox);
  let built = false;
  btn.addEventListener("click", () => {
    const on = btn.getAttribute("aria-pressed") === "true";
    btn.setAttribute("aria-pressed", String(!on));
    chartBox.hidden = !on;
    tableBox.hidden = on;
    if (!on && !built) { buildTable(tableBox); built = true; }
  });
  return { chartBox, tableBox };
}

/* ---------------------------------------------------------------------------
 * Time series with optional uncertainty band and event markers.
 * ------------------------------------------------------------------------- */

export function timeSeries(container, opts) {
  const {
    series = [],            // [{name, color, points:[[ts,v]], dash, width}]
    bands = [],             // [{name, color, points:[[ts,lo,hi]]}]
    events = [],            // [{ts, kind, color, label}]
    height = 240,
    yLabel = "",
    valueFmt = (v) => fmtNum(v, 1),
    yZero = false,
    now = null,
  } = opts;

  container.textContent = "";
  const width = Math.max(320, container.clientWidth || 640);
  const m = { top: 12, right: 62, bottom: 26, left: 46 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const svg = el("svg", {
    class: "chart", width: "100%", height,
    viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none",
    role: "img", "aria-label": yLabel || "time series",
  }, container);

  const allPts = series.flatMap((s) => s.points).filter((p) => p[1] !== null && isFinite(p[1]));
  const bandPts = bands.flatMap((b) => b.points || []);
  if (!allPts.length && !bandPts.length) {
    const t = el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, svg);
    t.textContent = "no data yet";
    return svg;
  }
  let tMin = Math.min(...allPts.map((p) => p[0]), ...bandPts.map((p) => p[0]));
  let tMax = Math.max(...allPts.map((p) => p[0]), ...bandPts.map((p) => p[0]));
  if (tMax === tMin) tMax = tMin + 1;
  let vMin = Math.min(...allPts.map((p) => p[1]), ...bandPts.map((p) => p[1]));
  let vMax = Math.max(...allPts.map((p) => p[1]), ...bandPts.map((p) => p[2]));
  if (!isFinite(vMin) || !isFinite(vMax)) { vMin = 0; vMax = 1; }
  if (yZero) vMin = Math.min(0, vMin);
  const pad = (vMax - vMin) * 0.08 || 1;
  vMin -= pad; vMax += pad;

  const X = (t) => m.left + ((t - tMin) / (tMax - tMin)) * iw;
  const Y = (v) => m.top + ih - ((v - vMin) / (vMax - vMin)) * ih;

  // gridlines: hairline, solid, recessive
  for (const tick of niceTicks(vMin, vMax, 4)) {
    const y = Y(tick);
    if (y < m.top - 1 || y > m.top + ih + 1) continue;
    el("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y, y2: y }, svg);
    const lbl = el("text", { x: m.left - 7, y: y + 3.5, "text-anchor": "end" }, svg);
    lbl.textContent = fmtNum(tick, Math.abs(tick) < 10 ? 1 : 0);
  }
  el("line", { class: "axis-line", x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }, svg);

  const nTicks = Math.max(2, Math.min(6, Math.floor(iw / 90)));
  for (let i = 0; i <= nTicks; i++) {
    const t = tMin + ((tMax - tMin) * i) / nTicks;
    const lbl = el("text", { x: X(t), y: m.top + ih + 16, "text-anchor": "middle" }, svg);
    lbl.textContent = fmtClock(t);
  }
  if (yLabel) {
    const yl = el("text", {
      class: "axis-label", x: m.left, y: m.top - 2, "text-anchor": "start",
    }, svg);
    yl.textContent = yLabel;
  }

  // uncertainty bands: the series hue as a ~10% wash, never a saturated block
  for (const b of bands) {
    const pts = (b.points || []).filter((p) => isFinite(p[1]) && isFinite(p[2]));
    if (pts.length < 2) continue;
    const top = pts.map((p) => `${X(p[0])},${Y(p[2])}`).join(" L");
    const bot = pts.slice().reverse().map((p) => `${X(p[0])},${Y(p[1])}`).join(" L");
    el("path", {
      d: `M${top} L${bot} Z`, fill: b.color, "fill-opacity": 0.13, stroke: "none",
    }, svg);
  }

  // event markers, drawn under the data
  for (const ev of events) {
    if (ev.ts < tMin || ev.ts > tMax) continue;
    const x = X(ev.ts);
    el("line", {
      x1: x, x2: x, y1: m.top, y2: m.top + ih, stroke: ev.color,
      "stroke-width": 1.5, "stroke-opacity": 0.55, "stroke-dasharray": "3 3",
    }, svg);
  }
  if (now !== null && now >= tMin && now <= tMax) {
    el("line", {
      x1: X(now), x2: X(now), y1: m.top, y2: m.top + ih,
      stroke: "var(--axis)", "stroke-width": 1,
    }, svg);
    const t = el("text", { x: X(now) + 4, y: m.top + 10 }, svg);
    t.textContent = "now";
  }

  // series
  const drawn = [];
  for (const s of series) {
    const pts = s.points.filter((p) => p[1] !== null && isFinite(p[1]));
    if (!pts.length) continue;
    // Break the path across gaps so a missing sensor reads as a gap, not as a
    // straight line through data that was never measured.
    const gapLimit = s.gapLimit || (tMax - tMin) / 12;
    let d = "";
    let prevT = null;
    for (const [t, v] of pts) {
      const cmd = prevT === null || t - prevT > gapLimit ? "M" : "L";
      d += `${cmd}${X(t).toFixed(1)},${Y(v).toFixed(1)}`;
      prevT = t;
    }
    el("path", {
      class: "mark-line", d, stroke: s.color, "stroke-width": s.width || 2,
      "stroke-dasharray": s.dash ? "5 4" : null,
    }, svg);
    const last = pts[pts.length - 1];
    el("circle", {
      class: "mark-dot", cx: X(last[0]), cy: Y(last[1]), r: 4, fill: s.color,
    }, svg);
    drawn.push({ s, pts, last });
  }

  // direct end-labels, sparingly: only the last value of each series
  const placed = [];
  for (const { s, last } of drawn) {
    let y = Y(last[1]) + 3.5;
    while (placed.some((p) => Math.abs(p - y) < 12)) y += 12;
    placed.push(y);
    const t = el("text", {
      class: "end-label", x: m.left + iw + 6, y: Math.min(y, m.top + ih),
    }, svg);
    t.textContent = valueFmt(last[1]);
  }

  // crosshair layer
  const cross = el("line", {
    class: "crosshair", y1: m.top, y2: m.top + ih, x1: -99, x2: -99,
  }, svg);
  const hit = el("rect", {
    class: "hit", x: m.left, y: m.top, width: iw, height: ih,
  }, svg);

  function nearest(tx) {
    const rows = [];
    for (const { s, pts } of drawn) {
      let best = null, bd = Infinity;
      for (const p of pts) {
        const d = Math.abs(p[0] - tx);
        if (d < bd) { bd = d; best = p; }
      }
      if (best && bd < (tMax - tMin) / 8) {
        rows.push({ name: s.name, color: s.color, dash: s.dash, value: valueFmt(best[1]) });
      }
    }
    return rows;
  }

  function onMove(e) {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * width;
    if (px < m.left || px > m.left + iw) { hideTip(); cross.setAttribute("x1", -99); cross.setAttribute("x2", -99); return; }
    const tx = tMin + ((px - m.left) / iw) * (tMax - tMin);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    const rows = nearest(tx);
    if (!rows.length) { hideTip(); return; }
    showTip(e.clientX, e.clientY, fmtClock(tx), rows);
  }
  hit.addEventListener("pointermove", onMove);
  hit.addEventListener("pointerleave", () => {
    hideTip(); cross.setAttribute("x1", -99); cross.setAttribute("x2", -99);
  });
  svg.setAttribute("tabindex", "0");
  svg.addEventListener("focus", () => {
    const rows = nearest(tMax);
    const r = svg.getBoundingClientRect();
    if (rows.length) showTip(r.right - 40, r.top + 30, fmtClock(tMax), rows);
  });
  svg.addEventListener("blur", hideTip);
  return svg;
}

/* ---------------------------------------------------------------------------
 * Grouped bars. One y-axis. Used for MAE per scope across arms.
 * ------------------------------------------------------------------------- */

export function groupedBars(container, opts) {
  const {
    groups = [],           // [{label, bars:[{name, value, color}]}]
    height = 260,
    yLabel = "",
    valueFmt = (v) => fmtNum(v, 3),
    lowerIsBetter = true,
  } = opts;
  container.textContent = "";
  const width = Math.max(340, container.clientWidth || 720);
  const m = { top: 14, right: 12, bottom: 62, left: 52 };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const svg = el("svg", {
    class: "chart", width: "100%", height, viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none", role: "img", "aria-label": yLabel || "grouped bars",
  }, container);

  const vals = groups.flatMap((g) => g.bars.map((b) => b.value)).filter((v) => isFinite(v));
  if (!vals.length) {
    const t = el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, svg);
    t.textContent = "no metrics yet";
    return svg;
  }
  const vMax = Math.max(...vals) * 1.12;
  const Y = (v) => m.top + ih - (v / vMax) * ih;

  for (const tick of niceTicks(0, vMax, 4)) {
    const y = Y(tick);
    if (y < m.top - 1) continue;
    el("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y, y2: y }, svg);
    const lbl = el("text", { x: m.left - 7, y: y + 3.5, "text-anchor": "end" }, svg);
    lbl.textContent = fmtNum(tick, vMax < 10 ? 2 : 1);
  }
  el("line", { class: "axis-line", x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }, svg);
  if (yLabel) {
    const yl = el("text", { class: "axis-label", x: m.left, y: m.top - 3 }, svg);
    yl.textContent = yLabel;
  }

  const gw = iw / groups.length;
  groups.forEach((g, gi) => {
    const n = g.bars.length || 1;
    const inner = gw * 0.82;
    // 2px surface gap between adjacent bars; cap thickness at 24px.
    const bw = Math.min(24, Math.max(3, inner / n - 2));
    const x0 = m.left + gi * gw + (gw - (bw + 2) * n + 2) / 2;
    const best = lowerIsBetter
      ? Math.min(...g.bars.map((b) => (isFinite(b.value) ? b.value : Infinity)))
      : Math.max(...g.bars.map((b) => (isFinite(b.value) ? b.value : -Infinity)));
    g.bars.forEach((b, bi) => {
      if (!isFinite(b.value)) return;
      const x = x0 + bi * (bw + 2);
      const y = Y(b.value);
      const h = Math.max(1, m.top + ih - y);
      const r = Math.min(4, bw / 2, h);
      // 4px rounded data-end, square at the baseline.
      const d = `M${x},${m.top + ih} L${x},${y + r} Q${x},${y} ${x + r},${y}` +
                ` L${x + bw - r},${y} Q${x + bw},${y} ${x + bw},${y + r}` +
                ` L${x + bw},${m.top + ih} Z`;
      const path = el("path", { d, fill: b.color }, svg);
      const hit = el("rect", {
        class: "hit", x: x - 3, y: m.top, width: bw + 6, height: ih,
      }, svg);
      const rows = [{ name: b.name, color: b.color, value: valueFmt(b.value) }];
      hit.addEventListener("pointermove", (e) => {
        path.setAttribute("fill-opacity", "0.78");
        showTip(e.clientX, e.clientY, g.label, rows);
      });
      hit.addEventListener("pointerleave", () => {
        path.setAttribute("fill-opacity", "1"); hideTip();
      });
      if (b.value === best) {
        const t = el("text", {
          class: "end-label", x: x + bw / 2, y: y - 5, "text-anchor": "middle",
        }, svg);
        t.textContent = valueFmt(b.value);
      }
    });
    const lbl = el("text", {
      x: m.left + gi * gw + gw / 2, y: m.top + ih + 15, "text-anchor": "middle",
    }, svg);
    lbl.textContent = g.label;
    if (g.sublabel) {
      const s2 = el("text", {
        x: m.left + gi * gw + gw / 2, y: m.top + ih + 28, "text-anchor": "middle",
      }, svg);
      s2.textContent = g.sublabel;
    }
  });
  return svg;
}

/* ---------------------------------------------------------------------------
 * Event lane: drift and adaptation events on a shared time axis.
 * ------------------------------------------------------------------------- */

export function eventLane(container, opts) {
  const { events = [], lanes = [], height = 120, tMin, tMax } = opts;
  container.textContent = "";
  const width = Math.max(320, container.clientWidth || 640);
  const m = { top: 10, right: 12, bottom: 24, left: 96 };
  const iw = width - m.left - m.right;
  const laneH = Math.max(18, (height - m.top - m.bottom) / Math.max(1, lanes.length));
  const svg = el("svg", {
    class: "chart", width: "100%", height, viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none", role: "img", "aria-label": "events over time",
  }, container);
  if (!events.length) {
    const t = el("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, svg);
    t.textContent = "no events in this window";
    return svg;
  }
  const t0 = tMin ?? Math.min(...events.map((e) => e.ts));
  const t1 = tMax ?? Math.max(...events.map((e) => e.ts));
  const X = (t) => m.left + ((t - t0) / Math.max(1e-6, t1 - t0)) * iw;

  lanes.forEach((lane, i) => {
    const y = m.top + i * laneH + laneH / 2;
    el("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y, y2: y }, svg);
    const lbl = el("text", { x: m.left - 8, y: y + 3.5, "text-anchor": "end" }, svg);
    lbl.textContent = lane.label;
  });
  for (const ev of events) {
    const li = lanes.findIndex((l) => l.key === ev.lane);
    if (li < 0) continue;
    const y = m.top + li * laneH + laneH / 2;
    const cx = X(ev.ts);
    const dot = el("circle", {
      class: "mark-dot", cx, cy: y, r: 5, fill: ev.color,
    }, svg);
    const hit = el("circle", { class: "hit", cx, cy: y, r: 12 }, svg);
    hit.addEventListener("pointermove", (e) => {
      dot.setAttribute("r", 6.5);
      showTip(e.clientX, e.clientY, fmtClock(ev.ts),
        [{ name: ev.label, color: ev.color, value: ev.value || "" }]);
    });
    hit.addEventListener("pointerleave", () => { dot.setAttribute("r", 5); hideTip(); });
  }
  const nTicks = Math.max(2, Math.min(6, Math.floor(iw / 90)));
  for (let i = 0; i <= nTicks; i++) {
    const t = t0 + ((t1 - t0) * i) / nTicks;
    const lbl = el("text", {
      x: X(t), y: height - 6, "text-anchor": "middle",
    }, svg);
    lbl.textContent = fmtClock(t);
  }
  return svg;
}

/* ---------------------------------------------------------------------------
 * Sparkline for stat tiles. No axes, no labels: the tile carries the number.
 * ------------------------------------------------------------------------- */

export function sparkline(container, values, color = COLORS.actual) {
  container.textContent = "";
  const pts = values.filter((v) => v !== null && isFinite(v));
  const width = Math.max(60, container.clientWidth || 160);
  const height = container.clientHeight || 30;
  const svg = el("svg", {
    class: "chart", width: "100%", height, viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none", "aria-hidden": "true",
  }, container);
  if (pts.length < 2) return svg;
  const lo = Math.min(...pts), hi = Math.max(...pts);
  const span = hi - lo || 1;
  const X = (i) => (i / (pts.length - 1)) * (width - 6) + 3;
  const Y = (v) => height - 3 - ((v - lo) / span) * (height - 6);
  const d = pts.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  el("path", { class: "mark-line", d, stroke: color, "stroke-width": 1.5 }, svg);
  el("circle", {
    class: "mark-dot", cx: X(pts.length - 1), cy: Y(pts[pts.length - 1]), r: 2.5, fill: color,
  }, svg);
  return svg;
}

export { html as h, el as svgEl };
