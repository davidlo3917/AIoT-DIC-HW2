import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

const METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso", "All"];
const MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"];
const ML = { "F-test": "F-test", MI: "Mutual info", RFE: "RFE", Backward: "Backward", Lasso: "Lasso path", All: "All features" };
const DS = {
  startups: {
    name: "50 Startups", file: "startups", digits: 0, unit: "US$",
    title: "Does R&D spend decide a startup's profit?",
    lede: "50 US startups: predict Profit from R&D, administration and marketing spend plus the state. Every feature-selection method, model and feature count k is fitted on the same 80/20 split and ranked below.",
  },
  boston: {
    name: "Boston Housing", file: "boston", digits: 2, unit: "US$1,000s",
    title: "What drives Boston house prices?",
    lede: "506 census tracts from 1970s Boston: predict the median home value from 13 neighbourhood features. The B column is derived from the share of Black residents, which is why scikit-learn removed this dataset; it is kept here only to compare with textbook results.",
  },
};

const S = { ds: "startups", base: {}, methods: new Set(METHODS), models: new Set(MODELS), kMode: "best", k: 2, rank: "cv", focus: null, lineModel: "OLS", whatif: {}, showAll: false };
const cur = () => S.base[S.ds];
const spec = () => DS[S.ds];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (v, d = spec().digits) => (v == null || !isFinite(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const sig = (v) => (v == null ? "—" : Math.abs(v) >= 1e4 ? Math.round(v).toLocaleString("en-US") : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : +v.toPrecision(4) + "");
const ml = () => (S.ds === "startups" ? 64 : 44);   // dollar figures need a wider left margin
const key = (c) => `${c.method}|${c.model}`;
const label = (c) => `${ML[c.method]} + ${c.model}`;

// ── ranking and "cells": one cell = method × model, k chosen by the mode ──────
const metricOf = (rank) => (c) => (rank === "cv" ? c.cv_rmse : c.test_rmse);
const score = (c) => metricOf(S.rank)(c);
const metricName = (rank = S.rank) => (rank === "cv" ? "CV RMSE" : "Test RMSE");
function cells(d = cur(), rank = S.rank) {
  const m = metricOf(rank), by = new Map();
  for (const c of d.combos) {
    if (!S.methods.has(c.method) || !S.models.has(c.model)) continue;
    if (S.kMode === "fixed" && c.method !== "All" && c.k !== S.k) continue;
    const b = by.get(key(c));
    if (!b || m(c) < m(b)) by.set(key(c), c);
  }
  return [...by.values()].sort((a, b) => m(a) - m(b));
}
const focused = (list) => list.find((c) => key(c) === S.focus) || list[0];
const baseline = (d) => (S.rank === "cv" ? d.bench.mean.cv : d.bench.mean.test);
const randMed = (d, k) => (S.rank === "cv" ? d.bench.random[k - 1].cv : d.bench.random[k - 1].test)[2];
const find = (d, method, k, model) => d.combos.find((c) => c.method === method && c.model === model && (k == null || c.k === k));

// ── colours come from CSS variables, so dark mode follows ─────────────────────
let C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
  C = { ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"), grid: v("--grid"), rule: v("--rule"), card: v("--card"), accent: v("--accent"), lo: v("--seq-lo"), hi: v("--seq-hi"),
    m: Object.fromEntries(METHODS.map((m, i) => [m, v(`--m${i + 1}`)])) };
}
const style = () => ({ background: "transparent", color: C.ink2, fontFamily: "IBM Plex Sans, system-ui, sans-serif", fontSize: "12px" });

// ── charts take their container's width; one ResizeObserver redraws ──────────
const draws = new Map();
const ro = new ResizeObserver((es) => es.forEach((e) => { const f = draws.get(e.target); if (f && Math.abs(e.contentRect.width - (e.target._w || 0)) > 4) paint(e.target, f); }));
function paint(el, f) { el._w = el.clientWidth; el.replaceChildren(f(Math.max(240, el.clientWidth))); }
function chart(el, f) { if (!draws.has(el)) ro.observe(el); draws.set(el, f); paint(el, f); }
function multiples(box, items, title, f) {
  box.querySelectorAll(".chart").forEach((c) => { ro.unobserve(c); draws.delete(c); });
  box.replaceChildren(...items.map((it) => {
    const wrap = document.createElement("div"), c = document.createElement("div");
    wrap.innerHTML = `<h4>${title(it)}</h4>`;
    c.className = "chart";
    wrap.append(c);
    return wrap;
  }));
  [...box.children].forEach((w, i) => chart(w.lastChild, (wd) => f(items[i], wd)));
}

// ── controls ─────────────────────────────────────────────────────────────────
function seg(box, options, value, onPick) {
  box.innerHTML = options.map(([v, t]) => `<button type="button" role="radio" aria-checked="${v === value}" data-v="${v}">${t}</button>`).join("");
  box.onclick = (e) => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); };
}
function chips(box, all, set, labelOf, color) {
  box.innerHTML = all.map((x) => `<button type="button" class="chip" aria-pressed="${set.has(x)}" data-v="${x}">${color ? `<span class="sw" style="--c:var(--m${METHODS.indexOf(x) + 1})"></span>` : ""}${labelOf(x)}</button>`).join("")
    + `<button type="button" class="chip all" data-v="*">Select all</button>`;
  box.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.v === "*") all.forEach((x) => set.add(x));
    else if (set.has(b.dataset.v)) { if (set.size > 1) set.delete(b.dataset.v); }   // the last one cannot be unchecked
    else set.add(b.dataset.v);
    render();
  };
}
function renderControls() {
  const d = cur();
  seg($("ds"), Object.entries(DS).map(([k, v]) => [k, v.name]), S.ds, (v) => { S.ds = v; S.focus = null; S.whatif = {}; render(); });
  chips($("methods"), METHODS, S.methods, (m) => ML[m], true);
  chips($("models"), MODELS, S.models, (m) => m, false);
  seg($("kmode"), [["best", "Best k"], ["fixed", "Fixed k"]], S.kMode, (v) => { S.kMode = v; render(); });
  seg($("rank"), [["cv", "CV RMSE"], ["test", "Test RMSE"]], S.rank, (v) => { S.rank = v; render(); });
  const k = $("k");
  k.max = d.meta.p; S.k = Math.min(S.k, d.meta.p); k.value = S.k; k.disabled = S.kMode !== "fixed";
  $("kout").textContent = S.kMode === "fixed" ? `k = ${S.k}` : "";
}
$("k").oninput = (e) => { S.k = +e.target.value; render(); };

// ── 1 + 2: business and data understanding ───────────────────────────────────
function renderHero(d, best) {
  const s = spec(), raw = Object.keys(d.rows[0]).length - 1;
  $("heroTitle").textContent = s.title;
  $("heroLede").textContent = s.lede;
  $("pills").innerHTML = [`${d.meta.rows} rows`, `${raw} raw features, ${d.meta.p} after one-hot`, `target: ${d.target}`, `${d.meta.missing} missing values`, `${d.meta.duplicates} duplicate rows`]
    .map((t) => `<span class="pill">${esc(t)}</span>`).join("") + `<span class="pill best">Best now: ${esc(label(best))}, k = ${best.k}</span>`;
}

// ── 3: data preparation, one line ────────────────────────────────────────────
function renderPipeline(d) {
  const c = d.config, txt = d.meta.text_columns;
  $("pipeline").innerHTML = [
    txt.length ? `One-hot encode ${txt.join(", ")}, drop the first level (${d.meta.p} columns)` : "No text columns, nothing to encode",
    `Split 80/20 with seed ${c.seed}: ${d.meta.n_train} train, ${d.meta.n_test} test`,
    "StandardScaler fitted on the training rows only",
    `Rank features inside each of ${c.folds} CV folds, so CV never sees validation rows`,
    "Fit every method × k × model, tune α by CV, score the test set once",
  ].map((t) => `<li>${esc(t)}</li>`).join("");
}

// ── 5: metric cards for the best cell ────────────────────────────────────────
function renderMetrics(d, best) {
  const b = d.bench.mean;
  $("metricsHead").innerHTML = `Best by ${metricName()}: ${esc(label(best))}, k = ${best.k}`;
  $("metrics").innerHTML = [
    ["CV RMSE", fmt(best.cv_rmse), `± ${fmt(best.cv_std)} SD over ${d.config.folds} folds`],
    ["Test RMSE", fmt(best.test_rmse), `${pct(1 - best.test_rmse / b.test)} below the mean baseline (${fmt(b.test)})`],
    ["Test R²", best.test_r2.toFixed(3), `share of ${d.target} variance explained on the test rows`],
    ["Adjusted R² (train)", best.adj_r2 == null ? "—" : best.adj_r2.toFixed(3), "penalises every extra coefficient"],
    ["Test MAE", fmt(best.test_mae), `mean absolute error, in ${spec().unit}`],
  ].map(([l, v, n]) => `<div class="card metric"><div class="label">${l}</div><b>${v}</b><span>${esc(n)}</span></div>`).join("");
}
function renderFeatures(d, list, f) {
  const best = list[0], gain = 1 - best.cv_rmse / d.bench.mean.cv;
  const beatsRand = best.method === "All" || best.cv_rmse < d.bench.random[best.k - 1].cv[2];
  const ok = gain >= 0.5 && beatsRand;
  const byCv = cells(d, "cv")[0], byTest = cells(d, "test")[0], same = key(byCv) === key(byTest) && byCv.k === byTest.k;
  $("features").innerHTML = `<span class="muted">Inspecting ${esc(label(f))}, k = ${f.k}:</span>${f.features.map((n) => `<span class="fchip">${esc(n)}</span>`).join("")}`;
  $("verdict").innerHTML = `<span class="verdict ${ok ? "ok" : "fail"}">${ok ? "✓ Meets" : "✗ Misses"} the success criterion: CV RMSE ${pct(gain)} below the mean baseline (target 50%), ${beatsRand ? "beats" : "does not beat"} random feature subsets</span>
    <span class="muted">${same ? "CV and test pick the same winner." : `By CV: ${esc(label(byCv))}, k = ${byCv.k}. By test: ${esc(label(byTest))}, k = ${byTest.k}. With only ${d.meta.n_test} test rows the test pick is optimistic, so CV ranks by default.`}</span>`;
}

// ── 5: RMSE vs k (one model at a time) and the method × model heatmap ────────
function renderCharts(d, list) {
  const base = baseline(d), models = MODELS.filter((m) => S.models.has(m));
  if (!models.includes(S.lineModel)) S.lineModel = models[0];
  seg($("lineModel"), models.map((m) => [m, m]), S.lineModel, (v) => { S.lineModel = v; render(); });
  const sel = METHODS.filter((m) => S.methods.has(m) && m !== "All");
  const band = d.bench.random.map((r) => ({ k: r.k, lo: (S.rank === "cv" ? r.cv : r.test)[1], hi: (S.rank === "cv" ? r.cv : r.test)[3] }));
  const rows = d.combos.filter((c) => c.model === S.lineModel && sel.includes(c.method)).map((c) => ({ k: c.k, v: score(c), method: c.method }));
  const all = find(d, "All", null, S.lineModel);
  const vals = d.combos.filter((c) => models.includes(c.model) && S.methods.has(c.method)).map(score);
  const ylo = Math.min(...vals) * 0.95, yhi = Math.max(...vals) * 1.1, meanIn = base <= yhi;
  $("linesNote").innerHTML = `${sel.map((m) => `<span class="dot" style="--c:var(--m${METHODS.indexOf(m) + 1})"></span>${ML[m]}`).join("")}<br>Band: random k-feature subsets with OLS, p10 to p90. Grey line: all features. ${meanIn ? "Dashed: mean baseline." : `Mean baseline ${fmt(base)} is above the chart.`}`;
  chart($("lines"), (w) => Plot.plot({ width: w, height: 300, marginLeft: ml(), style: style(), clip: true, x: { label: "Number of features k", ticks: d.meta.p, tickFormat: "d" }, y: { label: metricName(), grid: true, domain: [ylo, yhi] },
    marks: [Plot.areaY(band, { x: "k", y1: "lo", y2: "hi", fill: C.ink3, fillOpacity: 0.14 }),
      meanIn ? Plot.ruleY([base], { stroke: C.ink3, strokeDasharray: "4 4" }) : null,
      all && S.methods.has("All") ? Plot.ruleY([score(all)], { stroke: C.ink3, strokeWidth: 1.5 }) : null,
      Plot.line(rows, { x: "k", y: "v", z: "method", stroke: (x) => C.m[x.method], strokeWidth: 2 }),
      Plot.dot(rows, { x: "k", y: "v", fill: (x) => C.m[x.method], r: 3.5, stroke: C.card, tip: true, title: (x) => `${ML[x.method]}, k = ${x.k}\n${fmt(x.v)}` })] }));

  const hm = list.map((c) => ({ m: ML[c.method], model: c.model, v: score(c), gain: 1 - score(c) / base, k: c.k }));
  const g0 = Math.min(...hm.map((x) => x.gain)), g1 = Math.max(...hm.map((x) => x.gain));
  $("heatNote").textContent = `${metricName()} per cell, k in brackets. Darker = further below the mean baseline.`;
  chart($("heat"), (w) => Plot.plot({ width: w, height: 60 + new Set(hm.map((x) => x.m)).size * 42, marginLeft: 90, style: style(),
    x: { axis: "top", label: null, domain: models }, y: { label: null, domain: METHODS.filter((m) => S.methods.has(m)).map((m) => ML[m]) },
    color: { type: "linear", range: [C.lo, C.hi] },
    marks: [Plot.cell(hm, { x: "model", y: "m", fill: "gain", inset: 1.5, rx: 4, tip: true }),
      Plot.text(hm, { x: "model", y: "m", text: (x) => `${fmt(x.v)} (${x.k})`, fontSize: 11, fill: (x) => ((x.gain - g0) / (g1 - g0 || 1) > 0.5 ? C.card : C.ink) })] }));
}

// ── 5: leaderboard ───────────────────────────────────────────────────────────
function renderBoard(d, list, f) {
  const base = baseline(d);
  $("boardNote").textContent = `${list.length} cells (feature selection × model, ${S.kMode === "best" ? "each at its best k" : `k = ${S.k}`}), ranked by ${metricName()}. Click a row to inspect it below.`;
  const th = (r, t) => `<button type="button" data-rank="${r}" aria-pressed="${S.rank === r}">${t}</button>`;
  $("board").innerHTML = `<thead><tr><th>#</th><th class="txt">Feature selection</th><th class="txt">Model</th><th>k</th><th>${th("cv", "CV RMSE")}</th><th>± SD</th><th>${th("test", "Test RMSE")}</th><th>Test R²</th><th>Adj R² (train)</th><th>vs mean</th><th>vs random</th></tr></thead><tbody>`
    + list.filter((c, i) => S.showAll || i < 10 || key(c) === key(f)).map((c) => {
      const i = list.indexOf(c);
      const rnd = c.method === "All" ? "—" : score(c) < randMed(d, c.k) ? `<span class="yes">✓ beats</span>` : `<span class="no">✗ loses</span>`;
      return `<tr tabindex="0" data-key="${esc(key(c))}" class="${key(c) === key(f) ? "focus" : ""}"><td>${i + 1}</td><td class="txt"><span class="dot" style="--c:var(--m${METHODS.indexOf(c.method) + 1})"></span>${ML[c.method]}${i === 0 ? `<span class="badge">Best</span>` : ""}</td><td class="txt">${c.model}</td><td>${c.k}</td><td>${fmt(c.cv_rmse)}</td><td>${fmt(c.cv_std)}</td><td>${fmt(c.test_rmse)}</td><td>${c.test_r2.toFixed(3)}</td><td>${c.adj_r2 == null ? "—" : c.adj_r2.toFixed(3)}</td><td>${pct(1 - score(c) / base)}</td><td>${rnd}</td></tr>`;
    }).join("")
    + `<tr class="base"><td></td><td class="txt" colspan="3">Mean baseline: predict the training mean</td><td>${fmt(d.bench.mean.cv)}</td><td></td><td>${fmt(d.bench.mean.test)}</td><td colspan="4"></td></tr></tbody>`;
  $("showAll").hidden = list.length <= 10;
  $("showAll").textContent = S.showAll ? "Show top 10 only" : `Show all ${list.length}`;
}
$("board").onclick = (e) => {
  const r = e.target.closest("[data-rank]");
  if (r) { S.rank = r.dataset.rank; return render(); }
  const tr = e.target.closest("tr[data-key]");
  if (tr) { S.focus = tr.dataset.key; render(); }
};
$("board").onkeydown = (e) => { if (e.key === "Enter" && e.target.dataset.key) { S.focus = e.target.dataset.key; render(); } };
$("showAll").onclick = () => { S.showAll = !S.showAll; render(); };

// ── 4: equation and standardised coefficients for the focused cell ───────────
function renderModel(d, f) {
  const feats = d.meta.features, models = MODELS.filter((m) => S.models.has(m));
  $("eqNote").textContent = `${label(f)}, k = ${f.k}, original units.${f.features.some((n) => n.startsWith("State_")) ? " State_ columns are 0/1 against California as the reference." : ""}`;
  $("equation").innerHTML = `<code>${esc(d.target)} = ${sig(f.intercept)} ${f.features.map((n, i) => `${f.coef[i] < 0 ? "−" : "+"} ${sig(Math.abs(f.coef[i]))} × <span class="var">${esc(n)}</span>`).join(" ")}</code>`;
  const fits = models.map((m) => find(d, f.method, f.k, m)).filter(Boolean);
  const sd = (name) => d.meta.train_sd[feats.indexOf(name)];
  const std = fits.map((c) => c.features.map((name, i) => ({ feature: name, v: c.coef[i] * sd(name) })));
  const vs = std.flat().map((x) => x.v), dom = [Math.min(0, ...vs) * 1.1, Math.max(0, ...vs) * 1.1];
  $("coefNote").textContent = `The same ${f.k} features under every selected model. Bar = change in ${d.target} per one SD of the feature.`;
  multiples($("coefs"), fits.map((c, i) => [c, std[i]]), ([c]) => `${c.model}${Object.entries(c.hyper).map(([h, v]) => `, ${h} = ${sig(v)}`).join("")}`,
    ([, rows], w) => Plot.plot({ width: w, height: 30 + rows.length * 24, marginLeft: 110, style: style(), x: { domain: dom, label: null, grid: true }, y: { domain: rows.map((r) => r.feature), label: null },
      marks: [Plot.barX(rows, { x: "v", y: "feature", fill: C.accent, rx: 3, insetTop: 3, insetBottom: 3, tip: true }), Plot.ruleX([0], { stroke: C.ink3 })] }));
}

// ── 5: fit diagnostics for the focused cell ──────────────────────────────────
function renderFit(d, f) {
  const yt = d.y_test, pts = yt.map((y, i) => ({ y, p: f.pred[i], r: y - f.pred[i] })), col = C.m[f.method];
  const lo = Math.min(...yt, ...f.pred), hi = Math.max(...yt, ...f.pred);
  chart($("avp"), (w) => Plot.plot({ width: w, height: 280, marginLeft: ml(), style: style(), grid: true, x: { label: "Actual", domain: [lo, hi] }, y: { label: "Predicted", domain: [lo, hi] },
    marks: [Plot.line([[lo, lo], [hi, hi]], { stroke: C.ink3, strokeDasharray: "4 4" }), Plot.dot(pts, { x: "y", y: "p", r: 4, fill: col, stroke: C.card, strokeWidth: 1, tip: true })] }));
  chart($("resid"), (w) => Plot.plot({ width: w, height: 280, marginLeft: ml(), style: style(), grid: true, x: { label: "Predicted" }, y: { label: "Residual" },
    marks: [Plot.ruleY([0], { stroke: C.ink3, strokeDasharray: "4 4" }), Plot.dot(pts, { x: "p", y: "r", r: 4, fill: col, stroke: C.card, strokeWidth: 1, tip: true })] }));
}

// ── 6: deployment, the linear equation evaluated in the browser ──────────────
function renderDeploy(d, f) {
  const nums = f.features.filter((n) => !n.startsWith("State_")), hasState = f.features.some((n) => n.startsWith("State_"));
  const states = [...new Set(d.rows.map((r) => r.State).filter(Boolean))].sort();
  const vals = (n) => d.rows.map((r) => r[n]).sort((a, b) => a - b);
  $("deployNote").textContent = `${label(f)}, k = ${f.k}. Intercept + Σ coefficient × feature, computed here.`;
  const form = $("whatif");
  form.innerHTML = nums.map((n) => {
    const v = vals(n), lo = v[0], hi = v[v.length - 1], val = S.whatif[n] ?? v[Math.floor(v.length / 2)];
    return `<label><div><span>${esc(n)}</span><output>${sig(val)}</output></div><input type="range" name="${esc(n)}" min="${lo}" max="${hi}" step="${(hi - lo) / 200 || 1}" value="${val}"></label>`;
  }).join("") + (hasState ? `<label>State <select name="State">${states.map((s) => `<option ${s === (S.whatif.State || states[0]) ? "selected" : ""}>${s}</option>`).join("")}</select></label>` : "");
  const update = () => {
    const fd = new FormData(form);
    let y = f.intercept;
    f.features.forEach((n, i) => { const x = n.startsWith("State_") ? +(fd.get("State") === n.slice(6)) : +fd.get(n); S.whatif[n] = x; y += f.coef[i] * x; });
    if (hasState) S.whatif.State = fd.get("State");
    form.querySelectorAll("input[type=range]").forEach((r) => (r.previousElementSibling.lastChild.textContent = sig(+r.value)));
    $("prediction").innerHTML = `Predicted ${esc(d.target)} (${spec().unit})<b>${fmt(y)}</b>± ${fmt(f.test_rmse)} test RMSE`;
  };
  form.oninput = update;
  update();
}

// ── main ─────────────────────────────────────────────────────────────────────
function render() {
  const d = cur();
  if (!d) return;
  renderControls();
  const list = cells(d), f = focused(list);
  S.focus = key(f);
  renderHero(d, list[0]);
  renderPipeline(d);
  renderMetrics(d, list[0]);
  renderFeatures(d, list, f);
  renderCharts(d, list);
  renderBoard(d, list, f);
  renderModel(d, f);
  renderFit(d, f);
  renderDeploy(d, f);
}

readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); render(); });
try {
  const files = Object.values(DS).map((s) => fetch(`data/${s.file}.json`).then((r) => { if (!r.ok) throw new Error(`${r.status} data/${s.file}.json`); return r.json(); }));
  const [a, b] = await Promise.all(files);
  S.base = { startups: a, boston: b };
  $("status").textContent = "";
  render();
} catch (err) {
  $("status").className = "status error";
  $("status").textContent = `Results file missing (${err.message}). Run "python web/regress.py" to generate web/data/*.json, then serve with "python -m http.server -d web".`;
}
