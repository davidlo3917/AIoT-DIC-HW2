import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

const METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso"];
const MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"];
const ML = { "F-test": "F-test", MI: "Mutual info", RFE: "RFE", Backward: "Backward", Lasso: "Lasso path" };
const S = { ds: null, base: {}, method: "F-test", model: "OLS", k: null, whatif: {} };
const cur = () => S.base[S.ds];
const spec = () => cur().spec;
const desc = (n) => cur().meta.desc[n] || "";
// one-hot columns → [source column, level]; regress.py names them `${column}_${level}`
const dummies = (d) => Object.fromEntries(Object.entries(d.meta.groups).flatMap(([c, g]) => g.levels.slice(1).map((lv) => [`${c}_${lv}`, [c, lv]])));
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (v, d = spec().digits) => (v == null || !isFinite(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const sig = (v) => (v == null ? "—" : Math.abs(v) >= 1e4 ? Math.round(v).toLocaleString("en-US") : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : +v.toPrecision(4) + "");
const ml = () => (cur().bench.mean.test >= 1e4 ? 64 : 44);   // large figures (dollars) need a wider left margin
const label = (c) => `${ML[c.method]} + ${c.model}`;
const series = (d, method, model) => d.combos.filter((c) => c.method === method && c.model === model).sort((a, b) => a.k - b.k);
const minBy = (list, f) => list.reduce((a, b) => (f(b) < f(a) ? b : a));
const methods = () => (S.method === "*" ? METHODS : [S.method]);
// the combo every card describes: the chosen k, or each method's best k by CV RMSE; in compare mode the best method wins
function active(d) {
  const cands = methods().map((m) => (S.k ? series(d, m, S.model).find((c) => c.k === S.k) : minBy(series(d, m, S.model), (c) => c.cv_rmse)));
  return minBy(cands, (c) => c.cv_rmse);
}
const table = (head, rows) => `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.join("")}</tr>`).join("")}</tbody></table>`;
const td = (v, cls = "num") => `<td class="${cls}">${v}</td>`;

// colours come from CSS variables
let C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
  C = { ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"), card: v("--card"), accent: v("--accent"), good: v("--good"), m: Object.fromEntries(METHODS.map((m, i) => [m, v(`--m${i + 1}`)])) };
}
const style = () => ({ background: "transparent", color: C.ink2, fontFamily: "Inter, system-ui, sans-serif", fontSize: "12px" });

// charts take their container's width; one ResizeObserver redraws
const draws = new Map();
const ro = new ResizeObserver((es) => es.forEach((e) => { const f = draws.get(e.target); if (f && Math.abs(e.contentRect.width - (e.target._w || 0)) > 4) paint(e.target, f); }));
function paint(el, f) { el._w = el.clientWidth; el.replaceChildren(f(Math.max(240, el.clientWidth))); }
function chart(el, f) { if (!draws.has(el)) ro.observe(el); draws.set(el, f); paint(el, f); }

// ── controls ─────────────────────────────────────────────────────────────────
function seg(box, options, value, onPick) {
  box.innerHTML = options.map(([v, t, cls = ""]) => `<button type="button" role="radio" class="${cls}" aria-checked="${v === value}" data-v="${v}">${t}</button>`).join("");
  box.onclick = (e) => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); };
}
function renderControls(d, a) {
  const pick = (f) => (v) => { f(v); S.k = null; S.whatif = {}; render(); };
  seg($("ds"), Object.entries(S.base).map(([k, v]) => [k, esc(v.spec.name)]), S.ds, pick((v) => (S.ds = v)));
  seg($("methods"), [...METHODS.map((m, i) => [m, `<span class="sw" style="--c:var(--m${i + 1})"></span>${ML[m]}`]), ["*", `Compare all ${METHODS.length} methods`, "all"]], S.method, pick((v) => (S.method = v)));
  seg($("models"), MODELS.map((m) => [m, m]), S.model, pick((v) => (S.model = v)));
  const best = minBy(methods().flatMap((m) => series(d, m, S.model)), (c) => c.cv_rmse).k, p = d.meta.p;
  seg($("ks"), Array.from({ length: p }, (_, i) => i + 1).map((k) => [k, `k = ${k}${k === best ? " <small>★ Best by CV</small>" : k === p ? " <small>Full model</small>" : ""}`, k === best ? "best" : k === p ? "full" : ""]), a.k, (v) => { S.k = +v; S.whatif = {}; render(); });
  const c = d.config, txt = d.meta.text_columns;
  $("prep").textContent = `${txt.length ? `One-hot ${txt.join(", ")} with the first level dropped (${p} columns). ` : ""}80/20 split with seed ${c.seed}: ${d.meta.n_train} train, ${d.meta.n_test} test rows. StandardScaler fitted on the training rows. Features are ranked inside each of the ${c.folds} CV folds, so CV never sees validation rows.`;
}

// ── hero, metric cards, strip ────────────────────────────────────────────────
function renderHero(d) {
  const s = spec(), all = d.combos.filter((c) => c.method !== "All"), best = minBy(all, (c) => c.cv_rmse), full = series(d, "All", "OLS")[0];
  $("heroTitle").textContent = s.title;
  $("heroLede").textContent = s.lede;
  $("pills").innerHTML = [[`${d.meta.rows} rows, ${d.meta.p} features, target ${d.target}`], [`${METHODS.length} selection methods × ${MODELS.length} models × k = 1 … ${d.meta.p}`],
    [`★ Best by CV: ${label(best)}, k = ${best.k} (test RMSE ${fmt(best.test_rmse)})`, "best"],
    [`Full model, k = ${d.meta.p}: test RMSE ${fmt(full.test_rmse)}`, full.test_rmse > best.test_rmse ? "warn" : ""]]
    .map(([t, cls = ""]) => `<span class="pill ${cls}">${esc(t)}</span>`).join("");
}
function renderMetrics(d, a) {
  const b = d.bench.mean;
  $("metricsHead").textContent = `${label(a)}, k = ${a.k}`;
  $("metrics").innerHTML = [
    ["Test RMSE", fmt(a.test_rmse), `lower is better; ${pct(1 - a.test_rmse / b.test)} below the mean baseline (${fmt(b.test)})`],
    ["Test R²", a.test_r2.toFixed(4), `explains ${pct(a.test_r2)} of ${d.target} variance on the test rows`],
    ["Adjusted R² (train)", a.adj_r2 == null ? "—" : a.adj_r2.toFixed(4), "penalises every extra coefficient"],
    ["Test MAE", fmt(a.test_mae), `mean absolute error, in ${spec().unit}`],
    ["CV RMSE", fmt(a.cv_rmse), `± ${fmt(a.cv_std)} SD over ${d.config.folds} folds; the ranking metric`],
  ].map(([l, v, n]) => `<div class="card metric"><div class="label">${l}</div><b>${v}</b><span>${esc(n)}</span></div>`).join("");
}
function renderStrip(d, a) {
  const gain = 1 - a.cv_rmse / d.bench.mean.cv, beatsRand = a.cv_rmse < d.bench.random[a.k - 1].cv[2], ok = gain >= 0.5 && beatsRand;
  $("features").innerHTML = `<b>Active predictors (${a.k}):</b>${a.features.map((n) => `<span class="fchip" title="${esc(desc(n))}">${esc(n)}</span>`).join("")}`;
  $("verdict").innerHTML = `<span class="verdict ${ok ? "ok" : "fail"}">${ok ? "✓ Meets" : "✗ Misses"} the success criterion: CV RMSE ${pct(gain)} below the mean baseline (target 50%), ${beatsRand ? "beats" : "does not beat"} random ${a.k}-feature subsets</span>`;
}

// ── RMSE and R² by k ─────────────────────────────────────────────────────────
function renderCharts(d, a) {
  const rows = methods().flatMap((m) => series(d, m, S.model).map((c) => ({ k: c.k, rmse: c.test_rmse, r2: c.test_r2, method: m })));
  const legend = methods().map((m) => `<span class="dot" style="--c:var(--m${METHODS.indexOf(m) + 1})"></span>${ML[m]}`).join("");
  const lo = minBy(rows, (r) => r.rmse), hi = minBy(rows, (r) => -r.r2);
  const note = S.method === "*" ? " Lines overlap where methods pick the same features." : "";
  $("rmseNote").innerHTML = `Lower is better. Lowest test RMSE at k = ${lo.k}${S.method === "*" ? ` (${ML[lo.method]})` : ""}.${note} ${legend}`;
  $("r2Note").innerHTML = `Higher is better. Peak test R² at k = ${hi.k}${S.method === "*" ? ` (${ML[hi.method]})` : ""}.${note} ${legend}`;
  const line = (y, el, fmtY) => chart($(el), (w) => Plot.plot({ width: w, height: 300, marginLeft: ml(), style: style(), grid: true, x: { label: "Number of features k", ticks: d.meta.p, tickFormat: "d" }, y: { label: null },
    marks: [Plot.ruleX([a.k], { stroke: C.ink3, strokeDasharray: "3 3" }),
      Plot.line(rows, { x: "k", y, z: "method", stroke: (r) => C.m[r.method], strokeWidth: 2 }),
      Plot.dot(rows, { x: "k", y, fill: (r) => C.m[r.method], r: (r) => (r.k === a.k ? 5 : 3.5), stroke: C.card, tip: true, title: (r) => `${ML[r.method]}, k = ${r.k}\n${fmtY(r[y])}` })] }));
  line("rmse", "rmse", (v) => fmt(v));
  line("r2", "r2", (v) => v.toFixed(4));
}

// ── equation, coefficients, residuals ────────────────────────────────────────
function renderModel(d, a) {
  const feats = d.meta.features, hyper = Object.entries(a.hyper).map(([h, v]) => `${h} = ${sig(v)}`).join(", ");
  const dum = dummies(d), refs = [...new Set(a.features.filter((n) => dum[n]).map((n) => dum[n][0]))].map((c) => `${c}_ columns are 0/1 against ${c} = ${d.meta.groups[c].reference}`);
  $("eqNote").textContent = `${label(a)}, k = ${a.k}, original units${hyper ? `, ${hyper}` : ""}.${refs.length ? ` ${refs.join("; ")}.` : ""}`;
  $("equation").innerHTML = `<code>${esc(d.target)} = ${sig(a.intercept)} ${a.features.map((n, i) => `${a.coef[i] < 0 ? "−" : "+"} ${sig(Math.abs(a.coef[i]))} × <span class="var">${esc(n)}</span>`).join(" ")}</code>`;
  const sd = (n) => d.meta.train_sd[feats.indexOf(n)];
  $("coefNote").textContent = `Estimates in original units. Standardised = estimate × training SD, the change in ${d.target} per one SD of the predictor, comparable across rows.`;
  $("coefs").innerHTML = table(["Predictor", "Meaning", "Estimate", "Standardised", "Training SD"],
    [[td("Intercept (β₀)", "txt"), td(`baseline ${esc(d.target)} (${esc(desc(d.target))})`, "txt muted"), td(sig(a.intercept)), td("—"), td("—")],
      ...a.features.map((n, i) => [td(esc(n), "txt"), td(esc(desc(n)), "txt muted"), td(sig(a.coef[i])), td(sig(a.coef[i] * sd(n))), td(sig(sd(n)))])]);
  const pts = a.top.map(([row, y, p]) => ({ row, y, p, r: y - p }));   // regress.py keeps the 10 largest residuals
  $("residNote").textContent = `Top 10 of ${d.meta.n_test} test rows by absolute residual. High residual = larger than the test RMSE (${fmt(a.test_rmse)}).`;
  $("resid").innerHTML = table(["#", "Row", `Actual ${esc(d.target)}`, "Predicted", "Residual", "Abs error", "% error", "Status"],
    pts.map((x, i) => [td(i + 1, "txt"), td(`#${x.row}`, "txt"), td(fmt(x.y)), td(fmt(x.p)), td(`<span class="${x.r >= 0 ? "pos" : "neg"}">${x.r >= 0 ? "+" : "−"}${fmt(Math.abs(x.r))}</span>`), td(fmt(Math.abs(x.r))), td(x.y ? pct(Math.abs(x.r / x.y)) : "—"),
      td(Math.abs(x.r) > a.test_rmse ? `<span class="st hi">High residual</span>` : `<span class="st ok">In range</span>`, "txt")]));
}

// ── live predictor: the linear equation evaluated in the browser ─────────────
function renderDeploy(d, a) {
  const dum = dummies(d), nums = a.features.filter((n) => !dum[n]), cats = [...new Set(a.features.filter((n) => dum[n]).map((n) => dum[n][0]))];
  $("deployNote").textContent = `${label(a)}, k = ${a.k}. Intercept + Σ coefficient × feature, computed here as you drag.`;
  const form = $("whatif");
  form.innerHTML = nums.map((n) => {
    const [lo, mid, hi] = d.meta.ranges[n], val = S.whatif[n] ?? mid;
    return `<label title="${esc(desc(n))}"><div><span>${esc(n)}</span><output>${sig(val)}</output></div><input type="range" name="${esc(n)}" min="${lo}" max="${hi}" step="${(hi - lo) / 200 || 1}" value="${val}"></label>`;
  }).join("") + cats.map((c) => {
    const g = d.meta.groups[c], v = S.whatif[c] ?? g.reference;
    return `<label title="${esc(desc(`${c}_${g.levels[1]}`))}">${esc(c)} <select name="${esc(c)}">${g.levels.map((lv) => `<option ${lv === v ? "selected" : ""}>${esc(lv)}</option>`).join("")}</select></label>`;
  }).join("");
  const update = () => {
    const fd = new FormData(form);
    let y = a.intercept;
    a.features.forEach((n, i) => { const x = dum[n] ? +(fd.get(dum[n][0]) === dum[n][1]) : +fd.get(n); S.whatif[n] = x; y += a.coef[i] * x; });
    cats.forEach((c) => (S.whatif[c] = fd.get(c)));
    form.querySelectorAll("input[type=range]").forEach((r) => (r.previousElementSibling.lastChild.textContent = sig(+r.value)));
    $("prediction").innerHTML = `Predicted ${esc(d.target)} (${esc(spec().unit)})<b>${fmt(y)}</b>± ${fmt(a.test_rmse)} test RMSE`;
  };
  form.oninput = update;
  update();
}

// ── main ─────────────────────────────────────────────────────────────────────
function render() {
  const d = cur();
  if (!d) return;
  const a = active(d);
  renderControls(d, a);
  renderHero(d);
  renderMetrics(d, a);
  renderStrip(d, a);
  renderCharts(d, a);
  renderModel(d, a);
  renderDeploy(d, a);
}

readColors();
try {
  const presets = await (await fetch("presets.json")).json();
  const data = await Promise.all(presets.map((p) => fetch(`data/${p.key}.json`).then((r) => { if (!r.ok) throw new Error(`${r.status} data/${p.key}.json`); return r.json(); })));
  presets.forEach((p, i) => (S.base[p.key] = data[i]));
  S.ds = presets[0].key;
  $("status").textContent = "";
  render();
} catch (err) {
  $("status").className = "status error";
  $("status").textContent = `Results file missing (${err.message}). Run "python web/regress.py" to generate web/data/*.json, then serve with "python -m http.server -d web".`;
}
