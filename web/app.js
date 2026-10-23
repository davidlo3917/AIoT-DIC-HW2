import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

const METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso"];
const MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"];
const ML = { "F-test": "F-test", MI: "Mutual info", RFE: "RFE", Backward: "Backward", Lasso: "Lasso path" };
// the Gemini proxy (proxy/worker.mjs); the production URL is known after `wrangler deploy`
const PROXY = ["localhost", "127.0.0.1"].includes(location.hostname) ? "http://127.0.0.1:8787" : "https://aiot-dic-hw2-proxy.davidlo3917.workers.dev";
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
const best = (d) => minBy(d.combos.filter((c) => c.method !== "All"), (c) => c.cv_rmse);   // the recommended combination
const choose = (d) => { const b = best(d); Object.assign(S, { method: b.method, model: b.model, k: null, whatif: {} }); };   // every dataset opens on it
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
  const b = best(d), star = " <small>★ Best by CV</small>";
  seg($("ds"), Object.entries(S.base).map(([k, v]) => [k, esc(v.spec.name)]), S.ds, pick((v) => { S.ds = v; choose(cur()); }));
  seg($("methods"), [...METHODS.map((m, i) => [m, `<span class="sw" style="--c:var(--m${i + 1})"></span>${ML[m]}${m === b.method ? star : ""}`, m === b.method ? "best" : ""]), ["*", `Compare all ${METHODS.length} methods`, "all"]], S.method, pick((v) => (S.method = v)));
  seg($("models"), MODELS.map((m) => [m, `${m}${m === b.model ? star : ""}`, m === b.model ? "best" : ""]), S.model, pick((v) => (S.model = v)));
  const bk = minBy(methods().flatMap((m) => series(d, m, S.model)), (c) => c.cv_rmse).k, p = d.meta.p;
  seg($("ks"), Array.from({ length: p }, (_, i) => i + 1).map((k) => [k, `k = ${k}${k === bk ? star : k === p ? " <small>Full model</small>" : ""}`, k === bk ? "best" : k === p ? "full" : ""]), a.k, (v) => { S.k = +v; S.whatif = {}; render(); });
  $("ks").classList.toggle("dense", p > 15);
  const c = d.config, txt = d.meta.text_columns, f = d.prep.file, log = d.prep.log;
  $("prep").textContent = `${txt.length ? `One-hot ${txt.join(", ")} with the first level dropped (${p} columns). ` : ""}80/20 split with seed ${c.seed}: ${d.meta.n_train} train, ${d.meta.n_test} test rows. StandardScaler fitted on the training rows. Features are ranked inside each of the ${c.folds} CV folds, so CV never sees validation rows.`;
  $("logSum").textContent = `Preparation log: ${log.length ? `${log.length} step${log.length === 1 ? "" : "s"}` : "nothing to clean"}. Read ${f.rows.toLocaleString("en-US")} rows × ${f.columns} columns as ${f.encoding}, ${sepName(f.sep)}-separated; ${d.meta.missing.toLocaleString("en-US")} missing cells, ${d.meta.duplicates.toLocaleString("en-US")} duplicate rows.`;
  $("prepLog").innerHTML = log.map((l) => `<li>${esc(l)}</li>`).join("");
}

// ── hero, metric cards, strip ────────────────────────────────────────────────
function renderHero(d) {
  const s = spec(), b = best(d), full = series(d, "All", "OLS")[0];
  $("heroTitle").textContent = s.title;
  $("heroLede").textContent = s.lede;
  $("pills").innerHTML = [[`${d.meta.rows} rows, ${d.meta.p} features, target ${d.target}`], [`${METHODS.length} selection methods × ${MODELS.length} models × k = 1 … ${d.meta.p}`],
    [`★ Best by CV: ${label(b)}, k = ${b.k} (test RMSE ${fmt(b.test_rmse)})`, "best"],
    [`Full model, k = ${d.meta.p}: test RMSE ${fmt(full.test_rmse)}`, full.test_rmse > b.test_rmse ? "warn" : ""],
    ...(s.ai ? [["✦ Descriptions on this page were written by AI (Gemini): check them", "ai"]] : [])]
    .map(([t, cls = ""]) => `<span class="pill ${cls}">${esc(t)}</span>`).join("");
}
function renderSummary(d) {
  const s = d.summary;
  $("summary").hidden = !s;
  if (!s) return;
  $("sumNote").classList.toggle("busy", !!s.pending);
  $("sumNote").textContent = s.pending ? "Asking the AI (Gemini) to summarise the results…" : s.error ? `AI summary unavailable (${s.error}).`
    : `Written by AI (Gemini ${s.model}${s.date ? `, ${s.date}` : ""}) from the numbers on this page, not from the data itself. Check every claim against the cards below.`;
  $("sumBody").innerHTML = s.summary ? `<p>${esc(s.summary)}</p><ul>${s.findings.map((f) => `<li>${esc(f)}</li>`).join("")}</ul><p><b>Recommendation.</b> ${esc(s.recommendation)}</p>${s.caveats.length ? `<p class="muted">Caveats: ${s.caveats.map(esc).join(" ")}</p>` : ""}` : "";
}
function renderMetrics(d, a) {
  const b = d.bench.mean;
  $("metricsHead").textContent = `${label(a)}, k = ${a.k}`;
  $("metrics").innerHTML = [
    ["Test RMSE", fmt(a.test_rmse), `lower is better; ${pct(1 - a.test_rmse / b.test)} below the mean baseline (${fmt(b.test)})`],
    ["Test R²", a.test_r2.toFixed(4), `explains ${pct(a.test_r2)} of ${d.target} variance on the test rows`],
    ["Adjusted R² (train)", a.adj_r2 == null ? "—" : a.adj_r2.toFixed(4), "penalises every extra coefficient"],
    ["Test MAE", fmt(a.test_mae), `mean absolute error${spec().unit ? `, in ${spec().unit}` : ""}`],
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
  const line = (y, el, fmtY) => chart($(el), (w) => Plot.plot({ width: w, height: 300, marginLeft: ml(), style: style(), grid: true, x: { label: "Number of features k", ticks: Math.min(d.meta.p, 12), tickFormat: "d" }, y: { label: null },
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
  $("equation").innerHTML = `<code>${esc(d.target)} = ${sig(a.intercept)} ${a.features.map((n, i) => `<span class="term">${a.coef[i] < 0 ? "−" : "+"} ${sig(Math.abs(a.coef[i]))} × <span class="var">${esc(n)}</span></span>`).join(" ")}</code>`;
  const sd = (n) => d.meta.train_sd[feats.indexOf(n)];
  $("coefNote").textContent = `Estimates in original units. Standardised = estimate × training SD, the change in ${d.target} per one SD of the predictor, comparable across rows.`;
  $("coefs").innerHTML = table(["Predictor", "Meaning", "Estimate", "Standardised", "Training SD"],
    [[td("Intercept (β₀)", "txt"), td(`baseline ${esc(d.target)} (${esc(desc(d.target))})`, "txt muted"), td(sig(a.intercept)), td("—"), td("—")],
      ...a.features.map((n, i) => [td(esc(n), "txt"), td(esc(desc(n)), "txt muted"), td(sig(a.coef[i])), td(sig(a.coef[i] * sd(n))), td(sig(sd(n)))])]);
  // actual vs predicted on every test row; the predictions come from this combination's equation, like the live predictor
  const idx = a.features.map((n) => feats.indexOf(n));
  const pts = d.x_test.map((row, i) => { const p = a.intercept + idx.reduce((s, j, t) => s + row[j] * a.coef[t], 0); return { y: d.y_test[i], p, r: d.y_test[i] - p }; });
  const miss = pts.filter((x) => Math.abs(x.r) > a.test_rmse), under = miss.filter((x) => x.r > 0).length;
  $("residNote").textContent = `All ${pts.length} test rows. The dashed line is a perfect prediction; points above it are under-predicted. ${miss.length} rows miss by more than the test RMSE (${fmt(a.test_rmse)}), ${under} of them under-predicted.`;
  const all = pts.flatMap((x) => [x.y, x.p]), lo = Math.min(...all), hi = Math.max(...all);
  chart($("resid"), (w) => Plot.plot({ width: w, height: 340, marginLeft: ml(), style: style(), grid: true, x: { label: `Predicted ${d.target}` }, y: { label: `Actual ${d.target}` },
    marks: [Plot.line([[lo, lo], [hi, hi]], { stroke: C.ink3, strokeDasharray: "3 3" }),
      Plot.dot(pts, { x: "p", y: "y", r: 3, fill: C.m[a.method] || C.accent, fillOpacity: 0.55, stroke: C.card, strokeWidth: 0.5, tip: true, title: (x) => `actual ${fmt(x.y)}\npredicted ${fmt(x.p)}\nresidual ${x.r >= 0 ? "+" : "−"}${fmt(Math.abs(x.r))}` })] }));
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
    $("prediction").innerHTML = `Predicted ${esc(d.target)}${spec().unit ? ` (${esc(spec().unit)})` : ""}<b>${fmt(y)}</b>± ${fmt(a.test_rmse)} test RMSE`;
  };
  form.oninput = update;
  update();
}

// ── upload: profile the file in the worker, let the AI review the plan, run, show it as a dataset ──
const U = { file: null, resp: null, rules: null, ai: null, source: "rules", plan: null, edited: false };
let worker;
const jobs = [];   // the worker answers in the order the messages were sent
function call(msg, transfer = [], onProgress) {
  if (!worker) {
    worker = new Worker("worker.js", { type: "module" });
    worker.onmessage = ({ data: m }) => {
      if (m.type === "status") say(m.text, true);
      else if (m.type === "progress") jobs[0]?.onProgress?.(m);
      else if (m.type === "error") jobs.shift()?.reject(Object.assign(new Error(m.message), { data: m.data }));
      else jobs.shift()?.resolve(m);
    };
    worker.onerror = (e) => { e.preventDefault(); drop(`The worker failed: ${e.message || "could not load worker.js"}`); };
  }
  return new Promise((resolve, reject) => { jobs.push({ resolve, reject, onProgress }); worker.postMessage(msg, transfer); });
}
function drop(why) {   // Cancel, or a worker that could not load: a fresh worker reboots Pyodide on the next call
  worker?.terminate();
  worker = null;
  jobs.splice(0).forEach((j) => j.reject(new Error(why)));
}
const say = (text, busy = false) => { $("upStatus").textContent = text; $("upStatus").classList.toggle("busy", busy); };
const showErr = (e) => { say(""); $("upError").textContent = e.message; };
const sepName = (s) => (s === "\t" ? "tab" : `"${s}"`);
// every non-target column gets an action; a column that just stopped being the target is numeric by definition
const norm = (plan) => ({ target: plan.target, columns: Object.fromEntries(U.resp.profile.filter((p) => p.name !== plan.target).map((p) => [p.name, plan.columns[p.name] || { action: "numeric", reason: "numbers" }])) });
const fromSource = (src) => norm(src === "ai" ? { target: U.ai.plan.target || U.rules.target, columns: { ...U.rules.columns, ...U.ai.plan.columns } } : U.rules);

function openUpload() {
  $("upload").hidden = false;
  if (!U.file) call({ type: "boot" }).then(() => { if (!U.file) say("Python is ready. Choose a CSV."); }).catch(showErr);   // download while the user browses for the file
}
async function planFile(file) {
  if (!file) return;
  Object.assign(U, { file, resp: null, ai: null, source: "rules", plan: null, edited: false });
  $("upError").textContent = ""; $("planBox").hidden = true; say(`Reading ${file.name}…`, true);
  try {
    const buf = await file.arrayBuffer();
    const { json } = await call({ type: "plan", file: buf }, [buf]);
    if (U.file !== file) return;   // another file was picked meanwhile
    U.resp = JSON.parse(json); U.rules = U.resp.plan; U.plan = norm(U.rules);
  } catch (e) { if (U.file === file) showErr(e); return; }
  say(""); $("planBox").hidden = false;
  renderPlan();
  askAI(file);
}
async function ask(path, body) {   // the proxy's answer, or {error}
  try {
    const r = await fetch(`${PROXY}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(50_000) });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  } catch (e) { return { error: e.name === "TimeoutError" ? "timed out" : e.name === "TypeError" ? "proxy not reachable" : e.message }; }
}
async function askAI(file) {
  const d = await ask("/plan", U.resp);
  if (U.file !== file) return;
  U.ai = d;
  if (!d.error && !U.edited) { U.source = "ai"; U.plan = fromSource("ai"); }   // never overwrite edits the user made while waiting
  renderPlan();
}
async function summarise(d) {
  d.summary = { pending: true };
  renderSummary(d);
  d.summary = await ask("/summary", d.summary_input);
  if (cur() === d) renderSummary(d);
}
function renderPlan() {
  const { file, resp, plan, rules, ai } = U, f = resp.file, hasAI = ai && !ai.error;
  $("fileInfo").textContent = `${file.name}: ${f.rows.toLocaleString("en-US")} rows × ${f.columns} columns, ${f.encoding}, ${sepName(f.sep)}-separated`;
  $("target").innerHTML = resp.profile.filter((p) => p.numeric >= 0.95 && p.unique > 2).map((p) => p.name).map((c) => `<option ${c === plan.target ? "selected" : ""}>${esc(c)}</option>`).join("");
  const changed = (c) => (c === plan.target ? plan.target !== rules.target : plan.columns[c].action !== (rules.columns[c]?.action ?? "target"));
  const n = resp.profile.filter((p) => changed(p.name)).length, cols = `${n} column${n === 1 ? "" : "s"}`;
  $("source").hidden = !hasAI;
  if (hasAI) seg($("source"), [["rules", "Rule-based"], ["ai", `✦ AI suggestion (${esc(ai.model)})`]], U.source, (v) => { U.source = v; U.edited = false; U.plan = fromSource(v); renderPlan(); });
  $("aiNote").classList.toggle("busy", !ai);
  $("aiNote").textContent = ai?.error ? `AI unavailable (${ai.error}); the rule-based plan is used.`
    : !ai ? "Asking the AI (Gemini) to review the plan…"
    : U.source === "ai" ? `The AI ${n ? `changed ${cols} (highlighted)` : "kept every rule-based choice"}. Switch to Rule-based if it looks wrong.`
    : n ? `Rule-based plan; the target differs from the rules' pick (highlighted).` : "Rule-based plan.";
  $("plan").innerHTML = `<table><thead><tr>${["Column", "Type", "Missing", "Distinct", "Sample values", "Action", "Why"].map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${resp.profile.map((p) => {
    const c = p.name, a = plan.columns[c], w = hasAI && ai.spec.descriptions[c];
    const act = c === plan.target ? [td(`<span class="st ok">target</span>`, "txt"), td("the column to predict", "txt muted")]
      : [td(`${a.action}${a.levels ? ` <small>${a.levels} labels</small>` : ""}`, "txt"), td(esc(a.reason), "txt muted")];
    return `<tr class="${changed(c) ? "changed" : ""}">${[td(`${esc(c)}${w ? `<small>${esc(w)}</small>` : ""}`, "txt"), td(p.type, "txt"), td(pct(p.missing)), td(p.unique.toLocaleString("en-US")), td(p.samples.map(esc).join(", "), "txt samples"), ...act].join("")}</tr>`;
  }).join("")}</tbody></table>`;
}
async function runUpload() {
  const { file, resp, ai, plan } = U, hasAI = ai && !ai.error;
  const same = hasAI && ai.plan.target === plan.target;   // the AI's title, lede and unit describe the target it picked
  const spec = { name: (same && ai.spec.name) || file.name.replace(/\.[^.]+$/, "").slice(0, 40), descriptions: hasAI ? ai.spec.descriptions : {}, ai: !!hasAI,
    lede: `Uploaded file ${file.name}: ${resp.file.rows.toLocaleString("en-US")} rows and ${resp.file.columns} columns. ${plan.target} is predicted from the other columns, prepared as the column plan says.` };
  if (same) for (const k of ["title", "lede", "unit"]) if (ai.spec[k]) spec[k] = ai.spec[k];
  $("upError").textContent = ""; $("run").disabled = true; $("cancel").hidden = false; $("prog").hidden = false; $("prog").removeAttribute("value"); $("runNote").textContent = "Preparing the data…";
  const t0 = performance.now();
  try {
    const buf = await file.arrayBuffer();
    const { json, secs } = await call({ type: "run", file: buf, plan, spec }, [buf],
      ({ done, total }) => { $("prog").max = total; $("prog").value = done; $("runNote").textContent = `Fitted ${done.toLocaleString("en-US")} of ${total.toLocaleString("en-US")} models, ${Math.round((performance.now() - t0) / 1000)} s`; });
    S.base.upload = JSON.parse(json); S.ds = "upload"; choose(S.base.upload);
    render();
    summarise(S.base.upload);
    $("runNote").textContent = `Done in ${secs.toFixed(1)} s. The results live in this tab only and replace any earlier upload.`;
    document.querySelector(".hero").scrollIntoView({ behavior: "smooth" });   // the upload panel sits above the results
  } catch (e) { $("runNote").textContent = ""; showErr(e); }
  finally { $("run").disabled = false; $("cancel").hidden = true; $("prog").hidden = true; say(""); }
}
$("uploadBtn").onclick = () => { openUpload(); $("file").click(); };
$("pick").onclick = () => $("file").click();
$("file").onchange = (e) => { planFile(e.target.files[0]); e.target.value = ""; };
$("closeUp").onclick = () => { $("upload").hidden = true; };
$("run").onclick = runUpload;
$("cancel").onclick = () => drop("Cancelled. The Python runtime is reloaded on the next run.");
$("target").onchange = (e) => { U.edited = true; U.plan = norm({ ...U.plan, target: e.target.value }); renderPlan(); };
$("upload").ondragover = (e) => { e.preventDefault(); $("drop").classList.add("over"); };
$("upload").ondragleave = () => $("drop").classList.remove("over");
$("upload").ondrop = (e) => { e.preventDefault(); $("drop").classList.remove("over"); planFile(e.dataTransfer.files[0]); };

// ── main ─────────────────────────────────────────────────────────────────────
function render() {
  const d = cur();
  if (!d) return;
  const a = active(d);
  renderControls(d, a);
  renderHero(d);
  renderSummary(d);
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
  choose(cur());
  $("status").textContent = "";
  render();
} catch (err) {
  $("status").className = "status error";
  $("status").textContent = `Results file missing (${err.message}). Run "python web/regress.py" to generate web/data/*.json, then serve with "python -m http.server -d web".`;
}
