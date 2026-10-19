import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/+esm";

const METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso", "All"];
const MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"];
const ML = { "F-test": "F 檢定", MI: "互資訊", RFE: "RFE", Backward: "逐步後退", Lasso: "Lasso 嵌入", All: "全部特徵" };
const MNOTE = {
  "F-test": "過濾法：每個特徵單獨和目標做線性 F 檢定，分數高的先選。",
  MI: "過濾法：互資訊，連非線性的關係也看得到。",
  RFE: "包裝法：反覆訓練 OLS，每次剔除係數最小的特徵。",
  Backward: "包裝法：從全部特徵開始，每次拿掉「拿掉後 CV R² 最高」的那一個。",
  Lasso: "嵌入法：看 Lasso 路徑上誰最先得到非零係數。",
  All: "基準：不做選擇，全部特徵都用。",
};
const TEACHER = [
  ["1", "R&D Spend", 8274.868018, 0.946459, ["F-test", 1]],
  ["2", "R&D Spend, Marketing Spend", 8198.797191, 0.947439, ["F-test", 2]],
  ["3", "+ State_New York", 8309.059683, 0.946015, null],
  ["4", "+ State_Florida", 8409.916714, 0.944697, null],
  ["5", "+ State_California", 8409.916714, 0.944697, null],
  ["All 6", "+ Administration", 9137.990153, 0.934707, ["All", null]],
];
const DS = {
  startups: {
    name: "50 Startups", file: "startups", csv: "50_startups.csv", digits: 0, unit: "美元",
    kicker: "50 Startups：50 家美國新創公司的支出與利潤",
    title: "研發支出，<br>決定了新創的利潤嗎？",
    lede: "用研發、行政、行銷支出和所在州預測利潤。5 種特徵選擇、5 種線性迴歸、每一個特徵數 k 都跑過一遍，照 CRISP-DM 的六個階段看它們誰好、為什麼。",
    who: "創投或新創的財務人員：想知道錢花在哪裡最能換到利潤，以及要不要在意公司設在哪一州。",
    glossary: { "R&D Spend": "研發支出（美元）", Administration: "行政支出（美元）", "Marketing Spend": "行銷支出（美元）", State: "所在州（California / Florida / New York）", Profit: "利潤（美元），預測目標" },
  },
  boston: {
    name: "Boston Housing", file: "boston", csv: "boston.csv", digits: 2, unit: "千美元",
    kicker: "Boston Housing：1970 年代波士頓 506 個人口普查區",
    title: "什麼決定了<br>波士頓的房價？",
    lede: "13 個社區特徵預測房價中位數。同樣的 5 × 5 比較放到特徵更多的資料上，特徵選擇的差異會更明顯。",
    who: "都市規劃或不動產研究者：想知道哪些社區條件和房價最有關、一個線性模型能解釋多少。",
    glossary: {
      CRIM: "人均犯罪率", ZN: "大坪數住宅用地比例", INDUS: "非零售商業用地比例", CHAS: "是否鄰近查爾斯河（0/1）",
      NOX: "一氧化氮濃度", RM: "每戶平均房間數", AGE: "1940 年前建成的自住房比例", DIS: "到五個就業中心的加權距離",
      RAD: "高速公路可達性指數", TAX: "每萬美元的房產稅率", PTRATIO: "師生比", B: "由黑人比例換算的指標（見下方說明）",
      LSTAT: "低社經地位人口比例", MEDV: "自住房價中位數（千美元），預測目標",
    },
  },
};

const S = { ds: "startups", base: {}, live: {}, methods: new Set(METHODS), models: new Set(MODELS), kMode: "best", k: 2, rank: "cv", focus: null, whatif: {}, showAll: false };
const cur = () => S.live[S.ds] || S.base[S.ds];
const spec = () => DS[S.ds];
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = (v, d = spec().digits) => (v == null || !isFinite(v) ? "—" : v.toLocaleString("zh-TW", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v) => `${(v * 100).toFixed(1)}%`;
const sig = (v) => (v == null ? "—" : Math.abs(v) >= 1e4 ? Math.round(v).toLocaleString("zh-TW") : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : +v.toPrecision(4) + "");
const ml = () => (S.ds === "startups" ? 64 : 44);   // 美元數字比較寬
const key = (c) => `${c.method}|${c.model}`;
const label = (c) => `${ML[c.method]} + ${c.model}`;

// ── 排名與「格子」：一格 = 方法 × 模型，k 依模式決定 ────────────────────────────
const metricOf = (rank) => (c) => (rank === "cv" ? c.cv_rmse : c.test_rmse);
const score = (c) => metricOf(S.rank)(c);
const metricName = (rank = S.rank) => (rank === "cv" ? "CV RMSE（訓練集 5 折）" : "測試 RMSE");
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

// ── 顏色從 CSS 變數讀，深色模式跟著換 ─────────────────────────────────────────
let C = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement), v = (n) => cs.getPropertyValue(n).trim();
  C = { ink: v("--ink"), ink2: v("--ink-2"), ink3: v("--ink-3"), grid: v("--grid"), rule: v("--rule"), paper: v("--paper"), accent: v("--accent"),
    neg: v("--div-neg"), pos: v("--div-pos"), mid: v("--div-mid"), lo: v("--seq-lo"), hi: v("--seq-hi"),
    m: Object.fromEntries(METHODS.map((m, i) => [m, v(`--m${i + 1}`)])) };
}
const style = () => ({ background: "transparent", color: C.ink2, fontFamily: "Noto Sans TC, PingFang TC, sans-serif", fontSize: "12px" });
const halo = () => ({ fill: C.ink, stroke: C.paper, strokeWidth: 3 });

// ── 圖表：寬度跟著容器，一個 ResizeObserver 負責重畫 ─────────────────────────
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
const table = (head, rows) => `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((x) => `<td>${x}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

// ── 控制列 ──────────────────────────────────────────────────────────────────
function seg(box, options, value, onPick) {
  box.innerHTML = options.map(([v, t]) => `<button type="button" role="radio" aria-checked="${v === value}" data-v="${v}">${t}</button>`).join("");
  box.onclick = (e) => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); };
}
function chips(box, all, set, labelOf, color) {
  const avail = new Set(box.id === "methods" ? cur().methods : cur().models);
  box.innerHTML = all.map((x) => `<button type="button" class="chip" aria-pressed="${set.has(x)}" data-v="${x}" ${avail.has(x) ? "" : `disabled title="回到預設結果以啟用"`}>${color ? `<span class="sw" style="--c:var(--m${METHODS.indexOf(x) + 1})"></span>` : ""}${labelOf(x)}</button>`).join("")
    + `<button type="button" class="chip all" data-v="*">全選</button>`;
  box.onclick = (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    if (b.dataset.v === "*") all.filter((x) => avail.has(x)).forEach((x) => set.add(x));
    else if (set.has(b.dataset.v)) { if (set.size > 1) set.delete(b.dataset.v); }   // 最後一個不能取消
    else set.add(b.dataset.v);
    render();
  };
}
function renderControls() {
  const d = cur();
  seg($("ds"), Object.entries(DS).map(([k, v]) => [k, v.name]), S.ds, (v) => { S.ds = v; S.focus = null; S.whatif = {}; buildHyper(); render(); });
  chips($("methods"), METHODS, S.methods, (m) => ML[m], true);
  chips($("models"), MODELS, S.models, (m) => m, false);
  seg($("kmode"), [["best", "最佳 k"], ["fixed", "固定 k"]], S.kMode, (v) => { S.kMode = v; render(); });
  seg($("rank"), [["cv", "CV RMSE"], ["test", "測試 RMSE"]], S.rank, (v) => { S.rank = v; render(); });
  const k = $("k");
  k.max = d.meta.p; S.k = Math.min(S.k, d.meta.p); k.value = S.k; k.disabled = S.kMode !== "fixed";
  $("kout").textContent = S.kMode === "fixed" ? `k = ${S.k}` : "";
  document.querySelectorAll(".boston-only").forEach((el) => (el.hidden = S.ds !== "boston"));
}
$("k").oninput = (e) => { S.k = +e.target.value; render(); };

// ── 開場：最強特徵 vs 目標，迴歸線畫出來 ─────────────────────────────────────
function renderHero(d, best) {
  const s = spec(), f = d.meta.features[d.orders["F-test"][0]];
  $("heroKicker").textContent = s.kicker;
  $("heroTitle").innerHTML = s.title;
  $("heroLede").innerHTML = `${s.lede}<br><br>目前最佳：<b>${esc(label(best))}</b>，k = ${best.k}，${metricName()} ${fmt(score(best))}，比「預測平均值」好 ${pct(1 - score(best) / baseline(d))}。`;
  chart($("heroFig"), (w) => {
    const svg = Plot.plot({ width: w, height: Math.min(360, w * 0.7), marginLeft: ml(), style: style(), grid: true, x: { label: f }, y: { label: d.target },
      marks: [Plot.dot(d.rows, { x: f, y: d.target, r: 3.5, fill: C.ink2, fillOpacity: 0.55 }),
        Plot.linearRegressionY(d.rows, { x: f, y: d.target, stroke: C.accent, strokeWidth: 2.5, ci: 0, className: "fit" })] });
    svg.querySelectorAll(".fit path").forEach((p) => { p.setAttribute("pathLength", "1"); p.classList.add("draw"); });
    return svg;
  });
}

// ── 第一章 商業理解 ─────────────────────────────────────────────────────────
function renderBiz(d) {
  const s = spec(), g = s.glossary;
  $("biz").innerHTML = `<p>${esc(s.lede)}</p><p><b>誰會用：</b>${esc(s.who)}</p>
    <p><b>預測目標：</b>${esc(d.target)}（${esc(g[d.target])}）。<b>成功標準：</b>最佳組合的 CV RMSE 比「直接預測訓練集平均值」低 50% 以上，而且勝過隨機挑同樣數量的特徵。</p>
    <div class="table-wrap">${table(["欄位", "意義"], Object.entries(g).map(([k, v]) => [esc(k), esc(v)]))}</div>
    ${S.ds === "boston" ? `<p class="note">scikit-learn 從 1.2 版起移除了這份資料：B 欄是由「黑人居民比例」換算出來的，原作者假設種族隔離會影響房價。這裡保留 13 個欄位是為了和教科書結果對照；即時重跑可以把 B 刪掉。另外房價在 50 被截斷（16 筆），真實值可能更高。</p>` : ""}`;
}

// ── 第二章 資料理解 ─────────────────────────────────────────────────────────
function renderData(d) {
  const t = d.target, num = Object.keys(d.corr).filter((c) => c !== t);
  const top = num.reduce((a, b) => (Math.abs(d.corr[b][t]) > Math.abs(d.corr[a][t]) ? b : a));
  const raw = Object.keys(d.rows[0]).length - 1;
  $("eduLede").textContent = `共 ${d.meta.rows} 筆、${raw} 個特徵欄位。和 ${t} 最相關的是 ${top}（r = ${d.corr[top][t].toFixed(2)}）。`;
  const special = S.ds === "startups"
    ? [d.rows.filter((r) => r["R&D Spend"] === 0).length, "筆研發支出為 0"]
    : [d.rows.filter((r) => r.MEDV >= 50).length, "筆房價被截在 50"];
  $("quality").innerHTML = [[d.meta.rows, "筆資料"], [raw, "個特徵欄位"], [d.meta.missing, "個缺失值"], [d.meta.duplicates, "筆重複"], special]
    .map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");
  chart($("hist"), (w) => Plot.plot({ width: w, height: 260, style: style(), y: { grid: true, label: "筆數" }, x: { label: t },
    marks: [Plot.rectY(d.rows, Plot.binX({ y: "count" }, { x: t, thresholds: 12, fill: C.ink3, inset: 1, tip: true })), Plot.ruleY([0], { stroke: C.rule })] }));
  const cols = Object.keys(d.corr), cc = cols.flatMap((a) => cols.map((b) => ({ a, b, r: d.corr[a][b] })));
  chart($("corr"), (w) => Plot.plot({ width: w, height: Math.min(w, 60 + cols.length * 34), marginLeft: 110, marginBottom: 70, style: style(),
    x: { domain: cols, label: null, tickRotate: -40 }, y: { domain: cols, label: null },
    color: { type: "linear", domain: [-1, 0, 1], range: [C.neg, C.mid, C.pos] },
    marks: [Plot.cell(cc, { x: "a", y: "b", fill: "r", inset: 0.5, tip: true }),
      Plot.text(cc, { x: "a", y: "b", text: (x) => x.r.toFixed(2), fontSize: cols.length > 8 ? 9 : 11, fill: (x) => (Math.abs(x.r) > 0.6 ? C.paper : C.ink) })] }));
  multiples($("scatters"), num, (f) => `${esc(f)}　r = ${d.corr[f][t].toFixed(2)}`, (f, w) => Plot.plot({ width: w, height: 200, marginLeft: ml(), style: style(), grid: true, x: { label: null }, y: { label: null },
    marks: [Plot.dot(d.rows, { x: f, y: t, r: 2.5, fill: C.ink2, fillOpacity: 0.55, tip: true }), Plot.linearRegressionY(d.rows, { x: f, y: t, stroke: C.accent, ci: 0 })] }));
  const stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
  $("describe").innerHTML = table(["欄位", ...stats], Object.keys(d.describe).map((c) => [esc(c), ...stats.map((s) => sig(d.describe[c][s]))]));
}

// ── 第三章 資料準備 ─────────────────────────────────────────────────────────
function renderPrep(d) {
  const c = d.config, txt = d.meta.text_columns;
  $("prep").innerHTML = `<ul>
    ${txt.length ? `<li><b>One-hot encoding：</b>${txt.join("、")} 轉成 0/1 欄位${c.drop_first ? "，並刪除第一類（California 成為基準）。三個州欄位加起來永遠等於 1，全留著會和截距完全共線，這就是「虛擬變數陷阱」。" : "，三個州全部保留（虛擬變數陷阱：三欄相加 = 1，和截距共線；OLS 仍能算，但係數無法唯一決定）。"}</li>`
      : "<li><b>One-hot encoding：</b>沒有文字欄位；CHAS 本來就是 0/1。</li>"}
    <li><b>切分：</b><code>train_test_split(test_size=0.2, random_state=${c.seed})</code>，訓練 ${d.meta.n_train} 筆、測試 ${d.meta.n_test} 筆。測試集只在最後用一次。</li>
    <li><b>標準化：</b>${c.standardize ? "StandardScaler 只用訓練資料的平均與標準差，讓 Ridge / Lasso 的懲罰對每個特徵公平。" : "已關閉（Huber 仍會標準化，否則無法收斂）。Ridge / Lasso 的懲罰會偏向數值大的特徵。"}特徵排序一律在標準化後進行。</li>
    <li><b>不洩漏的特徵選擇：</b>在訓練集的每一個 CV fold 裡重新做特徵選擇，CV 分數才沒有偷看驗證資料。</li>
    ${S.ds === "boston" && (c.drop_b || c.drop_capped) ? `<li><b>即時設定：</b>${c.drop_b ? "已刪除 B 欄。" : ""}${c.drop_capped ? "已刪除房價 = 50 的列。" : ""}</li>` : ""}
  </ul>`;
  const o = d.onehot;
  $("onehotCap").textContent = txt.length ? "One-hot encoding：左邊是原始欄位，右邊是轉換後（前 6 筆）" : "One-hot encoding";
  $("onehot").innerHTML = txt.length
    ? table([...txt, "→", ...Object.keys(o.after[0])], o.before.map((r, i) => [...txt.map((t) => esc(r[t])), "", ...Object.values(o.after[i]).map((v) => v)]))
    : `<p class="muted">Boston 沒有需要編碼的欄位。</p>`;
  const test = new Set(d.test_index);
  $("splitCap").textContent = `80 / 20 切分：每格一筆，灰色 = 訓練 ${d.meta.n_train} 筆，深色 = 測試 ${d.meta.n_test} 筆`;
  $("split").innerHTML = d.rows.map((_, i) => `<i class="${test.has(i) ? "test" : ""}"></i>`).join("");
}

// ── 第四章 建模 ─────────────────────────────────────────────────────────────
function renderModel(d, f) {
  const feats = d.meta.features, k = f.k, shown = METHODS.filter((m) => S.methods.has(m) && m !== "All" && d.methods.includes(m));
  const picks = shown.map((m) => new Set(d.orders[m].slice(0, k)));
  const common = feats.filter((_, i) => picks.every((p) => p.has(i)));
  $("featLede").innerHTML = `${shown.map((m) => `<span class="dot" style="--c:var(--m${METHODS.indexOf(m) + 1})"></span><b>${ML[m]}</b>：${MNOTE[m]}`).join("<br>")}<br><br>數字 = 該方法給的名次，上色 = 在 k = ${k} 時被選中。${shown.length > 1 ? (common.length ? `${shown.length} 種方法都選了：${common.map(esc).join("、")}。` : "沒有任何特徵被所有方法同時選中。") : ""}`;
  const yorder = d.orders["F-test"].map((i) => feats[i]);
  const rm = shown.flatMap((m) => d.orders[m].map((fi, r) => ({ m: ML[m], key: m, feature: feats[fi], rank: r + 1, picked: r < k })));
  chart($("rankMatrix"), (w) => Plot.plot({ width: Math.min(w, 140 + shown.length * 110), height: 50 + feats.length * 30, marginLeft: 130, style: style(),
    x: { axis: "top", label: null, domain: shown.map((m) => ML[m]) }, y: { domain: yorder, label: null },
    marks: [Plot.cell(rm, { x: "m", y: "feature", fill: (x) => (x.picked ? C.m[x.key] : "transparent"), stroke: C.grid, inset: 1.5, rx: 4, tip: true,
      title: (x) => `${x.m}\n${x.feature}：第 ${x.rank} 名${x.picked ? "（選中）" : ""}` }),
      Plot.text(rm, { x: "m", y: "feature", text: "rank", ...halo(), fill: (x) => (x.picked ? C.ink : C.ink3) })] }));

  const models = MODELS.filter((m) => S.models.has(m) && d.models.includes(m));
  const fits = models.map((m) => find(d, f.method, f.k, m)).filter(Boolean);
  const sd = (name) => d.meta.train_sd[feats.indexOf(name)];
  const std = fits.map((c) => c.features.map((name, i) => ({ model: c.model, feature: name, v: c.coef[i] * sd(name) })));
  const vs = std.flat().map((x) => x.v), dom = [Math.min(0, ...vs) * 1.1, Math.max(0, ...vs) * 1.1];
  $("coefLede").textContent = `以 ${ML[f.method]} 選出的 ${f.k} 個特徵為準（第五章點選的焦點組合）。長條 = 標準化係數，也就是「特徵增加一個標準差，${d.target} 改變多少」，不同特徵可以直接比大小。`;
  multiples($("coefs"), fits.map((c, i) => [c, std[i]]), ([c]) => `${c.model}　${Object.entries(c.hyper).map(([h, v]) => `${h} = ${sig(v)}`).join("，")}`,
    ([, rows], w) => Plot.plot({ width: w, height: 40 + rows.length * 26, marginLeft: 120, style: style(), x: { domain: dom, label: null, grid: true }, y: { domain: rows.map((r) => r.feature), label: null },
      marks: [Plot.barX(rows, { x: "v", y: "feature", fill: C.ink2, rx: 3, insetTop: 4, insetBottom: 4, tip: true }), Plot.ruleX([0], { stroke: C.ink3 })] }));
  $("equation").innerHTML = `<div class="muted">${esc(label(f))}，k = ${f.k}，原始單位：</div>${esc(d.target)} = ${sig(f.intercept)} ${f.features.map((n, i) => `${f.coef[i] < 0 ? "−" : "+"} ${sig(Math.abs(f.coef[i]))} × ${esc(n)}`).join(" ")}`
    + (f.features.some((n) => n.startsWith("State_")) ? `<div class="muted">State_ 欄位是 0/1：例如 State_Florida 的係數 = Florida 相對於基準 California 的平均差距。</div>` : "");

  const fset = new Set(f.features);
  const tuned = (m) => (d.config.tuning === "cv" ? find(d, "All", null, m)?.hyper.alpha : d.config[{ Ridge: "alpha_ridge", Lasso: "alpha_lasso", ElasticNet: "alpha_enet" }[m]]);
  multiples($("paths"), ["Ridge", "Lasso", "ElasticNet"], (m) => `${m}${m === "ElasticNet" ? `（l1_ratio = ${d.paths.ElasticNet.l1_ratio}）` : ""}　α = ${sig(tuned(m))}`, (m, w) => {
    const p = d.paths[m], rows = feats.flatMap((name, i) => p.alphas.map((a, j) => ({ a, v: p.coefs[i][j], name, hi: fset.has(name) })));
    const ends = feats.map((name, i) => ({ name, v: p.coefs[i][p.alphas.indexOf(Math.min(...p.alphas))], a: Math.min(...p.alphas) })).filter((x) => fset.has(x.name));
    return Plot.plot({ width: w, height: 240, marginLeft: ml(), marginRight: 8, clip: true, style: style(), x: { type: "log", label: "α", grid: true }, y: { label: null, grid: true },
      marks: [Plot.ruleY([0], { stroke: C.rule }), Plot.line(rows.filter((r) => !r.hi), { x: "a", y: "v", z: "name", stroke: C.rule, strokeWidth: 1.5 }),
        Plot.line(rows.filter((r) => r.hi), { x: "a", y: "v", z: "name", stroke: C.ink2, strokeWidth: 2, tip: true, title: "name" }),
        Plot.text(ends, { x: "a", y: "v", text: "name", textAnchor: "start", dx: 4, fontSize: 10, ...halo() }),
        tuned(m) ? Plot.ruleX([tuned(m)], { stroke: C.accent, strokeWidth: 1.5 }) : null] });
  });
}

// ── 第五章 評估 ─────────────────────────────────────────────────────────────
function renderEval(d, list, f) {
  const base = baseline(d), best = list[0];
  $("evalLede").textContent = `${list.length} 個組合（特徵選擇 × 模型，${S.kMode === "best" ? "各自用最好的 k" : `k = ${S.k}`}），依${metricName()}排名。點表格任一列，第四到第六章都會改看那個組合。`;
  const byCv = cells(d, "cv")[0], byTest = cells(d, "test")[0], gain = 1 - best.cv_rmse / d.bench.mean.cv;
  const beatsRand = best.method === "All" || best.cv_rmse < d.bench.random[best.k - 1].cv[2];
  const ok = gain >= 0.5 && beatsRand;
  $("callouts").innerHTML = `
    <div class="callout"><b>${esc(label(best))}，k = ${best.k}</b>${metricName()} ${fmt(score(best))}，測試 R² ${best.test_r2.toFixed(3)}。</div>
    <div class="callout ${ok ? "ok" : "fail"}"><b>${ok ? "達成" : "未達成"}成功標準</b>CV RMSE 比預測平均值低 ${pct(gain)}（標準 50%）${beatsRand ? "，也勝過隨機挑選特徵。" : "，但沒有勝過隨機挑選特徵。"}</div>
    <div class="callout"><b>${key(byCv) === key(byTest) && byCv.k === byTest.k ? "CV 與測試的冠軍相同" : "CV 與測試的冠軍不同"}</b>依 CV：${esc(label(byCv))}（k = ${byCv.k}）；依測試：${esc(label(byTest))}（k = ${byTest.k}）。測試集只有 ${d.meta.n_test} 筆，從一堆組合裡挑測試分數最低的，那個分數會偏樂觀，所以預設用 CV 排名。</div>`;
  const th = (r, t) => `<button type="button" data-rank="${r}" aria-pressed="${S.rank === r}">${t}</button>`;
  $("board").innerHTML = `<thead><tr><th>#</th><th class="txt">特徵選擇</th><th class="txt">模型</th><th>k</th><th>${th("cv", "CV RMSE")}</th><th>± 標準差</th><th>${th("test", "測試 RMSE")}</th><th>測試 R²</th><th>調整後 R²（訓練）</th><th>勝過平均</th><th>勝過隨機</th></tr></thead><tbody>`
    + list.filter((c, i) => S.showAll || i < 10 || key(c) === key(f)).map((c) => {
      const i = list.indexOf(c);
      const rnd = c.method === "All" ? "—" : score(c) < randMed(d, c.k) ? `<span class="yes">✓ 是</span>` : `<span class="no">✗ 否</span>`;
      return `<tr tabindex="0" data-key="${esc(key(c))}" class="${key(c) === key(f) ? "focus" : ""}"><td>${i + 1}</td><td class="txt"><span class="dot" style="--c:var(--m${METHODS.indexOf(c.method) + 1})"></span>${ML[c.method]}${i === 0 ? `<span class="badge">最佳</span>` : ""}</td><td class="txt">${c.model}</td><td>${c.k}</td><td>${fmt(c.cv_rmse)}</td><td>${fmt(c.cv_std)}</td><td>${fmt(c.test_rmse)}</td><td>${c.test_r2.toFixed(3)}</td><td>${c.adj_r2 == null ? "—" : c.adj_r2.toFixed(3)}</td><td>${pct(1 - score(c) / base)}</td><td>${rnd}</td></tr>`;
    }).join("")
    + `<tr class="base"><td></td><td class="txt" colspan="3">基準：直接預測訓練集平均值</td><td>${fmt(d.bench.mean.cv)}</td><td></td><td>${fmt(d.bench.mean.test)}</td><td colspan="4"></td></tr></tbody>`;

  $("showAll").hidden = list.length <= 10;
  $("showAll").textContent = S.showAll ? "只顯示前 10 名" : `顯示全部 ${list.length} 個組合`;
  const hm = list.map((c) => ({ m: ML[c.method], model: c.model, v: score(c), gain: 1 - score(c) / base, k: c.k }));
  const g0 = Math.min(...hm.map((x) => x.gain)), g1 = Math.max(...hm.map((x) => x.gain));
  $("heatCap").textContent = `方法 × 模型的${metricName()}（格內數字，括號 = k）。顏色越深 = 比預測平均值好越多。`;
  chart($("heat"), (w) => Plot.plot({ width: w, height: 50 + new Set(hm.map((x) => x.m)).size * 44, marginLeft: 90, style: style(),
    x: { axis: "top", label: null, domain: MODELS.filter((m) => S.models.has(m)) }, y: { label: null, domain: METHODS.filter((m) => S.methods.has(m)).map((m) => ML[m]) },
    color: { type: "linear", range: [C.lo, C.hi] },
    marks: [Plot.cell(hm, { x: "model", y: "m", fill: "gain", inset: 1.5, rx: 4, tip: true }),
      Plot.text(hm, { x: "model", y: "m", text: (x) => (w < 560 ? fmt(x.v) : `${fmt(x.v)} (${x.k})`), fontSize: w < 560 ? 9 : 11, fill: (x) => ((x.gain - g0) / (g1 - g0 || 1) > 0.5 ? C.paper : C.ink) })] }));

  const yt = d.y_test, pts = yt.map((y, i) => ({ y, p: f.pred[i], r: y - f.pred[i] })), col = C.m[f.method];
  const lo = Math.min(...yt, ...f.pred), hi = Math.max(...yt, ...f.pred);
  $("focusCap").textContent = `焦點組合：${label(f)}，k = ${f.k}。上：實際 vs 預測（虛線 = 完美預測）；下：殘差。`;
  chart($("avp"), (w) => Plot.plot({ width: w, height: 240, marginLeft: ml(), style: style(), grid: true, x: { label: "實際值", domain: [lo, hi] }, y: { label: "預測值", domain: [lo, hi] },
    marks: [Plot.line([[lo, lo], [hi, hi]], { stroke: C.ink3, strokeDasharray: "4 4" }), Plot.dot(pts, { x: "y", y: "p", r: 4, fill: col, stroke: C.paper, strokeWidth: 1, tip: true })] }));
  chart($("resid"), (w) => Plot.plot({ width: w, height: 180, marginLeft: ml(), style: style(), grid: true, x: { label: "預測值" }, y: { label: "殘差" },
    marks: [Plot.ruleY([0], { stroke: C.ink3, strokeDasharray: "4 4" }), Plot.dot(pts, { x: "p", y: "r", r: 4, fill: col, stroke: C.paper, strokeWidth: 1, tip: true })] }));

  const mk = (c) => ({ k: c.k, v: score(c), method: c.method, label: ML[c.method] });
  const models = MODELS.filter((m) => S.models.has(m) && d.models.includes(m));
  const sel = METHODS.filter((m) => S.methods.has(m) && m !== "All");
  const band = d.bench.random.map((r) => ({ k: r.k, lo: (S.rank === "cv" ? r.cv : r.test)[1], hi: (S.rank === "cv" ? r.cv : r.test)[3] }));
  const vals = d.combos.filter((c) => models.includes(c.model) && S.methods.has(c.method)).map(score);
  const ylo = Math.min(...vals) * 0.95, yhi = Math.max(...vals) * 1.1, meanIn = base <= yhi;
  $("linesCap").innerHTML = `${metricName()} vs 特徵數 k，每個模型一張。${sel.map((m) => `<span class="dot" style="--c:var(--m${METHODS.indexOf(m) + 1})"></span>${ML[m]}`).join("　")}　灰帶 = 隨機挑 k 個特徵（OLS）的 p10–p90；灰色水平線 = 全部特徵。${meanIn ? "虛線 = 預測平均值。" : `預測平均值 = ${fmt(base)}（在圖外）。`}`;
  multiples($("lines"), models, (m) => m, (m, w) => {
    const rows = d.combos.filter((c) => c.model === m && sel.includes(c.method)).map(mk), all = find(d, "All", null, m);
    return Plot.plot({ width: w, height: 220, marginLeft: ml(), style: style(), clip: true, x: { label: "k", ticks: d.meta.p, tickFormat: "d" }, y: { label: null, grid: true, domain: [ylo, yhi] },
      marks: [Plot.areaY(band, { x: "k", y1: "lo", y2: "hi", fill: C.ink3, fillOpacity: 0.14 }),
        meanIn ? Plot.ruleY([base], { stroke: C.ink3, strokeDasharray: "4 4" }) : null,
        all && S.methods.has("All") ? Plot.ruleY([score(all)], { stroke: C.ink3, strokeWidth: 1.5 }) : null,
        Plot.line(rows, { x: "k", y: "v", z: "method", stroke: (x) => C.m[x.method], strokeWidth: 2 }),
        Plot.dot(rows, { x: "k", y: "v", fill: (x) => C.m[x.method], r: 3, stroke: C.paper, tip: true, title: (x) => `${x.label}，k = ${x.k}\n${fmt(x.v)}` })] });
  });

  const c = d.config, comparable = S.ds === "startups" && c.seed === 0 && c.test_size === 0.2;
  $("teacher").innerHTML = S.ds !== "startups" ? "" : `<h3>和老師的結果對照</h3><p class="prose">老師的表格：OLS、random_state=0、三個州欄位全留、依測試 RMSE 列出。第 1、2、All 列的特徵集合和這裡一樣，所以數字應該完全相同；第 3–5 列加入了州欄位，要在即時重跑把「刪除第一類」關掉才能對上。${comparable ? "" : `目前種子是 ${c.seed}，與老師的切分不同，無法對照。`}</p>`
    + `<div class="table-wrap">${table(["特徵數", "特徵", "老師 RMSE", "老師 R²", "這裡 RMSE", "相同？"], TEACHER.map(([n, feats, r, r2, ours]) => {
      const o = comparable && ours ? find(d, ours[0], ours[1] ?? null, "OLS") : null;
      return [n, `<span class="txt">${esc(feats)}</span>`, r.toFixed(6), r2.toFixed(6), o ? o.test_rmse.toFixed(6) : ours ? "—" : "需關閉刪除第一類", o ? (Math.abs(o.test_rmse - r) < 1e-3 ? `<span class="yes">✓</span>` : `<span class="no">✗</span>`) : ""];
    }))}</div>`;
}
$("board").onclick = (e) => {
  const r = e.target.closest("[data-rank]");
  if (r) { S.rank = r.dataset.rank; return render(); }
  const tr = e.target.closest("tr[data-key]");
  if (tr) { S.focus = tr.dataset.key; render(); }
};
$("showAll").onclick = () => { S.showAll = !S.showAll; render(); };
$("board").onkeydown = (e) => { if (e.key === "Enter" && e.target.dataset.key) { S.focus = e.target.dataset.key; render(); } };

// ── 第六章 部署：線性模型 → 在瀏覽器直接算 ───────────────────────────────────
function renderDeploy(d, f) {
  const nums = f.features.filter((n) => !n.startsWith("State_")), hasState = f.features.some((n) => n.startsWith("State_"));
  const states = [...new Set(d.rows.map((r) => r.State).filter(Boolean))].sort();
  const vals = (n) => d.rows.map((r) => r[n]).sort((a, b) => a - b);
  $("deployLede").textContent = `把第五章的焦點組合（${label(f)}，k = ${f.k}）拿來預測：拖動滑桿，預測值即時更新。線性模型的預測就是「截距 + Σ 係數 × 特徵」，所以直接在瀏覽器裡算。`;
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
    $("prediction").innerHTML = `<span>預測 ${esc(d.target)}（${spec().unit}）</span><b>${fmt(y)}</b><span>± ${fmt(f.test_rmse)}（測試 RMSE，約 68% 的誤差落在這個範圍內）</span>`;
  };
  form.oninput = update;
  update();
  const v = d.meta.versions;
  $("howbuilt").innerHTML = `<h3>這個網站怎麼運作</h3><p>每次推送到 GitHub，Actions 會用 Python ${v.python} + scikit-learn ${v.sklearn} 執行 <a href="regress.py">regress.py</a>：每種方法、每個 k、每個模型都在訓練集做 5 折 CV（Ridge / Lasso / ElasticNet 的 α 也用 CV 調），最後在測試集評估一次，結果存成 JSON，再部署到 GitHub Pages。頁面只是讀 JSON 畫圖。</p><p>「即時重跑」在瀏覽器裡用 Pyodide 跑同一份 regress.py，所以換種子、換超參數、改前處理都是真的重新訓練。</p>`;
}

// ── 即時重跑（Pyodide worker）────────────────────────────────────────────────
let worker;
const HP = [["alpha_ridge", "Ridge α", "Ridge"], ["alpha_lasso", "Lasso α", "Lasso"], ["alpha_enet", "ElasticNet α", "ElasticNet"]];
function buildHyper() {
  const d = S.base[S.ds];
  if (!d) return;
  const f = focused(cells(d));
  $("hyper").innerHTML = HP.map(([name, text, m]) => {
    const base = find(d, "All", null, m).hyper.alpha, start = find(d, f.method, f.k, m)?.hyper.alpha ?? base;
    return `<div class="hp"><div><span>${text}</span><output>${sig(start)}</output></div><input type="range" name="${name}" min="-2" max="2" step="0.05" value="${Math.max(-2, Math.min(2, Math.log10(start / base))).toFixed(2)}" data-base="${base}"></div>`;
  }).join("") + `<div class="hp"><div><span>ElasticNet l1_ratio</span><output>0.5</output></div><input type="range" name="l1_ratio" min="0.05" max="1" step="0.05" value="0.5"></div>
    <div class="hp"><div><span>Huber ε</span><output>1.35</output></div><input type="range" name="epsilon" min="1.05" max="3" step="0.05" value="1.35"></div>`;
}
const hpValue = (r) => (r.dataset.base ? +r.dataset.base * 10 ** +r.value : +r.value);
$("hyper").oninput = (e) => { const r = e.target; r.previousElementSibling.lastChild.textContent = sig(hpValue(r)); };
$("liveToggle").onclick = () => { const open = $("drawer").hidden; $("drawer").hidden = !open; $("liveToggle").setAttribute("aria-expanded", open); };
$("resetBtn").onclick = () => { delete S.live[S.ds]; S.focus = null; render(); };
$("liveForm").onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target), cfg = { tuning: "manual", seed: +fd.get("seed") || 0, folds: Math.min(10, Math.max(3, +fd.get("folds") || 5)) };
  ["drop_first", "standardize", "drop_b", "drop_capped"].forEach((n) => (cfg[n] = fd.has(n) && (S.ds === "boston" || !["drop_b", "drop_capped"].includes(n))));
  $("hyper").querySelectorAll("input").forEach((r) => (cfg[r.name] = hpValue(r)));
  const s = spec(), ds = S.ds, msg = $("liveMsg"), prog = $("prog"), btn = $("runBtn");
  btn.disabled = true; prog.hidden = false; prog.removeAttribute("value"); msg.textContent = "載入 Python（第一次約 31 MB）…";
  worker ||= new Worker("worker.js", { type: "module" });
  const csv = await fetch(`data/${s.csv}`).then((r) => r.text());
  worker.onmessage = ({ data: m }) => {
    if (m.type === "status") msg.textContent = m.text;
    if (m.type === "progress") { prog.max = m.total; prog.value = m.done; msg.textContent = `訓練中：${m.done} / ${m.total} 個模型`; }
    if (m.type === "error" || m.type === "done") { btn.disabled = false; prog.hidden = true; }
    if (m.type === "error") msg.textContent = `重跑失敗：${m.message}。頁面仍顯示原本的結果。`;
    if (m.type === "done") { S.live[ds] = JSON.parse(m.json); S.focus = null; msg.textContent = `完成，用了 ${m.secs} 秒。`; render(); }
  };
  worker.onerror = (err) => { btn.disabled = false; prog.hidden = true; msg.textContent = `無法啟動 Python（${err.message || "可能被瀏覽器或網路擋住"}）。頁面仍顯示原本的結果。`; };
  worker.postMessage({ csv, dataset: ds, cfg: JSON.stringify(cfg), methods: JSON.stringify([...S.methods]), models: JSON.stringify([...S.models]) });
};

// ── 主流程 ──────────────────────────────────────────────────────────────────
function render() {
  const d = cur();
  if (!d) return;
  renderControls();
  const list = cells(d), f = focused(list);
  S.focus = key(f);
  const live = S.live[S.ds];
  $("banner").hidden = !live;
  $("resetBtn").hidden = !live;
  if (live) $("banner").textContent = `目前顯示即時重跑的結果（種子 ${live.config.seed}，${live.config.folds} 折，手動超參數）。只有 OLS 的數字能和預設結果直接比較。按「回到預設結果」恢復。`;
  renderHero(d, list[0]);
  renderBiz(d);
  renderData(d);
  renderPrep(d);
  renderModel(d, f);
  renderEval(d, list, f);
  renderDeploy(d, f);
}

readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); render(); });
if (matchMedia("(max-width: 899px)").matches) $("rail").open = false;
try {
  const files = Object.values(DS).map((s) => fetch(`data/${s.file}.json`).then((r) => { if (!r.ok) throw new Error(`${r.status} data/${s.file}.json`); return r.json(); }));
  const [a, b] = await Promise.all(files);
  S.base = { startups: a, boston: b };
  $("status").textContent = "";
  buildHyper();
  render();
} catch (err) {
  $("status").className = "status error";
  $("status").textContent = `讀不到結果檔（${err.message}）。在本機請先執行 python web/regress.py 產生 web/data/*.json，再用 python -m http.server -d web 開啟。`;
}
