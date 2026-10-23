"""CRISP-DM 多元線性迴歸比較：任意 CSV → 欄位計畫 → 5 種特徵選擇 × 5 種線性模型 × 每個 k。

read()（編碼、分隔符號）→ profile()（每欄摘要）→ 計畫（啟發式，之後可由 AI 或使用者改）→ check_plan()（驗證計畫）
→ prepare()（轉成數值特徵並記錄每個決定）→ run()（切分、screen() 檢查洩漏、CV、測試、基準）
CI / 本機：`python web/regress.py` → 依 web/presets.json 產生每個預設資料集的 JSON（web/data/<key>.json）
瀏覽器：worker.js 用 Pyodide 載入本檔，呼叫 plan_json() 與 run_json()
"""
import csv
import io
import itertools
import json
import math
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV, HuberRegressor, Lasso, LassoCV, LinearRegression, lars_path
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error as rmse
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso", "All"]
MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]
ACTIONS = ["numeric", "category", "date", "extract-number", "drop"]
LIMITS = dict(max_mb=50, min_rows=30, max_rows=2000, max_features=40, max_levels=10, max_missing=0.5)   # max_rows：p=37 時瀏覽器裡約 19 秒（Chrome 實測）
CONFIG = dict(seed=0, test_size=0.2, folds=5)                       # 80/20 切分、5 折 CV；網頁的說明文字會讀
HYPER = {"OLS": [], "Ridge": ["alpha"], "Lasso": ["alpha"], "ElasticNet": ["alpha", "l1_ratio"], "Huber": ["epsilon"]}
TARGETS = ["price", "target", "y", "label", "count", "cnt", "sales", "profit", "revenue", "rent", "charges", "salary",
           "medv", "popularity", "selling_price", "score"]       # 依序比對欄名（不分大小寫），都沒有就用最後一個數值欄
ID_NAME = re.compile(r"(^|[_ -])(id|ID|Id)$|^(index|Unnamed: ?\d+)$|[a-z]Id$")
ZIP_NAME = re.compile(r"zip|postal|postcode", re.I)
DATE_LIKE = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|^\d{8}T\d{4,6}|^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}")
NUMBER = r"([-+]?\d*\.?\d+)"


class DataError(ValueError):
    """給使用者看的錯誤：網頁直接顯示 str(e)。"""


# ── 讀檔與欄位摘要 ───────────────────────────────────────────────────────────
def read(data):
    """bytes 或 str → (DataFrame, 檔案資訊)。處理 BOM、Windows-1252、, ; tab | 分隔。"""
    info = {}
    if isinstance(data, str):
        text, info["encoding"] = data.lstrip("﻿"), "UTF-8"
    else:
        data = bytes(data)
        if data[:4] == b"PK\x03\x04" or data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise DataError("This is an Excel file. Open it, save it as CSV, and upload the CSV.")
        if len(data) > LIMITS["max_mb"] * 1e6:
            raise DataError(f"The file is {len(data) / 1e6:.0f} MB; the limit is {LIMITS['max_mb']} MB.")
        try:
            text, info["encoding"] = data.decode("utf-8-sig"), "UTF-8"
        except UnicodeDecodeError:
            text, info["encoding"] = data.decode("cp1252", errors="replace"), "Windows-1252"
    if not text.strip():
        raise DataError("The file is empty.")
    try:
        info["sep"] = csv.Sniffer().sniff(text[:50_000], delimiters=",;\t|").delimiter
    except csv.Error:
        info["sep"] = ","
    info["decimal"] = "," if info["sep"] == ";" else "."             # ; 分隔的檔案多半來自小數點用逗號的地區
    try:
        df = pd.read_csv(io.StringIO(text), sep=info["sep"], skipinitialspace=True, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise DataError(f"Could not read this as CSV ({str(e).strip().splitlines()[0]}).") from None
    df.columns = [str(c).strip() or f"column {i + 1}" for i, c in enumerate(df.columns)]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if df.shape[1] < 2:
        raise DataError(f"Found {df.shape[1]} column. The file needs a target and at least one feature, "
                        "separated by commas, semicolons or tabs.")
    info.update(rows=len(df), columns=df.shape[1])
    return df.replace([np.inf, -np.inf], np.nan), info


def to_number(s, dec="."):
    """'$1,200'、'45%'、'US$ 3.5'、'1.234,5'（dec=","）→ float；轉不了 → NaN。"""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip().str.replace(r"^[A-Z]{0,3}\$|[\s€£¥%]", "", regex=True)
    t = t.str.replace("." if dec == "," else ",", "", regex=False).str.replace(",", ".", regex=False)  # 先拿掉千分位
    return pd.to_numeric(t, errors="coerce").astype(float)


def extract_number(s):
    """'8GB'、'1.37kg'、'2.5 GHz' → 第一個數字。"""
    return pd.to_numeric(s.astype("string").str.extract(NUMBER, expand=False), errors="coerce").astype(float)


def to_dates(s):
    with warnings.catch_warnings():                                 # 推不出格式時 pandas 會逐筆解析並警告
        warnings.simplefilter("ignore")
        return pd.to_datetime(s.astype("string"), errors="coerce", utc=True)


def profile(df, dec="."):
    """每欄的摘要：給啟發式、AI 與網頁上的計畫表。"""
    out = []
    for c in df.columns:
        s = df[c].dropna()
        num = to_number(s, dec)
        whole = num.dropna()
        p = dict(name=c, type="number" if pd.api.types.is_numeric_dtype(s) else "text", count=len(s),
                 missing=round(1 - len(s) / len(df), 4), unique=int(s.nunique()),
                 samples=[str(v)[:40] for v in s.drop_duplicates().head(5)],
                 numeric=round(float(num.notna().mean()), 4) if len(s) else 0.0,
                 integer=bool(len(whole)) and bool((whole == whole.round()).all()),
                 increasing=len(whole) > 1 and whole.is_monotonic_increasing and s.nunique() == len(s))
        if p["type"] == "text" and len(s):
            head = s.astype("string").head(500)
            rest = head.str.replace(NUMBER, "", n=1, regex=True).str.strip()
            p.update(date=round(float((head.str.contains(DATE_LIKE) & to_dates(head).notna()).mean()), 4),
                     extract=round(float(extract_number(head).notna().mean()), 4),
                     units=rest[rest != ""].value_counts().index[:4].tolist(), length=round(float(head.str.len().mean()), 1))
        out.append(p)
    return out


# ── 計畫：每欄怎麼處理 ───────────────────────────────────────────────────────
def suggest(p):
    """欄位摘要 → (動作, 理由)。只看欄位本身；目標欄另外處理。"""
    n, distinct = p["unique"], p["unique"] / max(p["count"], 1)
    if p["missing"] > LIMITS["max_missing"]:
        return "drop", f"{p['missing']:.0%} of values missing"
    if n <= 1:
        return "drop", "the same value in every row"
    if ID_NAME.search(p["name"]) and distinct > 0.9:
        return "drop", "identifier"
    if p["numeric"] >= 0.95:
        if p["increasing"] and p["integer"]:
            return "drop", "row number (unique and increasing)"
        if ZIP_NAME.search(p["name"]):
            return "category", "postal code: a label, not a quantity"
        return "numeric", "numbers" if p["type"] == "number" else "numbers stored as text ($, %, separators)"
    if p.get("date", 0) >= 0.9:
        return "date", "dates, split into year and month"
    if p.get("extract", 0) >= 0.9 and len(p["units"]) <= 3 and all(len(str(u)) <= 8 for u in p["units"]):   # 單位很短：GB、kg、GHz
        return "extract-number", f"numbers with a unit ({', '.join(map(str, p['units']))})"
    if distinct > 0.5 or p.get("length", 0) > 40:
        return "drop", "free text (mostly unique values)"
    top = LIMITS["max_levels"] - 1
    return "category", f"{n} labels" + (f"; the {top} most common kept, the rest grouped as Other" if n > top + 1 else "")


def pick_target(prof):
    ok = [p for p in prof if p["unique"] > 2 and suggest(p)[0] == "numeric"]
    if not ok:
        raise DataError("No column looks like a numeric target (numbers with more than 2 distinct values).")
    names = {p["name"].lower(): p["name"] for p in ok}
    exact = next((names[t] for t in TARGETS if t in names), None)
    part = next((p["name"] for t in TARGETS if len(t) > 3 for p in ok if t in p["name"].lower()), None)
    return exact or part or ok[-1]["name"]


def heuristic_plan(prof):
    target = pick_target(prof)
    return {"target": target, "columns": {p["name"]: dict(zip(["action", "reason"], suggest(p))) for p in prof if p["name"] != target}}


def check_plan(plan, prof):
    """計畫可能來自 AI 或使用者，一律當作不可信：欄名要存在、動作要合法，缺的欄位補預設。→ (計畫, 訊息)"""
    plan, by, notes = plan or {}, {p["name"]: p for p in prof}, []
    target = plan.get("target") or pick_target(prof)
    if target not in by:
        raise DataError(f'The target column "{target}" is not in the file.')
    if by[target]["numeric"] < 0.95:
        raise DataError(f'The target "{target}" is not numeric (samples: {", ".join(by[target]["samples"][:3])}). '
                        "Linear regression needs a numeric target.")
    given = plan.get("columns") or {}
    notes += [f'The plan mentions "{c}", which is not in the file; ignored.' for c in given if c not in by]
    cols = {}
    for p in prof:
        c, g = p["name"], given.get(p["name"]) or {}
        if c == target:
            continue
        if g.get("action") in ACTIONS:
            cols[c] = {"action": g["action"], "reason": str(g.get("reason") or "set in the plan")[:200]}
            if str(g.get("levels", "")).isdigit() and 2 <= int(g["levels"]) <= 100:
                cols[c]["levels"] = int(g["levels"])
        else:
            if g:
                notes.append(f'{c}: unknown action "{g.get("action")}" in the plan; used the default.')
            cols[c] = dict(zip(["action", "reason"], suggest(p)))
    return {"target": target, "columns": cols}, notes


# ── 依計畫轉成數值特徵 ───────────────────────────────────────────────────────
def labels(s, dec):
    """類別欄 → 字串標籤（缺值是 'Missing'），連同自然排序（數字照大小，其他照字母）。"""
    num = to_number(s, dec)
    fmt = lambda v: str(int(v)) if v == int(v) else f"{v:.10g}"     # 98178.0 → "98178"
    lab = num.map(fmt, na_action="ignore") if num.notna().sum() == s.notna().sum() else s.astype("string").str.strip()
    lab = lab.where(s.notna(), "Missing").astype(str)
    key = (lambda v: (v in ("Other", "Missing"), float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else math.inf, v))
    return lab, key


def prepare(df, plan, dec):
    """→ (保留的原始列, X 特徵表（可有 NaN）, y, 紀錄, 每個特徵的來源, one-hot 群組)"""
    log, t = [], plan["target"]
    y = to_number(df[t], dec)
    if y.isna().any():
        log.append(f"Dropped {int(y.isna().sum())} rows with no {t} value.")
    df, y = df[y.notna()], y[y.notna()]
    if len(df) < LIMITS["min_rows"]:
        raise DataError(f"Only {len(df)} rows have a {t} value; need at least {LIMITS['min_rows']}.")
    if y.nunique() <= 2:
        raise DataError(f"{t} has only {y.nunique()} distinct values, so this is a classification problem. "
                        "This lab fits linear regression, which needs a numeric target.")
    if len(df) > LIMITS["max_rows"]:
        log.append(f"Using a random {LIMITS['max_rows']:,} of {len(df):,} rows (seed {CONFIG['seed']}) to keep the run fast.")
        keep = df.sample(LIMITS["max_rows"], random_state=CONFIG["seed"]).index.sort_values()
        df, y = df.loc[keep], y.loc[keep]

    cols, feats, cats = {}, {}, {}
    for c, a in plan["columns"].items():
        s, act = df[c], a["action"]
        if act == "drop":
            log.append(f"{c}: dropped ({a['reason']}).")
        elif act == "category":
            cats[c] = a.get("levels", LIMITS["max_levels"])
        elif act == "date":
            d = to_dates(s)
            for part in ("year", "month"):
                cols[f"{c}_{part}"], feats[f"{c}_{part}"] = getattr(d.dt, part).astype(float), {"column": c, "part": part}
        else:
            cols[c], feats[c] = (to_number(s, dec) if act == "numeric" else extract_number(s)), {"column": c}
        lost = [f for f, src in feats.items() if src["column"] == c and (cols[f].isna() & s.notna()).any()]
        if lost:
            log.append(f"{c}: {int((cols[lost[0]].isna() & s.notna()).sum())} values could not be read as "
                       f"{'dates' if act == 'date' else 'numbers'}; treated as missing.")

    room = LIMITS["max_features"] - len(cols)
    if room < len(cats):
        raise DataError(f"{len(cols)} numeric and {len(cats)} category columns is more than the "
                        f"{LIMITS['max_features']}-feature limit. Drop some columns in the plan.")
    lab = {c: labels(df[c], dec) for c in cats}
    want = {c: min(cats[c], lab[c][0].nunique()) for c in cats}       # 每欄保留幾個類別（含 Other）
    while sum(want.values()) - len(want) > room:                       # 超過特徵上限 → 從類別最多的那欄開始減
        c = max(want, key=want.get)
        want[c] -= 1
    groups = {}
    for c in cats:
        s, key = lab[c]
        counts = s.value_counts()
        if len(counts) > want[c]:
            s = s.where(s.isin(counts.index[:want[c] - 1]), "Other")
            log.append(f"{c}: kept the {want[c] - 1} most common of {len(counts)} labels; the rest are grouped as Other"
                       + (f" (feature limit {LIMITS['max_features']})." if want[c] < cats[c] else "."))
        if (df[c].isna()).any():
            log.append(f"{c}: {int(df[c].isna().sum())} missing values kept as their own label, Missing.")
        order = sorted(s.unique(), key=key)
        groups[c] = {"levels": order, "reference": order[0]}
        for lv in order[1:]:                                           # 第一個類別當基準（drop_first）
            cols[f"{c}_{lv}"], feats[f"{c}_{lv}"] = (s == lv).astype(float), {"column": c, "level": lv}
    if not cols:
        raise DataError("No usable features: every column was dropped. Keep at least one column in the plan.")
    return df, pd.DataFrame(cols, index=df.index), y.to_numpy(float), log, feats, groups


def impute(Xtr, *others):
    """缺值補訓練列的中位數（整欄都缺 → 0）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nan_to_num(np.nanmedian(Xtr, axis=0))
    return [np.where(np.isnan(A), med, A) for A in (Xtr, *others)]


def screen(Xtr, ytr, cols, target):
    """只看訓練列：常數欄、洩漏目標的欄、可由其他欄線性組合出的欄 → {欄位 index: 理由}。"""
    A = impute(Xtr)[0]
    sd = A.std(0)
    out = {int(j): "constant in the training rows" for j in np.flatnonzero(sd < 1e-12)}
    Z, yz = (A - A.mean(0)) / np.where(sd < 1e-12, 1, sd), (ytr - ytr.mean()) / ytr.std()
    r = Z.T @ yz / len(yz)
    out |= {int(j): f"leaks the target: correlation {r[j]:+.3f} with {target}" for j in np.flatnonzero(np.abs(r) > 0.98) if j not in out}
    keep = [j for j in range(A.shape[1]) if j not in out]
    while len(keep) > 1 and len(yz) >= 2 * len(keep) + 10:             # 一起幾乎完美預測目標 → 拿掉 |t| 最大的那欄
        B = Z[:, keep]
        b = np.linalg.lstsq(B, yz, rcond=None)[0]
        r2 = 1 - ((yz - B @ b) ** 2).sum() / (yz ** 2).sum()
        if r2 < 0.999:
            break
        j = keep[int(np.argmax(np.nan_to_num(np.abs(b) / np.sqrt(np.abs(np.diag(np.linalg.pinv(B.T @ B)))))))]
        out[j] = f"with the other columns it reproduces {target} almost exactly (R² {r2:.4f}), so it is probably part of the target"
        keep.remove(j)
    if keep:
        R, piv = qr(Z[:, keep], mode="economic", pivoting=True)[1:]
        d = np.abs(np.diag(R))
        rank = int((d > 1e-6 * d[0]).sum())
        base = [keep[i] for i in piv[:rank]]
        for j in [keep[i] for i in piv[rank:]]:
            w = np.linalg.lstsq(Z[:, base], Z[:, j], rcond=None)[0]
            out[j] = "an exact combination of " + ", ".join(cols[base[i]] for i in np.argsort(-np.abs(w))[:3] if abs(w[i]) > 1e-6)
    return out


def words(feats, groups, target, desc):
    """每個特徵一句說明：原始欄位的說明加上轉換方式。"""
    out = {target: desc.get(target, "")}
    for f, src in feats.items():
        c = src["column"]
        w = desc.get(c, "")
        if "level" in src:
            lv = "not one of the kept labels" if src["level"] == "Other" else src["level"]
            out[f] = f"1 if {c} is {lv}, else 0 (reference: {groups[c]['reference']})" + (f". {c}: {w}" if len(w) > 2 * len(c) else "")   # 「zipcode: ZIP code」這種只是重複欄名
        elif "part" in src:
            out[f] = f"{src['part']} of {c}" + (f" ({w})" if w else "")
        else:
            out[f] = w
    return out


# ── 特徵排序與模型網格 ───────────────────────────────────────────────────────
def make_model(name, alpha=None, l1_ratio=None):
    """Lasso / ElasticNet：沒給 α 就用 CV 調，之後用調好的 α 固定。Huber 的 ε 固定 1.35（sklearn 預設）。"""
    if name == "Lasso":
        return Lasso(alpha, max_iter=20000, precompute=True) if alpha else LassoCV(eps=1e-6, cv=5, max_iter=20000)   # Gram：每輪 O(k²) 而不是 O(nk)
    if name == "ElasticNet":
        return (ElasticNet(alpha, l1_ratio=l1_ratio, max_iter=20000, precompute=True) if alpha
                else ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], eps=1e-6, cv=5, max_iter=20000))   # 預設 eps=1e-3 的 α 網格太粗，美元尺度下會過度收縮
    return HuberRegressor(epsilon=1.35, max_iter=1000)


def rank(Z, y):
    """每種方法給一個完整的特徵排序（top-k 會巢狀）。Z 一律已標準化、y 已置中。"""
    if Z.shape[1] == 1:                                             # RFE 至少要 2 個特徵
        return {m: [0] for m in METHODS}
    y = y - y.mean()
    order = lambda s: [int(i) for i in np.argsort(-np.asarray(s, float), kind="stable")]
    coefs = lars_path(Z, y, method="lasso")[2]                      # 每列 = 一個特徵在路徑上的係數
    entry = [np.flatnonzero(c)[0] if c.any() else np.inf for c in coefs]
    r = {"F-test": order(np.nan_to_num(f_regression(Z, y)[0])),
         "MI": order(mutual_info_regression(Z, y, random_state=CONFIG["seed"])),
         "RFE": order(-RFE(LinearRegression(), n_features_to_select=1).fit(Z, y).ranking_),
         "Lasso": order(-np.array(entry, float)),                    # 越早進入路徑越重要
         "All": list(range(Z.shape[1]))}
    keep, dropped = list(range(Z.shape[1])), []                    # backward：每次拿掉 OLS |t| 最小的那個（= p 值最大）
    while len(keep) > 1:
        A = Z[:, keep]
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        s2 = ((y - A @ b) ** 2).sum() / max(len(y) - len(keep) - 1, 1)
        t = np.nan_to_num(np.abs(b) / np.sqrt(s2 * np.abs(np.diag(np.linalg.pinv(A.T @ A)))))
        j = keep[int(np.argmin(t))]
        keep.remove(j)
        dropped.append(j)
    r["Backward"] = keep + dropped[::-1]
    return r


def ols_solver(A, y):
    """含截距 OLS 的 Gram 矩陣只算一次，之後每組特徵只解一個 (k+1)×(k+1) 系統；與 LinearRegression 同解，但快得多。"""
    B = np.column_stack([np.ones(len(A)), A])
    G, g = B.T @ B, B.T @ y

    def fit(f):
        i = [0, *[j + 1 for j in f]]
        b = np.linalg.lstsq(G[np.ix_(i, i)], g[i], rcond=None)[0]   # lstsq：欄位共線時取最小範數解，和 sklearn 一樣
        return SimpleNamespace(coef_=b[1:], intercept_=b[0], predict=lambda C: b[0] + C @ b[1:])
    return fit


def ridge_loo(Z, y, alphas=np.logspace(-3, 3, 25)):
    """與 RidgeCV（GCV 留一誤差、截距不懲罰）同解：Z 已置中，一次特徵分解算完所有 α 的留一誤差。"""
    lam, V = np.linalg.eigh(Z.T @ Z)
    Q, yc, b0 = Z @ V, y - y.mean(), y.mean()
    inv = 1 / (np.maximum(lam, 0)[:, None] + alphas)               # k × α
    qy = Q.T @ yc
    h = 1 / len(y) + (Q ** 2) @ inv                                 # 帽子矩陣對角線（含截距）
    loo = ((yc[:, None] - Q @ (inv * qy[:, None])) / (1 - h)) ** 2
    a = int(np.argmin(loo.mean(0)))
    coef = V @ (inv[:, a] * qy)
    return SimpleNamespace(coef_=coef, intercept_=b0, alpha_=float(alphas[a]), predict=lambda C: b0 + C @ coef)


def warm(est, prev, s_prev, s):
    """Huber 從上一個 k 的解出發（新特徵的係數從 0 開始）：同一個凸問題，lbfgs 少走很多步。"""
    if prev is None or hasattr(prev, "failed"):
        return est
    pos = {j: i for i, j in enumerate(s_prev)}
    est.set_params(warm_start=True)
    est.coef_ = np.array([prev.coef_[pos[j]] if j in pos else 0.0 for j in s])
    est.intercept_, est.scale_ = prev.intercept_, prev.scale_
    return est


def fit_grid(Xtr, ytr, Xev, tick):
    """每種方法 × 每個 k × 每個模型，在標準化的訓練列上配適 → {(方法, k, 模型): (模型, 特徵 index, 預測, 係數)}。"""
    Xtr, Xev = impute(Xtr, Xev)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zev = sc.transform(Xtr), sc.transform(Xev)
    orders = rank(Ztr, ytr)
    # ponytail: α 每個 fold 只用全部特徵調一次、所有 k 共用（快 10–15 倍）；小 k 會略為過度收縮，p=30 實測 CV RMSE 中位數 +0.5%、最多 +0.9%。
    # 要更準就改成每個 k 各自用 LassoCV/ElasticNetCV（舊版做法），代價是 Pyodide 裡要等 1–2 分鐘
    la, en = make_model("Lasso").fit(Ztr, ytr), make_model("ElasticNet").fit(Ztr, ytr)
    tuned = {"Lasso": dict(alpha=la.alpha_), "ElasticNet": dict(alpha=en.alpha_, l1_ratio=en.l1_ratio_), "Huber": {}}
    ols = ols_solver(Ztr, ytr)
    out, fits = {}, {}
    for m in METHODS:
        o = orders[m]
        for k in ([len(o)] if m == "All" else range(1, len(o) + 1)):
            f = o[:k]
            s = sorted(f)                                           # 不同方法常挑到同一組特徵 → 同一組只配一次
            for name in MODELS:
                if (tuple(s), name) not in fits:
                    try:
                        if name == "OLS":
                            est = ols(s)
                        elif name == "Ridge":
                            est = ridge_loo(Ztr[:, s], ytr)
                        else:
                            est = make_model(name, **tuned[name])
                            if name == "Huber" and k > 1:
                                prev = tuple(sorted(o[:k - 1]))
                                est = warm(est, fits.get((prev, name), (None,))[0], prev, s)
                            est.fit(Ztr[:, s], ytr)
                        fits[tuple(s), name] = est, est.predict(Zev[:, s])
                    except Exception as e:                          # 單一模型失敗不拖垮整個網格：改用平均值並標記
                        fits[tuple(s), name] = (SimpleNamespace(coef_=np.zeros(k), intercept_=ytr.mean(), failed=str(e)[:200]),
                                                np.full(len(Zev), ytr.mean()))
                est, pred = fits[tuple(s), name]
                out[m, k, name] = est, f, pred, est.coef_[[s.index(i) for i in f]]   # 係數排回 f（排名）的順序
                tick()
    return sc, out


# ── 基準 ───────────────────────────────────────────────────────────
def subsets(p, k, rng, n=20):
    if math.comb(p, k) <= n:
        return [list(s) for s in itertools.combinations(range(p), k)]
    seen = set()
    while len(seen) < n:
        seen.add(tuple(sorted(rng.choice(p, k, replace=False).tolist())))
    return [list(s) for s in seen]


def benchmarks(Xtr, ytr, Xte, yte, folds):
    """預測平均值、隨機特徵子集（OLS）。"""
    rng, p = np.random.default_rng(CONFIG["seed"]), Xtr.shape[1]
    q = lambda a: [float(np.mean(a)), *map(float, np.percentile(a, [10, 50, 90]))]
    err = lambda y, pred: float(np.sqrt(np.mean((y - pred) ** 2)))
    sc = StandardScaler().fit(Xtr)                                  # OLS 的預測不受欄位縮放影響；標準化只是讓 Gram 矩陣好解
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    per_fold, full = [(ols_solver(Ztr[a], ytr[a]), b) for a, b in folds], ols_solver(Ztr, ytr)
    rand = []
    for k in range(1, p + 1):
        S = subsets(p, k, rng)
        cv = [np.mean([err(ytr[b], fit(f).predict(Ztr[b][:, f])) for fit, b in per_fold]) for f in S]
        te = [err(yte, full(f).predict(Zte[:, f])) for f in S]
        rand.append(dict(k=k, n=len(S), cv=q(cv), test=q(te)))
    return {"mean": {"cv": float(np.mean([rmse(ytr[b], np.full(len(b), ytr[a].mean())) for a, b in folds])),
                     "test": float(rmse(yte, np.full(len(yte), ytr.mean())))},
            "random": rand}


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run(data, plan=None, progress=None, spec=None):
    df, info = read(data)
    prof = profile(df, info["decimal"])
    plan, log = check_plan(plan, prof)
    target = plan["target"]
    raw, X, y, more, feats, groups = prepare(df, plan, info["decimal"])
    log += more
    tr, te = train_test_split(np.arange(len(y)), test_size=CONFIG["test_size"], random_state=CONFIG["seed"])
    bad = screen(X.to_numpy(float)[tr], y[tr], list(X.columns), target)
    log += [f"{X.columns[j]}: dropped ({why})." for j, why in sorted(bad.items())]
    X = X.drop(columns=X.columns[sorted(bad)])
    if X.shape[1] == 0:
        raise DataError("No usable features left after the checks. See the preparation log.")
    cols, Xn = list(X.columns), X.to_numpy(float)
    p, n = len(cols), len(tr)
    if n < 2 * p + 10:
        raise DataError(f"{n} training rows is too few for {p} features (need at least {2 * p + 10}). "
                        "Drop some columns or use a file with more rows.")
    log += [f"{c}: {int(k)} missing values filled with the median of the training rows."
            for c, k in zip(cols, np.isnan(Xn).sum(0)) if k]
    Xtr, ytr, Xte, yte = Xn[tr], y[tr], Xn[te], y[te]
    folds = list(KFold(CONFIG["folds"], shuffle=True, random_state=CONFIG["seed"]).split(Xtr))
    total = sum(1 if m == "All" else p for m in METHODS) * len(MODELS) * (len(folds) + 1)
    done = [0]

    def tick():
        done[0] += 1
        if progress and (done[0] % 20 == 0 or done[0] == total):
            progress(done[0], total)

    cv = {}
    for a, b in folds:                                              # 補值、標準化、特徵選擇都在每個 fold 內重做 → 不洩漏
        for key, (_, _, pred, _) in fit_grid(Xtr[a], ytr[a], Xtr[b], tick)[1].items():
            cv.setdefault(key, []).append(rmse(ytr[b], pred))
    sc, out = fit_grid(Xtr, ytr, Xte, tick)
    Xtr, Xte, Xn = impute(Xtr, Xte, Xn)

    combos = []
    for (m, k, name), (est, f, _, c) in out.items():
        coef = c / sc.scale_[f]                                     # 換回原始單位
        b0 = float(est.intercept_ - (coef * sc.mean_[f]).sum())
        pred_te, pred_tr = Xte[:, f] @ coef + b0, Xtr[:, f] @ coef + b0
        r2_tr, nz = r2_score(ytr, pred_tr), int(np.count_nonzero(np.abs(coef) > 1e-12))
        worst = np.argsort(-np.abs(yte - pred_te), kind="stable")[:10]
        combos.append(dict(
            method=m, k=k, model=name, features=[cols[i] for i in f],
            cv_rmse=float(np.mean(cv[m, k, name])), cv_std=float(np.std(cv[m, k, name])),
            test_rmse=float(rmse(yte, pred_te)), test_mae=float(mean_absolute_error(yte, pred_te)),
            test_r2=float(r2_score(yte, pred_te)),
            adj_r2=float(1 - (1 - r2_tr) * (n - 1) / (n - nz - 1)) if n - nz - 1 > 0 else None,
            hyper={h: float(getattr(est, h + "_", getattr(est, h, 0))) for h in HYPER[name]},
            coef=coef.tolist(), intercept=b0,
            top=[[int(raw.index[te[i]]), float(yte[i]), float(f"{pred_te[i]:.6g}")] for i in worst],   # 殘差最大的 10 筆測試列
            **({"failed": est.failed} if hasattr(est, "failed") else {})))

    spec = {"name": "Uploaded data", "title": f"What drives {target}?", "lede": "", "unit": "",
            "digits": 0 if np.median(np.abs(y)) >= 100 else 2, "descriptions": {}, **(spec or {})}
    numeric = [c for c in cols if "level" not in feats[c]]
    return dict(
        target=target, spec=spec, config=CONFIG,
        prep=dict(file=info, profile=prof, plan=plan, log=log),
        meta=dict(rows=len(y), p=p, n_train=n, n_test=len(yte), features=cols,
                  text_columns=[c for c in groups if any(feats[f]["column"] == c for f in cols)],
                  groups={c: g for c, g in groups.items() if any(feats[f]["column"] == c for f in cols)},
                  desc=words({f: feats[f] for f in cols}, groups, target, spec["descriptions"]),
                  ranges={c: [float(v) for v in np.percentile(Xn[:, cols.index(c)], [0, 50, 100])] for c in numeric},
                  train_sd=sc.scale_.tolist(),
                  missing=int(df.isna().sum().sum()), duplicates=int(df.duplicated().sum())),
        combos=combos, bench=benchmarks(Xtr, ytr, Xte, yte, folds))


# ── 給 worker.js（Pyodide）用：檔案已寫進虛擬檔案系統，結果以 JSON 字串回傳 ──
def plan_json(path):
    df, info = read(Path(path).read_bytes())
    prof = profile(df, info["decimal"])
    return json.dumps({"file": info, "profile": prof, "plan": heuristic_plan(prof)}, allow_nan=False)


def run_json(path, plan=None, progress=None, spec=None):
    return json.dumps(run(Path(path).read_bytes(), json.loads(plan) if plan else None, progress=progress,
                          spec=json.loads(spec) if spec else None), separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    here = Path(__file__).parent / "data"
    for p in json.loads((here.parent / "presets.json").read_text()):
        res = run((here / p["csv"]).read_bytes(), p.get("plan"), spec=p["spec"],
                  progress=lambda d, t: print(f"\r{p['key']}: {d}/{t}", end="", file=sys.stderr))
        text = json.dumps(res, separators=(",", ":"), allow_nan=False)  # allow_nan=False：任何 NaN/inf 都會失敗
        (here / f"{p['key']}.json").write_text(text)
        print(f"\n{p['key']}: {res['meta']['rows']} rows, p = {res['meta']['p']}, {len(res['combos'])} combos, "
              f"{len(text) / 1e6:.2f} MB", file=sys.stderr)
        print("  " + "\n  ".join(res["prep"]["log"]), file=sys.stderr)
        assert len(text) < 3e6, "JSON 超過 3 MB"
