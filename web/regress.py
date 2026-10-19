"""CRISP-DM 多元線性迴歸比較：5 種特徵選擇 × 5 種線性模型 × 每個 k。

CI / 本機：`python web/regress.py` → 產生 web/data/{startups,boston}.json 並檢查老師的數字
"""
import io
import itertools
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.linear_model import (ElasticNet, ElasticNetCV, HuberRegressor, Lasso, LassoCV, LinearRegression, Ridge,
                                  RidgeCV, enet_path, lars_path, lasso_path)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

METHODS = ["F-test", "MI", "RFE", "Backward", "Lasso", "All"]
MODELS = ["OLS", "Ridge", "Lasso", "ElasticNet", "Huber"]
DATASETS = {"startups": ("50_startups.csv", "Profit"), "boston": ("boston.csv", "MEDV")}
DEFAULT = dict(seed=0, test_size=0.2, folds=5, drop_first=True, standardize=True, drop_b=False, drop_capped=False,
               tuning="cv", alpha_ridge=1.0, alpha_lasso=1.0, alpha_enet=1.0, l1_ratio=0.5, epsilon=1.35)
HYPER = {"OLS": [], "Ridge": ["alpha"], "Lasso": ["alpha"], "ElasticNet": ["alpha", "l1_ratio"], "Huber": ["epsilon"]}


def make_model(name, cfg):
    cv = cfg["tuning"] == "cv"
    if name == "OLS":
        return LinearRegression()
    if name == "Ridge":
        return RidgeCV(alphas=np.logspace(-3, 3, 25)) if cv else Ridge(cfg["alpha_ridge"])
    if name == "Lasso":
        return LassoCV(eps=1e-6, cv=5, max_iter=20000) if cv else Lasso(cfg["alpha_lasso"], max_iter=20000)
    if name == "ElasticNet":
        return (ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99], eps=1e-6, cv=5, max_iter=20000) if cv   # 預設 eps=1e-3 的 α 網格太粗，美元尺度下會過度收縮
                else ElasticNet(cfg["alpha_enet"], l1_ratio=cfg["l1_ratio"], max_iter=20000))
    return HuberRegressor(epsilon=cfg["epsilon"], max_iter=1000)


def prepare(csv_text, dataset, cfg):
    raw = pd.read_csv(io.StringIO(csv_text))
    if dataset == "boston":
        raw.columns = raw.columns.str.upper()
        raw = raw.drop(columns=["B"] if cfg["drop_b"] else [])
        raw = raw[raw.MEDV < 50] if cfg["drop_capped"] else raw
    target = DATASETS[dataset][1]
    text = [c for c in raw.columns if raw[c].dtype == object or pd.api.types.is_string_dtype(raw[c])]
    X = pd.get_dummies(raw.drop(columns=target), columns=text, drop_first=cfg["drop_first"], dtype=float)
    return raw.reset_index(drop=True), X.reset_index(drop=True), raw[target].to_numpy(float), text


def rank(Z, y, seed):
    """每種方法給一個完整的特徵排序（top-k 會巢狀）。Z 一律已標準化、y 已置中。"""
    y = y - y.mean()
    order = lambda s: [int(i) for i in np.argsort(-np.asarray(s, float), kind="stable")]
    coefs = lars_path(Z, y, method="lasso")[2]                      # 每列 = 一個特徵在路徑上的係數
    entry = [np.flatnonzero(c)[0] if c.any() else np.inf for c in coefs]
    r = {"F-test": order(np.nan_to_num(f_regression(Z, y)[0])),
         "MI": order(mutual_info_regression(Z, y, random_state=seed)),
         "RFE": order(-RFE(LinearRegression(), n_features_to_select=1).fit(Z, y).ranking_),
         "Lasso": order(-np.array(entry, float)),                    # 越早進入路徑越重要
         "All": list(range(Z.shape[1]))}
    keep, dropped = list(range(Z.shape[1])), []                    # backward：拿掉後 CV R² 最高的那個
    while len(keep) > 1:
        score = {j: cross_val_score(LinearRegression(), Z[:, [c for c in keep if c != j]], y, cv=5).mean() for j in keep}
        j = max(score, key=score.get)
        keep.remove(j)
        dropped.append(j)
    r["Backward"] = keep + dropped[::-1]
    return r


def scaled(name, cfg):
    return cfg["standardize"] or name == "Huber"   # ponytail: Huber 在原始尺度上 lbfgs 不收斂（Boston 實測），一律標準化


def fit_grid(Xtr, ytr, Xev, cfg, methods, models, tick):
    sc = StandardScaler().fit(Xtr)
    Ztr, Zev = sc.transform(Xtr), sc.transform(Xev)
    orders = rank(Ztr, ytr, cfg["seed"])
    out = {}
    for m in methods:
        o = orders[m]
        for k in ([len(o)] if m == "All" else range(1, len(o) + 1)):
            f = o[:k]
            for name in models:
                Atr, Aev = (Ztr, Zev) if scaled(name, cfg) else (Xtr, Xev)
                est = make_model(name, cfg).fit(Atr[:, f], ytr)
                out[m, k, name] = est, f, est.predict(Aev[:, f])
                tick()
    return orders, sc, out


def subsets(p, k, rng, n=50):
    if math.comb(p, k) <= n:
        return [list(s) for s in itertools.combinations(range(p), k)]
    seen = set()
    while len(seen) < n:
        seen.add(tuple(sorted(rng.choice(p, k, replace=False).tolist())))
    return [list(s) for s in seen]


def benchmarks(Xtr, ytr, Xte, yte, folds, seed, X_all, y_all):
    """預測平均值、隨機特徵子集（OLS）、全資料樣本內 R²。"""
    rng, p = np.random.default_rng(seed), Xtr.shape[1]
    q = lambda a: [float(np.mean(a)), *map(float, np.percentile(a, [10, 50, 90]))]
    ols = lambda A, y, B, f: LinearRegression().fit(A[:, f], y).predict(B[:, f])
    rand = []
    for k in range(1, p + 1):
        S = subsets(p, k, rng)
        cv = [np.mean([rmse(ytr[b], ols(Xtr[a], ytr[a], Xtr[b], f)) for a, b in folds]) for f in S]
        te = [rmse(yte, ols(Xtr, ytr, Xte, f)) for f in S]
        rand.append(dict(k=k, n=len(S), cv=q(cv), test=q(te)))
    return {"mean": {"cv": float(np.mean([rmse(ytr[b], np.full(len(b), ytr[a].mean())) for a, b in folds])),
                     "test": float(rmse(yte, np.full(len(yte), ytr.mean())))},
            "random": rand,
            "insample_r2": float(LinearRegression().fit(X_all, y_all).score(X_all, y_all))}


def paths(A, y, l1):
    """全特徵、訓練集上的 Ridge / Lasso / ElasticNet 係數路徑（ElasticNet 用調好的 l1_ratio）。"""
    y = y - y.mean()
    ridge_a = np.logspace(-3, 4, 40)
    ridge = np.array([Ridge(a).fit(A, y).coef_ for a in ridge_a]).T
    la, lc, _ = lasso_path(A, y, eps=1e-5, n_alphas=40)
    ea, ec, _ = enet_path(A, y, l1_ratio=l1, eps=1e-5, n_alphas=40)
    pack = lambda a, c: {"alphas": a.tolist(), "coefs": c.tolist()}
    return {"Ridge": pack(ridge_a, ridge), "Lasso": pack(la, lc), "ElasticNet": {**pack(ea, ec), "l1_ratio": l1}}


def run(csv_text, dataset, cfg=None, methods=METHODS, models=MODELS, progress=None):
    cfg = {**DEFAULT, **(cfg or {})}
    methods = [m for m in METHODS if m in methods]
    models = [m for m in MODELS if m in models]
    raw, X, y, text = prepare(csv_text, dataset, cfg)
    cols, Xn = list(X.columns), X.to_numpy(float)
    tr, te = train_test_split(np.arange(len(y)), test_size=cfg["test_size"], random_state=cfg["seed"])
    Xtr, ytr, Xte, yte = Xn[tr], y[tr], Xn[te], y[te]
    folds = list(KFold(cfg["folds"], shuffle=True, random_state=cfg["seed"]).split(Xtr))
    p, n = len(cols), len(ytr)
    total = sum(1 if m == "All" else p for m in methods) * len(models) * (len(folds) + 1)
    done = [0]

    def tick():
        done[0] += 1
        if progress and (done[0] % 20 == 0 or done[0] == total):
            progress(done[0], total)

    cv = {}
    for a, b in folds:                                              # 特徵選擇在每個 fold 內重做 → 不洩漏
        for key, (_, _, pred) in fit_grid(Xtr[a], ytr[a], Xtr[b], cfg, methods, models, tick)[2].items():
            cv.setdefault(key, []).append(rmse(ytr[b], pred))
    orders, sc, out = fit_grid(Xtr, ytr, Xte, cfg, methods, models, tick)

    combos = []
    for (m, k, name), (est, f, _) in out.items():
        s = scaled(name, cfg)
        coef = est.coef_ / sc.scale_[f] if s else est.coef_         # 換回原始單位
        b0 = float(est.intercept_ - (coef * sc.mean_[f]).sum()) if s else float(est.intercept_)
        pred_te, pred_tr = Xte[:, f] @ coef + b0, Xtr[:, f] @ coef + b0
        r2_tr, nz = r2_score(ytr, pred_tr), int(np.count_nonzero(np.abs(coef) > 1e-12))
        combos.append(dict(
            method=m, k=k, model=name, features=[cols[i] for i in f],
            cv_rmse=float(np.mean(cv[m, k, name])), cv_std=float(np.std(cv[m, k, name])),
            test_rmse=float(rmse(yte, pred_te)), test_mae=float(mean_absolute_error(yte, pred_te)),
            test_r2=float(r2_score(yte, pred_te)), train_r2=float(r2_tr),
            adj_r2=float(1 - (1 - r2_tr) * (n - 1) / (n - nz - 1)) if n - nz - 1 > 0 else None,
            hyper={h: float(getattr(est, h + "_", getattr(est, h, 0))) for h in HYPER[name]},
            coef=coef.tolist(), intercept=b0, pred=[float(f"{v:.6g}") for v in pred_te]))

    A = sc.transform(Xtr) if cfg["standardize"] else Xtr
    num = raw.select_dtypes("number")
    return dict(
        dataset=dataset, target=DATASETS[dataset][1], config=cfg, methods=methods, models=models,
        meta=dict(rows=len(raw), p=p, n_train=n, n_test=len(yte), features=cols, text_columns=text,
                  train_mean=sc.mean_.tolist(), train_sd=sc.scale_.tolist(),
                  missing=int(raw.isna().sum().sum()), duplicates=int(raw.duplicated().sum()),
                  versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__)),
        rows=json.loads(raw.to_json(orient="records")), test_index=te.tolist(), y_test=yte.tolist(),
        describe=json.loads(num.describe().round(4).to_json()), corr=json.loads(num.corr().round(4).to_json()),
        onehot=dict(before=json.loads(raw[text].head(6).to_json(orient="records")),
                    after=json.loads(X[[c for c in cols if any(c.startswith(t + "_") for t in text)]].head(6)
                                     .to_json(orient="records"))),
        orders={m: orders[m] for m in METHODS},
        combos=combos, bench=benchmarks(Xtr, ytr, Xte, yte, folds, cfg["seed"], Xn, y), paths=paths(A, ytr, next((c["hyper"]["l1_ratio"] for c in combos if c["method"] == "All" and c["model"] == "ElasticNet"), cfg["l1_ratio"])))



if __name__ == "__main__":
    here = Path(__file__).parent / "data"
    for key, (csv, _) in DATASETS.items():
        res = run((here / csv).read_text(), key, progress=lambda d, t: print(f"\r{key}: {d}/{t}", end="", file=sys.stderr))
        text = json.dumps(res, separators=(",", ":"), allow_nan=False)  # allow_nan=False：任何 NaN/inf 都會失敗
        (here / f"{key}.json").write_text(text)
        print(f"\n{key}: {len(res['combos'])} combos, {len(text) / 1e6:.2f} MB", file=sys.stderr)
        assert len(text) < 2e6, "JSON 超過 2 MB"
        if key == "startups":                                       # 老師表格第 1、2 列與全特徵列（random_state=0）
            got = {(c["method"], c["k"]): c["test_rmse"] for c in res["combos"] if c["model"] == "OLS"}
            for k, want in [(("F-test", 1), 8274.868018), (("F-test", 2), 8198.797191), (("All", res["meta"]["p"]), 9137.990153)]:
                assert abs(got[k] - want) < 1e-3, (k, got[k], want)
            print("teacher RMSE check: OK", file=sys.stderr)
