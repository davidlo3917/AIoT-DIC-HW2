"""`python tests/test_regress.py`：亂七八糟的 CSV 要嘛跑完、要嘛給看得懂的錯誤；另外檢查已知的 OLS 數字與 backward 的算法。"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import StandardScaler

import regress
from regress import DataError, run

HERE = Path(__file__).parent
rng = np.random.default_rng(0)


def frame(n=80, **extra):
    """y = 3a − 2b + 雜訊，再加上指定的欄位（值或 n → 值的函式）。"""
    a, b = rng.normal(size=n), rng.normal(size=n)
    cols = {k: v(n) if callable(v) else v for k, v in extra.items()}
    return pd.DataFrame({"a": a, "b": b, **cols, "y": 3 * a - 2 * b + rng.normal(size=n)})


def go(data, plan=None):
    res = run(data if isinstance(data, (bytes, str)) else data.to_csv(index=False), plan)
    json.dumps(res, allow_nan=False)                                # NaN / inf 不能流進 JSON
    assert all(np.isfinite(c["cv_rmse"]) and np.isfinite(c["test_rmse"]) for c in res["combos"])
    return res


def action(res, col):
    return res["prep"]["plan"]["columns"][col]["action"]


def logged(res, *words):
    text = " | ".join(res["prep"]["log"])
    for w in words:
        assert w in text, (w, text)


def fails(data, words, plan=None):
    try:
        go(data, plan)
    except DataError as e:
        assert words.lower() in str(e).lower(), (words, str(e))
        return
    raise AssertionError(f"expected a DataError containing {words!r}")


def test_known_numbers():
    """50 Startups 走一般流程（目標由欄名 Profit 自動選出）要重現已知的 F-test + OLS 測試 RMSE（k = 1、2 與全特徵）。"""
    res = go((HERE / "50_startups.csv").read_bytes())
    assert res["target"] == "Profit" and action(res, "State") == "category"
    got = {(c["method"], c["k"]): c["test_rmse"] for c in res["combos"] if c["model"] == "OLS"}
    for key, want in [(("F-test", 1), 8274.868018), (("F-test", 2), 8198.797191), (("All", res["meta"]["p"]), 9137.990153)]:
        assert abs(got[key] - want) < 1e-3, (key, got[key], want)


def test_backward_is_pvalue_elimination():
    """rank() 的 |t| 捷徑（標準化、無截距）= 原始尺度含截距的 OLS，每步刪 p 值最大者。"""
    df, _ = regress.read((HERE / "50_startups.csv").read_bytes())
    plan, _ = regress.check_plan(None, regress.profile(df))
    X, y = regress.prepare(df, plan, ".")[1:3]
    X = X.to_numpy(float)
    keep, dropped = list(range(X.shape[1])), []
    while len(keep) > 1:
        A = np.column_stack([np.ones(len(y)), X[:, keep]])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        dof = len(y) - A.shape[1]
        se = np.sqrt(((y - A @ b) ** 2).sum() / dof * np.diag(np.linalg.inv(A.T @ A)))
        j = keep[int(np.argmax(2 * st.t.sf(np.abs(b / se), dof)[1:]))]
        keep.remove(j)
        dropped.append(j)
    assert regress.rank(StandardScaler().fit_transform(X), y)["Backward"] == keep + dropped[::-1]


def test_ridge_loo_equals_ridgecv():
    """自己算的留一誤差 ridge 要和 sklearn RidgeCV 選到同一個 α、得到同一組係數（含共線欄）。"""
    from sklearn.linear_model import RidgeCV
    for n, p, collinear in [(60, 4, False), (300, 12, True), (2000, 30, False)]:
        X = rng.normal(size=(n, p))
        if collinear:
            X[:, -1] = X[:, 0] + 1e-3 * rng.normal(size=n)
        y = X[:, :3] @ [2.0, -1.0, 0.5] + rng.normal(size=n) * 3
        Z = StandardScaler().fit_transform(X)
        ref, ours = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Z, y), regress.ridge_loo(Z, y)
        assert ref.alpha_ == ours.alpha_, (n, p, ref.alpha_, ours.alpha_)
        assert np.allclose(ref.coef_, ours.coef_, rtol=1e-7, atol=1e-9) and abs(ref.intercept_ - ours.intercept_) < 1e-9


def test_semicolon_and_decimal_comma():
    res = go(frame(width=lambda n: rng.uniform(1, 9, n)).to_csv(sep=";", decimal=",", index=False))
    assert res["prep"]["file"]["sep"] == ";" and action(res, "a") == "numeric"
    assert 1 <= res["meta"]["ranges"]["width"][0] and res["meta"]["ranges"]["width"][2] <= 9


def test_tab_bom_and_windows_1252():
    df = frame(city=lambda n: rng.choice(["Café", "Crème", "Zürich"], n))
    assert go(df.to_csv(sep="\t", index=False))["prep"]["file"]["sep"] == "\t"
    res = go(df.to_csv(index=False).encode("utf-8-sig"))
    assert "a" in res["prep"]["plan"]["columns"]                    # BOM 不能黏在第一個欄名上
    res = go(df.to_csv(index=False).encode("cp1252"))
    assert res["prep"]["file"]["encoding"] == "Windows-1252" and "Café" in res["meta"]["groups"]["city"]["levels"]


def test_numbers_stored_as_text():
    df = frame(money=lambda n: [f"${v:,.2f}" for v in rng.uniform(500, 5000, n)],
               share=lambda n: [f"{v:.1f}%" for v in rng.uniform(0, 100, n)],
               ram=lambda n: [f"{v}GB" for v in rng.choice([4, 8, 16, 32], n)],
               weight=lambda n: [f"{v:.2f}kg" for v in rng.uniform(1, 3, n)])
    res = go(df)
    assert [action(res, c) for c in ["money", "share", "ram", "weight"]] == ["numeric", "numeric", "extract-number", "extract-number"]
    lo, _, hi = res["meta"]["ranges"]["money"]
    assert 500 <= lo and hi <= 5000 and res["meta"]["ranges"]["ram"][2] == 32


def test_ids_dates_and_free_text():
    df = frame(id=lambda n: rng.permutation(n) + 1000,
               sold=lambda n: pd.date_range("2020-01-01", periods=n, freq="5D").strftime("%Y%m%dT000000"),
               note=lambda n: [f"customer said thing number {i}" for i in range(n)])
    res = go(df.to_csv())                                            # to_csv() 會多一欄 Unnamed: 0（列號）
    assert [action(res, c) for c in ["Unnamed: 0", "id", "sold", "note"]] == ["drop", "drop", "date", "drop"]
    assert {"sold_year", "sold_month"} <= set(res["meta"]["features"])


def test_categories_postal_codes_and_constants():
    df = frame(zipcode=lambda n: rng.choice(np.arange(98001, 98031), n), city=lambda n: rng.choice(["A", "B", "C"], n), same=7)
    res = go(df)
    assert action(res, "zipcode") == "category" and action(res, "same") == "drop"
    assert sum(f.startswith("zipcode_") for f in res["meta"]["features"]) == 9    # 前 9 名 + Other，基準不算
    assert res["meta"]["groups"]["city"]["reference"] == "A"
    logged(res, "zipcode: kept the 9 most common of ")


def test_feature_limit_shrinks_categories():
    old = regress.LIMITS["max_features"]
    regress.LIMITS["max_features"] = 6
    try:
        res = go(frame(n=120, shop=lambda n: rng.choice(list("ABCDEFGH"), n)))
    finally:
        regress.LIMITS["max_features"] = old
    assert res["meta"]["p"] <= 6
    logged(res, "feature limit 6")


def test_missing_values():
    df = frame(n=100, sparse=lambda n: np.where(rng.random(n) < 0.8, np.nan, 1.0),
               color=lambda n: rng.choice(["red", "blue", None], n))
    df.loc[rng.choice(100, 10, replace=False), "a"] = np.nan
    df.loc[:4, "y"] = np.nan
    res = go(df)
    assert action(res, "sparse") == "drop" and "Missing" in res["meta"]["groups"]["color"]["levels"]
    logged(res, "Dropped 5 rows with no y value", "a: 10 missing values filled", "color: ")


def test_leakage_and_exact_combinations():
    df = frame(n=120)
    df["copy"] = df.y * 2 + 1                                       # 目標換個單位
    df["c"] = df.a + df.b                                           # 特徵之間完全共線
    res = go(df[["a", "b", "c", "copy", "y"]])
    logged(res, "copy: dropped (leaks the target", "an exact combination of")
    assert len(res["meta"]["features"]) == 2, res["prep"]["log"]


def test_single_feature():
    res = go(frame(n=60)[["a", "y"]])
    assert res["meta"]["p"] == 1 and {c["k"] for c in res["combos"]} == {1}


def test_sum_leak_like_bike_sharing():
    """casual + registered = cnt，兩欄各自的相關都不到 0.98，要靠「一起完美預測」抓出來。"""
    df = frame(n=200)
    df["casual"] = rng.normal(size=200) * 5                          # corr(registered, y) ≈ √(109/134) ≈ 0.90
    df["registered"] = df.a * 10 + rng.normal(size=200) * 3
    df["y"] = df.casual + df.registered
    assert df.registered.corr(df.y) < 0.98
    res = go(df)
    logged(res, "reproduces y almost exactly")


def test_refusals():
    fails(b"PK\x03\x04 fake excel", "Excel")
    fails(b"", "empty")
    fails("x\n1\n2\n3\n", "column")
    fails(frame(n=10), "at least 30")
    fails(frame(flag=lambda n: rng.integers(0, 2, n)), "classification", {"target": "flag"})
    fails(frame(name=lambda n: [f"n{i}" for i in range(n)]), "not numeric", {"target": "name"})
    fails(frame(), "not in the file", {"target": "nope"})
    fails(pd.DataFrame({"a": ["x", "y"] * 20, "b": ["p", "q"] * 20}), "numeric target")
    fails(pd.DataFrame(rng.normal(size=(40, 16))).add_prefix("x").assign(y=rng.normal(size=40)), "too few")


def test_untrusted_plan():
    plan = {"target": "y", "columns": {"ghost": {"action": "numeric"}, "a": {"action": "explode"}, "b": {"action": "drop", "reason": "x" * 999}}}
    res = go(frame(), plan)
    assert action(res, "a") == "numeric" and action(res, "b") == "drop" and len(res["prep"]["plan"]["columns"]["b"]["reason"]) == 200
    logged(res, '"ghost", which is not in the file', 'unknown action "explode"')


def test_sampling():
    old = regress.LIMITS["max_rows"]
    regress.LIMITS["max_rows"] = 200
    try:
        res = go(frame(n=500))
    finally:
        regress.LIMITS["max_rows"] = old
    assert res["meta"]["rows"] == 200
    logged(res, "Using a random 200 of 500 rows")


def test_one_failing_model_does_not_stop_the_grid():
    real = regress.make_model

    def broken(name, **kw):
        if name == "Huber":
            raise RuntimeError("boom")
        return real(name, **kw)
    regress.make_model = broken
    try:
        res = go(frame())
    finally:
        regress.make_model = real
    huber = [c for c in res["combos"] if c["model"] == "Huber"]
    assert huber and all(c.get("failed") == "boom" for c in huber)
    assert not any("failed" in c for c in res["combos"] if c["model"] != "Huber")


if __name__ == "__main__":
    for name, test in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        test()
        print("ok", name)
