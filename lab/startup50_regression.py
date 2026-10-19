# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "scikit-learn", "matplotlib"]
# ///
"""50 Startups — 多元線性回歸 + backward elimination 特徵選擇（CRISP-DM 四步驟）。

執行：uv run startup50_regression.py
　　　或 pip install pandas scikit-learn matplotlib && python startup50_regression.py
輸出：終端機表格 + startup50_results.png。測試版本：scikit-learn 1.9.1、pandas 3.0.6。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

URL = ("https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/"
       "raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv")
BLUE, GRID = "#2a78d6", "#e1e0d9"   # 資料點 / 曲線、格線
pd.set_option("display.width", 250)

# 中文字型：取系統上第一個有的，沒有就退回 DejaVu Sans
_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [f for f in ["PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC", "Heiti TC",
                                               "Arial Unicode MS"] if f in _fonts] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Step 1 載入資料 Load Data ────────────────────────────────────────────────────────
df = pd.read_csv(URL)
print(df.head(), "\n")
print(df.describe().round(1), "\n")
print("各州平均 Profit:\n" + df.groupby("State")["Profit"].mean().round(0).to_string(), "\n")
print("與 Profit 的相關係數:\n" + df.corr(numeric_only=True)["Profit"].round(3).to_string(), "\n")

# ── Step 2 前處理 Preprocessing ───────────────────────────────────────────────────────
# State one-hot encoding（不 drop_first，與老師的設定相同）；dtype=float 轉成數值
df = pd.get_dummies(df, columns=["State"], dtype=float)
X = df.drop(columns="Profit")   # 2-D 特徵矩陣
y = df["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)   # 40 / 10

# ── Step 3 建模 Build Model：LinearRegression + backward elimination（訓練集 5-fold CV）──
# n_features_to_select 必須 < 6，全部 6 個特徵另外 fit
order, prev = [], set()
for k in range(1, 6):
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="backward")
    kept = set(X.columns[sfs.fit(X_train, y_train).get_support()])
    assert prev <= kept, "backward 路徑應為巢狀"
    prev = kept
    order += [c for c in X.columns if c in kept and c not in order]   # 累加顯示順序
feature_sets = [(str(k), order[:k]) for k in range(1, 6)] + [("All 6", list(X.columns))]

# ── Step 4 評估 Evaluation：MSE / RMSE / MAE / R²（test、train、adjusted、5-fold CV）──────
n = len(X_train)
rows = []
for label, feats in feature_sets:
    model = LinearRegression().fit(X_train[feats], y_train)
    pred, pred_train = model.predict(X_test[feats]), model.predict(X_train[feats])
    r2_train = r2_score(y_train, pred_train)
    rows.append({
        "Number of Features": label,
        "Selected Features": "[" + ", ".join(feats) + "]",
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),      # sklearn 1.6 移除了 squared=False
        "R-squared": r2_score(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, pred_train)),
        "Train R2": r2_train,
        "Adj R2": 1 - (1 - r2_train) * (n - 1) / (n - len(feats) - 1),
        "CV RMSE": -cross_val_score(LinearRegression(), X_train[feats], y_train, cv=5,
                                    scoring="neg_root_mean_squared_error").mean(),
    })
    if label in ("2", "All 6"):
        print(f"[{label} features] intercept = {model.intercept_:.1f}；coefficients:")
        print(pd.Series(model.coef_, index=feats).round(4).to_string(), "\n")

results = pd.DataFrame(rows)
# 自我檢查：六個 RMSE 須與老師的結果一致
expected = [8274.868018, 8198.797191, 8309.059683, 8409.916714, 8409.916714, 9137.990153]
assert np.allclose(results["RMSE"], expected, atol=1e-3), results["RMSE"].tolist()
fmt = {"RMSE": "{:.6f}".format, "R-squared": "{:.6f}".format, "MSE": "{:,.0f}".format, "MAE": "{:,.1f}".format,
       "Train RMSE": "{:,.1f}".format, "Train R2": "{:.5f}".format, "Adj R2": "{:.5f}".format, "CV RMSE": "{:,.1f}".format}
print(results.to_string(index=False, formatters=fmt))

# 總覽圖：Step 1 散佈圖、RMSE / R² 折線圖（測試集）、表格、說明 → startup50_results.png
OUT = Path(__file__).with_name("startup50_results.png")
DESC = (
    "設定：50 家新創公司。State 做 one-hot encoding（3 個 dummy）→ 6 個特徵；train_test_split(test_size=0.2, random_state=0) → 訓練 40 筆 / 測試 10 筆。\n"
    "模型：LinearRegression + backward elimination（SequentialFeatureSelector，只用訓練集做 5-fold CV）挑出最好的 k 個特徵；RMSE 與 R² 都用 10 筆測試資料計算。\n"
    "解讀：只用 R&D Spend 就能解釋 94.6% 的利潤變異（r = 0.97）；再加 Marketing Spend 測試誤差最低（RMSE 8,199）；之後每多加一個特徵，測試誤差都變差：\n"
    "State 的 dummy 沒有訊號；第 4、5 列完全相同，因為第三個 dummy 是多餘的（dummy variable trap）。\n"
    "結論：特徵越多 ≠ 模型越好。訓練 R² 只會上升（0.945 → 0.950），測試 R² 卻在 k = 2 之後下降——這就是 40 筆資料上的 overfitting。"
)
fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(4, 6, height_ratios=[1, 1.1, 0.55, 0.3], hspace=0.5, wspace=0.65, top=0.94)
for i, col in enumerate(["R&D Spend", "Administration", "Marketing Spend"]):   # Step 1 散佈圖
    ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
    ax.scatter(df[col], df["Profit"], s=26, color=BLUE, edgecolor="white", linewidth=1)
    ax.set(xlabel=col, ylabel="Profit", title=f"Step 1: {col} vs Profit  (r = {df[col].corr(df['Profit']):.2f})")
    ax.grid(color=GRID, linewidth=0.8)
k = range(1, 7)
for i, col in enumerate(["RMSE", "R-squared"]):   # 老師的兩張圖（測試集）
    ax = fig.add_subplot(gs[1, 3 * i:3 * i + 3])
    ax.plot(k, results[col], marker="o", lw=2, color=BLUE)
    ax.set(title=f"{col} by Number of Features", xlabel="Number of Features", ylabel=col,
           xticks=list(k), xticklabels=["1", "2", "3", "4", "5", "All 6"])
    ax.grid(color=GRID, linewidth=0.8)
ax = fig.add_subplot(gs[2, :]); ax.axis("off")   # 表格
cols = ["Number of Features", "Selected Features", "RMSE", "R-squared"]
tbl = ax.table(cellText=[[r[c] if isinstance(r[c], str) else f"{r[c]:.6f}" for c in cols] for r in rows],
               colLabels=cols, colWidths=[0.13, 0.67, 0.10, 0.10], cellLoc="left", loc="upper center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
for c in range(4):
    tbl[0, c].set_text_props(weight="bold")
    tbl[2, c].set_facecolor("#e9f1fb")           # 最佳列
ax = fig.add_subplot(gs[3, :]); ax.axis("off")
ax.text(0, 1, DESC, va="top", ha="left", fontsize=9.5, linespacing=1.6)
fig.suptitle("50 Startups：多元線性回歸 + backward elimination 特徵選擇（測試集 10 筆）", fontsize=14, y=0.995)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n總覽圖已存成 {OUT.name}")
plt.show()
