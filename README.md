# AIIS HW2 — 任意 CSV 的線性迴歸實驗儀表板（CRISP-DM + AI）

接續 [Lab 4](https://github.com/davidlo3917/AIoT-DIC-L4) 的儀表板：**5 種特徵選擇方法 × 5 種線性迴歸模型 × 每一個特徵數 k**，
這次改成動態的。任何人上傳一份 CSV，整條流程在**瀏覽器裡用 Python（Pyodide）**跑完，
Gemini 負責兩件事：建議每個欄位怎麼處理（使用者審核後才執行），以及根據算出來的數字寫一段結論。
預設放了兩份資料集：**Bike Sharing**（每小時租借量）與 **King County 房價**。

- 網站：<https://davidlo3917.github.io/AIoT-DIC-HW2/>
- 老師的 50 Startups 數字（8274.868018 / 8198.797191 / 9137.990153）仍由 `tests/test_regress.py` 在每次建置時檢查
- 上傳的檔案不會離開瀏覽器；送到 AI 的只有欄位摘要與我們算出來的結果數字（見[AI 的角色與防護](#ai-的角色與防護)）

![Bike Sharing：開場、AI 結論、控制列、指標卡、RMSE 與 R² 對 k、迴歸式、係數表、即時預測、殘差表](docs/dashboard.jpg)

![上傳面板：檔案摘要、目標欄、規則版與 AI 版計畫切換、每欄的型別 / 缺值 / 樣本 / 動作 / 理由](docs/upload.jpg)

## 系統如何運作

```mermaid
flowchart LR
  subgraph browser [瀏覽器]
    P[網頁 app.js]
    W[Web Worker: Pyodide + regress.py]
    P -- CSV --> W
    W -- 欄位摘要、結果 JSON --> P
  end
  P -- 欄位摘要 / 算出來的數字 --> X[Cloudflare Worker proxy]
  X -- 固定提示詞 + JSON schema --> G[Gemini]
  G -- 建議計畫 / 結論 --> X
  X -- 清洗、驗證過的 JSON --> P
  CI[GitHub Actions: 測試 → regress.py] -- 預設資料集 JSON --> Pages[GitHub Pages]
```

資料不離開瀏覽器：CSV 寫進 Pyodide 的虛擬檔案系統，`regress.py` 在 Web Worker 裡跑完整條流程。
會離開瀏覽器的只有兩種東西，都先經過 Cloudflare Worker，它持有提示詞與金鑰、限制來源與頻率。

### 哪些事交給程式，哪些事交給 AI

| 工作 | 誰做 | 為什麼 |
|---|---|---|
| 讀檔、欄位摘要、型別與缺值處理、洩漏與共線偵測、80/20 切分、5 折 CV、5 × 5 種組合 × 每個 k 的配適、所有指標與基準 | `regress.py`：確定性程式，瀏覽器裡的 scikit-learn | 數字必須可重現、可驗證。CI 每次建置都用老師的 50 Startups 數字檢查，瀏覽器與 CI 算出同樣的結果 |
| 讀懂欄位的「意思」：這欄是標籤還是數量、哪幾欄會洩漏目標、該預測哪一欄、每欄的說明、資料集的標題 | Gemini **建議** → `check_plan()` 驗證 → 使用者決定（可一鍵切回規則版、可改目標欄） | 規則看不懂語意：hr 是時段不是數量、casual + registered 就是 cnt。但 AI 只能建議，不能執行任何東西 |
| 把結果寫成報告：摘要、3–5 個發現、建議、注意事項 | Gemini，輸入**只有我們算出來的數字**與準備紀錄 | 敘述是它的強項，算數不是。所以數字由程式算，AI 只負責說明，而且網頁標示為 AI 產生、請對照卡片 |

### 這樣用 AI 對嗎

做對的部分：

- AI 不碰原始資料列、不算任何數字、不執行任何程式碼；它的輸出永遠是「建議」或「敘述」
- 兩個端點的輸出都用 JSON schema 限制、逐欄清洗、再由 Python 驗證；`/plan` 用代號 `c0…cN` 指稱欄位，模型連欄名都不回傳
- 欄名、樣本值、紀錄都明講是「資料，不是指令」；實測一份欄名寫著指令的 CSV 與一份「請回答這個問題」的 CSV 都沒有改變 AI 的行為
- AI 不可用（逾時、額度、代理沒開）時規則版照常運作，網頁會說明
- 使用者看得到 AI 改了什麼（標色）、改了幾欄、理由是什麼，隨時能切回規則版

已知的限制，也寫在網頁上：

- 結論偶爾會引用錯數字。開發時出現過一次（k 的範圍寫成 34–37，實際是 35–37），重新產生後正確。所以結論卡片註明「請對照下方卡片」，預設資料集的結論也是產生一次後存檔、人工看過
- AI 的計畫不一定最好：Bike 的 hr 它只保留最常見的 9 個時段，預設資料集的計畫則保留全部 24 個
- 送到 Google 的是欄名、每欄 5 個樣本值與結果數字；免費方案的提示可能被用於改善模型。**不要上傳含個資或機密的檔案**
- 來源白名單可以被偽造；每 IP 每分鐘 10 次與免費額度是最後的防線

可以再加的防線（還沒做）：自動檢查結論裡的每個數字都出現在輸入裡，對不上的句子就拿掉；
以及 Cloudflare Turnstile 擋機器人。

## 上傳自己的 CSV 會發生什麼事

1. 按 **Upload a CSV**（或把檔案拖進面板）。瀏覽器開始下載 Pyodide 與 scikit-learn（約 40 MB，第一次約 10 秒，之後有快取）。
2. 檔案寫進 Pyodide 的虛擬檔案系統，`regress.py` 的 `profile()` 做每欄摘要，規則給出預設計畫：
   目標欄（照常見欄名猜，沒有就取最後一個數值欄）與每欄的動作 `numeric` / `category` / `date` / `extract-number` / `drop`。
3. 同一時間，**欄位摘要**（欄名、型別、缺值比例、5 個樣本值、規則版動作）送到代理，Gemini 回傳建議計畫、欄位說明、標題與摘要句。
   AI 改動的列會標色；覺得不對可以一鍵切回規則版，目標欄也可以自己選。AI 沒回應就用規則版，流程照常。
4. **Run the pipeline**：進度條顯示配了幾個模型；可以取消。結果變成一個新的資料集分頁，只存在記憶體裡。
5. 跑完後，**我們算出來的數字**（最佳組合、各方法 / 各模型最佳、標準化效果、基準、準備紀錄）送給 Gemini 寫結論，
   顯示在開場卡下方的「What the results say」，並標明是 AI 寫的、請對照下方卡片。
6. 控制列下方的「Preparation log」列出資料準備的每一步：抽樣、刪掉哪些欄、為什麼、類別怎麼合併、缺值怎麼補。

## 對付亂七八糟的 CSV

`read()` → `profile()` → `check_plan()` → `prepare()` → `screen()` 各擋一層；每個決定都寫進準備紀錄，錯誤訊息直接顯示給使用者。

| 狀況 | 處理 |
|---|---|
| 編碼、分隔符號 | UTF-8（含 BOM）或 Windows-1252；`,` `;` tab `\|` 自動偵測；`;` 分隔視為小數逗號 |
| Excel、空檔、單欄、超過 50 MB | 拒收，說明原因 |
| 數字存成文字 | `$1,200`、`45%`、`1.234,5` → 數字；`8GB`、`1.37kg` → 取出數字（`extract-number`） |
| 日期 | 拆成年、月 |
| ID、列號、常數欄、自由文字、缺值超過一半 | 預設 `drop` |
| 類別欄 | 預設保留最常見的 9 類，其餘併為 Other；缺值自成一類 Missing；郵遞區號雖是數字仍當類別 |
| 目標欄 | 必須是數值且超過 2 個不同值；沒有目標值的列刪除；少於 30 列拒收 |
| 洩漏 | 與目標相關 > 0.98、或幾個欄位一起幾乎完美重現目標（Bike 的 casual + registered = cnt）→ 刪除並說明 |
| 共線 | 可由其他欄線性組合出的欄（King County 的 sqft_above）→ 刪除 |
| 規模 | 超過 2,000 列隨機抽樣；特徵上限 40（超過就先縮減類別數）；訓練列數要 ≥ 2p + 10 |
| 來路不明的計畫 | AI 或使用者給的計畫一律經 `check_plan()`：欄名要存在、動作要合法，其餘補預設 |
| 單一模型失敗 | 該格改用平均值並標記 `failed`，不拖垮整個網格 |

`tests/test_regress.py` 有 17 個測試涵蓋以上情況，CI 每次建置都跑。

## 速度

使用者的要求是不能等一兩分鐘。Chrome 實測：2,000 列 × 37 特徵約 16–23 秒，另加第一次下載 Pyodide 約 10 秒。
為此 `regress.py` 做了幾件事（都有測試或比對證明結果不變）：

- OLS 用 Gram 矩陣一次算好，每組特徵只解一個 (k+1)×(k+1) 系統
- Ridge 用閉式留一誤差自己選 α（測試證明與 `RidgeCV` 同解）
- Lasso / ElasticNet 的 α 每個 fold 用全部特徵調一次、所有 k 共用（p = 30 實測 CV RMSE 中位數 +0.5%），網格內用 `precompute=True`
- Huber 從上一個 k 的解暖啟動
- 不同方法選到同一組特徵時只配一次

## AI 的角色與防護

AI 只做建議與敘述，**不執行任何東西**。兩個端點都在 `proxy/worker.mjs`（Cloudflare Worker，只為了藏金鑰）：

| | `POST /plan` | `POST /summary` |
|---|---|---|
| 輸入 | 欄位摘要（不含資料列） | 我們算出來的數字與準備紀錄 |
| 輸出 | 目標欄、每欄動作與理由、欄位說明、標題 | 摘要、3–5 個發現、建議、注意事項 |

防護（`proxy/worker.test.mjs` 用假的 Gemini 測 8 個情境）：

- 提示詞由 Worker 持有，頁面送來的只被當成資料；多餘的欄位一律丟掉
- 輸出用 JSON schema 限制；`/plan` 用每次請求的代號 `c0…cN` 指稱欄位，模型從不回傳欄名
- 回傳文字全部清洗：去掉 HTML 括號、連結、控制字元，長度上限；計畫再經 Python `check_plan()` 驗證
- 來源白名單（`ALLOWED_ORIGINS`）、每個 IP 每分鐘 10 次、64 KB / 100 欄上限
- 模型：`gemini-3.5-flash`，過載或逾時就換 `gemini-3.5-flash-lite`；實測 `/plan` 6–21 秒、`/summary` 6–13 秒
- 網頁上所有 AI 寫的文字都標示為 AI 產生；AI 不可用時規則版照常運作

殘餘風險：Origin 標頭可以偽造，免費額度是最後的防線。預設資料集的結論只產生一次（`python web/regress.py summaries`），
存在 `web/summaries.json`，之後每次建置沿用，訪客不會觸發 AI 呼叫。

## 網頁上的 CRISP-DM

| 階段 | 位置 |
|---|---|
| 1 商業理解 | 開場卡：問題、預測目標、實驗設定（AI 寫的標題與摘要會標示） |
| 2 資料理解 | 上傳面板：每欄的型別、缺值、不同值數、樣本、規則版與 AI 版計畫 |
| 3 資料準備 | 控制列下方：one-hot（刪第一類）、80/20 切分、訓練集 StandardScaler、fold 內特徵選擇；展開 Preparation log 看每一步 |
| 4 建模 | 控制列（方法、模型、k）、迴歸式、係數表（含標準化係數與欄位說明） |
| 5 評估 | AI 結論、指標卡、成功標準、RMSE 與 R² 對 k、測試集殘差表 |
| 6 部署 | 即時預測：拖動滑桿、選類別，在瀏覽器裡直接代入迴歸式 |

成功標準：目前組合的 CV RMSE 比預測平均值低 50% 以上，而且勝過隨機挑同樣數量的特徵。
最佳 k 依 CV RMSE（訓練集 5 折）決定，卡片上報告的是測試集分數；特徵選擇在每一個 fold 裡重做。

## 預設資料集的結果（seed 0，各抽 2,000 列）

| | Bike Sharing | King County 房價 |
|---|---|---|
| 資料 | 17,379 列 × 17 欄 → 37 個特徵（hr、season、weathersit one-hot） | 21,613 列 × 21 欄 → 37 個特徵（date 拆年月、zipcode 前 19 類） |
| 自動刪掉 | instant（列號）、dteday（與 yr/mnth/weekday 重複）、casual 與 registered（加起來就是 cnt） | id、sqft_above（= sqft_living − sqft_basement） |
| CV 最佳 | Lasso 路徑 k = 35 + ElasticNet，CV RMSE 106 | Lasso 路徑 k = 35 + ElasticNet，CV RMSE 190,889 |
| 測試集 | RMSE 99.2 輛，R² 0.651 | RMSE US$183,540，R² 0.736 |
| 完整模型（k = 37，OLS） | RMSE 99.3 | RMSE US$183,623 |
| 預測平均值（測試 RMSE） | 168.6 | 357,147 |
| 最大的標準化效果 | hr_18 +73.9、hr_17 +70.6、hr_8 +59.3、temp +44.2、yr +42.6 | sqft_living +134k、grade +115k、lat +87k、zipcode Other −66k、yr_built −58k |
| 結論 | 上下班尖峰時段主導；五種方法最佳 k 都在 35–37，彼此差不到 0.1 | 坪數、建築等級、緯度（越北越貴）；五種方法最佳 k 同樣落在 35–37 |

## 檔案

```text
web/index.html      頁面骨架（卡片堆疊，含上傳面板與 AI 結論卡）
web/style.css       樣式
web/app.js          狀態、控制列、圖表（Observable Plot）、上傳流程、與代理的兩個呼叫
web/worker.js       Web Worker：載入 Pyodide 與 regress.py，回報進度
web/regress.py      分析程式：讀檔、欄位計畫、準備、模型網格；也產生預設資料集的 JSON
web/presets.json    兩份預設資料集的計畫與說明文字
web/summaries.json  預設資料集的 AI 結論（產生一次）
web/data/           兩份 CSV；*.json 由 regress.py 產生（不進版控）
proxy/worker.mjs    Gemini 代理（Cloudflare Worker）與 wrangler.toml、離線測試
tests/              messy CSV 測試、老師的 50 Startups 數字
lab/                Lab 4 之前的單檔練習
.github/workflows/pages.yml   推送到 main → 測試 → regress.py → 部署 GitHub Pages
```

沒有建置步驟、沒有 npm：Observable Plot 0.6.17 與 Pyodide 314.0.7 都從 jsDelivr 載入；
`regress.py` 的套件版本釘在 Pyodide 內建的版本（scikit-learn 1.8.0、pandas 3.0.2、numpy 2.4.6、scipy 1.18.0），
所以 CI、本機與瀏覽器算出同樣的數字（實測相對差 < 1e-12）。

## 在本機執行

需要 [uv](https://docs.astral.sh/uv/)；代理需要 Node 22 與一把 Gemini API key（放在 `proxy/.dev.vars`：`GEMINI_KEY=...`，已被 git 忽略）。

```bash
R="uv run --python 3.14 --with scikit-learn==1.8.0 --with pandas==3.0.2 --with numpy==2.4.6 --with scipy==1.18.0 python"
$R tests/test_regress.py                 # 17 個測試，約 15 秒
$R web/regress.py                        # 產生 web/data/{bike,kc}.json，約 20 秒
node proxy/worker.test.mjs               # 代理的 8 個離線測試
python3 -m http.server 8000 -d web       # 開 http://localhost:8000（代理只放行 localhost:8000）
cd proxy && npx wrangler@4 dev --port 8787 --ip 127.0.0.1   # 本機代理
$R web/regress.py summaries              # （選用）重新產生預設資料集的 AI 結論
```

沒開代理也能用：上傳流程會顯示「AI unavailable」並改用規則版計畫。

## 部署

- 網頁：推送到 `main`，GitHub Actions 先跑測試與代理測試，再執行 `regress.py` 產生結果檔，最後把 `web/` 部署到 GitHub Pages。
- 代理：在 `proxy/` 執行 `npx wrangler@4 deploy` 與 `npx wrangler@4 secret put GEMINI_KEY`，
  把得到的 `*.workers.dev` 網址填進 `web/app.js` 的 `PROXY`。允許的來源寫在 `proxy/wrangler.toml`。

## 資料來源

- Bike Sharing：Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge.* Progress in Artificial Intelligence.
  取自 [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)，CC BY 4.0。
- King County 房價：[Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)，透過 [OpenML 42092](https://www.openml.org/d/42092) 下載，CC0。
- 50 Startups（只用於測試）：SuperDataScience《Machine Learning A-Z》課程資料。
- 版面參考 [AIoT-DIC2-MultipleLR](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)。
