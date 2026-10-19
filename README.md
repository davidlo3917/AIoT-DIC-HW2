# AIIS Lab 4 — Linear regression lab (CRISP-DM)

A one-page lab dashboard that compares **5 feature-selection methods × 5 linear regression models × every
feature count k** on **50 Startups** and **Boston Housing**, with an 80/20 train/test split. Pick a method
(or compare all five), a model and a k, and every card follows.

- Site: <https://davidlo3917.github.io/AIoT-DIC-L4/>
- Layout after [AIoT-DIC2-MultipleLR](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/); the numbers
  for F-test + OLS at k = 2 match that page (test RMSE 8,198.80, R² 0.9474).
- Last week's single-file exercise (backward elimination, reproducing the teacher's table) lives in [`lab/`](lab/)

![Dashboard: hero, controls, metric cards, RMSE and R² by k, equation, coefficients, live predictor, residuals](docs/dashboard.jpg)

## What is compared

| | Items |
|---|---|
| Feature selection | F-test (`f_regression`), mutual information, RFE, backward elimination (by CV R²), Lasso path (order of entry) |
| Models | OLS, Ridge, Lasso, ElasticNet, Huber (α for Ridge / Lasso / ElasticNet tuned by CV on the training set) |
| Feature count | every k = 1 … p; the page marks each method's best k by CV RMSE and the full model at k = p |
| Benchmarks | predict the training mean; random k-feature subsets with OLS; the teacher's table (asserted in CI) |

**The best k is chosen by CV RMSE (5-fold on the training set); the cards report the test score.**
50 Startups has only 10 test rows, so choosing by test RMSE would use the test set for model selection
and give an optimistic number. Feature selection is redone inside every CV fold, so CV never sees
validation rows.

## CRISP-DM on the page

| Phase | Where |
|---|---|
| 1 Business understanding | Hero: the question, the target, the setup (Boston: the ethics note on the B column) |
| 2 Data understanding | Hero pills: rows, features, target, best combination, full-model score |
| 3 Data preparation | One line under the controls: one-hot (drop first), 80/20 split with seed 0, StandardScaler on train, selection inside CV folds |
| 4 Modeling | Controls (method, model, k), fitted equation, coefficient table with standardised estimates |
| 5 Evaluation | Metric cards, success criterion, RMSE and R² by k, test-set residuals |
| 6 Deployment | Live predictor: sliders feed the fitted equation in the browser |

Success criterion: the active combination's CV RMSE is at least 50% below the mean baseline and beats
random feature subsets of the same size.

## Results (random_state = 0)

| | 50 Startups | Boston Housing |
|---|---|---|
| Best by CV | F-test k = 2 + Huber, CV RMSE 9,304, test R² 0.947 | RFE k = 11 + OLS, CV RMSE 4.61, test R² 0.590 |
| Best by test | F-test k = 2 + Lasso, test RMSE 8,199 | Lasso path k = 12 + OLS, test RMSE 5.78 |
| Mean baseline, test RMSE | 37,994 | 9.03 |
| Takeaway | R&D Spend + Marketing Spend is all it takes; all five methods pick the same features at every k, so "compare all" overlaps. State carries no signal. | Methods disagree; LSTAT and RM are the top two under every method. The textbook R² of 0.74 is in-sample on all rows; the held-out test R² is 0.59. |

Rows 1, 2 and "all" of the teacher's table (8274.868018 / 8198.797191 / 9137.990153) are asserted by
`regress.py` on every build.

## Files

```text
web/index.html   page skeleton (card stack)
web/style.css    styles
web/app.js       state, controls, charts (Observable Plot)
web/regress.py   the analysis; writes web/data/*.json and runs the asserts
web/data/        the two CSVs; *.json is generated and not committed
lab/             last week's single-file exercise
.github/workflows/pages.yml   push to main → run regress.py → deploy GitHub Pages
```

No build step, no npm. Observable Plot 0.6.17 is loaded from jsDelivr.

## Run locally

Needs [uv](https://docs.astral.sh/uv/). Generate the results, then serve `web/`:

```bash
uv run --python 3.14 --with scikit-learn==1.8.0 --with pandas==3.0.2 --with numpy==2.4.6 --with scipy==1.18.0 python web/regress.py
python3 -m http.server 8000 -d web   # open http://localhost:8000
```

## Deploy

Every push to `main` runs `regress.py` in GitHub Actions (a failed assert blocks the deploy) and publishes
`web/` to GitHub Pages.

## Data

- 50 Startups: SuperDataScience "Machine Learning A-Z" course data (the same file as `lab/`)
- Boston Housing: Harrison & Rubinfeld (1978), CSV from [selva86/datasets](https://github.com/selva86/datasets).
  scikit-learn removed this dataset in 1.2 because the B column is derived from the share of Black residents.
  It is kept here only to compare with textbook results, and the page says so.
