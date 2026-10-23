# Demand Forecasting: Foundation Models vs Classical Baselines

**Question:** across many series, when does a time-series foundation model (Chronos) beat classical and ML methods, and is it worth the compute?

**Data:** M5 (Walmart) daily unit sales. Current experiments use a **500-series subset** (stratified by store), 2011-01-29 to 2016-05-22.

**Environment:** Python 3.11.2, CPU only (no GPU). See [requirements.txt](requirements.txt).

---

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Setup, M5 data in `data/raw/` | done |
| 1 | Data prep: long format, hierarchy, demand segments | done |
| 2 | Rolling-origin backtest harness | done |
| 3 | AutoETS, AutoARIMA, LightGBM | done |
| 4 | N-HiTS, PatchTST | done |
| 5 | Chronos zero-shot and fine-tuned | done (calibrated and raw, all 5 folds) |
| 6 | Hierarchical reconciliation (bottom-up, MinTrace) | done for AutoETS, Chronos zero-shot, N-HiTS (OLS and structural WLS; no full-covariance MinT) |
| 7 | Conformal prediction intervals | done (90%, folds 2-5, in `results/intervals.csv`) |
| 8 | Evaluation (WAPE, RMSSE, coverage, slices) | done: `src/evaluate.py` |
| 9 | Streamlit dashboard | done: `streamlit run src/dashboard.py` |
| 10 | Write-up and decision guide | done: [docs/decision_guide.md](docs/decision_guide.md) |

See [PLAN.md](PLAN.md) for the full plan.

---

## Project layout

```
data/raw/                     M5 files (calendar, sales_train_*, sell_prices)
data/long.parquet             unique_id, ds, y, sell_price, snap
data/series_meta.csv          hierarchy columns + ADI, CV2, segment
src/prepare_data.py           Phase 1
src/backtest.py               Phase 2 harness + model registry
src/models_classical.py       AutoETS, AutoARIMA, LightGBM
src/models_deep.py            N-HiTS, PatchTST
src/models_foundation.py      Chronos zero-shot / fine-tuned (with volume calibration)
src/evaluate.py               Phase 8: WAPE, RMSE, bias, RMSSE, WRMSSE by model / segment / horizon / store
src/reconcile.py              Phase 6: bottom-up, MinTrace-OLS, MinTrace-WLS-structural
src/conformal.py              Phase 7: split-conformal 90% intervals
src/dashboard.py              Phase 9: Streamlit dashboard
src/significance.py           paired bootstrap over series (WAPE differences with 95% CI)
results/forecasts.csv         unique_id, cutoff, ds, model, y, yhat, lo, hi
results/runtime.csv           model, cutoff, seconds
results/metrics_*.csv         evaluate.py / reconcile.py / conformal.py outputs (overall, segment, horizon, store, hierarchy, intervals)
results/agg_forecasts.csv     base forecasts for the aggregate nodes (Phase 6)
results/reconciled.csv        reconciled bottom-level forecasts, model = <base>+<method>
results/intervals.csv         point forecasts with conformal lo/hi (folds 2-5)
results/*.backup.csv          copies taken before the Chronos recalibration
docs/                         flow diagrams and model explainers
```

## How to run

Not in git (see `.gitignore`): `data/raw/` (download M5 from Kaggle), `data/long.parquet`, and the large result CSVs (`forecasts*.csv`, `intervals.csv`, `reconciled.csv`). Regenerate them with the commands below. Set `DF_RUN=n3000` to write a separate run to `data/n3000` and `results/n3000`.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/prepare_data.py --n 500          # --n 0 for all 30,490 series
python src/backtest.py                       # all models, 5 folds, horizon 28
python src/backtest.py --models AutoETS Chronos_zeroshot --folds 3

$env:PYTHONPATH="src"; python src/reconcile.py   # adds agg_forecasts.csv, reconciled.csv, metrics_hierarchy.csv
python src/evaluate.py                       # metrics_overall/segment/horizon/store.csv
python src/conformal.py                      # intervals.csv, metrics_intervals.csv
streamlit run src/dashboard.py
```

The harness resumes: any (model, cutoff) already in `results/forecasts.csv` is skipped. If you change `--folds` or `--horizon`, delete `results/` first, because the resume check matches only model and cutoff.

---

## What each phase does

### Phase 1: data ([src/prepare_data.py](src/prepare_data.py))
- Samples series proportionally within each store, so the hierarchy stays balanced.
- Melts wide sales into long format and attaches price and SNAP flag.
- Drops the days before an item is first stocked (price is missing there).
- Computes ADI and CV² per series and labels it **smooth, erratic, intermittent or lumpy** (thresholds 1.32 and 0.49).
- 500-series subset: 373 intermittent, 82 lumpy, 29 smooth, 16 erratic. About 90% of the series are intermittent or lumpy, so smooth and erratic results are noisy.

### Phase 2: harness ([src/backtest.py](src/backtest.py))
- 5 rolling cutoffs, horizon 28 days, step 28 days. The last test window ends at the final date. Cutoffs: 2016-01-03, 01-31, 02-28, 03-27, 04-24.
- Every model is a function `fit_predict(train, future, h)` registered in `MODELS`.
- Leak guard: asserts that train stops at the cutoff and that the test window is exactly 28 days.
- Output rows are in the standard schema above. Flow diagram: [docs/phase2_harness_flow.png](docs/phase2_harness_flow.png).

### Phase 3: classical and ML baselines ([src/models_classical.py](src/models_classical.py))
- **AutoETS, AutoARIMA** (statsforecast): weekly seasonality, trained on the last ~2 years per series, forecasts clipped at 0. Explainer: [docs/AutoETS_AutoARIMA_explained.md](docs/AutoETS_AutoARIMA_explained.md). Diagram: [docs/phase3_plan.png](docs/phase3_plan.png).
- **LightGBM:** one global model, Tweedie loss, direct 28-day forecast using lags of at least 28 days (28/35/42/56), rolling stats, calendar, price, price change, SNAP, store and item. No recursion, so no leakage.

### Phase 4: deep models ([src/models_deep.py](src/models_deep.py))
- **N-HiTS, PatchTST** (neuralforecast): global models, input window 112 days, MSE loss (mean forecast), 400 steps, CPU, last ~2 years of data.
- MSE was used instead of a quantile loss because a median forecast is 0 for most zero-heavy series. Intervals come from conformal prediction in Phase 7.

### Phase 5: foundation model ([src/models_foundation.py](src/models_foundation.py))
- **Chronos-Bolt small** (`amazon/chronos-bolt-small`), CPU.
- **Zero-shot:** pretrained model, last 512 days as context, no training.
- **Fine-tuned:** 300 AdamW steps on windows whose targets end on or before the cutoff.
- **Point forecast:** mean of the 9 predicted quantiles (0.1 to 0.9). This misses the upper tail of zero-heavy demand and under-forecasts total volume by about 15%.
- **Volume calibration (added later):** each forecast is multiplied by actual ÷ forecast, measured on the last two 28-day windows of the training data (no leakage). The fine-tuned variant is trained without those two windows so its correction is also out-of-sample.
- The uncalibrated runs are kept in the results as `Chronos_zeroshot_raw` and `Chronos_finetuned_raw`.

---

## Results so far (500 series, 5 folds, horizon 28)

| Model | WAPE | RMSE | Bias | Total runtime |
|---|---|---|---|---|
| Chronos_finetuned_raw (uncalibrated) | 0.689 | 2.020 | -16.2% | 1,517 s |
| Chronos_zeroshot_raw (uncalibrated) | 0.699 | 2.069 | -15.4% | 34 s |
| Chronos_finetuned (calibrated) | 0.713 | 2.035 | -0.7% | 1,607 s |
| Chronos_zeroshot (calibrated) | 0.725 | 2.092 | -1.0% | 91 s |
| AutoETS | 0.730 | 2.143 | -3.1% | 104 s |
| N-HiTS | 0.736 | 2.079 | -2.9% | 176 s |
| AutoARIMA | 0.743 | 2.144 | -3.0% | 2,406 s |
| LightGBM | 0.747 | 2.109 | -5.8% | 24 s |
| PatchTST | 0.753 | 2.173 | -0.7% | 1,173 s |
| SeasonalNaive | 0.867 | 2.571 | -4.6% | 1 s |

Table is `results/metrics_overall.csv`, which also has RMSSE and revenue-weighted WRMSSE (all models sit between 0.70 and 0.72 RMSSE except SeasonalNaive at 0.93).

WAPE by segment, calibrated zero-shot Chronos vs AutoETS:

| Segment | AutoETS | Chronos_zeroshot | Chronos_finetuned | Chronos_zeroshot_raw |
|---|---|---|---|---|
| smooth | 0.423 | 0.423 | 0.412 | 0.401 |
| erratic | 0.637 | 0.661 | 0.642 | 0.626 |
| intermittent | 0.941 | 0.920 | 0.917 | 0.896 |
| lumpy | 0.859 | 0.862 | 0.838 | 0.833 |

### Findings so far
1. Every model beats SeasonalNaive by 0.11 to 0.18 WAPE.
2. AutoETS, N-HiTS and calibrated zero-shot Chronos are statistically tied (paired bootstrap over series, `results/significance.csv`). AutoARIMA, LightGBM and PatchTST are significantly worse than AutoETS, but by only 0.013 to 0.024 WAPE. Single seed per model.
3. AutoETS is about 23 times faster than AutoARIMA and no worse. PatchTST costs about 11 times more compute than N-HiTS for a worse score.
4. Uncalibrated Chronos looks clearly best, but part of that is bias: WAPE rewards forecasts that are too low on zero-heavy data. After correcting volume (bias -15% to -1%), its WAPE moves from 0.699 to 0.725 and its lead over AutoETS shrinks to noise level.
5. Calibrated zero-shot Chronos is still competitive with the best models at a fraction of the cost, and is clearly ahead only on intermittent demand (0.920 vs 0.941). It also has a lower RMSE than AutoETS, AutoARIMA and LightGBM.
6. Fine-tuning gained only 0.010 to 0.012 WAPE over zero-shot (calibrated 0.713 vs 0.725, uncalibrated 0.689 vs 0.699) for about 18 to 45 times the compute. Calibrated fine-tuned Chronos has the best RMSE (2.035) and the best calibrated WAPE. Its gain over AutoETS (-0.016, 95% CI -0.028 to -0.007) and over calibrated zero-shot (-0.012, CI -0.017 to -0.006) is statistically real, but small.
7. Intermittent and lumpy series stay hard for every model (WAPE about 0.8 to 0.97).
8. Reconciliation changes bottom-level WAPE by at most 0.003, which is noise. It helps the upper levels where it uses information from other levels: structural-WLS gives N-HiTS store WAPE 0.176 vs 0.183 base and Chronos zero-shot total 0.088 vs 0.096. Bottom-up is worse than base at the top for N-HiTS (total 0.104 vs 0.080), so use WLS or OLS if coherence is needed.
9. Conformal 90% intervals reach 0.89 to 0.90 coverage for every model, so interval width is the comparison: raw fine-tuned Chronos is narrowest (3.47), LightGBM widest among the non-naive models (3.73). SeasonalNaive over-covers (0.95) with the widest intervals (4.16).

### Caveats
- 500 series, a single training seed per model, no significance tests.
- Smooth and erratic segments hold only 45 series, so their numbers are noisy.
- WAPE and RMSE can rank models differently on intermittent data. Report both, plus bias.
- Chronos is on CPU with a small model. Larger Chronos or TimesFM variants were not tried.
- Reconciliation uses OLS and structural-WLS weights, not covariance-based MinT (no in-sample residuals are stored). Only 3 models were reconciled, and aggregate-node base forecasts come from running the same model on the aggregated series (N-HiTS trained on the aggregate nodes only).
- Conformal calibration uses at most 3 earlier folds, pooled across series within forecast-size bins. Fold 1 has no interval. Reconciled forecasts are clipped at 0.
- RMSSE scale uses the full history before each cutoff. The hierarchy is only the 500-series sample, not the full M5 hierarchy, so WRMSSE here is not comparable to published M5 numbers.

---

## Next steps
1. Rerun on a larger subset (for example 3,000 series), skipping AutoARIMA or running it overnight. Fine-tuned Chronos is about 5 min per fold at 500 series, so budget accordingly.
2. Several training seeds per model: the bootstrap only covers series sampling, not training randomness.
3. Try a larger Chronos variant, or run fine-tuning on a GPU.
