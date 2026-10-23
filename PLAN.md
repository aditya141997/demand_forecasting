# Demand Forecasting: Foundation Models vs Classical Baselines

**Question:** across thousands of series, when does a foundation model (Chronos/TimesFM) actually beat ARIMA/ETS/LightGBM, and is it worth the compute?

## Stack
`statsforecast`, `lightgbm`, `neuralforecast` (N-HiTS/PatchTST), `chronos-forecasting` (or `timesfm`), `hierarchicalforecast`, `MAPIE`/custom split-conformal, `streamlit`.

## Phases

| # | Phase | What you do | Output |
|---|-------|-------------|--------|
| 0 | Setup | Python **3.11.2**: `py -3.11 -m venv .venv`, activate, `pip install -r requirements.txt`; download M5 from Kaggle into `data/raw/` | env ready |
| 1 | Data | Melt M5 to long format `unique_id, ds, y`; keep 3 states / 10 stores / ~3k items (subset first, e.g. 500 series, then scale). Build hierarchy columns (state > store > dept > item). Tag each series as smooth / erratic / intermittent / lumpy (ADI & CV²). | `data/long.parquet`, `data/series_meta.csv` |
| 2 | Backtest harness | Rolling-origin CV: 4-5 cutoffs, horizon 28 days, step 28. One function `run_model(train, h) -> yhat (+ quantiles)` per model so every model plugs into the same loop. | `src/backtest.py` |
| 3 | Baselines | SeasonalNaive, AutoETS, AutoARIMA (statsforecast); LightGBM with lags 7/14/28, rolling means, calendar, price/SNAP features, global model across series. | forecasts per model |
| 4 | Deep | N-HiTS (start here: fast, strong), then PatchTST if time allows. Global model, MQLoss for quantiles. | forecasts |
| 5 | Foundation | Chronos-Bolt/T5 (or TimesFM): **zero-shot** first; then **fine-tune** on training windows only (no leakage past each cutoff). | forecasts |
| 6 | Reconciliation | Bottom-up, MinTrace (`mint_shrink`) via `hierarchicalforecast` on the best 2-3 models. | coherent forecasts |
| 7 | Intervals | Split-conformal: use the previous fold's residuals to calibrate intervals for the next fold (90% target). Report empirical coverage + width. | `lo`, `hi` columns |
| 8 | Evaluation | MAE, RMSE, WAPE, bias, RMSSE/WRMSSE, coverage, interval width, runtime. Slice by series type, hierarchy level, horizon step. | `results/metrics.csv` |
| 9 | Dashboard | `streamlit run dashboard/app.py` | this repo |
| 10 | Write-up | Decision guide: "use X when series are Y, because Z; cost = W". | README |

## Standard results schema (what every model writes)
`results/forecasts.csv`: `unique_id, cutoff, ds, model, y, yhat, lo, hi`
`results/series_meta.csv`: `unique_id, state, store, item, segment`
`results/runtime.csv`: `model, seconds, gpu` (optional)

The dashboard reads only these files, so any model that writes this schema shows up automatically.

## Suggested timeline
- Week 1: phases 0-3 (data + baselines + harness)
- Week 2: phases 4-5 (N-HiTS, Chronos zero-shot/fine-tuned)
- Week 3: phases 6-9 (reconciliation, conformal, dashboard)
- Week 4: slicing analysis and write-up

## Pitfalls to watch
- **Leakage:** fine-tuning and conformal calibration must only use data before each cutoff.
- **Intermittent demand:** MAPE explodes; use WAPE/RMSSE and report by segment. This is where foundation models often struggle.
- **Fair compute:** log runtime per model; a 1% gain at 100x cost is a finding.
- Start on a 500-series subset so every phase runs in minutes.

## Dashboard (`dashboard/`)
1. **Leaderboard:** metric table + bar chart across models, with runtime.
2. **Forecast explorer:** pick series and cutoff; actual vs forecasts with interval band.
3. **When does each model win?** Best model by segment, hierarchy level and horizon.
4. **Intervals:** coverage vs 90% target and width per model.
5. **Backtest folds:** metric stability across cutoffs.

Build it in phase 9 with Streamlit + Plotly, reading only the results files above.
