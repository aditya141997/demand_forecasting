# Decision guide: which model to use

Evidence: M5 subset, 500 series, 5 rolling folds, horizon 28 days, CPU only. Numbers are from `results/metrics_*.csv` and
`results/significance.csv` (paired bootstrap over series, 2,000 resamples, 95% interval).

## Recommendation

| Situation | Use | Why |
|---|---|---|
| Default for a large catalogue, no training budget | **Chronos-Bolt small, zero-shot, volume-calibrated** | WAPE 0.725, bias -1.0%, about 18 s per fold. Statistically tied with AutoETS, significantly better than N-HiTS, AutoARIMA, LightGBM and PatchTST. Best of the cheap models on intermittent series (0.920). |
| Accuracy matters most and you can pay for it | **Chronos-Bolt small, fine-tuned (300 steps), volume-calibrated** | Best valid score: WAPE 0.713, RMSE 2.035, bias -0.7%. Beats AutoETS by 0.016 WAPE (CI -0.028 to -0.007) and zero-shot by 0.012 (CI -0.017 to -0.006). Costs about 5 min per fold on CPU (about 18x zero-shot). |
| No foundation model available, need a classical baseline | **AutoETS** | WAPE 0.730 at about 21 s per fold. Tied with zero-shot Chronos. About 23x cheaper than AutoARIMA, which is significantly worse. |
| Need coherent forecasts across store, state and total | Any of the above plus **structural-WLS reconciliation** | Improves the upper levels (for example N-HiTS store WAPE 0.183 to 0.176, Chronos zero-shot total 0.096 to 0.088) and leaves the bottom level unchanged within 0.003. Avoid plain bottom-up at the top level. |
| Need uncertainty | Split-conformal 90% intervals on any model | Coverage 0.89 to 0.90 on folds 2 to 5 for every model. Narrowest intervals: fine-tuned Chronos. |

**Single best choice: calibrated fine-tuned Chronos if the compute is available, otherwise calibrated zero-shot Chronos.**
Do not use PatchTST (worst non-naive model, 11x the compute of N-HiTS) or AutoARIMA (slowest, no better than AutoETS).

## Always calibrate volume
Chronos-Bolt's quantile-mean point forecast under-forecasts volume by about 15% on zero-heavy demand. The uncalibrated
variants look best on WAPE (0.689 and 0.699) only because WAPE rewards low forecasts on such data. Do not compare those
numbers with the other models. After calibration the lead mostly disappears, so judge every model on WAPE, RMSE **and** bias.

## By series type
| Segment (share of sample) | Best WAPE (calibrated) | Notes |
|---|---|---|
| Intermittent (75%) | Chronos fine-tuned 0.917, zero-shot 0.920 | AutoETS 0.941, LightGBM 0.975. Chronos is the clear winner here. |
| Lumpy (16%) | Chronos fine-tuned 0.838 | Zero-shot Chronos (0.862) is level with AutoETS (0.859). |
| Smooth (6%) | Chronos fine-tuned 0.412, LightGBM 0.412, N-HiTS 0.412 | All close. Only 29 series, so treat as noisy. |
| Erratic (3%) | AutoARIMA 0.635, N-HiTS 0.634, AutoETS 0.637 | Chronos (0.642 to 0.661) is not ahead. 16 series, so treat as noisy. |

## Limits of this evidence
- 500 series, one training seed per model: the bootstrap covers series sampling, not training randomness.
- Smooth and erratic segments have 45 series between them.
- CPU only, small Chronos only. A GPU would cut fine-tuning cost sharply and a larger Chronos or TimesFM might change the ranking.
- Reconciliation used OLS and structural-WLS weights on 3 models. Full-covariance MinT was not run.
- Rerun on a larger subset (for example 3,000 series) before committing to a production choice.
