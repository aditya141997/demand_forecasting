# AutoETS and AutoARIMA explained

Both are **classical, per-series statistical models**. They fit one small model to each time series separately, using only that series' own history. "Auto" means the library searches for the best model form on its own, so you do not choose the settings by hand.

In this project they are the first real baselines: if a foundation model cannot beat them, it is not worth its compute cost.

---

## 1. ETS (Error, Trend, Seasonality)

### Idea
Demand is split into hidden components that evolve over time:

| Component | Meaning | Options |
|---|---|---|
| **Level** | the current "typical" demand | always present |
| **Trend** | whether the level is rising or falling | None (N), Additive (A), Additive damped (Ad) |
| **Seasonality** | repeating pattern (here weekly, 7 days) | None (N), Additive (A), Multiplicative (M) |
| **Error** | how noise combines with the signal | Additive (A), Multiplicative (M) |

Each model is named by three letters, `ETS(Error, Trend, Seasonal)`. Examples: `ETS(A,N,N)` is simple exponential smoothing; `ETS(A,A,A)` is additive Holt-Winters; `ETS(M,Ad,M)` is a damped-trend multiplicative model.

### How it works
The model keeps a running estimate of each component and updates it with every new observation. The simplest version, simple exponential smoothing:

```
level_t = alpha * y_t + (1 - alpha) * level_(t-1)
forecast for every future day = level_t
```

- `alpha` (0 to 1) is how fast the model forgets the past. A high alpha reacts quickly to recent sales; a low alpha averages over a long history.
- Adding a trend adds a second equation with its own parameter `beta`. A **damped** trend adds `phi < 1`, which makes the trend flatten out over the horizon instead of growing forever.
- Adding seasonality adds a third equation with parameter `gamma`, keeping one seasonal factor per day of the week.

### What "Auto" does
`AutoETS` fits all valid combinations of error, trend and seasonality (up to 30 models), estimates each one's parameters by maximum likelihood, and picks the one with the lowest **AICc**. AICc rewards a good fit but penalises extra parameters, so it guards against overfitting.

### Strengths
- Fast, stable and easy to explain ("level plus weekly pattern").
- Works well on smooth series with clear weekly seasonality.
- Gives principled prediction intervals (not used in this project until Phase 7, where we use conformal intervals for every model).

### Weaknesses for this data
- Multiplicative forms need strictly positive data, so they cannot be used on series with zeros. M5 has many zero-heavy series.
- Assumes a smooth, continuous signal. On intermittent demand (many zeros, occasional spikes) it produces a blurred average that is never right on any single day.
- Cannot use outside information such as price, SNAP days or events.

---

## 2. ARIMA (AutoRegressive Integrated Moving Average)

### Idea
Instead of explicit components, ARIMA says today's value is explained by **recent values** and **recent forecast errors**, after making the series stable.

`ARIMA(p, d, q)` has three parts:

| Part | Meaning |
|---|---|
| **AR(p)** AutoRegressive | `y_t` depends on its last `p` values |
| **I(d)** Integrated | difference the series `d` times (`y_t - y_(t-1)`) to remove trend and make it stationary |
| **MA(q)** Moving Average | `y_t` depends on the last `q` forecast errors |

Written out for the stationary (differenced) series `z_t`:

```
z_t = c + phi_1*z_(t-1) + ... + phi_p*z_(t-p)      (AR part)
        + theta_1*e_(t-1) + ... + theta_q*e_(t-q)  (MA part)
        + e_t                                      (new noise)
```

**Stationary** means the series' mean and variance do not drift over time. Differencing is what turns a trending series into a stationary one.

### Seasonal ARIMA
Weekly patterns are handled with a second set of terms at the seasonal lag: `ARIMA(p,d,q)(P,D,Q)[7]`. `P`, `D`, `Q` do the same job as `p`, `d`, `q`, but look back in steps of 7 days (the same weekday last week, two weeks ago, and so on).

### What "Auto" does
`AutoARIMA` follows the Hyndman-Khandakar procedure:
1. Use unit-root tests (such as KPSS) to choose `d` and `D`.
2. Start from a few simple models and **step through neighbours** (`p`, `q`, `P`, `Q` each moved by one), keeping any change that lowers AICc.
3. Stop when no neighbour improves AICc, and refit the winner.

### Strengths
- Flexible: can capture autocorrelation patterns that ETS cannot express.
- Often competitive on smooth, high-volume series.

### Weaknesses for this data
- **Slow.** It runs a search with many fits per series. Over 500 series and 5 folds, with about 1,900 days of history each, it is the slowest model in the project. We limit history to roughly 2 years to keep it manageable.
- Assumes Gaussian, continuous values. Count data with many zeros violates this.
- Forecasts can go negative, so we clip them at zero.
- Like ETS, it ignores price, SNAP and events unless exogenous regressors are added.

---

## 3. Side by side

| | AutoETS | AutoARIMA |
|---|---|---|
| Core idea | smoothed level, trend, season | past values and past errors |
| Handles weekly seasonality | yes, `season_length=7` | yes, seasonal terms at lag 7 |
| Model search | up to 30 combinations | stepwise over p, d, q, P, D, Q |
| Selection criterion | AICc | AICc |
| Speed | fast | slow |
| Interpretability | high (level/trend/season) | medium |
| Needs stationary data | no | yes (it differences for you) |
| Uses price/SNAP | no | no (not in our setup) |

Note: AICc values from ETS and ARIMA cannot be compared with each other, so we compare the two models on **held-out forecast accuracy** (WAPE) instead.

---

## 4. How we use them in this project (Phase 3)

```python
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

sf = StatsForecast(models=[AutoETS(season_length=7), AutoARIMA(season_length=7)],
                   freq="D", n_jobs=-1)
sf.fit(train[["unique_id", "ds", "y"]])
fc = sf.predict(h=28)           # then clip at 0
```

- Input is only `unique_id, ds, y`. Training data stops at the cutoff, as enforced by the harness.
- `season_length=7` tells both models the pattern repeats weekly.
- Output is renamed to `yhat`. `lo` and `hi` stay empty until Phase 7.

### What to expect on M5
- On **smooth and erratic** series (a small share of our subset), they should be competitive.
- On **intermittent and lumpy** series (about 90% of our subset), expect them to do poorly, because they forecast a smooth average for data that is mostly zeros. That is exactly where the comparison with LightGBM (which can use a Tweedie loss) and the foundation models becomes interesting.
- If we want a fairer classical baseline for intermittent demand, statsforecast also has Croston-style models (`CrostonOptimized`, `ADIDA`, `IMAPA`). They could be added later as extra entries in the harness.

---

## 5. Glossary

- **Stationary:** statistical behaviour (mean, variance) does not change over time.
- **Differencing:** replacing `y_t` with `y_t - y_(t-1)`.
- **AICc:** model-selection score that balances fit and complexity; lower is better.
- **Maximum likelihood:** choosing parameters that make the observed data most probable.
- **Damped trend:** a trend that flattens out as the forecast horizon grows.
- **Intermittent demand:** demand with many zero days and occasional non-zero sales.
