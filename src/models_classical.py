"""Phase 3 models: AutoETS, AutoARIMA (statsforecast) and a global LightGBM.

All follow the harness contract  fit_predict(train, future, h) -> unique_id, ds, yhat
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

HISTORY_DAYS = 730  # statsforecast / LightGBM train on the last ~2 years only


# ---------------------------------------------------------------- statsforecast
def _stats_model(model, train: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    cutoff = train["ds"].max()
    t = train[train["ds"] > cutoff - pd.Timedelta(days=HISTORY_DAYS)]
    sf = StatsForecast(models=[model], freq="D", n_jobs=-1)
    fc = sf.forecast(df=t[["unique_id", "ds", "y"]], h=h)
    name = type(model).__name__
    fc = fc.rename(columns={name: "yhat"})
    fc["yhat"] = fc["yhat"].clip(lower=0)
    out = future[["unique_id", "ds"]].merge(fc[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="left")
    return out


def auto_ets(train, future, h):
    return _stats_model(AutoETS(season_length=7), train, future, h)


def auto_arima(train, future, h):
    return _stats_model(AutoARIMA(season_length=7), train, future, h)


# ---------------------------------------------------------------- LightGBM
LAGS = [28, 35, 42, 56]  # all >= h, so one model predicts the whole horizon without recursion
FEATURES = ([f"lag_{k}" for k in LAGS]
            + ["rmean_7", "rmean_28", "rstd_28", "nz_28", "dow", "dom", "month", "price",
               "price_chg", "snap", "store", "item"])


def _features(panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """panel: train + future rows (future y = NaN), contiguous daily per series."""
    p = panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    g = p.groupby("unique_id")["y"]
    for k in LAGS:
        p[f"lag_{k}"] = g.shift(k)
    base = g.shift(h)  # newest value usable for every day of the horizon
    gb = base.groupby(p["unique_id"])
    p["rmean_7"] = gb.transform(lambda s: s.rolling(7, min_periods=1).mean())
    p["rmean_28"] = gb.transform(lambda s: s.rolling(28, min_periods=1).mean())
    p["rstd_28"] = gb.transform(lambda s: s.rolling(28, min_periods=2).std())
    p["nz_28"] = gb.transform(lambda s: (s > 0).rolling(28, min_periods=1).mean())
    p["dow"], p["dom"], p["month"] = p["ds"].dt.dayofweek, p["ds"].dt.day, p["ds"].dt.month
    p["price"] = p["sell_price"]
    p["price_chg"] = p["sell_price"] / p.groupby("unique_id")["sell_price"].shift(7)
    sp = p["unique_id"].str.split("__", n=1, expand=True)
    p["store"] = sp[0].astype("category").cat.codes
    p["item"] = sp[1].astype("category").cat.codes
    return p


def lightgbm_direct(train: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    cutoff = train["ds"].max()
    hist = train[train["ds"] > cutoff - pd.Timedelta(days=HISTORY_DAYS + max(LAGS))]
    fut = future.copy()
    fut["y"] = np.nan
    p = _features(pd.concat([hist, fut], ignore_index=True), h)

    is_future = p["ds"] > cutoff
    fit = p[~is_future & (p["ds"] > cutoff - pd.Timedelta(days=HISTORY_DAYS)) & p["lag_56"].notna()]
    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.1, n_estimators=300, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=0, verbose=-1)
    model.fit(fit[FEATURES], fit["y"], categorical_feature=["store", "item"])

    out = p.loc[is_future, ["unique_id", "ds"]].copy()
    out["yhat"] = np.clip(model.predict(p.loc[is_future, FEATURES]), 0, None)
    return out
