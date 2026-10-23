"""Phase 2: rolling-origin backtest harness.

Every model is a function  fit_predict(train, future, h) -> DataFrame
    train   unique_id, ds, y, sell_price, snap   (ds <= cutoff only)
    future  unique_id, ds, sell_price, snap      (the h days after the cutoff)
    returns unique_id, ds, yhat [, lo, hi]

Register it in MODELS and it is run on identical folds, with resume support.

Usage:
    python src/backtest.py                              # all models, 5 folds, h=28
    python src/backtest.py --models SeasonalNaive --folds 3
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from paths import DATA_DIR, RESULTS_DIR  # noqa: E402

DATA, RESULTS = DATA_DIR / "long.parquet", RESULTS_DIR
KEYS = ["model", "cutoff"]


# ---------------------------------------------------------------- models
def seasonal_naive(train: pd.DataFrame, future: pd.DataFrame, h: int) -> pd.DataFrame:
    """Repeat the last 7 observed days (weekly seasonality)."""
    last = train.sort_values("ds").groupby("unique_id").tail(7).copy()
    last["dow"] = last["ds"].dt.dayofweek
    out = future[["unique_id", "ds"]].copy()
    out["dow"] = out["ds"].dt.dayofweek
    out = out.merge(last[["unique_id", "dow", "y"]], on=["unique_id", "dow"], how="left")
    out["yhat"] = out["y"].fillna(0.0)
    return out[["unique_id", "ds", "yhat"]]


from models_classical import auto_arima, auto_ets, lightgbm_direct  # noqa: E402
from models_deep import nhits, patchtst  # noqa: E402
from models_foundation import chronos_finetuned, chronos_zeroshot  # noqa: E402

MODELS = {
    "SeasonalNaive": seasonal_naive,
    "AutoETS": auto_ets,
    "AutoARIMA": auto_arima,
    "LightGBM": lightgbm_direct,
    "N-HiTS": nhits,
    "PatchTST": patchtst,
    "Chronos_zeroshot": chronos_zeroshot,
    "Chronos_finetuned": chronos_finetuned,
}


# ---------------------------------------------------------------- folds
def make_cutoffs(last_date: pd.Timestamp, n_folds: int, h: int) -> list[pd.Timestamp]:
    """Last fold's test window ends at last_date; earlier folds step back h days."""
    return [last_date - pd.Timedelta(days=h * k) for k in range(n_folds, 0, -1)]


def split(df: pd.DataFrame, cutoff: pd.Timestamp, h: int):
    end = cutoff + pd.Timedelta(days=h)
    train = df[df["ds"] <= cutoff]
    test = df[(df["ds"] > cutoff) & (df["ds"] <= end)]
    # leak guard
    assert train["ds"].max() <= cutoff, "train contains data after cutoff"
    assert test["ds"].nunique() == h, f"test window has {test['ds'].nunique()} days, expected {h}"
    assert test["ds"].min() > cutoff, "test overlaps train"
    return train, test


# ---------------------------------------------------------------- runner
def load_done() -> set:
    f = RESULTS / "forecasts.csv"
    if not f.exists():
        return set()
    d = pd.read_csv(f, usecols=KEYS, parse_dates=["cutoff"]).drop_duplicates()
    return set(zip(d["model"], d["cutoff"]))


def append(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run(models: list[str], n_folds: int, h: int) -> None:
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    cutoffs = make_cutoffs(df["ds"].max(), n_folds, h)
    done = load_done()
    print(f"{df.unique_id.nunique():,} series | cutoffs: {[str(c.date()) for c in cutoffs]} | h={h}")

    for cutoff in cutoffs:
        train, test = split(df, cutoff, h)
        future = test.drop(columns="y")
        for name in models:
            if (name, cutoff) in done:
                print(f"  skip   {name:<18} {cutoff.date()} (already done)")
                continue
            t0 = time.perf_counter()
            pred = MODELS[name](train, future, h)
            secs = time.perf_counter() - t0

            for c in ("lo", "hi"):
                if c not in pred:
                    pred[c] = np.nan
            res = test[["unique_id", "ds", "y"]].merge(pred, on=["unique_id", "ds"], how="left")
            # cold start: an item first stocked inside the test window has no history, so no model can forecast it
            cold = res["yhat"].isna() & ~res["unique_id"].isin(train["unique_id"].unique())
            if cold.any():
                print(f"    {name}: {res.loc[cold, 'unique_id'].nunique()} series without history -> forecast 0")
                res.loc[cold, "yhat"] = 0.0
            assert res["yhat"].notna().all(), f"{name} returned no forecast for some series/days"
            res.insert(1, "cutoff", cutoff)
            res.insert(3, "model", name)
            append(RESULTS / "forecasts.csv",
                   res[["unique_id", "cutoff", "ds", "model", "y", "yhat", "lo", "hi"]])
            append(RESULTS / "runtime.csv",
                   pd.DataFrame([dict(model=name, cutoff=cutoff, seconds=round(secs, 2))]))
            print(f"  done   {name:<18} {cutoff.date()}  {secs:6.1f}s  rows={len(res):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=28)
    a = ap.parse_args()
    run(a.models, a.folds, a.horizon)
