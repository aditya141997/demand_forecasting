"""Phase 8: evaluation of results/forecasts.csv.

Metrics: WAPE, RMSE, bias, RMSSE, WRMSSE (revenue-weighted), per-series MASE-free.
RMSSE scale = mean squared one-step naive error on each series' history up to the cutoff.
WRMSSE weights = revenue (units * price) over the 28 days before the cutoff, per fold.

Writes results/metrics_overall.csv, metrics_segment.csv, metrics_horizon.csv, metrics_store.csv.
Usage:  python src/evaluate.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from paths import DATA_DIR as DATA, RESULTS_DIR as RESULTS  # noqa: E402


def fold_scale_and_weight(long: pd.DataFrame, cutoffs) -> pd.DataFrame:
    rows = []
    for c in cutoffs:
        tr = long[long["ds"] <= c].sort_values(["unique_id", "ds"])
        d2 = tr.groupby("unique_id")["y"].diff().pow(2)
        scale = d2.groupby(tr["unique_id"]).mean().rename("scale")
        last = tr[tr["ds"] > c - pd.Timedelta(days=28)]
        rev = (last["y"] * last["sell_price"]).groupby(last["unique_id"]).sum().rename("rev")
        f = pd.concat([scale, rev], axis=1).reset_index()
        f["cutoff"] = c
        f["w"] = f["rev"] / f["rev"].sum() if f["rev"].sum() > 0 else 1 / len(f)
        rows.append(f)
    return pd.concat(rows)


def load() -> pd.DataFrame:
    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["cutoff", "ds"])
    long = pd.read_parquet(DATA / "long.parquet")
    meta = pd.read_csv(DATA / "series_meta.csv")[["unique_id", "segment", "store_id", "state_id", "cat_id"]]
    sw = fold_scale_and_weight(long, fc["cutoff"].unique())
    fc = fc.merge(sw, on=["unique_id", "cutoff"], how="left").merge(meta, on="unique_id", how="left")
    fc["h"] = (fc["ds"] - fc["cutoff"]).dt.days
    fc["err"] = fc["yhat"] - fc["y"]
    fc["se"] = fc["err"] ** 2
    fc["ae"] = fc["err"].abs()
    return fc


def summarise(g: pd.DataFrame) -> pd.Series:
    # RMSSE per (series, fold), then average; WRMSSE weights per fold
    s = g.groupby(["unique_id", "cutoff"]).agg(mse=("se", "mean"), scale=("scale", "first"), w=("w", "first"))
    s = s[s["scale"] > 0]
    rmsse = np.sqrt(s["mse"] / s["scale"])
    wrmsse = (rmsse * s["w"]).groupby("cutoff").sum() / s["w"].groupby("cutoff").sum()
    return pd.Series({
        "WAPE": g["ae"].sum() / g["y"].sum(),
        "RMSE": np.sqrt(g["se"].mean()),
        "Bias_%": 100 * g["err"].sum() / g["y"].sum(),
        "RMSSE": rmsse.mean(),
        "WRMSSE": wrmsse.mean(),
        "n_series": g["unique_id"].nunique(),
    })


def table(fc: pd.DataFrame, by: str | None) -> pd.DataFrame:
    keys = ["model"] + ([by] if by else [])
    out = fc.groupby(keys).apply(summarise, include_groups=False).reset_index()
    return out.sort_values(keys[::-1] if by else "WAPE")


if __name__ == "__main__":
    fc = load()
    rt = pd.read_csv(RESULTS / "runtime.csv").groupby("model")["seconds"].sum().rename("seconds")
    overall = table(fc, None).merge(rt, on="model", how="left").sort_values("WAPE")
    overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
    table(fc, "segment").to_csv(RESULTS / "metrics_segment.csv", index=False)
    fc["h_bin"] = pd.cut(fc["h"], [0, 7, 14, 21, 28], labels=["1-7", "8-14", "15-21", "22-28"])
    table(fc, "h_bin").to_csv(RESULTS / "metrics_horizon.csv", index=False)
    table(fc, "store_id").to_csv(RESULTS / "metrics_store.csv", index=False)
    pd.set_option("display.width", 200)
    print(overall.round(3).to_string(index=False))
    seg = pd.read_csv(RESULTS / "metrics_segment.csv")
    print(seg.pivot(index="model", columns="segment", values="WAPE").round(3).to_string())
