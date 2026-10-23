"""Phase 7: split-conformal prediction intervals (90%) on top of every model's point forecast.

For fold k the calibration set is every residual (y - yhat) from folds before k, i.e. only data that
was already observed at that cutoff. Fold 1 has no earlier fold, so it gets no interval.
Residual quantiles are taken per forecast-size bin (Mondrian): zero-heavy series with a small yhat
get narrow intervals, high-volume series get wide ones.  lo = max(0, yhat + q_lo), hi = yhat + q_hi.

Writes results/intervals.csv and results/metrics_intervals.csv (coverage, mean width, interval score).
Usage:  python src/conformal.py [--alpha 0.1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import RESULTS_DIR as RESULTS  # noqa: E402
BINS = [-np.inf, 0.25, 0.5, 1.0, 2.0, np.inf]
MIN_CAL = 200


def conformal(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    df["bin"] = pd.cut(df["yhat"], BINS, labels=False)
    df["res"] = df["y"] - df["yhat"]
    cutoffs = sorted(df["cutoff"].unique())
    parts = []
    for i, c in enumerate(cutoffs[1:], start=1):
        cal, cur = df[df["cutoff"].isin(cutoffs[:i])], df[df["cutoff"] == c].copy()
        q = cal.groupby("bin")["res"].agg(
            q_lo=lambda r: r.quantile(alpha / 2), q_hi=lambda r: r.quantile(1 - alpha / 2),
            n="size")
        q.loc[q["n"] < MIN_CAL, ["q_lo", "q_hi"]] = np.nan
        qa = cal["res"].quantile([alpha / 2, 1 - alpha / 2]).to_numpy()   # pooled fallback
        cur = cur.join(q[["q_lo", "q_hi"]], on="bin")
        cur["q_lo"], cur["q_hi"] = cur["q_lo"].fillna(qa[0]), cur["q_hi"].fillna(qa[1])
        cur["lo"] = (cur["yhat"] + cur["q_lo"]).clip(lower=0)
        cur["hi"] = cur["yhat"] + cur["q_hi"]
        parts.append(cur)
    return pd.concat(parts)


def interval_metrics(g: pd.DataFrame, alpha: float) -> pd.Series:
    below, above = g["y"] < g["lo"], g["y"] > g["hi"]
    score = (g["hi"] - g["lo"]) + 2 / alpha * (g["lo"] - g["y"]).clip(lower=0) \
        + 2 / alpha * (g["y"] - g["hi"]).clip(lower=0)
    return pd.Series({"coverage": 1 - (below | above).mean(), "width": (g["hi"] - g["lo"]).mean(),
                      "interval_score": score.mean(), "n": len(g)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    alpha = ap.parse_args().alpha

    fc = pd.concat([pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["cutoff", "ds"]),
                    pd.read_csv(RESULTS / "reconciled.csv", parse_dates=["cutoff", "ds"])])
    out = pd.concat([conformal(g, alpha) for _, g in fc.groupby("model")])
    cols = ["unique_id", "cutoff", "ds", "model", "y", "yhat", "lo", "hi"]
    out[cols].to_csv(RESULTS / "intervals.csv", index=False)

    m = out.groupby("model").apply(interval_metrics, alpha=alpha, include_groups=False).reset_index()
    m.sort_values("interval_score").to_csv(RESULTS / "metrics_intervals.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"target coverage {1 - alpha:.0%}, folds 2-5\n")
    print(m.sort_values("interval_score").round(3).to_string(index=False))
