"""Phase 6: hierarchical reconciliation (bottom-up, MinTrace-OLS, MinTrace-WLS-structural).

Hierarchy over the sampled series:  Total > State > Store > Store x Category > Item-Store (bottom).
Bottom base forecasts are reused from results/forecasts.csv (so numbers match the main table).
Upper levels get their own base forecast: the same model is run on the aggregated series
(sum of sales, mean price, max SNAP) and cached in results/agg_forecasts.csv (resumable).

Reconciliation is  y_tilde = S P y_hat  with
    bottom_up   P = [0 | I]
    ols         P = (S'S)^-1 S'
    wls_struct  P = (S' W^-1 S)^-1 S' W^-1,  W = diag(S 1)   (variance proportional to #bottom series)
MinT with a full / shrunk covariance needs in-sample residuals, which the harness does not store.
Reconciled forecasts are clipped at 0 (clipping keeps coherence only approximately).

Writes results/reconciled.csv (bottom level only, model = "<base>+<method>"),
results/metrics_hierarchy.csv (WAPE / RMSE / bias per level, base vs reconciled).
Usage:  python src/reconcile.py [--models AutoETS Chronos_zeroshot N-HiTS]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DATA, MODELS, RESULTS, make_cutoffs, split

ROOT = Path(__file__).resolve().parents[1]
H = 28
LEVELS = ["total", "state", "store", "store_cat", "bottom"]


def build_hierarchy(meta: pd.DataFrame):
    """Return bottom ids, all node ids, level of each node, dense S [n_nodes, n_bottom]."""
    meta = meta.sort_values("unique_id").reset_index(drop=True)
    bottom = meta["unique_id"].tolist()
    keys = {
        "total": pd.Series("TOTAL", index=meta.index),
        "state": "state:" + meta["state_id"],
        "store": "store:" + meta["store_id"],
        "store_cat": "storecat:" + meta["store_id"] + "|" + meta["cat_id"],
    }
    node_ids, node_level, rows = [], [], []
    for lvl in LEVELS[:-1]:
        for k in sorted(keys[lvl].unique()):
            node_ids.append(k)
            node_level.append(lvl)
            rows.append((keys[lvl] == k).to_numpy(np.float64))
    S_bottom = np.eye(len(bottom))
    S = np.vstack(rows + [S_bottom])
    node_ids += bottom
    node_level += ["bottom"] * len(bottom)
    return bottom, node_ids, node_level, S, {lvl: keys[lvl] for lvl in LEVELS[:-1]}


def aggregate_long(long: pd.DataFrame, meta: pd.DataFrame, keys: dict) -> pd.DataFrame:
    m = meta.sort_values("unique_id").reset_index(drop=True)
    parts = []
    for lvl, key in keys.items():
        mp = pd.DataFrame({"unique_id": m["unique_id"], "node": key.to_numpy()})
        g = long.merge(mp, on="unique_id").groupby(["node", "ds"]).agg(
            y=("y", "sum"), sell_price=("sell_price", "mean"), snap=("snap", "max")).reset_index()
        parts.append(g.rename(columns={"node": "unique_id"}))
    return pd.concat(parts, ignore_index=True)


def projection(S: np.ndarray, method: str) -> np.ndarray:
    n, m = S.shape
    if method == "bottom_up":
        P = np.zeros((m, n))
        P[:, n - m:] = np.eye(m)
        return P
    w = np.ones(n) if method == "ols" else S.sum(axis=1)
    SW = S.T / w
    return np.linalg.solve(SW @ S, SW)


def base_agg_forecasts(models, long, meta, keys, cutoffs) -> pd.DataFrame:
    f = RESULTS / "agg_forecasts.csv"
    done = pd.read_csv(f, parse_dates=["cutoff", "ds"]) if f.exists() else pd.DataFrame()
    agg = aggregate_long(long, meta, keys)
    for cutoff in cutoffs:
        train, test = split(agg, cutoff, H)
        for name in models:
            if len(done) and ((done["model"] == name) & (done["cutoff"] == cutoff)).any():
                continue
            pred = MODELS[name](train, test.drop(columns="y"), H)[["unique_id", "ds", "yhat"]]
            pred["cutoff"], pred["model"] = cutoff, name
            done = pd.concat([done, pred], ignore_index=True)
            done.to_csv(f, index=False)
            print(f"  agg base  {name:<18} {cutoff.date()}")
    return done, agg


def wape_rmse_bias(g: pd.DataFrame) -> pd.Series:
    e = g["yhat"] - g["y"]
    return pd.Series({"WAPE": e.abs().sum() / g["y"].sum(), "RMSE": np.sqrt((e ** 2).mean()),
                      "Bias_%": 100 * e.sum() / g["y"].sum()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["AutoETS", "Chronos_zeroshot", "N-HiTS"])
    models = ap.parse_args().models

    long = pd.read_parquet(DATA)
    meta = pd.read_csv(DATA.parent / "series_meta.csv")
    bottom, node_ids, node_level, S, keys = build_hierarchy(meta)
    cutoffs = make_cutoffs(long["ds"].max(), 5, H)
    agg_fc, agg = base_agg_forecasts(models, long, meta, keys, cutoffs)

    fc = pd.read_csv(RESULTS / "forecasts.csv", parse_dates=["cutoff", "ds"])
    actual = pd.concat([long[["unique_id", "ds", "y"]], agg[["unique_id", "ds", "y"]]])
    level_of = dict(zip(node_ids, node_level))
    methods = ["bottom_up", "ols", "wls_struct"]
    Ps = {m: projection(S, m) for m in methods}

    recon_rows, metric_rows = [], []
    for name in models:
        for cutoff in cutoffs:
            b = fc[(fc["model"] == name) & (fc["cutoff"] == cutoff)]
            a = agg_fc[(agg_fc["model"] == name) & (agg_fc["cutoff"] == cutoff)]
            base = pd.concat([a[["unique_id", "ds", "yhat"]], b[["unique_id", "ds", "yhat"]]])
            wide = base.pivot(index="unique_id", columns="ds", values="yhat").reindex(node_ids)
            dates = wide.columns
            Y = wide.to_numpy()
            variants = {"base": Y}
            for m in methods:
                variants[m] = np.clip(S @ (Ps[m] @ Y), 0, None)
            for tag, arr in variants.items():
                df = pd.DataFrame(arr, index=node_ids, columns=dates).rename_axis("unique_id")
                df = df.reset_index().melt("unique_id", var_name="ds", value_name="yhat")
                df["ds"] = pd.to_datetime(df["ds"])
                df = df.merge(actual, on=["unique_id", "ds"], how="left")
                df["level"], df["model"], df["method"] = df["unique_id"].map(level_of), name, tag
                df["cutoff"] = cutoff
                metric_rows.append(df)
                if tag != "base":
                    r = df[df["level"] == "bottom"].copy()
                    r["model"] = f"{name}+{tag}"
                    recon_rows.append(r[["unique_id", "cutoff", "ds", "model", "y", "yhat"]])

    allm = pd.concat(metric_rows)
    out = (allm.groupby(["model", "method", "level"]).apply(wape_rmse_bias, include_groups=False)
           .reset_index())
    out["level"] = pd.Categorical(out["level"], LEVELS, ordered=True)
    out = out.sort_values(["model", "level", "method"])
    out.to_csv(RESULTS / "metrics_hierarchy.csv", index=False)
    pd.concat(recon_rows).assign(lo=np.nan, hi=np.nan).to_csv(RESULTS / "reconciled.csv", index=False)

    pd.set_option("display.width", 200)
    for name in models:
        print(f"\n{name}  WAPE by level")
        print(out[out["model"] == name].pivot(index="level", columns="method", values="WAPE")
              [["base"] + methods].round(3).to_string())
