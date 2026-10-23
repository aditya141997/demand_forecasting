"""Paired bootstrap over series: is model A's WAPE really different from the reference model's?

Resamples series (all folds of a series stay together) 2,000 times and reports the WAPE difference
(model - reference) with a 95% interval. Interval excluding 0 => difference is not noise at this sample size.
Writes results/significance.csv.   Usage: python src/significance.py [--ref AutoETS]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paths import RESULTS_DIR as RESULTS  # noqa: E402
B = 2000

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="AutoETS")
    ref = ap.parse_args().ref

    fc = pd.read_csv(RESULTS / "forecasts.csv", usecols=["unique_id", "model", "y", "yhat"])
    fc["ae"] = (fc["yhat"] - fc["y"]).abs()
    per = fc.groupby(["model", "unique_id"]).agg(ae=("ae", "sum"), y=("y", "sum")).reset_index()
    ae = per.pivot(index="unique_id", columns="model", values="ae")
    y = per.pivot(index="unique_id", columns="model", values="y")[ref].to_numpy()
    n = len(ae)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))
    ysum = y[idx].sum(axis=1)

    ref_w = ae[ref].to_numpy()[idx].sum(axis=1) / ysum
    rows = []
    for m in ae.columns:
        w = ae[m].to_numpy()[idx].sum(axis=1) / ysum
        d = w - ref_w
        rows.append(dict(model=m, WAPE=ae[m].sum() / y.sum(), diff_vs_ref=ae[m].sum() / y.sum() - ae[ref].sum() / y.sum(),
                         ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5)))
    out = pd.DataFrame(rows).sort_values("WAPE")
    out["significant"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    out.to_csv(RESULTS / "significance.csv", index=False)
    print(f"reference: {ref}, {n} series, {B} bootstrap resamples\n")
    print(out.round(4).to_string(index=False))
