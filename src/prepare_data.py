"""Phase 1: M5 wide sales -> long format + hierarchy + demand-type segments.

Usage:
    python src/prepare_data.py              # 500-series stratified subset
    python src/prepare_data.py --n 0        # all 30,490 series

Outputs:
    data/long.parquet       unique_id, ds, y, sell_price, snap
    data/series_meta.csv    unique_id, state_id, store_id, cat_id, dept_id, item_id,
                            n_obs, adi, cv2, segment
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from paths import DATA_DIR  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", DATA_DIR
HIER = ["state_id", "store_id", "cat_id", "dept_id", "item_id"]
ADI_CUT, CV2_CUT = 1.32, 0.49  # Syntetos-Boylan classification thresholds


def load_sales(n: int, seed: int) -> pd.DataFrame:
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    if n and n < len(sales):
        # sample proportionally within each store so the hierarchy stays balanced
        frac = n / len(sales)
        sales = (sales.groupby("store_id", group_keys=False)
                 .sample(frac=frac, random_state=seed).reset_index(drop=True))
    sales["unique_id"] = sales["store_id"] + "__" + sales["item_id"]
    return sales


def to_long(sales: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=["unique_id", "state_id", "store_id", "item_id"],
                      value_vars=day_cols, var_name="d", value_name="y")
    long = long.merge(cal[["d", "date", "wm_yr_wk", "snap_CA", "snap_TX", "snap_WI"]], on="d")
    long["snap"] = np.select(
        [long.state_id == "CA", long.state_id == "TX", long.state_id == "WI"],
        [long.snap_CA, long.snap_TX, long.snap_WI])
    long = long.rename(columns={"date": "ds"})
    long["ds"] = pd.to_datetime(long["ds"])

    prices = pd.read_csv(RAW / "sell_prices.csv")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # M5 convention: price is NaN until the item is first stocked -> drop that leading period
    long = long[long.groupby("unique_id")["sell_price"].transform(lambda s: s.notna().cummax())]
    return (long[["unique_id", "ds", "y", "sell_price", "snap"]]
            .sort_values(["unique_id", "ds"]).reset_index(drop=True))


def segment(long: pd.DataFrame) -> pd.DataFrame:
    """ADI = mean gap between non-zero sales; CV2 = squared CV of non-zero sizes."""
    def stats(g: pd.Series) -> pd.Series:
        nz = g[g > 0]
        adi = len(g) / max(len(nz), 1)
        cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        return pd.Series({"n_obs": len(g), "adi": adi, "cv2": cv2})

    s = long.groupby("unique_id")["y"].apply(stats).unstack()
    s["segment"] = np.select(
        [(s.adi < ADI_CUT) & (s.cv2 < CV2_CUT), (s.adi < ADI_CUT), (s.cv2 < CV2_CUT)],
        ["smooth", "erratic", "intermittent"], default="lumpy")
    return s.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="series to keep (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cal = pd.read_csv(RAW / "calendar.csv")
    sales = load_sales(a.n, a.seed)
    long = to_long(sales, cal)
    meta = sales[["unique_id"] + HIER].merge(segment(long), on="unique_id")

    OUT.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT / "long.parquet", index=False)
    meta.to_csv(OUT / "series_meta.csv", index=False)

    print(f"series: {meta.unique_id.nunique():,}  rows: {len(long):,}  "
          f"{long.ds.min().date()} -> {long.ds.max().date()}")
    print(meta.segment.value_counts().to_string())


if __name__ == "__main__":
    main()
