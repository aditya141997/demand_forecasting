"""Phase 9: Streamlit dashboard over results/*.csv.

Run:  streamlit run src/dashboard.py   (after python src/evaluate.py, reconcile.py, conformal.py)
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
from paths import DATA_DIR, RESULTS_DIR as R  # noqa: E402

st.set_page_config(page_title="Demand forecasting benchmark", layout="wide")
st.title("Demand forecasting: Chronos vs classical baselines")
st.caption("M5 subset, 500 series, 5 rolling folds, horizon 28 days. `_raw` = Chronos without volume calibration.")


@st.cache_data
def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(R / name, **kw)


overall, seg = read("metrics_overall.csv"), read("metrics_segment.csv")
hor, store = read("metrics_horizon.csv"), read("metrics_store.csv")
hier, ivl = read("metrics_hierarchy.csv"), read("metrics_intervals.csv")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Leaderboard", "Slices", "Hierarchy", "Intervals", "Series explorer"])

with tab1:
    metric = st.selectbox("Metric", ["WAPE", "RMSE", "RMSSE", "WRMSSE", "Bias_%"], key="m1")
    d = overall.sort_values(metric)
    st.plotly_chart(px.bar(d, x=metric, y="model", orientation="h", text_auto=".3f",
                           color=d["model"].str.endswith("_raw").map({True: "uncalibrated", False: "calibrated / other"}),
                           labels={"color": ""}).update_yaxes(autorange="reversed"), width="stretch")
    st.dataframe(overall.round(3), width="stretch", hide_index=True)
    st.plotly_chart(px.scatter(overall, x="seconds", y="WAPE", text="model", log_x=True,
                               title="Accuracy vs total compute (s, 5 folds, log scale)"), width="stretch")
    st.caption("WAPE rewards under-forecasting on zero-heavy data: compare raw vs calibrated Chronos and check Bias_%.")

with tab2:
    metric = st.selectbox("Metric", ["WAPE", "RMSE", "RMSSE", "Bias_%"], key="m2")
    c1, c2 = st.columns(2)
    for col, df, key, title in [(c1, seg, "segment", "By demand segment"), (c1, hor, "h_bin", "By horizon step (days)"),
                                (c2, store, "store_id", "By store")]:
        p = df.pivot(index="model", columns=key, values=metric)
        col.subheader(title)
        col.plotly_chart(px.imshow(p, text_auto=".2f", aspect="auto", color_continuous_scale="Blues"),
                         width="stretch")
    st.caption("Segment sizes: 373 intermittent, 82 lumpy, 29 smooth, 16 erratic. Smooth/erratic are noisy.")

with tab3:
    metric = st.selectbox("Metric", ["WAPE", "RMSE", "Bias_%"], key="m3")
    model = st.selectbox("Base model", sorted(hier["model"].unique()))
    d = hier[hier["model"] == model]
    st.plotly_chart(px.bar(d, x="level", y=metric, color="method", barmode="group",
                           category_orders={"level": ["total", "state", "store", "store_cat", "bottom"],
                                            "method": ["base", "bottom_up", "ols", "wls_struct"]}),
                    width="stretch")

with tab4:
    st.write("Split-conformal 90% intervals, calibrated on earlier folds only (folds 2-5).")
    st.dataframe(ivl.round(3), width="stretch", hide_index=True)
    st.plotly_chart(px.scatter(ivl, x="width", y="coverage", text="model", title="Coverage vs width (target 0.90)"),
                    width="stretch")


@st.cache_data
def intervals() -> pd.DataFrame:
    return pd.read_csv(R / "intervals.csv", parse_dates=["cutoff", "ds"])


with tab5:
    iv = intervals()
    meta = pd.read_csv(DATA_DIR / "series_meta.csv")
    c1, c2 = st.columns(2)
    segsel = c1.selectbox("Segment", sorted(meta["segment"].unique()))
    uid = c2.selectbox("Series", meta.loc[meta["segment"] == segsel, "unique_id"].tolist())
    models = st.multiselect("Models", sorted(iv["model"].unique()), default=["AutoETS", "Chronos_zeroshot"])
    cutoff = st.selectbox("Fold cutoff", sorted(iv["cutoff"].unique()), index=3, format_func=lambda x: str(x.date()))
    d = iv[(iv["unique_id"] == uid) & (iv["cutoff"] == cutoff) & iv["model"].isin(models)]
    fig = px.line(d, x="ds", y="yhat", color="model")
    for m_, g in d.groupby("model"):
        fig.add_scatter(x=g["ds"], y=g["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=g["ds"], y=g["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                        opacity=0.2, name=f"{m_} 90%", hoverinfo="skip")
    act = d.drop_duplicates("ds")
    fig.add_scatter(x=act["ds"], y=act["y"], mode="markers+lines", name="actual", line=dict(color="black", dash="dot"))
    st.plotly_chart(fig, width="stretch")
