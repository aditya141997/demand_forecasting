"""Phase 4 models: N-HiTS and PatchTST (neuralforecast), global across all series.

Harness contract: fit_predict(train, future, h) -> unique_id, ds, yhat
Point forecast uses MSE (mean demand); intervals come from conformal in Phase 7.
"""
import logging
import warnings

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MSE
from neuralforecast.models import NHITS, PatchTST

warnings.filterwarnings("ignore")
for name in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.ERROR)

HISTORY_DAYS = 730
MAX_STEPS = 400


def _run(model_cls, train: pd.DataFrame, future: pd.DataFrame, h: int, **kw) -> pd.DataFrame:
    cutoff = train["ds"].max()
    t = train[train["ds"] > cutoff - pd.Timedelta(days=HISTORY_DAYS)][["unique_id", "ds", "y"]]
    model = model_cls(h=h, input_size=4 * h, loss=MSE(), scaler_type="standard", max_steps=MAX_STEPS,
                      batch_size=64, windows_batch_size=512, random_seed=1,
                      enable_progress_bar=False, enable_model_summary=False, logger=False,
                      accelerator="cpu", **kw)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=t)
    fc = nf.predict().rename(columns={model_cls.__name__: "yhat"})
    fc["yhat"] = fc["yhat"].clip(lower=0)
    return future[["unique_id", "ds"]].merge(fc[["unique_id", "ds", "yhat"]], on=["unique_id", "ds"], how="left")


def nhits(train, future, h):
    return _run(NHITS, train, future, h)


def patchtst(train, future, h):
    return _run(PatchTST, train, future, h, patch_len=16, stride=8)
