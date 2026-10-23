"""Phase 5 models: Chronos-Bolt zero-shot and fine-tuned (amazon/chronos-bolt-small).

Harness contract: fit_predict(train, future, h) -> unique_id, ds, yhat

Point forecast = mean of the 9 predicted quantiles (0.1 ... 0.9), times a global volume correction.
Bolt predicts a quantile grid, not a mean, and the median is 0 for most zero-heavy M5 series, so the
quantile average is used. It still misses the upper tail (about -15% volume), so the correction is
measured on the last CAL_WINDOWS windows of the training data and applied to the forecast.
Fine-tuning only ever sees windows whose target ends on or before the cutoff (no leakage).
"""
import warnings

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, torch.get_num_threads()))

MODEL_ID = "amazon/chronos-bolt-small"
CONTEXT = 512        # zero-shot context length (days)
FT_CONTEXT = 256     # fine-tune context length (days)
FT_STEPS = 300
FT_BATCH = 32
FT_LR = 5e-5
FT_HISTORY = 730     # target windows come from the last ~2 years before the cutoff
BATCH = 128
CAL_WINDOWS = 2      # most recent h-day windows of the training data used for volume calibration


def _load() -> BaseChronosPipeline:
    return BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float32)


def _series(train: pd.DataFrame) -> dict[str, np.ndarray]:
    t = train.sort_values(["unique_id", "ds"])
    return {u: g["y"].to_numpy(np.float32) for u, g in t.groupby("unique_id", sort=True)}


def _forecast(pipe: BaseChronosPipeline, ctxs: list, h: int) -> np.ndarray:
    """Mean of the 9 predicted quantiles for every context -> [n_series, h]."""
    preds = []
    for i in range(0, len(ctxs), BATCH):
        batch = [torch.from_numpy(c[-CONTEXT:]) for c in ctxs[i:i + BATCH]]
        with torch.no_grad():
            preds.append(pipe.predict(batch, prediction_length=h).mean(dim=1).numpy())
    return np.clip(np.concatenate(preds), 0, None)


def _calibration(pipe: BaseChronosPipeline, series: dict, h: int) -> float:
    """Volume correction = actual / forecast over the last CAL_WINDOWS windows of the training data.

    Quantile averaging (0.1-0.9) misses the upper tail of zero-heavy demand, so raw forecasts
    under-shoot total volume. Only data up to the cutoff is used.
    """
    act, fc = 0.0, 0.0
    for k in range(1, CAL_WINDOWS + 1):
        ids = [u for u, y in series.items() if len(y) > k * h + 28]
        ctx = [series[u][:len(series[u]) - k * h] for u in ids]
        pred = _forecast(pipe, ctx, h)
        end = lambda y: len(y) - (k - 1) * h
        actual = np.stack([series[u][end(series[u]) - h:end(series[u])] for u in ids])
        act += actual.sum()
        fc += pred.sum()
    return float(act / fc) if fc > 0 else 1.0


def _predict(pipe: BaseChronosPipeline, series: dict, future: pd.DataFrame, h: int,
             scale: float = 1.0) -> pd.DataFrame:
    ids = list(series)
    yhat = _forecast(pipe, [series[u] for u in ids], h) * scale   # [n_series, h]

    dates = np.sort(future["ds"].unique())
    out = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, len(ids)),
                        "yhat": yhat.ravel()})
    return future[["unique_id", "ds"]].merge(out, on=["unique_id", "ds"], how="left")


def chronos_zeroshot(train, future, h):
    pipe, series = _load(), _series(train)
    return _predict(pipe, series, future, h, scale=_calibration(pipe, series, h))


# ---------------------------------------------------------------- fine-tuning
def _batch(series: dict, ids: list, rng: np.random.Generator, h: int):
    ctx = np.full((FT_BATCH, FT_CONTEXT), np.nan, np.float32)
    tgt = np.full((FT_BATCH, h), np.nan, np.float32)
    for b in range(FT_BATCH):
        y = series[ids[rng.integers(len(ids))]]
        lo = max(h, len(y) - FT_HISTORY)               # target must start inside the recent window
        e = int(rng.integers(lo, len(y) - h + 1)) if len(y) - h + 1 > lo else len(y) - h
        c = y[max(0, e - FT_CONTEXT):e]
        ctx[b, FT_CONTEXT - len(c):] = c
        tgt[b] = y[e:e + h]
    return torch.from_numpy(ctx), torch.from_numpy(tgt)


def chronos_finetuned(train, future, h):
    pipe = _load()
    series = _series(train)
    # fine-tune without the calibration windows so the volume correction is measured out-of-sample
    fit_series = {u: y[:len(y) - CAL_WINDOWS * h] for u, y in series.items()}
    fit_series = {u: y for u, y in fit_series.items() if len(y) > FT_CONTEXT // 4 + h}
    ids = list(fit_series)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    model = pipe.model
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=FT_STEPS)
    for _ in range(FT_STEPS):
        ctx, tgt = _batch(fit_series, ids, rng, h)
        loss = model(context=ctx, target=tgt, target_mask=~torch.isnan(tgt)).loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return _predict(pipe, series, future, h, scale=_calibration(pipe, series, h))
