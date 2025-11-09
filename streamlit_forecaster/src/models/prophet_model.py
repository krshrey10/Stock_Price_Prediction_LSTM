# src/models/prophet_model.py
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from prophet import Prophet
except Exception as e:  # pragma: no cover
    Prophet = None

def fit_predict(train_close: pd.DataFrame, future_df: pd.DataFrame):
    """Prophet using close price (history is a DataFrame with column 'close').
    future_df: index with future timestamps. Returns dict yhat/lo/hi arrays."""
    if Prophet is None:
        raise RuntimeError("Prophet is not installed")

    df = (
        pd.DataFrame({"ds": train_close.index, "y": train_close["close"].astype(float).values})
        .dropna()
    )
    m = Prophet(seasonality_mode="additive", daily_seasonality=False, weekly_seasonality=True)
    m.fit(df)
    future = pd.DataFrame({"ds": list(future_df.index)})
    fc = m.predict(future)
    return {
        "yhat": fc["yhat"].astype(float).to_numpy(),
        "lo": fc["yhat_lower"].astype(float).to_numpy(),
        "hi": fc["yhat_upper"].astype(float).to_numpy(),
    }
