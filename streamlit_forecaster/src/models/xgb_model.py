# src/models/xgb_model.py
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

def _prep_xy(df: pd.DataFrame):
    ycols = [c for c in df.columns if c.startswith("y")]
    X = df.drop(columns=ycols, errors="ignore")
    y = df["y_t+1"].astype(float).values if "y_t+1" in df else None
    return X, y

def fit_predict(train: pd.DataFrame, future_like: pd.DataFrame):
    if XGBRegressor is None:
        raise RuntimeError("xgboost not installed")

    Xtr, ytr = _prep_xy(train)
    Xte, _ = _prep_xy(future_like)

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=42
    )
    model.fit(Xtr, ytr, verbose=False)
    yhat = model.predict(Xte)
    return {"yhat": np.asarray(yhat, float), "model": model, "Xte": Xte}
