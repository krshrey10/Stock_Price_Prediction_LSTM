# src/models/arima.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pmdarima as pm

def fit_predict(y: pd.Series, steps: int = 1, seasonal: bool = False, m: int = 0):
    """Fit auto_arima on a univariate series (y) and predict `steps` ahead.
    Returns dict with yhat, lo, hi arrays (float)."""
    y = pd.Series(y).astype(float).dropna()
    model = pm.auto_arima(
        y, seasonal=seasonal, m=m or 0, suppress_warnings=True, error_action="ignore"
    )
    fc, conf = model.predict(n_periods=steps, return_conf_int=True, alpha=0.2)  # 80% PI
    lo = conf[:, 0]
    hi = conf[:, 1]
    return {"yhat": np.asarray(fc, float), "lo": np.asarray(lo, float), "hi": np.asarray(hi, float)}
