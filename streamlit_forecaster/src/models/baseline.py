# src/models/baseline.py
import numpy as np
import pandas as pd
from typing import Iterable, Union, Any

SeriesLike = Union[pd.Series, Iterable[float], Any]

def _last_numeric_value(x: SeriesLike) -> float:
    # pandas Series
    if isinstance(x, pd.Series):
        s = pd.to_numeric(x, errors="coerce").dropna()
        if s.empty:
            raise ValueError("No numeric values found in Series")
        return float(s.iloc[-1])

    # pandas DataFrame -> first numeric column
    if isinstance(x, pd.DataFrame):
        num = x.select_dtypes(include="number")
        if num.shape[1] == 0:
            raise ValueError("No numeric columns in DataFrame")
        return float(num.iloc[-1, 0])

    # scalar
    try:
        return float(x)  # will work for numeric scalars
    except Exception:
        pass

    # generic iterable – walk from the end to find a numeric value
    try:
        arr = list(x)
        for v in reversed(arr):
            try:
                return float(v)
            except Exception:
                continue
    except Exception:
        pass

    raise ValueError("Could not find a numeric last value in input.")

def predict_last_value(series: SeriesLike, steps: int = 1):
    last = _last_numeric_value(series)
    return np.repeat(last, steps)

def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, **kwargs):
    steps = len(test_df)
    yhat = predict_last_value(train_df["close"], steps=steps)
    return {"yhat": yhat}
