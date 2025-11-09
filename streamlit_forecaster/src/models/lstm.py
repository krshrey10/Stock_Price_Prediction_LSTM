# src/models/lstm.py
# --- quiet & stable TF BEFORE any TF/Keras import ---
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF INFO logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # disable oneDNN custom ops for reproducibility

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    keras = layers = None

def fit_predict(y: pd.Series, steps: int = 1, lookback: int = 30, epochs: int = 3):
    """Simple LSTM on univariate series (y). Forecast iteratively for `steps`."""
    if keras is None:
        raise RuntimeError("TensorFlow is not installed")

    y = pd.Series(y).astype(float).dropna().values
    if len(y) <= lookback + 1:
        raise ValueError("Series too short for LSTM")

    X, T = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback:i])
        T.append(y[i])
    X = np.asarray(X, float)[..., None]  # (n, lookback, 1)
    T = np.asarray(T, float)

    model = keras.Sequential(
        [layers.Input(shape=(lookback, 1)),
         layers.LSTM(32), layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, T, epochs=epochs, verbose=0, batch_size=32)

    # iterative forecast
    buf = y[-lookback:].copy()
    out = []
    for _ in range(steps):
        pred = model.predict(buf.reshape(1, lookback, 1), verbose=0).ravel()[0]
        out.append(pred)
        buf = np.roll(buf, -1)
        buf[-1] = pred
    return {"yhat": np.asarray(out, float)}
