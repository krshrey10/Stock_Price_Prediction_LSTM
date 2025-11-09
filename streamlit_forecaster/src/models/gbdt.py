from __future__ import annotations

import re
import time
import numpy as np
import pandas as pd
import lightgbm as lgb


def _sanitize_feature_names(cols):
    """
    LightGBM complains if feature names contain tuples or special JSON chars.
    Convert everything to safe strings like: ('a', 'b') -> "a__b".
    """
    safe = []
    for c in cols:
        if isinstance(c, tuple):
            s = "__".join(map(str, c))
        else:
            s = str(c)
        # strip risky chars
        s = re.sub(r"[{}\\[\\]\":,]", "_", s)
        safe.append(s)
    return safe


def fit_predict_quantiles(
    train: pd.DataFrame,
    future_like: pd.DataFrame,
    quantiles=(0.1, 0.5, 0.9),
    params=None,
):
    """
    Train LightGBM quantile models and predict on `future_like`.

    Returns dict with lo, yhat, hi + extras:
      - runtime_s
      - feature_importance (gain) from median model

    Notes:
    - n_estimators must be passed only via the constructor (avoids LightGBM spam).
    - verbosity=-1 + callbacks=log_evaluation(period=0) silences logs.
    - Feature names are sanitized to avoid 'special JSON characters' errors.
    """
    params = (params or {}).copy()

    # --- number of trees in ONE place ---
    n_estimators = int(params.pop("n_estimators", params.pop("num_boost_round", 300)))

    # --- small-data friendly defaults ---
    default_params = dict(
        learning_rate=0.05,
        num_leaves=min(31, max(7, int(np.sqrt(max(1, len(train)))))),
        min_child_samples=max(5, int(0.01 * max(1, len(train)))),
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        max_depth=-1,
        min_split_gain=0.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        verbosity=-1,
        force_col_wise=True,
    )
    for k, v in default_params.items():
        params.setdefault(k, v)

    # --- prepare matrices + safe column names ---
    y = train["y_t+1"].values
    X = train.drop(columns=["y_t+1"], errors="ignore").copy()
    Xf = future_like.drop(columns=["y_t+1"], errors="ignore").copy()

    safe_cols = _sanitize_feature_names(X.columns)
    X.columns = safe_cols
    Xf.columns = safe_cols

    outs, models = {}, {}
    t0 = time.time()
    for q, key in zip(quantiles, ["lo", "yhat", "hi"]):
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=float(q),
            n_estimators=n_estimators,  # only here
            **params,
        )
        model.fit(X, y, callbacks=[lgb.log_evaluation(period=0)])
        outs[key] = model.predict(Xf)
        models[key] = model

    runtime_s = time.time() - t0

    # feature importance from the median model
    fi = None
    try:
        gains = models["yhat"].booster_.feature_importance(importance_type="gain")
        fi = pd.Series(gains, index=safe_cols).sort_values(ascending=False)
    except Exception:
        pass

    return {
        "lo": outs["lo"],
        "yhat": outs["yhat"],
        "hi": outs["hi"],
        "runtime_s": float(runtime_s),
        "feature_importance": fi,
    }
