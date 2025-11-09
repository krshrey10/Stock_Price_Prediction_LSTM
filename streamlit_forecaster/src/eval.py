# src/eval.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

import altair as alt
import matplotlib.pyplot as plt

from typing import Iterable, Dict, Tuple, List, Optional

# project imports
from .models import gbdt, baseline
try:
    from .models import arima, prophet_model, xgb_model, lstm
except Exception:
    arima = prophet_model = xgb_model = lstm = None


# ---------- utilities shared with app ----------
def _close_series_1d(prices_df: pd.DataFrame) -> pd.Series:
    s = prices_df["close"] if "close" in prices_df.columns else prices_df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s.squeeze(), errors="coerce")


def convert_to_price_space(yhat: pd.Series, last_price: float, target_type: str) -> pd.Series:
    if target_type == "Price":
        return yhat
    if target_type == "Delta":
        return last_price + yhat
    if target_type == "Return":
        return last_price * (1.0 + yhat)
    return last_price + yhat


# ---------- Diebold–Mariano test ----------
def _dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2) -> Tuple[float, float]:
    """
    Simple DM test (two-sided) with Newey–West HAC variance for forecast horizon h.
    Returns: (DM statistic, p-value)
    """
    # loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power
    d = d - np.nanmean(d)
    n = len(d)
    if n < 5:
        return (np.nan, np.nan)

    # Newey-West variance with lag (h-1)
    lag = h - 1
    gamma0 = np.nanmean(d * d)
    cov = 0.0
    for j in range(1, lag + 1):
        g = np.nanmean(d[j:] * d[:-j])
        cov += 2.0 * (1.0 - j / (lag + 1)) * g
    var = gamma0 + cov
    if var <= 0 or np.isnan(var):
        return (np.nan, np.nan)

    stat = np.nanmean(d) / math.sqrt(var / n)
    # two-sided normal p-value
    try:
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / math.sqrt(2))))
    except Exception:
        p = np.nan
    return (float(stat), float(p))


# ---------- model adapters used in backtest ----------
def _predict_with_model(name: str,
                        train_feat: pd.DataFrame,
                        future_like: pd.DataFrame,
                        cfg: dict) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """Returns (yhat, lo, hi) in *model's native target space*"""
    idx = future_like.index

    if name == "GBDT":
        q = gbdt.fit_predict_quantiles(
            train_feat,
            future_like,
            quantiles=tuple(cfg["models"]["gbdt"]["quantiles"]),
            params=cfg["models"]["gbdt"]["params"],
        )
        return (pd.Series(q["yhat"], index=idx),
                pd.Series(q["lo"], index=idx),
                pd.Series(q["hi"], index=idx))

    if name == "Naive":
        # repeat last observed close from *prices*, but we don't have it here.
        # caller will replace yhat later (special handling in backtest)
        return (pd.Series(np.nan, index=idx), None, None)

    if name == "ARIMA" and arima is not None:
        a = arima.fit_predict(train_feat["y_t+1"].dropna(), steps=len(idx))
        return (pd.Series(a["yhat"], index=idx), pd.Series(a["lo"], index=idx), pd.Series(a["hi"], index=idx))

    if name == "Prophet" and prophet_model is not None:
        # prophet expects raw prices; adapter handled by caller (we need prices)
        raise RuntimeError("Prophet adapter requires prices. Not supported inside generic backtest.")

    if name == "XGBoost" and xgb_model is not None:
        x = xgb_model.fit_predict(train_feat, future_like)
        return (pd.Series(x["yhat"], index=idx), None, None)

    if name == "LSTM" and lstm is not None:
        l = lstm.fit_predict(train_feat["y_t+1"].dropna(), steps=len(idx))
        return (pd.Series(l["yhat"], index=idx), None, None)

    return (pd.Series(np.nan, index=idx), None, None)


# ---------- walk-forward backtest ----------
def backtest_multi_horizon(
    prices: pd.DataFrame,
    feat: pd.DataFrame,
    models: List[str],
    horizons: Iterable[int],
    cfg: dict,
    target_type: str,
    min_train: int = 252,
    step: int = 1,
) -> Dict[int, pd.DataFrame]:
    """
    Expanding-window walk-forward forecasts for each horizon.
    Returns: dict[h] -> long DataFrame with columns:
        date, model, yhat(price), true(price), error
    """
    close = _close_series_1d(prices)
    horizons = list(horizons)
    out: Dict[int, List[Tuple[pd.Timestamp, str, float, float, float]]] = {h: [] for h in horizons}

    n = len(feat)
    for t in range(min_train, n - max(horizons), step):
        # train end at t (exclusive), forecast for future dates t .. t+h-1
        train_feat = feat.iloc[:t].copy()
        last_price = float(close.iloc[t - 1])

        for h in horizons:
            future_idx = feat.index[t : t + h]
            if len(future_idx) < h:
                continue
            future_like = pd.DataFrame(index=future_idx, data=train_feat.iloc[-h:])

            # true prices for horizon
            true_h = close.loc[future_idx]

            for m in models:
                if m == "Naive":
                    yhat = pd.Series([last_price] * h, index=future_idx)
                    pr = yhat  # already price space
                elif m == "Prophet" and prophet_model is not None:
                    try:
                        p = prophet_model.fit_predict(prices[["close"]].iloc[:t], pd.DataFrame(index=future_idx))
                        pr = convert_to_price_space(pd.Series(p["yhat"], index=future_idx), last_price, target_type)
                    except Exception:
                        continue
                else:
                    try:
                        native_yhat, _, _ = _predict_with_model(m, train_feat, future_like, cfg)
                        pr = convert_to_price_space(native_yhat, last_price, target_type)
                    except Exception:
                        continue

                # record only the last point (t+h-1) for a fair "h-step ahead" error
                d = future_idx[-1]
                out[h].append((d, m, float(pr.iloc[-1]), float(true_h.iloc[-1]), float(pr.iloc[-1] - true_h.iloc[-1])))

    # to frames
    frames = {}
    for h in horizons:
        fr = pd.DataFrame(out[h], columns=["date", "model", "yhat", "true", "err"]).sort_values(["date", "model"])
        frames[h] = fr
    return frames


def dm_matrix_for_horizon(fr: pd.DataFrame) -> pd.DataFrame:
    """Pairwise DM-test p-values across models for one horizon."""
    mat = pd.DataFrame(index=sorted(fr["model"].unique()), columns=sorted(fr["model"].unique()), dtype=float)
    for i in mat.index:
        for j in mat.columns:
            if i == j:
                mat.loc[i, j] = np.nan
            else:
                e1 = fr.loc[fr["model"] == i, "err"].values
                e2 = fr.loc[fr["model"] == j, "err"].values
                # align lengths
                n = min(len(e1), len(e2))
                stat, p = _dm_test(e1[-n:], e2[-n:], h=1, power=2)
                mat.loc[i, j] = p
    return mat


# ---------- rolling feature importance (GBDT) ----------
def rolling_feature_importances_gbdt(
    feat: pd.DataFrame,
    cfg: dict,
    every: int = 20,
    min_train: int = 252,
) -> pd.DataFrame:
    """
    Train GBDT on expanding window every `every` steps; collect LightGBM feature importances.
    Returns wide frame: index = timestamp of model, columns = features.
    """
    cols = [c for c in feat.columns if c not in ("y_t+1",)]
    imp_rows = []
    idx = []
    n = len(feat)
    for t in range(min_train, n, every):
        train = feat.iloc[:t].copy()
        # small wrapper that exposes fitted model via gbdt.fit_predict_quantiles
        q = gbdt.fit_predict_quantiles(
            train, pd.DataFrame(index=train.index[-1:], data=train.iloc[-1:]),
            quantiles=(0.5,), params=cfg["models"]["gbdt"]["params"],
            return_model=True  # <- ensure your gbdt helper supports this flag
        )
        model = q["model"]
        try:
            imp = dict(zip(model.feature_name_, model.feature_importance()))
        except Exception:
            continue
        imp_rows.append([imp.get(c, 0.0) for c in cols])
        idx.append(feat.index[t-1])

    if not imp_rows:
        return pd.DataFrame()

    return pd.DataFrame(imp_rows, index=idx, columns=cols)


# ---------- SHAP (GBDT) ----------
def shap_summary_gbdt(train_feat: pd.DataFrame, cfg: dict, max_rows: int = 2000) -> Optional[pd.DataFrame]:
    """
    Compute mean(|SHAP|) per feature for the last fitted GBDT on train_feat.
    Returns DataFrame columns: feature, mean_abs_shap
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    q = gbdt.fit_predict_quantiles(
        train_feat, pd.DataFrame(index=train_feat.index[-1:], data=train_feat.iloc[-1:]),
        quantiles=(0.5,), params=cfg["models"]["gbdt"]["params"], return_model=True
    )
    model = q["model"]

    # sample to keep it light
    X = train_feat.drop(columns=["y_t+1"]).tail(max_rows).copy()
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X, check_additivity=False)
        # LightGBM regressor -> array (n, f)
        if isinstance(values, list):
            values = values[0]
        mean_abs = np.mean(np.abs(values), axis=0)
        return pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        return None


# ---------- High-level panel ----------
def diagnostics_panel_for_ticker(
    st,                     # streamlit module (passed from caller)
    prices: pd.DataFrame,
    feat: pd.DataFrame,
    models: List[str],
    horizons: Iterable[int],
    cfg: dict,
    target_type: str,
):
    """Render advanced panel inside an expander for a single ticker."""
    with st.expander("Advanced evaluation & diagnostics", expanded=False):

        # 1) Multi-horizon backtest
        st.markdown("### Walk-forward backtest (expanding window)")
        frames = backtest_multi_horizon(prices, feat, models, horizons, cfg, target_type)
        for h in horizons:
            fr = frames[h]
            if fr.empty:
                st.info(f"No backtest rows for horizon={h}.")
                continue

            # per-date errors table
            fr_show = fr.copy()
            fr_show["AE"] = fr_show["err"].abs()
            fr_show["APE%"] = (fr_show["AE"] / fr_show["true"].replace(0, np.nan)) * 100
            st.markdown(f"**H = {h}** — {len(fr):,} evaluations")
            st.dataframe(fr_show.round(3), use_container_width=True)

            # summary metrics by model
            met = (fr_show.groupby("model")
                   .agg(RMSE=("err", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                        MAE=("AE", "mean"),
                        MAPE=("APE%", "mean"))
                   .reset_index())
            st.dataframe(met.round(3), use_container_width=True)

            # DM matrix
            dm = dm_matrix_for_horizon(fr)
            st.markdown("**DM-test p-values (rows vs columns; lower = significant difference)**")
            st.dataframe(dm, use_container_width=True)

        # 2) Residuals & QQ from 1-day horizon
        h1 = frames[horizons[0]] if len(horizons) else pd.DataFrame()
        if not h1.empty:
            st.markdown("### Residual diagnostics (H=1)")
            # pick first model (or GBDT if present)
            order = ["GBDT"] + [m for m in h1["model"].unique() if m != "GBDT"]
            sel = order[0] if order[0] in h1["model"].unique() else sorted(h1["model"].unique())[0]
            res = h1.loc[h1["model"] == sel].sort_values("date")
            resid = res["err"].values

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Error-by-time — {sel}**")
                err_df = res[["date", "err"]].rename(columns={"err": "error"})
                line = alt.Chart(err_df).mark_line().encode(x="date:T", y="error:Q")
                st.altair_chart(line, use_container_width=True)
            with c2:
                st.markdown(f"**QQ plot — {sel}**")
                # Normal QQ via numpy
                r = pd.Series(resid).dropna().values
                if len(r) >= 8:
                    from scipy.stats import probplot  # if missing scipy, fall back
                    try:
                        fig, ax = plt.subplots()
                        probplot(r, dist="norm", plot=ax)
                        st.pyplot(fig, use_container_width=True)
                    except Exception:
                        # fallback manual
                        r_sorted = np.sort(r)
                        q = np.linspace(0.01, 0.99, len(r_sorted))
                        z = pd.Series(q).apply(lambda p: np.sqrt(2)*np.erfinv(2*p-1))
                        qq_df = pd.DataFrame({"theoretical": z, "sample": r_sorted})
                        st.altair_chart(alt.Chart(qq_df).mark_point().encode(x="theoretical:Q", y="sample:Q"),
                                        use_container_width=True)
                else:
                    st.info("Not enough residuals for QQ plot.")

            # Scatter actual vs predicted
            st.markdown("### Scatter — actual vs predicted (H=1)")
            scat = res.rename(columns={"yhat": "pred", "true": "actual"})
            st.altair_chart(
                alt.Chart(scat).mark_circle(size=35, opacity=0.7).encode(
                    x=alt.X("actual:Q", title="Actual"),
                    y=alt.Y("pred:Q", title="Predicted"),
                    tooltip=["date:T", "pred:Q", "actual:Q"],
                ),
                use_container_width=True,
            )

        # 3) Rolling feature importances (GBDT)
        if "GBDT" in models:
            st.markdown("### Rolling feature importances — GBDT")
            imp = rolling_feature_importances_gbdt(feat, cfg, every=20, min_train=252)
            if imp.empty:
                st.info("Could not collect feature importances (check that gbdt helper supports return_model=True).")
            else:
                heat_df = imp.reset_index().melt(id_vars="index", var_name="feature", value_name="importance")
                heat_df = heat_df.rename(columns={"index": "date"})
                heat = alt.Chart(heat_df).mark_rect().encode(
                    x=alt.X("date:T", title="Model timestamp"),
                    y=alt.Y("feature:N"),
                    color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["date:T", "feature:N", alt.Tooltip("importance:Q", format=".2f")],
                )
                st.altair_chart(heat, use_container_width=True)

            # 4) SHAP explainability (last GBDT)
            st.markdown("### SHAP explainability — last GBDT (mean |SHAP|)")
            shap_df = shap_summary_gbdt(feat.iloc[:-1].copy(), cfg, max_rows=2000)
            if shap_df is None or shap_df.empty:
                st.info("SHAP not available (install `shap`) or model/params unsupported.")
            else:
                top = shap_df.head(20)
                st.altair_chart(
                    alt.Chart(top).mark_bar().encode(
                        x=alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
                        y=alt.Y("feature:N", sort="-x"),
                        tooltip=["feature:N", alt.Tooltip("mean_abs_shap:Q", format=".3f")],
                    ),
                    use_container_width=True,
                )
