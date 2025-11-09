# --- make project root importable ---
import sys, os, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF INFO logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # disable oneDNN custom ops for reproducibility

# --- regular imports ---
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import altair as alt
from datetime import timedelta, datetime

# Plotly availability
try:
    import plotly.express as px  # noqa: F401
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    st.warning("Plotly not available — interactive plots disabled")

# quiet LightGBM logs a bit
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# data / features
from src.data import load_prices
from src.features import make_feature_frame

# models (GBDT + Naive are always present)
from src.models import gbdt, baseline
try:
    from src.models import arima, prophet_model, xgb_model, lstm
except Exception:
    arima = prophet_model = xgb_model = lstm = None

# diagnostics panel
from src.eval import diagnostics_panel_for_ticker


# ---------- small helpers ----------
def _close_series_1d(prices_df: pd.DataFrame) -> pd.Series:
    """Return 1-D numeric Series for 'close' column, robust to (n,1) shapes."""
    s = prices_df["close"] if "close" in prices_df.columns else prices_df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.to_numeric(s.squeeze(), errors="coerce")


def infer_target_type(y: pd.Series) -> str:
    """
    Heuristic to infer y-space used by models that consume y_t+1 from features.
    """
    s = pd.to_numeric(y.dropna(), errors="coerce")
    if s.empty:
        return "Price"
    q95 = s.abs().quantile(0.95)
    mean_abs = s.abs().mean()
    if q95 < 0.5:
        return "Return"
    if mean_abs < 3:
        return "Delta"
    return "Price"


def convert_to_price_space(yhat: pd.Series, last_price: float, target_type: str) -> pd.Series:
    if target_type == "Price":
        return yhat
    if target_type == "Delta":
        return last_price + yhat
    if target_type == "Return":
        return last_price * (1.0 + yhat)
    # fallback
    return last_price + yhat


def _format_pct(x) -> str:
    try:
        return f"{x:+.2f}%"
    except Exception:
        return ""


# === quick, light backtest for hover metrics (per model) ===
@st.cache_data(show_spinner=False)
def quick_backtest_metrics(
    feat: pd.DataFrame,
    prices: pd.DataFrame,
    target_type: str,
    models: list,
    horizon: int,
    n_last: int = 20,
) -> dict:
    """
    Tiny walk-forward backtest over the last `n_last` targets (1-step ahead) for the
    enabled models. Returns {model: {'MAE':..., 'RMSE':..., 'MAPE':...}}
    NOTE: kept intentionally light to avoid slowing the app. Uses 1-step horizon only.
    """
    n_last = max(5, min(n_last, 60))
    y_true, y_pred = {m: [] for m in models}, {m: [] for m in models}

    close = _close_series_1d(prices)
    X_all = feat.copy()
    idx = X_all.index
    if len(idx) < (n_last + 10):
        return {}

    for t in range(len(idx) - n_last, len(idx)):
        # training set up to t-1
        train_slice = X_all.iloc[:t].copy()
        if train_slice.empty:
            continue

        # guard: need at least 2 closes to get "last observed" before t
        hist = close.loc[:idx[t]].dropna()
        if len(hist) < 2:
            continue
        last_price = float(hist.iloc[-2])

        future_like = pd.DataFrame(index=[idx[t]], data=train_slice.iloc[-1:])  # mimic 1-step future row
        y_true_val = float(close.loc[idx[t]])

        for m in models:
            if m == "Naive":
                y_pred[m].append(last_price)
            elif m == "GBDT":
                try:
                    out = gbdt.fit_predict_quantiles(
                        train_slice, future_like, quantiles=(0.1, 0.5, 0.9), params=None
                    )
                    yhat = float(pd.Series(out["yhat"], index=future_like.index).iloc[0])
                    yhat_price = float(convert_to_price_space(pd.Series([yhat]), last_price, target_type).iloc[0])
                    y_pred[m].append(yhat_price)
                except Exception:
                    y_pred[m].append(np.nan)
            else:
                # keep light; extend here if you want more models in hover metrics
                y_pred[m].append(np.nan)

        for m in models:
            y_true[m].append(y_true_val)

    metrics = {}
    for m in models:
        yt = np.array(y_true[m], dtype="float64")
        yp = np.array(y_pred[m], dtype="float64")
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() == 0:
            continue
        mae = np.mean(np.abs(yp[mask] - yt[mask]))
        rmse = float(np.sqrt(np.mean((yp[mask] - yt[mask]) ** 2)))
        mape = float(np.mean(np.abs((yp[mask] - yt[mask]) / yt[mask])) * 100.0)
        metrics[m] = {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}
    return metrics


# ---------- Streamlit page ----------
st.set_page_config(page_title="Stock Forecaster", layout="wide")

cfg_path = os.path.join(ROOT, "config.yaml")
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("Settings")

# (NEW) multi-ticker selector
all_tickers = cfg["data"]["tickers"]
default_tickers = all_tickers[:1] if isinstance(all_tickers, list) else [all_tickers]
TICKERS = st.sidebar.multiselect("Tickers", options=all_tickers, default=default_tickers)

H = st.sidebar.number_input("Horizon (days)", min_value=1, max_value=7, value=cfg["split"]["horizon"])

# target space selector (+ auto)
st.sidebar.subheader("Target type used when training models that consume y_t+1")
target_type_choice = st.sidebar.radio(" ", ["Auto", "Price", "Delta", "Return"], index=0, horizontal=False)

# model picker
available_models = ["GBDT", "Naive"]
if arima:          available_models.append("ARIMA")
if prophet_model:  available_models.append("Prophet")
if xgb_model:      available_models.append("XGBoost")
if lstm:           available_models.append("LSTM")
MODELS = st.sidebar.multiselect("Models", available_models, default=["GBDT", "Naive"])

# plot controls
st.sidebar.subheader("Plot")
show_bands = st.sidebar.checkbox("Show confidence bands", value=True)
yscale = st.sidebar.radio("Y axis scale", ["Linear", "Log"], index=0)
zoom_days = st.sidebar.slider("Auto-zoom (last N days)", 90, 540, 180, 30)
marker_points = st.sidebar.checkbox("Show points", value=True)
DEBUG = st.sidebar.checkbox("Debug", value=False)

# -------------------- PER-TICKER RENDERER --------------------
def render_ticker_block(ticker: str):
    # ---------- DATA ----------
    prices = load_prices(ticker, cfg["data"]["start"], cfg["data"]["end"], cfg["data"]["interval"])
    feat = make_feature_frame(prices, cfg)

    # split: use all but last H for training; forecast next H dates
    if len(feat) <= H:
        st.warning("Not enough data to forecast this horizon.")
        return
    train = feat.iloc[:-H].copy()
    future_index = feat.index[-H:]

    # compute final target_type (Auto or manual)
    inferred = infer_target_type(train.get("y_t+1", pd.Series(dtype=float)))
    target_type = inferred if target_type_choice == "Auto" else target_type_choice

    # last close
    last_close = float(_close_series_1d(prices).dropna().iloc[-1])
    st.metric("Last close", f"{last_close:,.2f}")

    # ---------- FORECASTS ----------
    pred_rows = []
    runtime_map = {}
    gbdt_importance = None

    # GBDT
    if "GBDT" in MODELS:
        t0 = time.time()
        q = gbdt.fit_predict_quantiles(
            train,
            pd.DataFrame(index=future_index, data=train.iloc[-H:]),
            quantiles=tuple(cfg["models"]["gbdt"]["quantiles"]),
            params=cfg["models"]["gbdt"]["params"],
        )
        runtime_map["GBDT"] = float(q.get("runtime_s", time.time() - t0))
        fi = q.get("feature_importance", None)
        if isinstance(fi, (pd.Series, pd.DataFrame)):
            gbdt_importance = fi

        last_price_train = float(_close_series_1d(prices).iloc[:-H].dropna().iloc[-1]) if H > 0 else last_close
        yhat = convert_to_price_space(pd.Series(q["yhat"], index=future_index), last_price_train, target_type)
        lo   = convert_to_price_space(pd.Series(q["lo"],   index=future_index), last_price_train, target_type)
        hi   = convert_to_price_space(pd.Series(q["hi"],   index=future_index), last_price_train, target_type)
        for d in future_index:
            pred_rows.append((d, "GBDT", float(yhat.loc[d]), float(lo.loc[d]), float(hi.loc[d])))

    # Naive
    if "Naive" in MODELS:
        t0 = time.time()
        close_ser = _close_series_1d(prices)
        close_train = close_ser.iloc[:-H] if H > 0 else close_ser
        last_val = float(close_train.dropna().iloc[-1])
        for d in future_index:
            pred_rows.append((d, "Naive", float(last_val), np.nan, np.nan))
        runtime_map["Naive"] = float(time.time() - t0)

    # ARIMA
    if arima and "ARIMA" in MODELS:
        try:
            t0 = time.time()
            a_out = arima.fit_predict(train["y_t+1"].dropna(), steps=H)
            runtime_map["ARIMA"] = float(time.time() - t0)
            last_price_train = float(_close_series_1d(prices).iloc[:-H].dropna().iloc[-1]) if H > 0 else last_close
            yhat = convert_to_price_space(pd.Series(a_out["yhat"], index=future_index), last_price_train, target_type)
            lo   = convert_to_price_space(pd.Series(a_out["lo"],   index=future_index), last_price_train, target_type)
            hi   = convert_to_price_space(pd.Series(a_out["hi"],   index=future_index), last_price_train, target_type)
            for d in future_index:
                pred_rows.append((d, "ARIMA", float(yhat.loc[d]), float(lo.loc[d]), float(hi.loc[d])))
        except Exception as e:
            st.info(f"ARIMA skipped: {e}")

    # Prophet
    if prophet_model and "Prophet" in MODELS:
        try:
            t0 = time.time()
            p_out = prophet_model.fit_predict(prices[["close"]].iloc[:-H], pd.DataFrame(index=future_index))
            runtime_map["Prophet"] = float(time.time() - t0)
            yhat = pd.Series(p_out["yhat"], index=future_index)
            lo   = pd.Series(p_out["lo"],   index=future_index)
            hi   = pd.Series(p_out["hi"],   index=future_index)
            for d in future_index:
                pred_rows.append((d, "Prophet", float(yhat.loc[d]), float(lo.loc[d]), float(hi.loc[d])))
        except Exception as e:
            st.info(f"Prophet skipped: {e}")

    # XGBoost
    if xgb_model and "XGBoost" in MODELS:
        try:
            t0 = time.time()
            x_out = xgb_model.fit_predict(train, pd.DataFrame(index=future_index, data=train.iloc[-H:]))
            runtime_map["XGBoost"] = float(time.time() - t0)
            last_price_train = float(_close_series_1d(prices).iloc[:-H].dropna().iloc[-1]) if H > 0 else last_close
            yhat = convert_to_price_space(pd.Series(x_out["yhat"], index=future_index), last_price_train, target_type)
            for d in future_index:
                pred_rows.append((d, "XGBoost", float(yhat.loc[d]), np.nan, np.nan))
        except Exception as e:
            st.info(f"XGBoost skipped: {e}")

    # LSTM
    if lstm and "LSTM" in MODELS:
        try:
            t0 = time.time()
            l_out = lstm.fit_predict(train["y_t+1"].dropna(), steps=H)
            runtime_map["LSTM"] = float(time.time() - t0)
            last_price_train = float(_close_series_1d(prices).iloc[:-H].dropna().iloc[-1]) if H > 0 else last_close
            yhat = convert_to_price_space(pd.Series(l_out["yhat"], index=future_index), last_price_train, target_type)
            for d in future_index:
                pred_rows.append((d, "LSTM", float(yhat.loc[d]), np.nan, np.nan))
        except Exception as e:
            st.info(f"LSTM skipped: {e}")

    pred_df_long = (
        pd.DataFrame(pred_rows, columns=["date", "model", "yhat", "lo", "hi"])
          .set_index("date")
          .sort_index()
    )

    # relative change vs last close
    if not pred_df_long.empty:
        pred_df_long = pred_df_long.assign(rel_pct=(pred_df_long["yhat"] - last_close) / last_close * 100.0)

    # backtest metrics (hover)
    bt_metrics = quick_backtest_metrics(feat, prices, target_type, MODELS, H, n_last=20)

    # table-friendly
    tbl = (
        pred_df_long.reset_index()
        .rename(columns={"yhat": "Point", "lo": "PI_lo", "hi": "PI_hi", "rel_pct": "Rel_% from last"})
        .sort_values(["date", "model"])
    )
    if "model" in tbl.columns:
        # per-model runtime
        tbl["runtime_s"] = tbl["model"].map(runtime_map).round(3)

    # persist per-run CSV
    run_dir = os.path.join(ROOT, "outputs", "runs")
    os.makedirs(run_dir, exist_ok=True)
    run_path = os.path.join(run_dir, f"{ticker}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{H}d.csv")
    try:
        tbl.to_csv(run_path, index=False)
    except Exception:
        pass

    # ---------- PLOTTING ----------
    st.subheader(f"Next {H} days forecast — {ticker}")

    hist_all = prices[["close"]].copy()
    if zoom_days:
        cutoff = hist_all.index.max() - timedelta(days=int(zoom_days))
        hist = hist_all.loc[hist_all.index >= cutoff]
    else:
        hist = hist_all

    hist_df = hist.reset_index().rename(columns={"Date": "date", "Close": "close"})
    hist_df.columns = ["date", "close"]

    # History line
    hist_line = alt.Chart(hist_df).mark_line(size=2.2).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y(
            "close:Q",
            title="close, PI_lo, PI_hi, yhat",
            scale=alt.Scale(type="log") if yscale == "Log" else alt.Scale(type="linear"),
        ),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=".2f", title=ticker)],
    ).properties(height=360)

    layers = [hist_line]

    # Forecast layers
    if not pred_df_long.empty:
        def _hover_text(row):
            m = row["model"]
            bt = bt_metrics.get(m, {})
            if not bt:
                return f"runtime: {runtime_map.get(m, 0):.3f}s"
            return (f"runtime: {runtime_map.get(m, 0):.3f}s | "
                    f"MAE: {bt.get('MAE', np.nan):.3f} | "
                    f"RMSE: {bt.get('RMSE', np.nan):.3f} | "
                    f"MAPE: {bt.get('MAPE', np.nan):.2f}%")

        fcst = pred_df_long.reset_index().copy()
        fcst["hover"] = fcst.apply(_hover_text, axis=1)
        fcst["rel_str"] = fcst["rel_pct"].map(_format_pct)

        if show_bands:
            band_chart = alt.Chart(fcst.dropna(subset=["lo", "hi"])).mark_area(opacity=0.25).encode(
                x="date:T",
                y="lo:Q",
                y2="hi:Q",
                color=alt.Color("model:N", legend=None),
            )
            layers.append(band_chart)

        line_chart = alt.Chart(fcst).mark_line(size=2.4, strokeDash=[6, 3]).encode(
            x="date:T",
            y=alt.Y(
                "yhat:Q",
                scale=alt.Scale(type="log") if yscale == "Log" else alt.Scale(type="linear"),
            ),
            color=alt.Color("model:N", legend=alt.Legend(title="Models", orient="top")),
            tooltip=[
                alt.Tooltip("date:T"),
                "model:N",
                alt.Tooltip("yhat:Q", format=".2f", title="Point"),
                alt.Tooltip("lo:Q", format=".2f", title="PI_lo"),
                alt.Tooltip("hi:Q", format=".2f", title="PI_hi"),
                alt.Tooltip("rel_str:N", title="Δ vs last"),
                alt.Tooltip("hover:N", title="Backtest / runtime"),
            ],
        )
        layers.append(line_chart)

        if marker_points:
            point_chart = alt.Chart(fcst).mark_point(size=45).encode(
                x="date:T",
                y="yhat:Q",
                color="model:N",
                tooltip=[
                    alt.Tooltip("date:T"),
                    "model:N",
                    alt.Tooltip("yhat:Q", format=".2f", title="Point"),
                    alt.Tooltip("lo:Q", format=".2f", title="PI_lo"),
                    alt.Tooltip("hi:Q", format=".2f", title="PI_hi"),
                    alt.Tooltip("rel_str:N", title="Δ vs last"),
                    alt.Tooltip("hover:N", title="Backtest / runtime"),
                ],
            )
            layers.append(point_chart)

        # vertical line at last historical timestamp
        vline = alt.Chart(pd.DataFrame({"date": [hist.index.max()]})).mark_rule(strokeDash=[2, 2]).encode(x="date:T")
        layers.append(vline)

    chart = alt.layer(*layers)
    st.altair_chart(chart.interactive(bind_y=False), use_container_width=True)

    # ---------- TABLE ----------
    st.caption("Uncertainty shown via model-specific intervals (GBDT quantiles; ARIMA/Prophet confidence).")
    if "Rel_% from last" in tbl.columns:
        tbl["Rel_% from last"] = tbl["Rel_% from last"].map(lambda x: f"{x:.2f}%")
    st.dataframe(tbl, use_container_width=True)

    st.caption(f"Run file saved to: `outputs/runs/{os.path.basename(run_path)}`")
    st.download_button(
        "Download CSV",
        data=tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_forecast_{H}d.csv",
        mime="text/csv",
    )

    # Feature importance (GBDT)
    if gbdt_importance is not None and len(getattr(gbdt_importance, "index", [])) > 0:
        st.subheader("GBDT — Feature importance (gain)")
        fi = gbdt_importance
        if isinstance(fi, pd.Series):
            fi = fi.reset_index()
            fi.columns = ["feature", "gain"]
            fi = fi.sort_values("gain", ascending=False)
        st.dataframe(fi, use_container_width=True)

    # Quick HTML report
    if st.button("Export HTML report", key=f"export_html_{ticker}"):
        html = f"""
        <html><head><meta charset='utf-8'><title>{ticker} report</title></head>
        <body>
          <h1>{ticker} — {H}-day Forecast</h1>
          <p><b>Target type:</b> {target_type}</p>
          <p><b>Last close:</b> {last_close:,.2f}</p>
          {tbl.to_html(index=False)}
          <p><i>Generated by Stock Forecaster.</i></p>
        </body></html>
        """
        out = os.path.join(ROOT, "outputs", f"{ticker}_report.html")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        st.success(f"Saved {out}. Open in a browser and use Print → Save as PDF.")

    # Diagnostics
    with st.expander("Advanced evaluation & diagnostics", expanded=False):
        diagnostics_panel_for_ticker(
            st=st,
            prices=prices,
            feat=feat,
            models=MODELS,
            horizons=[1, 3, 5],
            cfg=cfg,
            target_type=target_type,
        )

    # Debug extras
    if DEBUG:
        cols = [c for c in train.columns if c in ("close", "y_t+1")]
        st.write("Tail of training targets:")
        st.write(train[cols].tail())
        st.write(f"Inferred target type: {inferred} | Used: {target_type}")
        st.write("Per-model runtime (s):", runtime_map)
        st.write("Backtest metrics:", bt_metrics)


# -------------------- RENDER TABS FOR SELECTED TICKERS --------------------
if not TICKERS:
    st.info("Pick at least one ticker in the sidebar to start.")
else:
    tabs = st.tabs(TICKERS)
    for tkr, tab in zip(TICKERS, tabs):
        with tab:
            st.markdown(f"### {tkr}")
            render_ticker_block(tkr)
