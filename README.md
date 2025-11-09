# Stock Price Prediction (LSTM + Multi-Model Streamlit Forecaster)

This repository contains two major components:

1) **Deep-learning stock prediction with LSTM**  
2) **Full-stack Streamlit application for multi-model stock forecasting**

Both are included under the same project to demonstrate progression from
a single-model research notebook → to a production-style interactive app.

---

## ✅ 1) Highlights

### ✅ Latest — Streamlit Multi-Model Forecasting App
`/streamlit_forecaster/`

A complete, interactive forecasting system supporting multiple models:

✅ LightGBM GBDT (quantile forecasting)  
✅ Naive baseline  
✅ ARIMA  
✅ Facebook Prophet  
✅ XGBoost  
✅ LSTM integration  
✅ Interactive uncertainty bands (quantiles / confidence)  
✅ Auto feature engineering (returns, stats, MACD, RSI…)  
✅ Auto target-space inference (Price / Delta / Return)  
✅ Lightweight backtesting w/ MAE / RMSE / MAPE  
✅ Feature importance panel (GBDT gain)  
✅ Export predictions → CSV  
✅ Save run history → `outputs/runs/`  
✅ One-click HTML report export  
✅ Interactive Plot (Altair)  

> **LIVE VIEW**
- Time series plot w/ predictions + confidence bands
- Hover tooltips show: runtime, MAE, RMSE, MAPE, % change vs last
- Toggle models, log/linear scale, show/hide points, zoom

> **Tech Stack**
- Python
- Streamlit
- LightGBM
- Prophet
- XGBoost
- NumPy / Pandas / YAML
- Plotly / Altair

---

## ✅ 2) LSTM Notebook

Early exploration using a pure LSTM forecaster.

Located at:
