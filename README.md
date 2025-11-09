## 📈 Stock Price Prediction
### (LSTM + Multi-Model Streamlit Forecaster)

This repository contains two major components:

Deep-learning stock prediction using LSTM

Full-stack Streamlit application for multi-model stock forecasting

Together, they demonstrate progression from a single-model research prototype → to a production-style interactive forecasting app.

---

### ✅ 1) Multi-Model Forecasting Web App (Latest)

Path: streamlit_forecaster/

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

---

### 🖥️ Live Features

Time-series plot w/ predicted values + confidence bands

Hover tooltips with runtime + backtest metrics

Toggle:

Models

Log/linear scale

Show/hide points

Time-window zoom

### 🛠️ Tech Stack

Python

Streamlit

LightGBM

Prophet

XGBoost

NumPy / Pandas / YAML

Plotly / Altair

---

### ✅ 2) LSTM Stock Prediction Notebook

## Path:
Stock_Price_Prediction_LSTM.ipynb

## Features:

Yahoo Finance data ingestion

Data preprocessing: scaling + windowing

LSTM model architecture

Forecast visualization

RMSE evaluation

Saved model weights (.h5)

Saved scaler (.pkl)

This notebook served as the foundation before expanding into a multi-model forecasting system.

---

## ✅ Folder Structure

.
├── streamlit_forecaster/
│ ├── app/
│ ├── src/
│ ├── config.yaml
│ ├── requirements.txt
│ └── README_streamlit.md
│
├── Stock_Price_Prediction_LSTM.ipynb
├── lstm_stock_predictor.h5
├── scaler.pkl
└── README.md (this file)

---

## ✅ Running Streamlit App

### 1) Create venv
bash
python -m venv .venv
### 2) Activate
Windows:.\.venv\Scripts\activate
### 3) Install dependencies
pip install -r streamlit_forecaster/requirements.txt
### 4) Run
streamlit run streamlit_forecaster/app/streamlit_app.py

---

## ✅ Screenshots
<img width="1905" height="918" alt="image" src="https://github.com/user-attachments/assets/c312f963-76d8-4f87-a0b8-1e6c093f772b" />
<img width="1897" height="841" alt="image" src="https://github.com/user-attachments/assets/ebb8ad03-a4eb-4a7b-9182-cbd02db53879" />

---

## ✅ Future Work

Add transformer-based prediction

Add multi-asset portfolio analytics

Online learning / continual training

Model explainability → SHAP

---

## ✅ Author
Shreya K R
Data Science student passionate about time-series forecasting, NLP, and full-stack ML.

---

## ⭐ If you like this project

Give the repository a star ⭐ on GitHub—it helps a lot!
