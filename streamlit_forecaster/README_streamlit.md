# Stock Price Forecaster — Minimal, Rigorous Baselines

**Goals:** walk-forward backtests, honest baselines, uncertainty bands, and a clean app/API.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_backtest.py --ticker AAPL --horizon 1
streamlit run app/streamlit_app.py
# API
uvicorn app.api:app --reload
```

## Methods

- **Baselines:** Naïve (last value), MA(5)
- **Classical:** AutoARIMA, Prophet (yearly/weekly seasonality)
- **ML/DL:** LightGBM (quantile) with engineered features; LSTM (tiny demo)

## Evaluation

- Walk-forward expanding window (`min_train_period`, `step`) with t+1 (or 7-day) target.
- Metrics: MAE, RMSE, MAPE per fold.
- **Diebold–Mariano test**: each model vs Prophet (MSE loss) across out-of-sample forecasts.

## Uncertainty

- Prophet 80% intervals; LightGBM quantile regression (10/50/90%). Displayed in Streamlit.
