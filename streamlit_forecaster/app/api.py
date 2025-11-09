from fastapi import FastAPI, HTTPException
import yaml
import pandas as pd
import numpy as np
from src.data import load_prices
from src.features import make_feature_frame
from src.models import prophet_model, gbdt

app = FastAPI(title="Stock Forecaster API")

cfg = yaml.safe_load(open('config.yaml'))

@app.get("/forecast")
def forecast(ticker: str = "AAPL", h: int = 1):
    if h < 1 or h > 7:
        raise HTTPException(400, detail="h must be between 1 and 7")
    prices = load_prices(ticker, cfg['data']['start'], cfg['data']['end'], cfg['data']['interval'])
    feat = make_feature_frame(prices, cfg)
    train = feat.iloc[:-h]
    future_idx = feat.index[-h:]

    p_out = prophet_model.fit_predict(train[['close']], pd.DataFrame(index=future_idx, data={'close': np.nan}))
    q_out = gbdt.fit_predict_quantiles(train, pd.DataFrame(index=future_idx, data=train.iloc[-h:]))

    out = pd.DataFrame({
        'date': future_idx,
        'prophet': p_out['yhat'].values.tolist(),
        'prophet_lo': p_out['lo'].values.tolist(),
        'prophet_hi': p_out['hi'].values.tolist(),
        'gbdt': q_out['yhat'].values.tolist(),
        'gbdt_lo': q_out['lo'].values.tolist(),
        'gbdt_hi': q_out['hi'].values.tolist(),
    })
    return out.to_dict(orient='list')
