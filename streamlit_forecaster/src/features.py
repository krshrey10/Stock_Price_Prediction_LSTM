import pandas as pd
import numpy as np

def _rsi(close: pd.Series, period: int = 14):
    delta = close.diff()
    up = (delta.clip(lower=0)).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series, f=12, s=26, signal=9):
    ema_f = close.ewm(span=f, adjust=False).mean()
    ema_s = close.ewm(span=s, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def make_feature_frame(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = prices.copy()

    # core features (keep 'close'!)
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['roll_mean_5'] = df['close'].rolling(5).mean()
    df['roll_std_5'] = df['close'].rolling(5).std()
    df['rsi_14'] = _rsi(df['close'], 14)
    macd, sig, hist = _macd(df['close'])
    df['macd'] = macd
    df['macd_sig'] = sig
    df['macd_hist'] = hist

    # targets in *price units*
    df['y_t+1'] = df['close'].shift(-1)
    df['y7']    = df['close'].shift(-7)

    return df.dropna()
