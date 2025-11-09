import pandas as pd
import yfinance as yf

def load_prices(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df = df.rename(columns=str.lower)
    df = df[['close']].rename(columns={'close': 'close'})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.dropna()
    return df
