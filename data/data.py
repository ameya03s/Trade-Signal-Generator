"""Helpers to download single-ticker OHLCV data and persist locally."""

import yfinance as yf
import pandas as pd
import os

def get_data(ticker, start, end):
    """Download OHLCV data for a ticker and return a cleaned DataFrame.

    Persists a CSV under `data/raw/{ticker}.csv` for reproducibility.

    Args:
        ticker: Symbol, e.g. "AAPL".
        start: ISO date string inclusive start.
        end: ISO date string exclusive end.

    Returns:
        DataFrame indexed by date with OHLCV columns.
    """
    os.makedirs("data", exist_ok=True) # check to see if this directory exists
    os.makedirs("data/raw", exist_ok=True)

    data = yf.download(ticker, start=start, end=end)
    data = data.xs(ticker, level=1, axis=1) # flattens ticker data
    data.columns.name = None # fixes indexing within columns
    # drops all NaN values from these columns
    data = data.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    if data.empty:
        print(f"[!] No data for {ticker}")
        return pd.DataFrame()
    
    path = f"data/raw/{ticker}.csv"
    data.to_csv(path) # saves a csv in that path to be reloaded later
    return pd.read_csv(path, parse_dates=["Date"])