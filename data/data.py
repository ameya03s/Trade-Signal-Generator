import yfinance as yf
import pandas as pd
import os

def get_data(ticker, start, end):
    os.makedirs("data", exist_ok=True)
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
    data.to_csv(path)
    return pd.read_csv(path, parse_dates=["Date"])