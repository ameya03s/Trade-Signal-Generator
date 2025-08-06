import yfinance as yf
import pandas as pd
import os

def get_data(ticker, start, end):
    os.makedirs("data", exist_ok=True)

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        print(f"[!] No data for {ticker}")
        return pd.DataFrame()
    
    path = f"data/{ticker}.csv"
    data.to_csv(path)
    df_raw = pd.read_csv(path, parse_dates=["Date"])
    df = df_raw.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df