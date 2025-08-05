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
    return pd.read_csv(path)