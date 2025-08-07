import pandas as pd
import numpy as np
from data import data

FEATURES = ["sma_20", "ema_20", "log_return", "rolling_std_20", "rsi_14", 
            "lag_1","lag_5", "volume_avg_10", "zscore_close"]

# sma_20: 20-day simple moving average of Close
# rsi_14: 14-day RSI of Close, custom implementation
# log_return: log difference of consecutive Close prices
# rolling_std_10: 10-day rolling standard deviation (volatility)
# lag_1: yesterday's Close

def add_features(df):
    df["sma_20"] = df["Close"].rolling(20).mean() # .rolling() executes taking the sum of prices of the last 20 days
    df["ema_20"] = df["Close"].ewm(span=20).mean() # ewm does exponential weighting
    s = pd.Series(df["Close"].shift(1)) # lines up yesterday's close with today
    df["log_return"] = np.log(df["Close"].div(s)) # log difference of yesterday and today's close
    return df