import pandas as pd
import numpy as np
from data import data

FEATURES = ["sma_20", "ema_20", "log_return", "rolling_std_20", "rsi_14", 
            "lag_1","lag_5", "volume_avg_10", "zscore_close", "atr_14"]

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
    df["rolling_std_20"] = df["Close"].rolling(20).std() # 20-day rolling standard deviation
    df['rsi_14'] = add_rsi(df)
    df["rolling_std_10"] = df["Close"].rolling(10).std() # 10-day rolling standard deviation
    df["lag_1"] = df["Close"].shift(1) # yesterday's close
    df["lag_5"] = df["Close"].shift(5) # 5 days ago
    df['volume_avg_10'] = df['Volume'].rolling(10).mean() # 10-day average volume
    df["zscore_close"] = (df["Close"] - df["sma_20"]) / df["rolling_std_20"] # z-score of close
    df["atr_14"] = get_atr(df, s) # 14-day ATR

    clean = df.dropna(subset=FEATURES)
    return clean

def add_rsi(df):
    changes = df["Close"].diff()
    gains = changes.clip(lower=0).rolling(14).mean()
    losses = changes.clip(upper=0).abs().rolling(14).mean()
    rs = gains.div(losses)
    df = 100 - (100/(1+rs))
    return df

def get_atr(df, s):
    r1 = (df["High"]-df['Low'])
    r2 = (df["High"] - s).abs()
    r3 = (df["Low"] - s).abs()
    tr = pd.concat([r1, r2, r3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean() # wilder's smoothing for ATR
    return atr