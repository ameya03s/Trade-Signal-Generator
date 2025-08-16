import pandas
import numpy as np

# price column: "Close"
# atr column: "atr_14" (must already be in df)
# horizon: 1 (tomorrow vs today)
# k (threshold multiplier): leave as a function arg; you’ll sweep on train only

def sanity_checks(df, price, atr, k, horizon):
    assert not df.empty, print(f"DataFrame is EMPTY - No labels generated")
    assert price in df.columns, print(f"Error: Closing price not found")
    assert atr in df.columns, print(f"Error: ATR not found")
    assert k > 0, print("Threshold invalid")
    assert horizon >= 1, print("Future Returns invalid")
    assert (df[price] > 0).all(), print("Negative price found in data - Data is invalid")

def add_labels(df, price, atr, k, horizon):
    sanity_checks(df, price, atr, k , horizon)
    future_log_return = df['log_return'].shift(-1)
    atr_log = np.log(1+(df[atr]/df[price]))
    threshold = k * atr_log
    significant = np.abs(future_log_return) >= threshold
    pos = future_log_return > 0
    labels = np.where(~significant, 0, np.where(pos, 2, 1))

    return labels