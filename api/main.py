from data.data import get_data
from features.features import add_features
import os

def main():
    ticker = "AAPL"
    start = "2024-01-01"
    end = "2025-01-01"

    df = get_data(ticker, start, end)
    df = add_features(df)

    os.makedirs("data/processed", exist_ok=True)
    processed_path = f"data/processed/{ticker}_{start}_{end}.csv"
    df.to_csv(processed_path, index=False)

    print(df)

if __name__ == "__main__":
    main()