from data.data import get_data
from features.features import add_features

def main():
    df = get_data("AAPL", "2024-01-01", "2025-01-01")
    df = add_features(df)
    print(df.head())

if __name__ == "__main__":
    main()