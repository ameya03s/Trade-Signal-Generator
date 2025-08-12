from data.data import get_data
from features.features import get_feature_list, add_features
from labeler import add_labels
from train import load_data,data_split, prep_features_labels, train
import os

def main():
    ticker = "AAPL"
    start = "2024-01-01"
    end = "2025-01-01"

    df = get_data(ticker, start, end)
    df = add_features(df)
    df['labels'] = add_labels(df, "Close", "atr_14", 0.5, 1)

    os.makedirs("data/processed", exist_ok=True)
    processed_path = f"data/processed/{ticker}_{start}_{end}.csv"
    df.to_csv(processed_path, index=False)

    load_df = load_data(processed_path)
    df_train, df_eval = data_split(load_df)
    df_train = df_train.dropna(subset=get_feature_list() + ["labels"])
    df_eval = df_eval.dropna(subset=get_feature_list()+['labels'])
    x_train, y_train = prep_features_labels(df_train, get_feature_list())
    x_eval, y_eval = prep_features_labels(df_eval, get_feature_list())

    train(x_train, x_eval, y_train, y_eval)

if __name__ == "__main__":
    main()