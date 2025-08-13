from api.dataset_prep import *
from train import train

def main():
    ticker = "AAPL"
    start = "2024-01-01"
    end = "2025-01-01"

    df_train, df_eval = prep_data(ticker, start, end)
    x_train, y_train, x_eval, y_eval = prep_train_set(df_train, df_eval)

    train(x_train, x_eval, y_train, y_eval)

if __name__ == "__main__":
    main()