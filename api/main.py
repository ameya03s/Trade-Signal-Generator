from api.dataset_prep import *
from train import train
from backtest import *

def main():
    ticker = "XLV"
    start = "2024-01-01"
    end = "2025-01-01"

    df_train, df_eval = prep_data(ticker, start, end)
    x_train, y_train, x_eval, y_eval = prep_train_set(df_train, df_eval)

    acc = train(x_train, y_train, x_eval, y_eval)
    print(f"{ticker}: {acc}")

    starting_cash = 100000
    df = pd.read_csv("data/processed/eval_XLV.csv")
    df = prep_prices(df)
    labels = load_predictions(x_eval)

    print(f"Cash: {starting_cash}")
    print(f"Shares: 0")
    print(y_eval)

    final_cash, shares, actions = simulate(starting_cash, df, labels)

    print(f"Cash: {final_cash}")
    print(f"Shares: {shares}")
    print(actions)



if __name__ == "__main__":
    main()