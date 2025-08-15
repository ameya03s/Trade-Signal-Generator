import numpy as np
import pandas as pd
import joblib

def load_predictions(x_test):
    model = joblib.load('model/model.joblib')
    return model.predict(x_test)

def prep_prices(df):
    df = df.shift(-1)
    df = df.dropna()
    return df

def get_price(df, date):
    return df.loc[date]['Open']

def calc_min_cash(cash):
    return 0.1 * cash

def calc_shares(cash, price):
    spending = float(0.25 * cash)
    num_shares = int(spending/price) # rounded down to ensure we're not overspending
    return num_shares

def buy(cash, shares, price):
    amount_to_buy = calc_shares(cash, price)
    shares += amount_to_buy
    cash -= amount_to_buy * price

def num_to_sell(min_cash, cash, shares, price):
    if cash > min_cash:
        return int(0.15 * shares)
    else:
        cash_needed = float(min_cash - cash)
        return int(cash_needed/price)

def sell(cash, min_cash, shares, price):
    shares_to_sell = num_to_sell(min_cash, cash, shares, price)
    shares -= shares_to_sell
    cash += shares_to_sell * price

def simulate(cash, df, labels):
    shares = 0
    for day in range(len(labels)):
        label = labels.iloc[day]
        price = df.iloc[day]['Open']
        min_cash = calc_min_cash(cash)

        if label == 2:
            # buy logic
            if cash < min_cash:
                continue
            else:
                buy(cash, shares, price)
        elif label == 1:
            # sell logic
            if shares == 0:
                continue
            else:
                sell(cash, min_cash, shares, price)
        else:
            # hold stocks
            continue
        # TODO: RUN TESTS ON THIS RIGHT AFTER OPENING
