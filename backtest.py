import numpy
import pandas
import joblib

# def load_predictions(x_test):
#     model = joblib.load('model/model.joblib')
#     return model.predict(x_test)

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

def calc_spending(cash, price):
    shares = calc_shares(cash, price)
    return shares * price

def do_we_have_funds(cash, min_cash, spending):
    net_funds = cash - min_cash
    if net_funds < spending:
        return False
    else:
        return True

def buy(cash, shares, price):
    amount_to_buy = calc_shares(cash, price)
    shares += amount_to_buy
    cash -= amount_to_buy * price
    return cash, shares

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
    return cash, shares

def simulate(cash, df, labels):
    shares = 0
    portfolio_values = []
    trades = []
    for day in range(len(labels)):
        label = labels.iloc[day]
        price = df.iloc[day]['Open']
        min_cash = calc_min_cash(cash)

        if label == 2:
            # buy logic
            if cash < min_cash:
                continue
            else:
                if do_we_have_funds(cash, min_cash, calc_spending(cash, price)):
                    cash, shares = buy(cash, shares, price)
                    trades.append(2)
                else:
                    continue
        elif label == 1:
            # sell logic
            if shares == 0:
                continue
            else:
                trades.append(1)
                cash, shares = sell(cash, min_cash, shares, price)
        else:
            # hold stocks
            continue

        portfolio_values.append( cash + (shares * price))
    
    return cash, shares, portfolio_values, trades
