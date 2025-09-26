"""Simple rule-based backtester to evaluate label-driven trading actions.

Labels convention:
  2 = buy, 1 = sell, 0 = hold
"""

def prep_prices(df):
    """Shift and drop NaNs so that `Open` represents next session's trade price."""
    df = df.shift(-1)
    df = df.dropna()
    return df

def get_price(df, date):
    """Get the `Open` price for a specific date index."""
    return df.loc[date]['Open']

def calc_min_cash(cash):
    """Reserve 10% of cash as minimum liquidity buffer."""
    return 0.1 * cash

def calc_shares(cash, price):
    """Compute a position size targeting 25% of current cash, integer shares."""
    spending = float(0.25 * cash)
    num_shares = int(spending/price) # rounded down to ensure we're not overspending
    return num_shares

def calc_spending(cash, price):
    """Dollar amount spent when buying computed shares at price."""
    shares = calc_shares(cash, price)
    return shares * price

def do_we_have_funds(cash, min_cash, spending):
    """Return True if cash beyond buffer covers intended spending."""
    net_funds = cash - min_cash
    if net_funds < spending:
        return False
    else:
        return True

def buy(cash, shares, price):
    """Buy shares per sizing rule and update cash/position."""
    amount_to_buy = calc_shares(cash, price)
    shares += amount_to_buy
    cash -= amount_to_buy * price
    return cash, shares

def num_to_sell(min_cash, cash, shares, price):
    """Determine number of shares to sell to restore liquidity or take profits."""
    if cash > min_cash:
        return int(0.15 * shares)
    else:
        cash_needed = float(min_cash - cash)
        return int(cash_needed/price)

def sell(cash, min_cash, shares, price):
    """Sell shares per sizing rule and update cash/position."""
    shares_to_sell = num_to_sell(min_cash, cash, shares, price)
    shares -= shares_to_sell
    cash += shares_to_sell * price
    return cash, shares

def simulate(cash, df, labels):
    """Simulate trading over label sequence.

    Args:
        cash: Starting cash balance.
        df: Price DataFrame (expects `Open`).
        labels: Series of action labels aligned to `df` rows.

    Returns:
        final_cash, shares, portfolio_values (list), trades (list of actions)
    """
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
                trades.append(0)
                continue
            else:
                if do_we_have_funds(cash, min_cash, calc_spending(cash, price)):
                    cash, shares = buy(cash, shares, price)
                    trades.append(2)
                else:
                    trades.append(0)
                    continue
        elif label == 1:
            # sell logic
            if shares == 0:
                trades.append(0)
                continue
            else:
                trades.append(1)
                cash, shares = sell(cash, min_cash, shares, price)
        else:
            # hold stocks
            trades.append(0)
            continue

        portfolio_values.append( cash + (shares * price))
    
    return cash, shares, portfolio_values, trades
