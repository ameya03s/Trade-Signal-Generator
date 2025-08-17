import numpy as np
import pandas as pd

def calc_total_return(starting_cash, portfolio_values):
    return (portfolio_values[-1] - starting_cash)/starting_cash

def calc_daily_return(portfolio_values):
    portfolio_values = pd.Series(portfolio_values)
    pct_change = portfolio_values.pct_change().dropna()
    return pct_change

def calc_sharpe_ratio(returns):
    risk_free_rate = 0.0005
    mean_return = returns.mean()
    std_dev = returns.std()
    return (mean_return - risk_free_rate) / std_dev

def calc_win_rate(returns):
    return (returns > 0).mean()