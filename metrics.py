"""Performance metrics for portfolio backtests."""

import pandas as pd

def calc_total_return(starting_cash, portfolio_values):
    """Compute total return relative to starting cash."""
    return (portfolio_values[-1] - starting_cash)/starting_cash

def calc_daily_return(portfolio_values):
    """Compute daily percentage returns from portfolio values list."""
    portfolio_values = pd.Series(portfolio_values)
    pct_change = portfolio_values.pct_change().dropna()
    return pct_change

def calc_sharpe_ratio(returns):
    """Compute Sharpe ratio against a fixed daily risk-free rate."""
    risk_free_rate = 0.0005
    mean_return = returns.mean()
    std_dev = returns.std()
    return (mean_return - risk_free_rate) / std_dev

def calc_win_rate(returns):
    """Fraction of positive return periods."""
    return (returns > 0).mean()