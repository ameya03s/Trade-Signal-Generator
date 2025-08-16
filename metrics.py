import numpy
import pandas

def calc_total_return(starting_cash, portfolio_values):
    return (portfolio_values[-1] - starting_cash)/starting_cash

def calc_sharpe_ratio(total_return):
    return
