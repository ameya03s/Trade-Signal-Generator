from data.data import get_data
import pandas as pd

def get_dataset():
    aapl = get_data('AAPL', '2020-01-01', '2023-12-31')
    aapl['ticker'] = 'AAPL'
    meta = get_data('META', '2020-01-01', '2023-12-31')
    meta['ticker'] = 'META'
    nvda = get_data('NVDA', '2020-01-01', '2023-12-31')
    nvda['ticker'] = 'NVDA'
    tsla = get_data('TSLA', '2020-01-01', '2023-12-31')
    tsla['ticker'] = 'TSLA'
    googl = get_data('GOOGL', '2020-01-01', '2023-12-31')
    googl['ticker'] = 'GOOGL'
    msft = get_data('MSFT', '2020-01-01', '2023-12-31')
    msft['ticker'] = 'MSFT'
    xlv = get_data('XLV', '2020-01-01', '2023-12-31')
    xlv['ticker'] = 'XLV'
    amd = get_data('AMD', '2020-01-01', '2023-12-31')
    amd['ticker'] = 'AMD'
    amzn = get_data('AMZN', '2020-01-01', '2023-12-31')
    amzn['ticker'] = 'AMZN'

    df = pd.concat([aapl, meta, nvda, tsla, googl, msft, xlv, amd, amzn], ignore_index=True)
    return df

