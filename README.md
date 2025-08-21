# Trade Signal Generator

Generate short-horizon trading signals from daily OHLCV data, train a multi-class classifier, and backtest a simple execution strategy with clear, explainable metrics.

## Why this project stands out
- Feature-engineered, label-driven pipeline that is easy to reason about end-to-end
- Practical, readable implementation that demonstrates disciplined Python, data wrangling, and ML for time series
- Clear separation of concerns: data → features → labels → model → backtest → metrics

## Highlights
- Multi-asset training set (AAPL, META, NVDA, TSLA, GOOGL, MSFT, XLV, AMD, AMZN)
- Robust feature set: SMA/EMA, log returns, volatility, RSI, volume trends, ATR, lags, z-score
- Labeling via ATR-scaled future log returns (3 classes: hold, sell, buy)
- XGBoost classifier with multi:softmax objective
- Simple backtest that sizes positions conservatively and reports total return, Sharpe, win rate

## Tech stack
- Python, Pandas, NumPy, XGBoost
- yfinance (market data sourcing)

## Setup
- Python 3.10+
- Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
Run the end-to-end pipeline (defaults in `api/main.py`: `ticker=TSLA`, `start=2024-01-01`, `end=2025-01-01`):
```bash
python -m api.main
# or
python api/main.py
```

What happens:
1) Build a 9-ticker training set (2020–2023)
2) Create an eval set for your chosen `ticker`/dates
3) Engineer features and 3-class labels
4) Train XGBoost and print eval accuracy
5) Backtest the eval period and print summary metrics

## Pipeline overview
- Data: `data/data.py:get_data` uses yfinance and caches CSVs in `data/raw/`
- Train set: `api/dataset_prep.get_dataset()` concatenates 9 tickers
- Features: `features/features.py:add_features` builds the feature list defined in `FEATURES`
- Labels: `labeler.add_labels` uses ATR-adjusted thresholds on future log returns
- Model: `train.train` fits an `xgboost.XGBClassifier` and reports accuracy
- Backtest: `backtest.simulate` executes next-day-open rules and tracks portfolio
- Metrics: `metrics.py` reports total return, Sharpe, and win rate

## Labeling details (3-class)
Using next-day log return and ATR in log space:
- If `|future_log_return| < k * log(1 + ATR/Close)` → class 0 (hold)
- Else positive → class 2 (buy); negative → class 1 (sell)
- Tunable sensitivity via `k` (default 0.5) and `horizon` (default 1)

## Backtest rules (execution at next-day Open)
- Maintain a 10% cash buffer; buy only if buffer maintained
- Buy: target spend 25% of current cash when signal=buy
- Sell: reduce 15% of current shares, or more to restore buffer when signal=sell

## Customize
- Switch eval ticker/dates in `api/main.py` (e.g., MSFT, 2023–2024)
- Adjust label sensitivity `k` and `horizon` in `api/dataset_prep.py`
- Add/remove features in `features/features.py` and `FEATURES`

## Repository structure
```
api/
  main.py            # Orchestrates: data → features → labels → train → backtest → metrics
  dataset_prep.py    # Dataset builders for train/eval
data/
  data.py            # yfinance download + CSV cache
  ticker_data.py     # Multi-ticker train set assembly
features/
  features.py        # Feature engineering
labeler.py           # Label creation via ATR-scaled thresholds
train.py             # XGBoost training + eval accuracy
backtest.py          # Rule-based trading simulator
metrics.py           # Return metrics (Sharpe, win rate, total return)
```

## Design choices
- Favor clarity over cleverness: explicit feature list and labeling function
- Avoids overfitting tricks: simple, strong baseline with XGBoost
- Reproducible data sourcing with local CSV cache

## Notes & limitations
- No transaction costs/slippage; deterministic rules for educational purposes
- Time split is ticker/date based; consider adding rolling CV for production research
- Model persistence intentionally omitted to keep the repo focused on pipeline clarity

## License
Add your preferred license here (e.g., MIT).