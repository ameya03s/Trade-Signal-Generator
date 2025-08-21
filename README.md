
---

## Setup

- Python 3.10+
- Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

Run the end-to-end pipeline (defaults in `api/main.py`: `ticker=TSLA`, `start=2024-01-01`, `end=2025-01-01`):

```bash
python -m api.main
# or
python api/main.py
```

What it does:
1. Builds a multi-ticker training set (AAPL, META, NVDA, TSLA, GOOGL, MSFT, XLV, AMD, AMZN; 2020-01-01 to 2023-12-31).
2. Creates an eval set for your chosen `ticker`/date range.
3. Engineers features and 3-class labels on both sets.
4. Trains an XGBoost classifier and prints eval accuracy.
5. Runs a simple backtest on the eval period and prints:
   - final cash, shares, actions count, final portfolio value
   - total return, Sharpe ratio, win rate

---

## Data pipeline

- Raw data fetch: `data/data.py:get_data(ticker, start, end)` uses `yfinance`, saves to `data/raw/{ticker}.csv`. Requires columns: `Open, High, Low, Close, Volume`.
- Train dataset: `api/dataset_prep.get_dataset()` concatenates 9 tickers for 2020–2023.
- Features: `features/features.py:add_features(df)` adds:
  - `sma_20`, `ema_20`, `log_return`, `rolling_std_20`, `rsi_14`
  - `lag_1`, `lag_5`, `volume_avg_10`, `zscore_close`, `atr_14`
  - The model’s feature set is defined by `get_feature_list()`.
- Labels: `labeler.add_labels(df, price="Close", atr="atr_14", k=0.5, horizon=1)`:
  - Compute future log return (t+1 vs t)
  - Compute ATR in log space `atr_log = log(1 + ATR/Close)`
  - If `|future_log_return| < k * atr_log` → `0` (hold)
  - Else positive → `2` (buy); negative → `1` (sell)

---

## Model training

- `train.train(x_train, y_train, x_eval, y_eval)`:
  - `xgboost.XGBClassifier(objective="multi:softmax", n_estimators=100, max_depth=3, learning_rate=0.1)`
  - Fits on the multi-ticker train set, reports accuracy on eval set
  - Model saving is currently commented out; to persist:
    - Uncomment in `train.py`:
      ```python
      os.makedirs("model", exist_ok=True)
      joblib.dump(model, 'model/model.joblib')
      ```
    - Then load with `joblib.load('model/model.joblib')`

---

## Backtesting

- Inputs: `backtest.simulate(cash, df, labels)`
  - Use `backtest.prep_prices(df)` to shift to next-day pricing and drop NaNs
  - Uses next-day `Open` for trade execution
- Positioning and cash rules:
  - Minimum cash buffer: `min_cash = 10% of current cash`
  - Buy (label=2): attempt to spend `25% of current cash` if `cash - min_cash` is sufficient
  - Sell (label=1): sell `15% of current shares`, or more if needed to restore `min_cash`
  - Hold (label=0): no trade
- Outputs: `(final_cash, shares, portfolio_values, trades)`

---

## Metrics

- `metrics.calc_total_return(starting_cash, portfolio_values)`
- `metrics.calc_daily_return(portfolio_values)` (pct changes)
- `metrics.calc_sharpe_ratio(returns)` (uses a small fixed risk-free rate)
- `metrics.calc_win_rate(returns)` (fraction of positive daily returns)

---

## Customization

- Change eval universe/dates in `api/main.py`:
  ```python
  ticker = "MSFT"
  start = "2023-01-01"
  end = "2024-01-01"
  ```
- Adjust label sensitivity (threshold `k`, horizon) in `api/dataset_prep.py`:
  ```python
  df_eval['labels'] = add_labels(df_eval, "Close", "atr_14", k=0.75, horizon=1)
  ```
- Tweak features in `features/features.py` and ensure they’re included in `FEATURES` via `get_feature_list()`.

---

## Notes and limitations

- This is a simple educational/sandbox pipeline; no transaction costs, slippage, or borrow/interest modeling.
- No train/eval time split beyond ticker/date separation; consider adding rolling CV.
- `backtest` executes at next-day open and uses deterministic rules; not a broker-grade simulator.
- Make sure `FEATURES` matches the columns you actually compute.

---

## License

Add your preferred license here (e.g., MIT).