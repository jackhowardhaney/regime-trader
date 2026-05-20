# regime-trader

A Python algorithmic trading bot that uses a Hidden Markov Model (HMM) to detect market regimes (bull / bear / volatile) and dynamically adjusts portfolio allocation using volatility-based strategies.

## Project Structure

```
regime-trader/
├── config/           # settings.yaml and credentials template
├── core/             # HMM engine, regime strategies, risk manager, signal generator
├── broker/           # Alpaca API wrapper, order executor, position tracker
├── data/             # Market data fetching and feature engineering
├── monitoring/       # Structured logging, terminal dashboard, alerts
├── backtest/         # Walk-forward backtester, performance analytics, stress tests
├── tests/            # Unit and integration tests
└── main.py           # Entry point
```

## Setup

1. Copy `.env.example` to `.env` and fill in your Alpaca API keys:
   ```
   cp .env.example .env
   ```

2. Copy `config/credentials.yaml.example` to `config/credentials.yaml` and fill in alert credentials (optional).

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

**Paper trading (live):**
```
python main.py --config config/settings.yaml --paper
```

**Backtest:**
```
python main.py --config config/settings.yaml --backtest
```

**Run tests:**
```
pytest tests/
```

## Configuration

All parameters are in `config/settings.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `broker` | Symbols, timeframe, paper vs live |
| `hmm` | Regime detection hyperparameters |
| `strategy` | Vol-based allocation thresholds |
| `risk` | Position limits, drawdown halts |
| `backtest` | Walk-forward window sizes, slippage |
| `monitoring` | Dashboard refresh, alert rate limits |
