# Forex RL Trader

A reinforcement learning trading agent for forex, built with a custom Gymnasium environment backed by a realistic trade simulator, MetaTrader 5 for historical data, and Stable-Baselines3 for training.

---

## Overview

The agent observes a sliding window of OHLCV candles plus its current trade state, then decides to Buy, Sell, or Hold on a single forex pair. All training is done on historical data through a simulator that models slippage, spread, stop-loss / take-profit breaches, and position lifecycle — so the agent learns from realistic P&L, not idealised fills.

```
MT5 (data) → Preprocessor → TradingEnv (Gymnasium) → SB3 Agent (PPO)
                                   ↑
                           TradeSimulator
```

---

## Project Structure

```
forex_rl_trader/
├── data/
│   ├── raw/                  # Raw OHLCV files from MT5
│   ├── generator.py          # Synthetic data generator for testing
│   └── downloader.py         # MT5 connection and bar download
│
├── env/
│   ├── trading_env.py        # Custom Gymnasium environment
│   ├── simulator.py          # Trade simulator (slippage, spread, SL/TP)
│   └── preprocessor.py       # Feature scaling and normalization
│
├── agents/
│   ├── train.py              # Training script with SB3 (PPO)
│   └── evaluate.py           # Backtest and performance metrics
│
├── config/
│   ├── config.yaml           # Hyperparameters, paths, environment settings
│   └── symbols.yaml          # Forex pair configurations
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   └── 02_backtest_visualization.ipynb  # Backtest visualization
│
├── tests/
│   ├── test_simulator.py     # Unit tests for trade simulator
│   ├── test_trading_env.py   # Unit tests for trading environment
│   ├── test_preprocessor.py  # Unit tests for preprocessor
│   └── test_integration.py   # End-to-end integration tests
│
├── models/                   # Saved model checkpoints (.zip)
├── logs/                     # TensorBoard logs and training outputs
│
├── main.py                   # CLI entry point
├── requirements.txt
├── .env                      # MT5 credentials (git-ignored)
└── .gitignore
```

---

## Installation

### Prerequisites

- Python 3.10+
- MetaTrader 5 (optional, for real data download)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd forex_rl_trader

# Install dependencies
pip install -r requirements.txt

# For MT5 data download (optional)
pip install MetaTrader5
```

### Environment Variables

Create a `.env` file for MT5 credentials (optional):

```bash
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
```

---

## Quick Start

### 1. Generate Synthetic Data (No MT5 Required)

```bash
# Generate 10,000 bars of EURUSD data
python main.py generate --symbol EURUSD --samples 10000
```

### 2. Train the Agent

```bash
# Train for 100,000 timesteps
python main.py train --data data/raw/EURUSD.npy --timesteps 100000
```

### 3. Evaluate the Model

```bash
# Run backtest evaluation
python main.py evaluate --model models/ppo_trading_final.zip --data data/raw/EURUSD.npy --episodes 50
```

---

## CLI Commands

### Generate Data

Generate synthetic forex data for testing without MT5:

```bash
python main.py generate --symbol EURUSD --samples 10000 --output data/raw
```

Options:
- `--symbol`: Currency pair (default: EURUSD)
- `--samples`: Number of bars (default: 10000)
- `--output`: Output directory (default: data/raw)
- `--seed`: Random seed (default: 42)

### Download Data (Requires MT5)

Download historical data from MetaTrader 5:

```bash
python main.py download --symbols EURUSD GBPUSD --timeframe H1 --start 2024-01-01 --end 2024-12-31
```

Options:
- `--symbols`: List of currency pairs
- `--timeframe`: M1, M5, M15, M30, H1, H4, D1, W1, MN1
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--output`: Output directory

### Train

Train the RL agent:

```bash
python main.py train --data data/raw/EURUSD.npy --timesteps 100000 --seed 42
```

Options:
- `--data`: Path to OHLCV data file (required)
- `--config`: Path to config file (default: config/config.yaml)
- `--timesteps`: Training timesteps (overrides config)
- `--model`: Path to existing model for continued training
- `--output`: Model output directory (default: models)
- `--seed`: Random seed (default: 42)

### Evaluate

Evaluate a trained model:

```bash
python main.py evaluate --model models/ppo_trading_final.zip --data data/raw/EURUSD.npy --episodes 50
```

Options:
- `--model`: Path to trained model (required)
- `--data`: Path to OHLCV data file (required)
- `--config`: Path to config file (default: config/config.yaml)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Render environment during evaluation
- `--output`: Results output directory (default: data/evaluation)

---

## Configuration

### Environment Settings (`config/config.yaml`)

```yaml
environment:
  window_size: 10           # Observation window (candles)
  initial_balance: 10000.0  # Starting balance
  spread: 0.0001            # 1 pip for EURUSD
  slippage_prob: 0.3        # 30% chance of slippage
  slippage_range: [0.00001, 0.0005]

training:
  total_timesteps: 1000000
  save_freq: 100000         # Checkpoint frequency
  eval_freq: 50000          # Evaluation frequency
```

### Agent Hyperparameters

```yaml
agent:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
```

---

## Features

### Trade Simulator

Realistic forex trade execution with:
- **Spread modeling**: Bid/ask spread simulation
- **Slippage**: Probabilistic slippage on order execution
- **Stop-loss / Take-profit**: Automatic breach detection
- **Position lifecycle**: Open, close, reverse operations

### Trading Environment

Gymnasium-compatible environment with:
- **Observation space**: OHLCV window + position state + account state
- **Action space**: Buy (1), Sell (2), Hold (0)
- **Reward function**: P&L-based with cost penalties

### Preprocessor

Feature engineering and normalization:
- **Price scaling**: Log returns, relative pricing, or z-score
- **Volume scaling**: Log transform or min-max normalization
- **Technical indicators**: Price range, momentum, shadows

---

## Performance Metrics

The evaluation script calculates:

| Metric | Description |
|--------|-------------|
| **Win Rate** | Percentage of profitable trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Annualized return / Max drawdown |
| **Expectancy** | Expected profit per trade |

---

## Notebooks

### Exploratory Data Analysis (`notebooks/01_eda.ipynb`)

- Price and volume distributions
- Returns analysis with normal fit
- Rolling volatility
- Candlestick pattern statistics
- Correlation analysis

### Backtest Visualization (`notebooks/02_backtest_visualization.ipynb`)

- Equity curve plotting
- Drawdown visualization
- Trade outcome analysis
- Rolling metrics (win rate, P&L)
- Monthly returns heatmap

---

## Testing

Run all tests:

```bash
# Unit tests
python -m pytest tests/test_simulator.py -v
python -m pytest tests/test_trading_env.py -v
python -m pytest tests/test_preprocessor.py -v

# Integration tests
python tests/test_integration.py
```

---

## TensorBoard

View training logs:

```bash
tensorboard --logdir logs/
```

---

## Example Workflow

```bash
# 1. Generate test data
python main.py generate --symbol EURUSD --samples 50000

# 2. Train agent (quick test)
python main.py train --data data/raw/EURUSD.npy --timesteps 50000

# 3. Evaluate
python main.py evaluate --model models/ppo_trading_final.zip --data data/raw/EURUSD.npy --episodes 100

# 4. View results
python -c "import json; print(json.load(open('data/evaluation/metrics_*.json')))"

# 5. Open notebooks for visualization
# Open notebooks/02_backtest_visualization.ipynb in Jupyter
```

---

## Requirements

```
gymnasium>=0.28.0
stable-baselines3>=2.0.0
MetaTrader5>=5.0.0 (optional)
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
tensorboard>=2.14.0
pytest>=7.0.0 (for testing)
```

---

## License

MIT

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/ -v`
5. Submit a pull request
