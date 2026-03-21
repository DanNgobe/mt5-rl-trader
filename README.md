# forex_rl_trader

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

## Project structure

```
forex_rl_trader/
├── data/
│   ├── raw/                  # Raw OHLCV files from MT5
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
├── models/                   # Saved model checkpoints (.zip)
├── logs/                     # TensorBoard logs and training outputs
├── notebooks/                # EDA, backtest analysis, visualizations
├── tests/                    # Unit tests for env, simulator, agents
│
├── main.py                   # CLI entry point
├── requirements.txt
├── .env                      # MT5 credentials (git-ignored)
└── .gitignore
```

## Requirements

```
gymnasium
stable-baselines3
MetaTrader5
pandas
numpy
pyyaml
python-dotenv
tensorboard
```