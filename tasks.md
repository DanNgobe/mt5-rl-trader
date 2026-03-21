# Development Tasks

## Phase 1: Core Infrastructure

### 1.1 Data Pipeline
- [ ] Set up MT5 connection and credentials (`.env`)
- [ ] Implement `data/downloader.py` for historical OHLCV download
- [ ] Create sample raw data files in `data/raw/`

### 1.2 Trade Simulator
- [ ] Build `env/simulator.py` with realistic order execution
  - Slippage and spread modeling
  - Stop-loss / take-profit breach detection
  - Position lifecycle management

### 1.3 Trading Environment
- [ ] Implement `env/trading_env.py` (Gymnasium interface)
  - Observation space (OHLCV window + trade state)
  - Action space (Buy, Sell, Hold)
  - Reward function (P&L based)
- [ ] Add `env/preprocessor.py` for feature normalization

---

## Phase 2: Agent Training

### 2.1 Configuration
- [ ] Create `config/config.yaml` (hyperparameters, paths)
- [ ] Create `config/symbols.yaml` (forex pair specs)

### 2.2 Training Pipeline
- [ ] Implement `agents/train.py` with SB3 PPO
- [ ] Set up TensorBoard logging to `logs/`
- [ ] Save model checkpoints to `models/`

### 2.3 Evaluation
- [ ] Implement `agents/evaluate.py` for backtesting
- [ ] Calculate performance metrics (Sharpe, max drawdown, win rate)

---

## Phase 3: Testing & Validation

### 3.1 Unit Tests
- [ ] Test trade simulator logic
- [ ] Test environment step/reset functions
- [ ] Test preprocessor transformations

### 3.2 Integration Tests
- [ ] End-to-end training run (small episode count)
- [ ] Backtest validation against known results

---

## Phase 4: Analysis & CLI

### 4.1 Notebooks
- [ ] EDA notebook for price data exploration
- [ ] Backtest visualization notebook (equity curves, trade markers)

### 4.2 CLI Interface
- [ ] Build `main.py` entry point with commands:
  - `download` — fetch data from MT5
  - `train` — train agent
  - `evaluate` — run backtest
  - `plot` — generate visualizations

---

## Phase 5: Documentation & Polish

### 5.1 Docs
- [ ] Expand README with usage examples
- [ ] Add docstrings to core modules

### 5.2 Cleanup
- [ ] Finalize `.gitignore`
- [ ] Verify `requirements.txt`
- [ ] Run full training + evaluation cycle

---

## Milestones

| Milestone | Phases | Goal |
|-----------|--------|------|
| M1 | Phase 1 | Working env with realistic simulation |
| M2 | Phase 2 | Trainable agent with logging |
| M3 | Phase 3 | Tested and validated pipeline |
| M4 | Phase 4–5 | Usable CLI with analysis tools |
