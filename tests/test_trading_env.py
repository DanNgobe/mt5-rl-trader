"""
tests/test_trading_env.py
-------------------------
Unit tests for TradingEnv logic.
"""

import numpy as np
import pytest
from core.simulator import Direction, SymbolSpec
from env.trading_env import TradingEnv

@pytest.fixture
def mock_data():
    n = 200
    raw_close = np.linspace(1.1000, 1.1200, n)
    # Simple log returns
    close_lr = np.zeros(n)
    close_lr[1:] = np.log(raw_close[1:] / raw_close[:-1])
    
    obs_arrays = {
        "close_log_returns": close_lr,
        "rsi": np.zeros(n),
        "atr": np.zeros(n),
        "ema_ratio": np.zeros(n),
        "bollinger": np.zeros(n),
        "momentum": np.zeros((n, 3)),
        "session": np.zeros((n, 4)),
    }
    return obs_arrays, raw_close

@pytest.fixture
def symbol_spec():
    return SymbolSpec(
        name          = "EURUSD",
        pip_value     = 0.0001,
        pip_location  = 4,
        contract_size = 100_000,
        spread_pips   = 2.0,
        min_lot       = 0.01,
        max_lot       = 100.0,
        margin_rate   = 0.01,
    )

@pytest.fixture
def env(mock_data, symbol_spec):
    obs_arrays, raw_close = mock_data
    return TradingEnv(
        obs_arrays      = obs_arrays,
        raw_close       = raw_close,
        symbol_spec     = symbol_spec,
        obs_config      = {
            "price_lags": [1, 2],
            "indicators": {
                "rsi":      {"enabled": True}, 
                "atr":      {"enabled": False},
                "momentum": {"enabled": False},
                "session":  {"enabled": False}
            }
        },
        initial_balance = 10_000.0,
        lot_tiers       = [0.1, 1.0],
        reward_mode     = "step",
        max_positions   = 5,
    )

def test_reset(env):
    obs, info = env.reset(seed=42)
    assert obs.shape[0] > 0
    assert info["balance"] == 10_000.0
    assert env._step == env._min_step

def test_action_masking_limits(env):
    env.reset()
    # Masking for n_positions limit
    env.max_positions = 1
    # Open 1 BUY
    env.step(1) # OPEN BUY 0.1
    mask = env.action_masks()
    
    # 0=HOLD, 1=OPEN_BUY_01, 2=OPEN_SELL_01, 3=CLOSE_BUY_01, 4=CLOSE_SELL_01,
    # 5=OPEN_BUY_10, 6=OPEN_SELL_10, 7=CLOSE_BUY_10, 8=CLOSE_SELL_10, 9=CLOSE_ALL
    
    # Open indices (1, 2, 5, 6) should be masked
    assert not mask[1]
    assert not mask[2]
    assert not mask[5]
    assert not mask[6]
    
    # Close BUY 0.1 (3) should be valid
    assert mask[3]
    # Close ALL (9) should be valid
    assert mask[9]

def test_action_masking_invalid_closes(env):
    env.reset()
    # No positions open
    mask = env.action_masks()
    # ALL close actions (3, 4, 7, 8, 9) should be masked
    assert not mask[3]
    assert not mask[4]
    assert not mask[7]
    assert not mask[8]
    assert not mask[9]

def test_reward_step_mode(env):
    env.reset()
    # Bullish price move: 1.1000 -> 1.1200 over 200 steps
    # At step 2 (min_step), price is ~1.1002
    
    # Open 1.0 lot LONG (index 5)
    # entry_price = 1.1002 + 0.0001 = 1.1003
    _, reward_open, _, _, _ = env.step(5)
    
    # Spread cost in step mode: actual_spread_norm * (scale - 1)
    # scale is 2.0, so cost = spread_norm * 1.0
    # spread = 0.0002 / 10000 = 0.00002
    # reward_open should be approximately -0.00002 + step_return
    assert reward_open < 0
    
    # Step again (HOLD=0)
    # Price increases -> equity increases -> positive reward
    _, reward_hold, _, _, _ = env.step(0)
    assert reward_hold > 0

def test_margin_call(env):
    env.reset()
    env.max_drawdown_pct = 0.01 # 1% crash = terminal
    
    # Price is 1.1002. Open 10.0 lot LONG (we only have 1.0 in lot tiers fixture, but let's override)
    env.lot_tiers = [10.0]
    # Re-init action map shortcut or just use the existing 0.1/1.0
    
    # Let's use 1.0 lot and crash the price in mock_data
    env.initial_balance = 1000.0
    env.max_drawdown_pct = 0.05 # $50 drawdown
    
    env.reset()
    # Force a position
    env._sim.open_position(1.1000, Direction.LONG, 1.0) # entry 1.1001
    
    # Crash price to 1.0900 -> PnL = (1.0900 - 1.1001) * 100,000 = -1010.0
    env.raw_close[env._step] = 1.0500
    
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert reward <= -env.drawdown_penalty

def test_observation_values(env):
    env.reset()
    obs = env._observation()
    # Lagged returns
    # Indicators
    # Position slots (mostly 0)
    # Account state
    assert len(obs) == env._obs_dim
    assert not np.isnan(obs).any()
