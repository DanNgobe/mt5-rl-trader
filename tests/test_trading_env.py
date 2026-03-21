"""
Unit tests for the trading environment.
"""

import numpy as np
import pytest

from env.trading_env import TradingEnv
from env.simulator import PositionType


class TestTradingEnvInit:
    """Tests for TradingEnv initialization."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
            [1.1025, 1.1035, 1.1020, 1.1030, 1500],
            [1.1030, 1.1040, 1.1025, 1.1035, 1600],
            [1.1035, 1.1045, 1.1030, 1.1040, 1700],
            [1.1040, 1.1050, 1.1035, 1.1045, 1800],
            [1.1045, 1.1055, 1.1040, 1.1050, 1900],
            [1.1050, 1.1060, 1.1045, 1.1055, 2000],
            [1.1055, 1.1065, 1.1050, 1.1060, 2100],
            [1.1060, 1.1070, 1.1055, 1.1065, 2200],
            [1.1065, 1.1075, 1.1060, 1.1070, 2300],
            [1.1070, 1.1080, 1.1065, 1.1075, 2400],
        ])
    
    def test_init_basic(self):
        """Test basic environment initialization."""
        env = TradingEnv(self.sample_data, window_size=5)
        
        assert env.window_size == 5
        assert env.initial_balance == 10000.0
        assert env.action_space.n == 3
    
    def test_init_custom_balance(self):
        """Test initialization with custom balance."""
        env = TradingEnv(self.sample_data, window_size=5, initial_balance=50000.0)
        
        assert env.initial_balance == 50000.0
    
    def test_init_insufficient_data(self):
        """Test that insufficient data raises error."""
        small_data = self.sample_data[:5]
        
        with pytest.raises(ValueError):
            TradingEnv(small_data, window_size=5)
    
    def test_observation_space_shape(self):
        """Test observation space has correct shape."""
        env = TradingEnv(self.sample_data, window_size=5)
        
        # OHLCV: 5 * 5 = 25, Position: 6, Account: 2 = 33
        expected_dim = 5 * 5 + 6 + 2
        assert env.observation_space.shape[0] == expected_dim


class TestTradingEnvReset:
    """Tests for TradingEnv reset."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
            [1.1025, 1.1035, 1.1020, 1.1030, 1500],
            [1.1030, 1.1040, 1.1025, 1.1035, 1600],
            [1.1035, 1.1045, 1.1030, 1.1040, 1700],
            [1.1040, 1.1050, 1.1035, 1.1045, 1800],
            [1.1045, 1.1055, 1.1040, 1.1050, 1900],
            [1.1050, 1.1060, 1.1045, 1.1055, 2000],
            [1.1055, 1.1065, 1.1050, 1.1060, 2100],
            [1.1060, 1.1070, 1.1055, 1.1065, 2200],
            [1.1065, 1.1075, 1.1060, 1.1070, 2300],
            [1.1070, 1.1080, 1.1065, 1.1075, 2400],
        ])
    
    def test_reset_returns_observation(self):
        """Test that reset returns observation and info."""
        env = TradingEnv(self.sample_data, window_size=5)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
    
    def test_reset_initializes_balance(self):
        """Test that reset initializes balance correctly."""
        env = TradingEnv(self.sample_data, window_size=5, initial_balance=50000.0)
        env.reset()
        
        assert env.balance == 50000.0
        assert env.equity == 50000.0
    
    def test_reset_clears_position(self):
        """Test that reset clears any open position."""
        env = TradingEnv(self.sample_data, window_size=5)
        env.reset()
        
        # Manually open a position
        env.simulator.open_position(1.1000, PositionType.LONG)
        assert env.simulator.has_position
        
        # Reset should clear it
        env.reset()
        assert not env.simulator.has_position
    
    def test_reset_seed(self):
        """Test that reset with seed is reproducible."""
        env1 = TradingEnv(self.sample_data, window_size=5)
        env2 = TradingEnv(self.sample_data, window_size=5)
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        assert np.array_equal(obs1, obs2)


class TestTradingEnvStep:
    """Tests for TradingEnv step."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
            [1.1025, 1.1035, 1.1020, 1.1030, 1500],
            [1.1030, 1.1040, 1.1025, 1.1035, 1600],
            [1.1035, 1.1045, 1.1030, 1.1040, 1700],
            [1.1040, 1.1050, 1.1035, 1.1045, 1800],
            [1.1045, 1.1055, 1.1040, 1.1050, 1900],
            [1.1050, 1.1060, 1.1045, 1.1055, 2000],
            [1.1055, 1.1065, 1.1050, 1.1060, 2100],
            [1.1060, 1.1070, 1.1055, 1.1065, 2200],
            [1.1065, 1.1075, 1.1060, 1.1070, 2300],
            [1.1070, 1.1080, 1.1065, 1.1075, 2400],
        ])
    
    def test_step_hold_action(self):
        """Test step with HOLD action."""
        env = TradingEnv(self.sample_data, window_size=5)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not env.simulator.has_position
    
    def test_step_buy_action(self):
        """Test step with BUY action."""
        env = TradingEnv(self.sample_data, window_size=5, slippage_prob=0.0)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(1)  # BUY
        
        assert env.simulator.has_position
        assert env.simulator.position.type == PositionType.LONG
    
    def test_step_sell_action(self):
        """Test step with SELL action."""
        env = TradingEnv(self.sample_data, window_size=5, slippage_prob=0.0)
        env.reset()
        
        obs, reward, terminated, truncated, info = env.step(2)  # SELL
        
        assert env.simulator.has_position
        assert env.simulator.position.type == PositionType.SHORT
    
    def test_step_reverse_position(self):
        """Test reversing a position."""
        env = TradingEnv(self.sample_data, window_size=5, slippage_prob=0.0, spread=0.0)
        env.reset()
        
        # Open long
        env.step(1)  # BUY
        assert env.simulator.position.type == PositionType.LONG
        
        # Reverse to short
        env.step(2)  # SELL
        assert env.simulator.position.type == PositionType.SHORT
    
    def test_step_returns_correct_observation_shape(self):
        """Test that step returns observation with correct shape."""
        env = TradingEnv(self.sample_data, window_size=5)
        env.reset()
        
        obs, _, _, _, _ = env.step(0)
        
        assert obs.shape == env.observation_space.shape
    
    def test_step_updates_current_step(self):
        """Test that step advances current step."""
        env = TradingEnv(self.sample_data, window_size=5)
        env.reset()
        
        initial_step = env._current_step
        env.step(0)
        
        assert env._current_step == initial_step + 1


class TestTradingEnvTermination:
    """Tests for episode termination conditions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create longer data series
        np.random.seed(42)
        n_samples = 100
        base_price = 1.1000
        returns = np.random.randn(n_samples) * 0.001
        prices = base_price * np.cumprod(1 + returns)
        
        self.sample_data = np.column_stack([
            prices,  # open (approximate)
            prices * 1.001,  # high
            prices * 0.999,  # low
            prices,  # close
            np.random.randint(1000, 2000, n_samples),  # volume
        ])
    
    def test_termination_at_end_of_data(self):
        """Test that episode terminates at end of data."""
        env = TradingEnv(self.sample_data, window_size=10)
        env.reset()
        
        # Step until end
        for _ in range(self.sample_data.shape[0] - env.window_size):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated:
                break
        
        assert terminated or env._current_step >= self.sample_data.shape[0] - 1
    
    def test_termination_on_margin_call(self):
        """Test that episode terminates on margin call."""
        env = TradingEnv(self.sample_data, window_size=10, initial_balance=10000.0)
        env.reset()
        
        # Force a large loss by manipulating simulator
        env.simulator._cumulative_pnl = -6000  # 60% loss
        
        # Update balance
        env.balance = env.initial_balance + env.simulator.cumulative_pnl
        env.equity = env.balance
        
        # Should trigger termination check
        assert env._check_termination()


class TestTradingEnvEpisodeStats:
    """Tests for episode statistics."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
            [1.1025, 1.1035, 1.1020, 1.1030, 1500],
            [1.1030, 1.1040, 1.1025, 1.1035, 1600],
            [1.1035, 1.1045, 1.1030, 1.1040, 1700],
            [1.1040, 1.1050, 1.1035, 1.1045, 1800],
            [1.1045, 1.1055, 1.1040, 1.1050, 1900],
            [1.1050, 1.1060, 1.1045, 1.1055, 2000],
            [1.1055, 1.1065, 1.1050, 1.1060, 2100],
            [1.1060, 1.1070, 1.1055, 1.1065, 2200],
            [1.1065, 1.1075, 1.1060, 1.1070, 2300],
            [1.1070, 1.1080, 1.1065, 1.1075, 2400],
        ])
    
    def test_stats_no_trades(self):
        """Test stats when no trades were made."""
        env = TradingEnv(self.sample_data, window_size=5)
        env.reset()
        
        # Only hold
        for _ in range(5):
            env.step(0)
        
        stats = env.get_episode_stats()
        
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
    
    def test_stats_with_trades(self):
        """Test stats after making trades."""
        env = TradingEnv(
            self.sample_data,
            window_size=5,
            spread=0.0,
            slippage_prob=0.0,
        )
        env.reset()
        
        # Open and close a profitable long trade
        env.step(1)  # BUY at ~1.1020
        
        # Simulate price increase and close
        for _ in range(3):
            env.step(0)  # HOLD
        
        env.step(0)  # This should close if we had SL/TP set
        
        stats = env.get_episode_stats()
        
        assert "total_trades" in stats
        assert "winning_trades" in stats
        assert "losing_trades" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats


class TestTradingEnvObservation:
    """Tests for observation construction."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_data = np.array([
            [1.1000, 1.1010, 1.0990, 1.1005, 1000],
            [1.1005, 1.1015, 1.1000, 1.1010, 1100],
            [1.1010, 1.1020, 1.1005, 1.1015, 1200],
            [1.1015, 1.1025, 1.1010, 1.1020, 1300],
            [1.1020, 1.1030, 1.1015, 1.1025, 1400],
            [1.1025, 1.1035, 1.1020, 1.1030, 1500],
            [1.1030, 1.1040, 1.1025, 1.1035, 1600],
            [1.1035, 1.1045, 1.1030, 1.1040, 1700],
            [1.1040, 1.1050, 1.1035, 1.1045, 1800],
            [1.1045, 1.1055, 1.1040, 1.1050, 1900],
            [1.1050, 1.1060, 1.1045, 1.1055, 2000],
            [1.1055, 1.1065, 1.1050, 1.1060, 2100],
            [1.1060, 1.1070, 1.1055, 1.1065, 2200],
            [1.1065, 1.1075, 1.1060, 1.1070, 2300],
            [1.1070, 1.1080, 1.1065, 1.1075, 2400],
        ])
    
    def test_observation_contains_price_window(self):
        """Test that observation contains OHLCV window."""
        env = TradingEnv(self.sample_data, window_size=5)
        obs, _ = env.reset()
        
        # First part of observation should be OHLCV data
        ohlcv_part = obs[:5 * 5]
        assert len(ohlcv_part) == 25
    
    def test_observation_contains_position_state(self):
        """Test that observation contains position state."""
        env = TradingEnv(self.sample_data, window_size=5, spread=0.0, slippage_prob=0.0)
        env.reset()
        
        # No position initially
        obs_no_pos, _ = env.reset()
        position_start = 5 * 5
        has_position = obs_no_pos[position_start]
        assert has_position == 0.0
        
        # Open position
        env.step(1)  # BUY
        obs_with_pos, _, _, _, _ = env.step(0)
        has_position = obs_with_pos[position_start]
        assert has_position == 1.0
    
    def test_observation_contains_account_state(self):
        """Test that observation contains account state."""
        env = TradingEnv(self.sample_data, window_size=5)
        obs, _ = env.reset()
        
        # Last 2 elements should be account state (balance, equity normalized)
        account_start = 5 * 5 + 6
        account_state = obs[account_start:]
        
        assert len(account_state) == 2
        # Initial balance and equity should be normalized to ~1.0
        assert np.allclose(account_state, 1.0, atol=0.1)
