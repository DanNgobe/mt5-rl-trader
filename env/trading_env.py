"""
Trading environment for forex RL agent.

Custom Gymnasium environment that wraps the trade simulator
and provides OHLCV observations with trade state.
"""

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulator import TradeSimulator, PositionType, ActionType, Trade


class TradingEnv(gym.Env):
    """
    Forex trading environment with realistic trade simulation.
    
    The agent observes a sliding window of OHLCV candles plus
    its current trade state, and decides to Buy, Sell, or Hold.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 10,
        initial_balance: float = 10000.0,
        spread: float = 0.0001,
        slippage_prob: float = 0.3,
        slippage_range: tuple[float, float] = (0.00001, 0.0005),
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: OHLCV data array of shape (n_samples, 5) or (n_samples, 6)
                  where columns are [open, high, low, close, volume, (spread)]
            window_size: Number of candles in observation window
            initial_balance: Starting account balance
            spread: Fixed spread for trade simulator
            slippage_prob: Probability of slippage
            slippage_range: Min/max slippage range
            render_mode: Render mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.render_mode = render_mode
        
        # Validate data shape
        if len(data.shape) != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")
        
        self.n_features = data.shape[1]
        self.n_samples = data.shape[0]
        
        # Ensure we have enough data for the window
        if self.n_samples <= self.window_size:
            raise ValueError(
                f"Data has {self.n_samples} samples, need more than window_size {window_size}"
            )
        
        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # - OHLCV window: window_size * n_features
        # - Position state: [has_position, position_type, unrealized_pnl_norm, entry_price_norm, sl_norm, tp_norm]
        # - Account state: [balance_norm, equity_norm]
        obs_dim = (
            window_size * self.n_features  # OHLCV window
            + 6  # Position state
            + 2  # Account state
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Initialize simulator
        self.simulator = TradeSimulator(
            spread=spread,
            slippage_prob=slippage_prob,
            slippage_range=slippage_range,
        )
        
        # Account state
        self.balance = initial_balance
        self.equity = initial_balance
        
        # Episode state
        self._current_step = 0
        self._terminated = False
        self._truncated = False
        
        # Normalization constants (set during reset)
        self._price_scale = 1.0
        self._pnl_scale = 100.0  # Normalize P&L by $100
        
        # Trade history for episode
        self._episode_trades: list[Trade] = []
        
        # For rendering
        self._render_frame = None
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the current observation vector.
        
        Returns:
            Observation array of shape (obs_dim,)
        """
        # Get OHLCV window
        window_start = max(0, self._current_step - self.window_size + 1)
        window_end = self._current_step + 1
        ohlcv_window = self.data[window_start:window_end]
        
        # Pad if necessary (at episode start)
        if len(ohlcv_window) < self.window_size:
            padding = np.zeros(
                (self.window_size - len(ohlcv_window), self.n_features)
            )
            ohlcv_window = np.vstack([padding, ohlcv_window])
        
        # Flatten OHLCV window
        ohlcv_flat = ohlcv_window.flatten()
        
        # Get position state from simulator
        current_price = self._get_current_price()
        pos_state = self.simulator.get_state(current_price)
        
        # Normalize position state
        position_state = np.array([
            float(pos_state["has_position"]),
            pos_state["position_type"],
            pos_state["unrealized_pnl"] / self._pnl_scale,
            (pos_state["entry_price"] - self._price_center) * self._price_scale if pos_state["entry_price"] > 0 else 0.0,
            (pos_state["stop_loss"] - self._price_center) * self._price_scale if pos_state["stop_loss"] > 0 else 0.0,
            (pos_state["take_profit"] - self._price_center) * self._price_scale if pos_state["take_profit"] > 0 else 0.0,
        ], dtype=np.float32)
        
        # Account state (normalized)
        account_state = np.array([
            self.balance / self.initial_balance,
            self.equity / self.initial_balance,
        ], dtype=np.float32)
        
        # Concatenate all parts
        observation = np.concatenate([
            ohlcv_flat.astype(np.float32),
            position_state,
            account_state,
        ])
        
        return observation
    
    def _get_current_price(self) -> float:
        """Get current close price."""
        return float(self.data[self._current_step, 3])  # Close is column 3
    
    @property
    def _price_center(self) -> float:
        """Reference price for normalization."""
        return float(self.data[0, 3])  # First close price
    
    def _calculate_reward(self, action: int, closed_trade: Optional[Trade]) -> float:
        """
        Calculate reward for the current step.
        
        Reward is based on:
        - Realized P&L from closed trades
        - Change in unrealized P&L
        - Small penalty for holding positions (encourages active trading)
        
        Args:
            action: Action taken
            closed_trade: Trade that was closed this step (if any)
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Realized P&L from closed trade
        if closed_trade is not None:
            reward += closed_trade.pnl / self._pnl_scale
        
        # Small penalty for spread/slippage costs (learn to minimize trading costs)
        if closed_trade is not None:
            total_costs = (closed_trade.spread + closed_trade.slippage) * 10000  # In pips
            reward -= total_costs * 0.01  # Small penalty per pip of costs
        
        # Holding penalty (optional, encourages closing unprofitable positions)
        if self.simulator.has_position:
            pos_state = self.simulator.get_state(self._get_current_price())
            if pos_state["unrealized_pnl"] < 0:
                reward -= 0.001  # Small penalty for holding losing position
        
        return reward
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.
        
        Episode terminates when:
        - No more data
        - Balance drops below minimum threshold (margin call)
        """
        # End of data
        if self._current_step >= self.n_samples - 1:
            return True
        
        # Margin call (balance too low)
        if self.equity < self.initial_balance * 0.5:  # 50% drawdown limit
            return True
        
        return False
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._terminated = False
        self._truncated = False
        
        current_price = self._get_current_price()
        closed_trade: Optional[Trade] = None
        
        # Execute action
        if action == ActionType.BUY.value:
            if not self.simulator.has_position:
                # Open long position
                self.simulator.open_position(
                    market_price=current_price,
                    position_type=PositionType.LONG,
                    size=1.0,  # 1 lot
                )
            elif self.simulator.position.type == PositionType.SHORT:
                # Reverse: close short and open long
                closed_trade, _ = self.simulator.reverse_position(current_price)
        
        elif action == ActionType.SELL.value:
            if not self.simulator.has_position:
                # Open short position
                self.simulator.open_position(
                    market_price=current_price,
                    position_type=PositionType.SHORT,
                    size=1.0,
                )
            elif self.simulator.position.type == PositionType.LONG:
                # Reverse: close long and open short
                closed_trade, _ = self.simulator.reverse_position(current_price)
        
        else:  # HOLD
            # Check for SL/TP breaches
            closed_trade = self.simulator.check_breaches(current_price)
        
        # Update account state
        self.balance = self.initial_balance + self.simulator.cumulative_pnl
        unrealized_pnl = self.simulator.position.unrealized_pnl(current_price) if self.simulator.position else 0.0
        self.equity = self.balance + unrealized_pnl
        
        # Track closed trades
        if closed_trade is not None:
            self._episode_trades.append(closed_trade)
        
        # Calculate reward
        reward = self._calculate_reward(action, closed_trade)
        
        # Advance to next step
        self._current_step += 1
        
        # Check termination
        self._terminated = self._check_termination()
        
        # Get observation
        observation = self._get_observation()
        
        # Info dict
        info = {
            "step": self._current_step,
            "balance": self.balance,
            "equity": self.equity,
            "has_position": self.simulator.has_position,
            "closed_trade": closed_trade,
            "total_trades": len(self._episode_trades),
        }
        
        return observation, reward, self._terminated, self._truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset simulator
        self.simulator.reset_full()
        
        # Reset account
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        
        # Reset episode state
        self._current_step = self.window_size  # Start after warm-up period
        self._terminated = False
        self._truncated = False
        self._episode_trades = []
        
        # Set normalization constants
        self._price_scale = 10000.0 / self._price_center  # Normalize to ~1
        self._pnl_scale = self.initial_balance * 0.1  # Normalize to 10% of balance
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "step": self._current_step,
            "balance": self.balance,
            "equity": self.equity,
        }
        
        return observation, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None
        
        # Simple text rendering for now
        if self.render_mode == "human":
            print(f"Step: {self._current_step}/{self.n_samples}")
            print(f"Balance: ${self.balance:.2f}, Equity: ${self.equity:.2f}")
            if self.simulator.has_position:
                pos = self.simulator.position
                pos_type = "LONG" if pos.type == PositionType.LONG else "SHORT"
                print(f"Position: {pos_type} @ {pos.entry_price:.5f}")
            else:
                print("No open position")
            print("-" * 40)
        
        return self._render_frame
    
    def close(self):
        """Clean up resources."""
        pass
    
    def get_episode_stats(self) -> dict:
        """
        Get statistics for the current/completed episode.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self._episode_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "max_drawdown": 0.0,
            }
        
        trades = self._episode_trades
        pnls = [t.pnl for t in trades]
        winning = sum(1 for p in pnls if p > 0)
        losing = sum(1 for p in pnls if p < 0)
        
        # Calculate running drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = float(np.max(drawdowns))
        
        return {
            "total_trades": len(trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": winning / len(trades) if trades else 0.0,
            "total_pnl": float(sum(pnls)),
            "avg_pnl": float(np.mean(pnls)),
            "max_drawdown": max_drawdown,
            "final_balance": self.balance,
            "final_equity": self.equity,
        }
