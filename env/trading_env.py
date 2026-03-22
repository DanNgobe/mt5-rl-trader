"""
Gymnasium environment wrapping the MT5 hedging simulator.

Observation
-----------
    [ohlcv_window (flattened)]  +  [position_slots]  +  [account_state]

    ohlcv_window   : window_size × n_features  (log-return normalised)
    position_slots : max_positions × 5          (see TradeSimulator.position_state_vector)
    account_state  : 2                          (balance_norm, equity_norm)

Action space — MultiDiscrete([4, 3])
-------------------------------------
    axis-0  direction : 0=HOLD  1=BUY  2=SELL  3=CLOSE
    axis-1  lot_tier  : 0=0.01  1=0.02  2=0.05

    HOLD ignores the lot tier.
    CLOSE targets the oldest open position with the matching lot size.

Reward
------
    r_t = Δequity_t / initial_balance

    This is the change in total equity (realised + unrealised) normalised
    by the starting balance.  It directly penalises losses and rewards
    gains without requiring a hand-crafted reward function, and it avoids
    the "hold-losers-forever" pathology of realised-only rewards.
"""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulator import (
    Action,
    ClosedTrade,
    Direction,
    LOT_TIERS,
    SymbolSpec,
    TradeSimulator,
)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Forex trading environment simulating an MT5 hedging account.

    Parameters
    ----------
    data : np.ndarray
        Pre-processed OHLCV array of shape (n_samples, n_features).
        Must already be log-return normalised (use preprocessor.preprocess).
    raw_close : np.ndarray
        Raw (un-normalised) close prices of shape (n_samples,).
        Used only for P&L calculations; never shown to the agent.
    symbol_spec : SymbolSpec
        Instrument specification (pip value, spread, contract size, etc.).
    window_size : int
        Number of candles in the observation window.
    initial_balance : float
        Starting account balance in account currency.
    max_positions : int
        Hard cap on simultaneous open positions.
    slippage_prob : float
        Probability of slippage on any fill.
    slippage_range : tuple[float, float]
        (min, max) slippage in price units when it fires.
    render_mode : str or None
        'human' for stdout rendering.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        data:            np.ndarray,
        raw_close:       np.ndarray,
        symbol_spec:     SymbolSpec,
        window_size:     int = 10,
        initial_balance: float = 10_000.0,
        max_positions:   int = 3,
        slippage_prob:   float = 0.3,
        slippage_range:  tuple[float, float] = (0.00001, 0.0005),
        render_mode:     Optional[str] = None,
    ):
        super().__init__()

        # ---------------------------------------------------------------
        # Validate inputs
        # ---------------------------------------------------------------
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got {data.shape}")
        if len(raw_close) != len(data):
            raise ValueError(
                f"raw_close length {len(raw_close)} != data length {len(data)}"
            )
        if data.shape[0] <= window_size:
            raise ValueError(
                f"data has {data.shape[0]} rows but window_size={window_size}; "
                "need more rows than the window."
            )

        self.data            = np.asarray(data,      dtype=np.float32)
        self.raw_close       = np.asarray(raw_close, dtype=np.float64)
        self.spec            = symbol_spec
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.max_positions   = max_positions
        self.render_mode     = render_mode

        n_samples, n_features = self.data.shape
        self.n_samples  = n_samples
        self.n_features = n_features

        # ---------------------------------------------------------------
        # Spaces
        # ---------------------------------------------------------------

        # MultiDiscrete([4, 3]):  [direction, lot_tier]
        self.action_space = spaces.MultiDiscrete([4, 3])

        obs_dim = (
            window_size * n_features    # OHLCV candle window
            + max_positions * 5         # position slot vector
            + 2                         # account state
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---------------------------------------------------------------
        # Simulator (created once; reset() reinitialises it)
        # ---------------------------------------------------------------
        self._sim = TradeSimulator(
            symbol_spec    = symbol_spec,
            max_positions  = max_positions,
            slippage_prob  = slippage_prob,
            slippage_range = slippage_range,
        )

        # ---------------------------------------------------------------
        # Episode state (initialised properly in reset())
        # ---------------------------------------------------------------
        self._step:          int   = 0
        self._balance:       float = initial_balance
        self._prev_equity:   float = initial_balance
        self._episode_trades: list[ClosedTrade] = []

    # -------------------------------------------------------------------
    # Gym interface
    # -------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Pass the seeded RNG into the simulator so fills are reproducible
        self._sim._rng = self.np_random

        self._sim.reset()
        self._step           = self.window_size   # skip the warm-up period
        self._balance        = self.initial_balance
        self._prev_equity    = self.initial_balance
        self._episode_trades = []

        obs  = self._observation()
        info = self._info()
        logger.debug("Environment reset. Starting at step %d.", self._step)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Array of shape (2,) → [direction_idx, lot_tier_idx].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        direction_idx, lot_idx = int(action[0]), int(action[1])
        direction_action       = Action(direction_idx)
        lot_size               = LOT_TIERS[lot_idx]
        current_price          = self._current_raw_price()

        closed_trade: Optional[ClosedTrade] = None

        # ---------------------------------------------------------------
        # Execute action
        # ---------------------------------------------------------------
        if direction_action == Action.BUY:
            pos = self._sim.open_position(current_price, Direction.LONG,  lot_size)
            if pos is None:
                logger.debug("BUY ignored: at position cap.")

        elif direction_action == Action.SELL:
            pos = self._sim.open_position(current_price, Direction.SHORT, lot_size)
            if pos is None:
                logger.debug("SELL ignored: at position cap.")

        elif direction_action == Action.CLOSE:
            closed_trade = self._sim.close_position(current_price, lot_size)
            if closed_trade is None:
                logger.debug(
                    "CLOSE %.2f ignored: no matching position.", lot_size
                )
            else:
                self._episode_trades.append(closed_trade)

        # HOLD → no action

        # ---------------------------------------------------------------
        # Update account
        # ---------------------------------------------------------------
        self._balance = (
            self.initial_balance + self._sim.cumulative_pnl
        )
        unrealized   = self._sim.total_unrealized_pnl(current_price)
        equity       = self._balance + unrealized

        # ---------------------------------------------------------------
        # Reward: Δequity normalised by initial balance
        # ---------------------------------------------------------------
        reward           = (equity - self._prev_equity) / self.initial_balance
        self._prev_equity = equity

        # ---------------------------------------------------------------
        # Advance step
        # ---------------------------------------------------------------
        self._step += 1

        terminated = self._is_terminated(equity)
        truncated  = False

        # Force-close all positions at episode end so P&L is accurate
        if terminated and self._sim.has_positions:
            forced = self._sim.close_all(self._current_raw_price())
            self._episode_trades.extend(forced)
            # Recompute equity after forced close
            self._balance  = self.initial_balance + self._sim.cumulative_pnl
            equity         = self._balance
            self._prev_equity = equity

        obs  = self._observation()
        info = self._info(closed_trade=closed_trade, equity=equity)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        price  = self._current_raw_price()
        equity = self._balance + self._sim.total_unrealized_pnl(price)
        print(
            f"[{self.spec.name}] step={self._step}/{self.n_samples}  "
            f"balance={self._balance:.2f}  equity={equity:.2f}  "
            f"open_positions={self._sim.n_positions}"
        )

    def close(self) -> None:
        pass

    # -------------------------------------------------------------------
    # Episode statistics (call after episode ends)
    # -------------------------------------------------------------------

    def episode_stats(self) -> dict:
        """Return summary statistics for the completed episode."""
        trades = self._episode_trades
        if not trades:
            return {
                "total_trades": 0,
                "win_rate":     0.0,
                "total_pnl":    0.0,
                "avg_pnl":      0.0,
                "max_drawdown": 0.0,
                "final_balance": self._balance,
            }

        pnls    = np.array([t.pnl for t in trades])
        wins    = int((pnls > 0).sum())
        cumsum  = np.cumsum(pnls)
        peak    = np.maximum.accumulate(cumsum)
        dd      = float(np.max(peak - cumsum))

        return {
            "total_trades":  len(trades),
            "winning_trades": wins,
            "losing_trades":  len(trades) - wins,
            "win_rate":       wins / len(trades),
            "total_pnl":      float(pnls.sum()),
            "avg_pnl":        float(pnls.mean()),
            "max_drawdown":   dd,
            "final_balance":  self._balance,
        }

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _current_raw_price(self) -> float:
        idx = min(self._step, self.n_samples - 1)
        return float(self.raw_close[idx])

    def _is_terminated(self, equity: float) -> bool:
        if self._step >= self.n_samples:
            return True
        if equity < self.initial_balance * 0.5:   # 50 % drawdown → margin call
            logger.debug("Margin call triggered at step %d.", self._step)
            return True
        return False

    def _observation(self) -> np.ndarray:
        # OHLCV window
        start      = max(0, self._step - self.window_size)
        end        = self._step
        window     = self.data[start:end]

        if len(window) < self.window_size:
            pad    = np.zeros(
                (self.window_size - len(window), self.n_features), dtype=np.float32
            )
            window = np.vstack([pad, window])

        ohlcv_flat = window.flatten()

        # Position state
        price    = self._current_raw_price()
        pos_vec  = self._sim.position_state_vector(price, self.max_positions)

        # Account state
        unrealized   = self._sim.total_unrealized_pnl(price)
        equity       = self._balance + unrealized
        account_vec  = np.array(
            [
                self._balance / self.initial_balance,
                equity        / self.initial_balance,
            ],
            dtype=np.float32,
        )

        return np.concatenate([ohlcv_flat, pos_vec, account_vec])

    def _info(
        self,
        closed_trade: Optional[ClosedTrade] = None,
        equity: Optional[float] = None,
    ) -> dict:
        price = self._current_raw_price()
        if equity is None:
            equity = self._balance + self._sim.total_unrealized_pnl(price)
        return {
            "step":          self._step,
            "price":         price,
            "balance":       self._balance,
            "equity":        equity,
            "open_positions": self._sim.n_positions,
            "total_trades":  len(self._episode_trades),
            "closed_trade":  closed_trade,
        }