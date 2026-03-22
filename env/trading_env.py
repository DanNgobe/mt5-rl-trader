"""
Gymnasium environment wrapping the MT5 hedging simulator.

Observation
-----------
    [ohlcv_window (flattened)]  +  [position_slots]  +  [account_state]

    ohlcv_window   : window_size × n_features  (log-return normalised)
    position_slots : max_positions × 5
    account_state  : 2  (balance_norm, equity_norm)

Action space — MultiDiscrete([5, 3])
-------------------------------------
    axis-0  direction : 0=HOLD  1=BUY  2=SELL  3=CLOSE_LONG  4=CLOSE_SHORT
    axis-1  lot_tier  : 0=0.01  1=0.02  2=0.05

Reward — sparse, fires only on trade close or invalid action
-------------------------------------------------------------
    On trade close:
        r = pnl_norm
          - drawdown_penalty_scale  * max(0, -mae_pnl) / initial_balance
          - missed_profit_scale     * max(0, mfe_pnl - pnl) / initial_balance

    On invalid action (open at cap, close non-existent lot/direction):
        r = invalid_action_penalty  (flat negative constant)

    All other steps:
        r = 0.0
"""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.simulator import (
    Action,
    ClosedTrade,
    Direction,
    LOT_TIERS,
    OrderResult,
    SymbolSpec,
    TradeSimulator,
)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        data:                    np.ndarray,
        raw_close:               np.ndarray,
        symbol_spec:             SymbolSpec,
        window_size:             int   = 10,
        initial_balance:         float = 10_000.0,
        max_positions:           int   = 3,
        slippage_prob:           float = 0.3,
        slippage_range:          tuple[float, float] = (0.00001, 0.0005),
        invalid_action_penalty:  float = -0.01,
        drawdown_penalty_scale:  float = 1.0,
        missed_profit_scale:     float = 0.5,
        render_mode:             Optional[str] = None,
    ):
        super().__init__()

        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got {data.shape}")
        if len(raw_close) != len(data):
            raise ValueError(f"raw_close length {len(raw_close)} != data length {len(data)}")
        if data.shape[0] <= window_size:
            raise ValueError(f"data has {data.shape[0]} rows but window_size={window_size}")

        self.data            = np.asarray(data,      dtype=np.float32)
        self.raw_close       = np.asarray(raw_close, dtype=np.float64)
        self.spec            = symbol_spec
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.max_positions   = max_positions
        self.render_mode     = render_mode

        self.invalid_action_penalty = invalid_action_penalty
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.missed_profit_scale    = missed_profit_scale

        self.n_samples, self.n_features = self.data.shape

        self.action_space = spaces.MultiDiscrete([5, 3])
        obs_dim = window_size * self.n_features + max_positions * 5 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._sim = TradeSimulator(
            symbol_spec    = symbol_spec,
            max_positions  = max_positions,
            slippage_prob  = slippage_prob,
            slippage_range = slippage_range,
        )

        self._step:           int              = 0
        self._balance:        float            = initial_balance
        self._episode_trades: list[ClosedTrade] = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._sim._rng       = self.np_random
        self._sim.reset()
        self._step           = self.window_size
        self._balance        = self.initial_balance
        self._episode_trades = []
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        direction_idx, lot_idx = int(action[0]), int(action[1])
        direction_action       = Action(direction_idx)
        lot_size               = LOT_TIERS[lot_idx]
        current_price          = self._current_raw_price()

        reward       = 0.0
        closed_trade = None

        if direction_action == Action.BUY:
            result = self._sim.open_position(current_price, Direction.LONG, lot_size)
            reward = self._order_reward(result)

        elif direction_action == Action.SELL:
            result = self._sim.open_position(current_price, Direction.SHORT, lot_size)
            reward = self._order_reward(result)

        elif direction_action == Action.CLOSE_LONG:
            result = self._sim.close_position(current_price, Direction.LONG, lot_size)
            reward = self._order_reward(result)
            if result.trade is not None:
                closed_trade = result.trade
                self._episode_trades.append(closed_trade)

        elif direction_action == Action.CLOSE_SHORT:
            result = self._sim.close_position(current_price, Direction.SHORT, lot_size)
            reward = self._order_reward(result)
            if result.trade is not None:
                closed_trade = result.trade
                self._episode_trades.append(closed_trade)

        # Update MFE/MAE on all open positions after action is resolved
        self._sim.update_excursions(current_price)

        self._balance = self.initial_balance + self._sim.cumulative_pnl
        equity        = self._balance + self._sim.total_unrealized_pnl(current_price)

        self._step   += 1
        terminated    = self._is_terminated(equity)

        if terminated and self._sim.has_positions:
            forced = self._sim.close_all(self._current_raw_price())
            self._episode_trades.extend(forced)
            self._balance = self.initial_balance + self._sim.cumulative_pnl
            equity        = self._balance

        obs  = self._observation()
        info = self._info(closed_trade=closed_trade, equity=equity)

        if terminated:
            info["episode_stats"] = self.episode_stats()

        return obs, float(reward), terminated, False, info

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

    def episode_stats(self) -> dict:
        trades = self._episode_trades
        if not trades:
            return {
                "total_trades":  0,
                "winning_trades": 0,
                "losing_trades":  0,
                "win_rate":      0.0,
                "total_pnl":     0.0,
                "avg_pnl":       0.0,
                "max_drawdown":  0.0,
                "final_balance": self._balance,
            }
        pnls   = np.array([t.pnl for t in trades])
        wins   = int((pnls > 0).sum())
        cumsum = np.cumsum(pnls)
        peak   = np.maximum.accumulate(cumsum)
        return {
            "total_trades":   len(trades),
            "winning_trades": wins,
            "losing_trades":  len(trades) - wins,
            "win_rate":       wins / len(trades),
            "total_pnl":      float(pnls.sum()),
            "avg_pnl":        float(pnls.mean()),
            "max_drawdown":   float(np.max(peak - cumsum)),
            "final_balance":  self._balance,
        }

    def _order_reward(self, result: OrderResult) -> float:
        """Compute sparse reward from an OrderResult."""
        if result.invalid:
            return self.invalid_action_penalty

        if result.trade is None:
            return 0.0  # successful open — no reward yet

        t   = result.trade
        ib  = self.initial_balance

        pnl_norm         = t.pnl / ib
        drawdown_penalty = self.drawdown_penalty_scale * max(0.0, -t.mae_pnl) / ib
        missed_penalty   = self.missed_profit_scale    * max(0.0, t.mfe_pnl - t.pnl) / ib

        return pnl_norm - drawdown_penalty - missed_penalty

    def _current_raw_price(self) -> float:
        return float(self.raw_close[min(self._step, self.n_samples - 1)])

    def _is_terminated(self, equity: float) -> bool:
        if self._step >= self.n_samples:
            return True
        if equity < self.initial_balance * 0.5:
            logger.debug("Margin call at step %d.", self._step)
            return True
        return False

    def _observation(self) -> np.ndarray:
        start  = max(0, self._step - self.window_size)
        window = self.data[start:self._step]
        if len(window) < self.window_size:
            pad    = np.zeros((self.window_size - len(window), self.n_features), dtype=np.float32)
            window = np.vstack([pad, window])

        price       = self._current_raw_price()
        pos_vec     = self._sim.position_state_vector(price, self.max_positions)
        unrealized  = self._sim.total_unrealized_pnl(price)
        equity      = self._balance + unrealized
        account_vec = np.array(
            [self._balance / self.initial_balance, equity / self.initial_balance],
            dtype=np.float32,
        )
        return np.concatenate([window.flatten(), pos_vec, account_vec])

    def _info(self, closed_trade: Optional[ClosedTrade] = None, equity: Optional[float] = None) -> dict:
        price = self._current_raw_price()
        if equity is None:
            equity = self._balance + self._sim.total_unrealized_pnl(price)
        return {
            "step":           self._step,
            "price":          price,
            "balance":        self._balance,
            "equity":         equity,
            "open_positions": self._sim.n_positions,
            "total_trades":   len(self._episode_trades),
            "closed_trade":   closed_trade,
        }
