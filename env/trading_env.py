"""
Gymnasium environment wrapping the MT5 hedging simulator.

Observation vector (all features configurable via obs_config)
-------------------------------------------------------------
    [close_log_returns at sparse lags]   e.g. lags=[1,2,4,8,24]
    [rsi]                                normalised to [-1, 1]
    [atr]                                ATR / close (volatility ratio)
    [ema_ratio]                          EMA(fast)/EMA(slow) - 1
    [bollinger]                          Bollinger %B normalised [-1,1]
    [momentum_5, momentum_20, momentum_50]
    [hour_sin, hour_cos, dow_sin, dow_cos]
    [position_slots]                     max_positions × 3
    [balance_norm, equity_norm]

    Observation dimension is computed dynamically from obs_config +
    max_positions — changing either in config.yaml automatically resizes
    the space.

Action space — MultiDiscrete([5, 3])
-------------------------------------
    axis-0  direction : 0=HOLD  1=BUY  2=SELL  3=CLOSE_LONG  4=CLOSE_SHORT
    axis-1  lot_tier  : 0=0.01  1=0.02  2=0.05

Reward
------
    On open:    -(round_trip_spread_cost) / initial_balance
    On close:   pnl_norm - drawdown_penalty - missed_profit_penalty
    On invalid: invalid_action_penalty  (flat negative)
    Every step: step_reward_scale * unrealized_pnl_delta / initial_balance
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
from env.preprocessor import obs_dim_from_config

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        obs_arrays:              dict,
        raw_close:               np.ndarray,
        symbol_spec:             SymbolSpec,
        obs_config:              dict,
        initial_balance:         float = 10_000.0,
        max_positions:           int   = 3,
        slippage_prob:           float = 0.3,
        slippage_range:          tuple[float, float] = (0.00001, 0.0005),
        invalid_action_penalty:  float = -0.01,
        drawdown_penalty_scale:  float = 1.0,
        missed_profit_scale:     float = 0.5,
        step_reward_scale:       float = 0.1,
        render_mode:             Optional[str] = None,
    ):
        super().__init__()

        n = len(raw_close)
        if n < 2:
            raise ValueError("raw_close must have at least 2 rows")

        self.obs_arrays      = obs_arrays
        self.raw_close       = np.asarray(raw_close, dtype=np.float64)
        self.spec            = symbol_spec
        self.obs_config      = obs_config
        self.initial_balance = initial_balance
        self.max_positions   = max_positions
        self.render_mode     = render_mode
        self.n_samples       = n

        self.invalid_action_penalty = invalid_action_penalty
        self.drawdown_penalty_scale = drawdown_penalty_scale
        self.missed_profit_scale    = missed_profit_scale
        self.step_reward_scale      = step_reward_scale

        # Sparse lag indices (bars back from current step)
        self._price_lags: list[int] = obs_config.get("price_lags", [1, 2, 4, 8, 24])
        # Minimum step needed so all lags are valid
        self._min_step = max(self._price_lags)

        obs_dim = obs_dim_from_config(obs_config, max_positions)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([5, 3])

        self._sim = TradeSimulator(
            symbol_spec    = symbol_spec,
            max_positions  = max_positions,
            slippage_prob  = slippage_prob,
            slippage_range = slippage_range,
        )

        self._step:           int               = 0
        self._balance:        float             = initial_balance
        self._prev_unrealized: float            = 0.0
        self._episode_trades: list[ClosedTrade] = []

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._sim._rng        = self.np_random
        self._sim.reset()
        self._step            = self._min_step
        self._balance         = self.initial_balance
        self._prev_unrealized = 0.0
        self._episode_trades  = []
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        direction_idx, lot_idx = int(action[0]), int(action[1])
        direction_action       = Action(direction_idx)
        lot_size               = LOT_TIERS[lot_idx]
        current_price          = self._current_price()

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

        self._sim.update_excursions(current_price)

        self._balance = self.initial_balance + self._sim.cumulative_pnl
        unrealized    = self._sim.total_unrealized_pnl(current_price)
        equity        = self._balance + unrealized

        # Step shaping: reward unrealized PnL improving, penalise deterioration
        reward += self.step_reward_scale * (unrealized - self._prev_unrealized) / self.initial_balance
        self._prev_unrealized = unrealized

        self._step += 1
        terminated  = self._is_terminated(equity)

        if terminated and self._sim.has_positions:
            close_price = self._current_price()
            for trade in self._sim.close_all(close_price):
                self._episode_trades.append(trade)
                reward += self._order_reward(OrderResult(success=True, trade=trade))
            self._balance         = self.initial_balance + self._sim.cumulative_pnl
            self._prev_unrealized = 0.0
            equity                = self._balance

        obs  = self._observation()
        info = self._info(closed_trade=closed_trade, equity=equity)
        if terminated:
            info["episode_stats"] = self.episode_stats()

        return obs, float(reward), terminated, False, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        price  = self._current_price()
        equity = self._balance + self._sim.total_unrealized_pnl(price)
        print(
            f"[{self.spec.name}] step={self._step}/{self.n_samples}  "
            f"balance={self._balance:.2f}  equity={equity:.2f}  "
            f"positions={self._sim.n_positions}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Episode stats
    # ------------------------------------------------------------------

    def episode_stats(self) -> dict:
        trades = self._episode_trades
        if not trades:
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
                "max_drawdown": 0.0, "final_balance": self._balance,
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

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _order_reward(self, result: OrderResult) -> float:
        """
        On open:    charge estimated round-trip spread cost.
        On close:   PnL norm - MAE penalty - missed profit penalty.
        Invalid:    flat penalty.
        """
        if result.invalid:
            return self.invalid_action_penalty

        if result.trade is None:
            # Charge round-trip spread upfront (open + anticipated close)
            return -(result.position.spread_paid * 2.0) / self.initial_balance

        t  = result.trade
        ib = self.initial_balance

        pnl_norm         = t.pnl / ib
        drawdown_penalty = self.drawdown_penalty_scale * max(0.0, -t.mae_pnl) / ib

        if t.mfe_pnl > 0:
            capture_ratio  = t.pnl / t.mfe_pnl
            missed_penalty = self.missed_profit_scale * max(0.0, 1.0 - capture_ratio)
        else:
            missed_penalty = 0.0

        return pnl_norm - drawdown_penalty - missed_penalty

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        parts = []
        t     = min(self._step, self.n_samples - 1)  # Clamp to valid index range
        ind   = self.obs_config.get("indicators", {})

        # 1. Sparse lagged close log returns
        close_lr = self.obs_arrays["close_log_returns"]
        for lag in self._price_lags:
            idx = max(0, t - lag)
            parts.append(float(close_lr[idx]))

        # 2. Indicators — index current step, fall back to 0 if missing
        if ind.get("rsi",       {}).get("enabled", True):
            parts.append(float(self.obs_arrays["rsi"][t]))

        if ind.get("atr",       {}).get("enabled", True):
            parts.append(float(self.obs_arrays["atr"][t]))

        if ind.get("ema_ratio", {}).get("enabled", True):
            parts.append(float(self.obs_arrays["ema_ratio"][t]))

        if ind.get("bollinger", {}).get("enabled", True):
            parts.append(float(self.obs_arrays["bollinger"][t]))

        if ind.get("momentum",  {}).get("enabled", True):
            mom = self.obs_arrays["momentum"][t]   # shape (n_periods,)
            parts.extend(mom.tolist())

        if ind.get("session",   {}).get("enabled", True):
            parts.extend(self.obs_arrays["session"][t].tolist())

        # 3. Position slots: [direction, lot_norm, unrealized_pnl_norm] × max_positions
        price = self._current_price()
        for i in range(self.max_positions):
            if i < len(self._sim.positions):
                pos = self._sim.positions[i]
                parts.append(float(pos.direction))
                parts.append(pos.lot_size / LOT_TIERS[-1])   # norm to [0,1]
                parts.append(pos.unrealized_pnl(price, self.spec.contract_size) / self.initial_balance)
            else:
                parts.extend([0.0, 0.0, 0.0])

        # 4. Account state
        unrealized = self._sim.total_unrealized_pnl(price)
        equity     = self._balance + unrealized
        parts.append(self._balance / self.initial_balance)
        parts.append(equity        / self.initial_balance)

        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_price(self) -> float:
        return float(self.raw_close[min(self._step, self.n_samples - 1)])

    def _is_terminated(self, equity: float) -> bool:
        if self._step >= self.n_samples:
            return True
        if equity < self.initial_balance * 0.5:
            logger.debug("Margin call at step %d.", self._step)
            return True
        return False

    def _info(self, closed_trade: Optional[ClosedTrade] = None,
              equity: Optional[float] = None) -> dict:
        price = self._current_price()
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
