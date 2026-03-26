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
    On open:      -(spread_paid * spread_cost_scale) / initial_balance
    On close:     pnl / initial_balance
    On invalid:   -invalid_action_penalty  (flat)
    Every step:   -holding_cost_per_lot * sum(lot sizes)  if positions open
                  -flat_penalty_per_step                  if flat
    On terminal:  (final_equity - initial_balance) / initial_balance
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
        invalid_action_penalty:  float = 0.001,
        holding_cost_per_lot:    float = 0.0001,
        flat_penalty_per_step:   float = 0.0,
        spread_cost_scale:       float = 2.0,
        wrong_lot_penalty:       float = 0.0002,
        portfolio_offset_factor: float = 0.0,
        max_drawdown_pct:        float = 0.5,
        episode_length:          Optional[int] = None,
        random_start:            bool  = False,
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
        self.holding_cost_per_lot   = holding_cost_per_lot
        self.flat_penalty_per_step  = flat_penalty_per_step
        self.spread_cost_scale      = spread_cost_scale
        self.wrong_lot_penalty      = wrong_lot_penalty
        self.portfolio_offset_factor = portfolio_offset_factor
        self.max_drawdown_pct       = max_drawdown_pct
        self.episode_length         = episode_length
        self.random_start           = random_start

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
        self._episode_trades: list[ClosedTrade] = []
        self._start_step:     int               = self._min_step
        self._end_step:       int               = n

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._sim._rng        = self.np_random
        self._sim.reset()
        self._balance         = self.initial_balance
        self._episode_trades  = []

        if self.random_start and self.episode_length is not None:
            # Pick a random window — ensure enough room for lags at the start
            # and a full episode_length window after the start point.
            max_start = self.n_samples - self.episode_length - self._min_step
            if max_start < self._min_step:
                # Dataset too short for the window — fall back to full run
                self._start_step = self._min_step
                self._end_step   = self.n_samples
            else:
                self._start_step = int(self.np_random.integers(self._min_step, max_start + 1))
                self._end_step   = self._start_step + self.episode_length
        else:
            self._start_step = self._min_step
            self._end_step   = self.n_samples

        self._step = self._start_step
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        direction_idx, lot_idx = int(action[0]), int(action[1])
        direction_action       = Action(direction_idx)
        lot_size               = LOT_TIERS[lot_idx]
        current_price          = self._current_price()

        reward       = 0.0
        closed_trade = None

        # Update excursions first so MFE/MAE reflect the current bar's price
        # before any close (including terminal close_all) reads them.
        self._sim.update_excursions(current_price)

        if direction_action == Action.BUY:
            result = self._sim.open_position(current_price, Direction.LONG, lot_size, self._step)
            reward = self._order_reward(result)

        elif direction_action == Action.SELL:
            result = self._sim.open_position(current_price, Direction.SHORT, lot_size, self._step)
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

        self._balance = self.initial_balance + self._sim.cumulative_pnl
        unrealized    = self._sim.total_unrealized_pnl(current_price)
        equity        = self._balance + unrealized

        # Holding cost: scales with total lots held, discourages overtrading
        total_lots = sum(p.lot_size for p in self._sim.positions)
        if total_lots > 0:
            reward -= self.holding_cost_per_lot * total_lots
        else:
            # Flat penalty: nudges agent to engage when no positions are open
            reward -= self.flat_penalty_per_step

        self._step += 1
        terminated  = self._is_terminated(equity)

        if terminated and self._sim.has_positions:
            close_price = self._current_price()
            for trade in self._sim.close_all(close_price):
                self._episode_trades.append(trade)
                # Do NOT add trade.pnl here — the terminal bonus below uses
                # final balance which already includes these PnLs via cumulative_pnl.
                # Adding them here would double-count every forced close.
            self._balance = self.initial_balance + self._sim.cumulative_pnl
            equity        = self._balance

        if terminated:
            # Terminal bonus: overall episode return as a final shaping signal.
            # Uses only the realised balance (all positions closed above), so no double-count.
            reward += (self._balance - self.initial_balance) / self.initial_balance

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
            f"[{self.spec.name}] step={self._step}/{self._end_step}  "
            f"balance={self._balance:.2f}  equity={equity:.2f}  "
            f"positions={self._sim.n_positions}"
        )

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Action masking (MaskablePPO / sb3-contrib)
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask over the MultiDiscrete([5, 3]) action space.
        
        MaskablePPO expects shape (n_actions_per_dim,) = (5+3,) = (8,).
        Structure: [dir_mask (5 elements), lot_mask (3 elements)]
          - dir_mask[i]: whether direction i is valid
          - lot_mask[j]: whether lot tier j is valid
        
        Masking rules for direction:
          - HOLD (0): always valid
          - BUY (1): valid only if not at max_positions
          - SELL (2): valid only if not at max_positions
          - CLOSE_LONG (3): valid only if long position(s) exist
          - CLOSE_SHORT (4): valid only if short position(s) exist
        
        For lot tiers: we allow all tiers universally, since MaskablePPO
        doesn't natively support joint masking (lot validity depends on direction).
        The environment will reject invalid lot-direction combinations via
        the reward penalty system.
        """
        positions   = self._sim.positions
        at_cap      = self._sim.n_positions >= self.max_positions
        has_long    = any(p.direction == Direction.LONG for p in positions)
        has_short   = any(p.direction == Direction.SHORT for p in positions)

        # Mask for 5 direction choices
        dir_mask = np.array([
            True,           # Action.HOLD: always valid
            not at_cap,     # Action.BUY: valid if not at capacity
            not at_cap,     # Action.SELL: valid if not at capacity
            has_long,       # Action.CLOSE_LONG: valid if longs exist
            has_short,      # Action.CLOSE_SHORT: valid if shorts exist
        ], dtype=bool)

        # Mask for 3 lot tier choices (all always valid at this level)
        # Individual lot matching for CLOSE actions is handled via reward penalties
        lot_mask = np.ones(3, dtype=bool)

        # Concatenate direction and lot masks
        return np.concatenate([dir_mask, lot_mask])

    # ------------------------------------------------------------------
    # Episode stats
    # ------------------------------------------------------------------

    def episode_stats(self) -> dict:
        all_trades = self._episode_trades
        vol_trades = [t for t in all_trades if not t.forced]   # agent-chosen closes
        frc_trades = [t for t in all_trades if t.forced]       # episode-end forced closes

        def _trade_stats(trades) -> dict:
            if not trades:
                return {"count": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                        "total_pnl": 0.0, "avg_pnl": 0.0}
            pnls = np.array([t.pnl for t in trades])
            wins = int((pnls > 0).sum())
            return {
                "count":     len(trades),
                "wins":      wins,
                "losses":    len(trades) - wins,
                "win_rate":  wins / len(trades),
                "total_pnl": float(pnls.sum()),
                "avg_pnl":   float(pnls.mean()),
            }

        vol_stats = _trade_stats(vol_trades)
        frc_stats = _trade_stats(frc_trades)

        # Drawdown uses all trades (forced closes affect real equity)
        if all_trades:
            all_pnls = np.array([t.pnl for t in all_trades])
            cumsum   = np.cumsum(all_pnls)
            peak     = np.maximum.accumulate(cumsum)
            max_dd   = float(np.max(peak - cumsum))
        else:
            max_dd = 0.0

        return {
            # Voluntary (agent-chosen) trade metrics — primary quality signal
            "total_trades":   vol_stats["count"],
            "winning_trades": vol_stats["wins"],
            "losing_trades":  vol_stats["losses"],
            "win_rate":       vol_stats["win_rate"],
            "total_pnl":      vol_stats["total_pnl"],
            "avg_pnl":        vol_stats["avg_pnl"],
            # Forced close summary — informational
            "forced_trades":      frc_stats["count"],
            "forced_total_pnl":   frc_stats["total_pnl"],
            # Financial metrics include everything
            "max_drawdown":   max_dd,
            "final_balance":  self._balance,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _order_reward(self, result: OrderResult, direction: Optional[Direction] = None) -> float:
        """
        On open:    charge estimated round-trip spread cost.
        On close:   pnl / initial_balance, optionally offset by other positions' unrealized gains.
        Invalid:
          - no_matching_lot (right direction, wrong lot tier): small penalty.
          - everything else (no position in direction, open at cap): full penalty.
        
        Portfolio offset (grid/martingale):
          When closing a losing trade, reward is adjusted by other positions' unrealized PnL.
          Formula: close_reward = trade_pnl + (portfolio_offset_factor * other_unrealized_pnl)
        """
        if result.invalid:
            if result.reason == "no_matching_lot":
                # Agent tried to close the right direction but picked the wrong lot tier.
                # Intent is correct — apply a small nudge, not a full deterrent.
                return -self.wrong_lot_penalty
            return -self.invalid_action_penalty

        if result.trade is None:
            # Charge spread cost upfront — scale configurable (default round-trip = 2.0)
            return -(result.position.spread_paid * self.spread_cost_scale) / self.initial_balance

        # BASE CLOSE REWARD
        trade_pnl = result.trade.pnl / self.initial_balance
        
        # PORTFOLIO-AWARE ADJUSTMENT (grid/martingale support)
        if self.portfolio_offset_factor > 0 and trade_pnl < 0:
            # Trade is losing — offset with other positions' unrealized gains
            current_price = self._current_price()
            other_unrealized = sum(
                p.unrealized_pnl(current_price, self.spec.contract_size)
                for p in self._sim.positions
            ) / self.initial_balance
            
            # Only apply offset if portfolio (excluding this trade) is net positive
            if other_unrealized > 0:
                offset_reward = trade_pnl + (self.portfolio_offset_factor * other_unrealized)
                return offset_reward
        
        return trade_pnl

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

        # 3. Position slots: [direction*lot_size, unrealized_pnl_norm, bars_open_norm] × max_positions
        # Sorted by ticket to guarantee stable slot assignment — closing a position never
        # shifts remaining positions into different slots, preventing spurious state changes.
        price = self._current_price()
        sorted_positions = sorted(self._sim.positions, key=lambda p: p.ticket)
        for i in range(self.max_positions):
            if i < len(sorted_positions):
                pos = sorted_positions[i]
                bars_open = max(0, self._step - pos.open_step)
                parts.append(float(pos.direction) * pos.lot_size)
                parts.append(pos.unrealized_pnl(price, self.spec.contract_size) / self.initial_balance)
                parts.append(bars_open / self.n_samples)
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
        if self._step >= self._end_step:
            return True
        if equity < self.initial_balance * (1.0 - self.max_drawdown_pct):
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
