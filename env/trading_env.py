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
    [position_slots]   len(lot_tiers)*2 slots × 3 features each
    [balance_norm, equity_norm]

Action space — Discrete(2 + 4 * len(lot_tiers))
-------------------------------------------------
    0               = HOLD
    1 + tier*4      = OPEN_BUY   lot_tiers[tier]
    2 + tier*4      = OPEN_SELL  lot_tiers[tier]
    3 + tier*4      = CLOSE_BUY  lot_tiers[tier]
    4 + tier*4      = CLOSE_SELL lot_tiers[tier]
    n_actions - 1   = CLOSE_ALL

Reward
------
    On open:      -(spread_paid * spread_cost_scale) / initial_balance
    On close:     pnl / initial_balance  (sparse mode)
    Every step:   equity_delta / initial_balance  (step mode, replaces close reward)
                  -holding_cost_per_lot * sum(lot sizes)  if positions open
                  -flat_penalty_per_step                  if flat
"""

import logging
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.simulator import (
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
        lot_tiers:               list  = None,
        slippage_prob:           float = 0.3,
        slippage_range:          tuple[float, float] = (0.00001, 0.0005),
        holding_cost_per_lot:    float = 0.0001,
        flat_penalty_per_step:   float = 0.0,
        spread_cost_scale:       float = 2.0,
        reward_mode:             str   = "sparse",  # "sparse" | "step"
        portfolio_offset_factor: float = 0.0,
        volatility_penalty_multiplier: float = 0.0,  # 0.0 to disable variance penalty
        drawdown_penalty:        float = 5.0,
        random_start:            bool  = False,
        max_drawdown_pct:        float = 0.5,
        episode_length:          Optional[int] = None,
        max_positions:           Optional[int] = 10,
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
        self.lot_tiers       = list(lot_tiers) if lot_tiers is not None else list(LOT_TIERS)
        self.render_mode     = render_mode
        self.n_samples       = n

        self.holding_cost_per_lot    = holding_cost_per_lot
        self.flat_penalty_per_step   = flat_penalty_per_step
        self.spread_cost_scale       = spread_cost_scale
        self.reward_mode             = reward_mode
        self.portfolio_offset_factor = portfolio_offset_factor
        self.volatility_penalty_multiplier = volatility_penalty_multiplier
        self.drawdown_penalty        = drawdown_penalty
        self.max_drawdown_pct        = max_drawdown_pct
        self.episode_length          = episode_length
        self.random_start            = random_start
        self.max_positions           = max_positions

        # Pre-compute drawdown threshold — avoids a multiply every step
        self._drawdown_threshold = initial_balance * (1.0 - max_drawdown_pct)

        # Action map: action_idx → (type, Direction, lot_size)
        # 0 = HOLD
        # 1 + i*4 = OPEN_BUY
        # 2 + i*4 = OPEN_SELL
        # 3 + i*4 = CLOSE_BUY
        # 4 + i*4 = CLOSE_SELL
        # n_actions - 1 = CLOSE_ALL
        self._action_map: dict[int, tuple[str, Direction, float]] = {}
        for i, lot in enumerate(self.lot_tiers):
            self._action_map[1 + i * 4] = ("OPEN",  Direction.LONG,  lot)
            self._action_map[2 + i * 4] = ("OPEN",  Direction.SHORT, lot)
            self._action_map[3 + i * 4] = ("CLOSE", Direction.LONG,  lot)
            self._action_map[4 + i * 4] = ("CLOSE", Direction.SHORT, lot)

        # Pre-split action indices by type for O(1) single-pass masking
        self._open_indices:  list[int] = [idx for idx, (t, _, _) in self._action_map.items() if t == "OPEN"]
        self._close_indices: list[tuple[int, Direction, float]] = [
            (idx, d, l) for idx, (t, d, l) in self._action_map.items() if t == "CLOSE"
        ]

        self.n_actions = 2 + 4 * len(self.lot_tiers)
        # n_slots is the number of position slots in the observation vector.
        # If max_positions is unlimited (None or 0), default to 10 slots.
        self.n_slots   = max_positions if (max_positions is not None and max_positions > 0) else 10

        self._price_lags: list[int] = obs_config.get("price_lags", [1, 2, 4, 8, 24])
        self._min_step = max(self._price_lags)

        # Cache obs dimension and indicator flags — avoids dict lookups every step
        self._obs_dim = obs_dim_from_config(obs_config, self.n_slots)
        ind = obs_config.get("indicators", {})
        self._use_rsi      = ind.get("rsi",       {}).get("enabled", True)
        self._use_atr      = ind.get("atr",       {}).get("enabled", True)
        self._use_ema      = ind.get("ema_ratio",  {}).get("enabled", True)
        self._use_boll     = ind.get("bollinger",  {}).get("enabled", True)
        self._use_momentum = ind.get("momentum",   {}).get("enabled", True)
        self._use_session  = ind.get("session",    {}).get("enabled", True)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._sim = TradeSimulator(
            symbol_spec    = symbol_spec,
            lot_tiers      = self.lot_tiers,
            slippage_prob  = slippage_prob,
            slippage_range = slippage_range,
        )

        self._step:           int               = 0
        self._balance:        float             = initial_balance
        self._prev_equity:    float             = initial_balance
        self._episode_trades: list[ClosedTrade] = []
        # deque(maxlen=50): O(1) append+pop vs O(n) list.pop(0)
        self._returns_buffer: deque             = deque([0.0] * 50, maxlen=50)
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
        # Pre-fill with zeros so std starts at 0, not noisy cold-start values
        self._returns_buffer  = deque([0.0] * 50, maxlen=50)
        self._prev_equity     = self.initial_balance

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

    def step(self, action: int):
        action        = int(action)
        current_price = self._current_price()

        reward       = 0.0
        closed_trade = None

        self._sim.update_excursions(current_price)

        if action != 0:  # 0 = HOLD
            if action == self.n_actions - 1:  # CLOSE_ALL
                if self._sim.has_positions:
                    norm_denom = self._prev_equity if self._prev_equity > 0 else self.initial_balance
                    offset_pool = sum(
                        p.unrealized_pnl(current_price, self.spec.contract_size)
                        for p in self._sim._positions
                    ) / norm_denom

                    for pos in list(self._sim._positions):
                        result = self._sim.close_position(current_price, pos.direction, pos.lot_size)
                        if result.trade is not None:
                            self._episode_trades.append(result.trade)
                            closed_trade = result.trade
                        reward += self._order_reward(result, is_close_all=True, offset_pool=offset_pool)
            else:
                act_type, direction, lot_size = self._action_map[action]
                if act_type == "CLOSE":
                    result = self._sim.close_position(current_price, direction, lot_size)
                    reward = self._order_reward(result, is_close_all=False)
                    if result.trade is not None:
                        closed_trade = result.trade
                        self._episode_trades.append(closed_trade)
                else:  # OPEN
                    result = self._sim.open_position(current_price, direction, lot_size, self._step)
                    reward = self._order_reward(result, is_close_all=False)

        self._balance = self.initial_balance + self._sim.cumulative_pnl
        unrealized    = self._sim.total_unrealized_pnl(current_price)
        equity        = self._balance + unrealized

        # Cost application based on reward mode
        total_lots = sum(p.lot_size for p in self._sim._positions)
        if self.reward_mode == "step":
            if total_lots > 0:
                reward -= self.holding_cost_per_lot * total_lots
            else:
                reward -= self.flat_penalty_per_step

            # Step-mode: add per-bar equity delta on top of any open/close reward
            norm_denom  = self._prev_equity if self._prev_equity > 0 else self.initial_balance
            step_return = (equity - self._prev_equity) / norm_denom
            reward     += step_return

            # Differential Sharpe Ratio / Variance Penalty
            if self.volatility_penalty_multiplier > 0:
                self._returns_buffer.append(step_return)
                volatility = float(np.std(self._returns_buffer))
                reward    -= volatility * self.volatility_penalty_multiplier
        else:
            # Sparse mode: holding cost is delayed until close, flat penalty if flat
            if total_lots == 0:
                reward -= self.flat_penalty_per_step

        # Apply Terminal Bankrupt Penalty
        if equity < self._drawdown_threshold:
            reward -= self.drawdown_penalty

        self._prev_equity = equity
        self._step       += 1
        terminated        = self._is_terminated(equity)

        if terminated and self._sim.has_positions:
            close_price = self._current_price()
            for trade in self._sim.close_all(close_price):
                self._episode_trades.append(trade)
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
        Single-pass masking for the explicit Open/Close action space.
        Pre-computed index lists (_open_indices, _close_indices) avoid
        repeatedly scanning the full action map.
        """
        mask     = np.ones(self.n_actions, dtype=bool)
        positions = self._sim._positions
        n_pos     = len(positions)

        # OPEN actions: mask if at position limit
        if self.max_positions and n_pos >= self.max_positions:
            for idx in self._open_indices:
                mask[idx] = False

        # CLOSE / CLOSE_ALL actions
        if n_pos == 0:
            mask[self.n_actions - 1] = False  # CLOSE_ALL
            for idx, _, _ in self._close_indices:
                mask[idx] = False
        else:
            pos_info = [(p.direction, p.lot_size) for p in positions]
            for idx, direction, lot_size in self._close_indices:
                if not any(d == direction and abs(l - lot_size) < 1e-9 for d, l in pos_info):
                    mask[idx] = False

        return mask

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
            "forced_trades":    frc_stats["count"],
            "forced_total_pnl": frc_stats["total_pnl"],
            # Financial metrics include everything
            "max_drawdown":  max_dd,
            "final_balance": self._balance,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _order_reward(self, result: OrderResult, is_close_all: bool = False, offset_pool: float = 0.0) -> float:
        """
        sparse mode: charge spread on open, return pnl on close.
        step mode:   charge spread on open only — pnl is captured by the equity delta.
        """
        norm_denom = self._prev_equity if self._prev_equity > 0 else self.initial_balance

        if result.trade is None:
            # Open: charge spread cost
            actual_spread_norm = result.position.spread_paid / norm_denom
            if self.reward_mode == "step":
                # In step mode, spread is already reflected in equity drop.
                # Only explicitly penalize the extra scale
                extra_scale = max(0.0, self.spread_cost_scale - 1.0)
                return -(actual_spread_norm * extra_scale)
            else:
                return -(actual_spread_norm * self.spread_cost_scale)

        if self.reward_mode == "step":
            # PnL already reflected in equity delta this step — don't double-count
            return 0.0

        trade_pnl = result.trade.pnl / norm_denom

        # Apply holding costs if sparse
        bars_open    = max(0, self._step - result.trade.open_step)
        holding_cost = self.holding_cost_per_lot * result.trade.lot_size * bars_open
        trade_pnl   -= holding_cost

        if is_close_all and self.portfolio_offset_factor > 0 and trade_pnl < 0:
            if offset_pool > 0:
                trade_pnl += self.portfolio_offset_factor * offset_pool
                trade_pnl  = min(0.0, trade_pnl)  # Exploit fix: cap at 0

        return trade_pnl

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self) -> np.ndarray:
        # Write directly into a pre-allocated array — avoids list growth + np.array() call
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        t   = min(self._step, self.n_samples - 1)
        i   = 0  # write cursor

        # 1. Sparse lagged close log returns
        close_lr = self.obs_arrays["close_log_returns"]
        for lag in self._price_lags:
            obs[i] = close_lr[max(0, t - lag)]
            i += 1

        # 2. Indicators (flags cached at __init__ — no dict lookup per step)
        if self._use_rsi:
            obs[i] = self.obs_arrays["rsi"][t];      i += 1
        if self._use_atr:
            obs[i] = self.obs_arrays["atr"][t];      i += 1
        if self._use_ema:
            obs[i] = self.obs_arrays["ema_ratio"][t]; i += 1
        if self._use_boll:
            obs[i] = self.obs_arrays["bollinger"][t]; i += 1
        if self._use_momentum:
            mom = self.obs_arrays["momentum"][t]
            n_mom = len(mom)
            obs[i:i + n_mom] = mom
            i += n_mom
        if self._use_session:
            sess = self.obs_arrays["session"][t]
            n_sess = len(sess)
            obs[i:i + n_sess] = sess
            i += n_sess

        # 3. Position slots: [direction*lot_size, unrealized_pnl_norm, bars_open_norm] × n_slots
        # Sorted by ticket to guarantee stable slot assignment.
        price    = self._current_price()
        cs       = self.spec.contract_size
        ib       = self.initial_balance
        ns       = self.n_samples
        step     = self._step
        sorted_positions = sorted(self._sim._positions, key=lambda p: p.ticket)
        for k in range(self.n_slots):
            if k < len(sorted_positions):
                pos = sorted_positions[k]
                obs[i]     = (float(pos.direction) * pos.lot_size) * 10.0
                obs[i + 1] = (pos.unrealized_pnl(price, cs) / ib) * 10.0
                obs[i + 2] = max(0, step - pos.open_step) / ns
            # else: stays 0.0 from np.zeros
            i += 3

        # 4. Account state
        unrealized = self._sim.total_unrealized_pnl(price)
        equity     = self._balance + unrealized
        obs[i]     = (self._balance - ib) / ib * 10.0
        obs[i + 1] = (equity       - ib) / ib * 10.0

        return obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_price(self) -> float:
        return float(self.raw_close[min(self._step, self.n_samples - 1)])

    def _is_terminated(self, equity: float) -> bool:
        if self._step >= self._end_step:
            return True
        if equity < self._drawdown_threshold:
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
