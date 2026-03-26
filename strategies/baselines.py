"""
strategies/baselines.py
-----------------------
Concrete baseline strategies for benchmarking the RL agent.

RandomStrategy
    Samples uniformly from env.n_actions each step.

MACrossStrategy
    Classic moving-average crossover on raw close prices.
    - fast MA crosses above slow MA → BUY  (open long)
    - fast MA crosses below slow MA → SELL (open short)
    - Opposite crossover while a position is open → toggle-close then flip
    - No signal → HOLD
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.simulator import Direction
from .base import BaseStrategy

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomStrategy(BaseStrategy):
    """
    Uniformly random actions — the absolute floor baseline.

    Parameters
    ----------
    seed : int or None
    lot_tier : int
        Fixed lot tier index used for MA cross (ignored for random).
    """

    name = "random"

    def __init__(self, seed: Optional[int] = 42, lot_tier: int = 0):
        self._rng      = np.random.default_rng(seed)
        self._lot_tier = lot_tier

    def reset(self) -> None:
        pass

    def act(self, env: "TradingEnv") -> int:
        return int(self._rng.integers(0, env.n_actions))


# ---------------------------------------------------------------------------
# Moving-average crossover
# ---------------------------------------------------------------------------

class MACrossStrategy(BaseStrategy):
    """
    Simple dual moving-average crossover strategy.

    Parameters
    ----------
    fast : int
        Fast MA period in bars (default 10).
    slow : int
        Slow MA period in bars (default 50).
    lot_tier : int
        Tier index into env.lot_tiers used for new positions (default 0).
    """

    name = "ma_cross"

    def __init__(self, fast: int = 10, slow: int = 50, lot_tier: int = 0):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow}).")
        self._fast     = fast
        self._slow     = slow
        self._lot_tier = lot_tier
        self._prev_fast_above: Optional[bool] = None
        self._pending_action:  Optional[int]  = None

    def reset(self) -> None:
        self._prev_fast_above = None
        self._pending_action  = None

    def act(self, env: "TradingEnv") -> int:
        if self._pending_action is not None:
            action               = self._pending_action
            self._pending_action = None
            return action

        prices = self._prices_up_to_now(env)
        if len(prices) < self._slow:
            return self._hold()

        fast_ma    = float(prices[-self._fast:].mean())
        slow_ma    = float(prices[-self._slow:].mean())
        fast_above = fast_ma > slow_ma

        if self._prev_fast_above is None:
            self._prev_fast_above = fast_above
            return self._hold()

        crossover_up   = fast_above and not self._prev_fast_above
        crossover_down = not fast_above and self._prev_fast_above
        self._prev_fast_above = fast_above

        positions = env._sim.positions

        if crossover_up:
            short_pos = next((p for p in positions if p.direction == Direction.SHORT), None)
            if short_pos is not None:
                # Find tier index in env.lot_tiers matching this position's lot
                close_tier = min(range(len(env.lot_tiers)),
                                 key=lambda i: abs(env.lot_tiers[i] - short_pos.lot_size))
                self._pending_action = self._buy(self._lot_tier)
                return self._sell(close_tier)   # toggle-close the short
            elif env._sim.n_positions == 0:
                return self._buy(self._lot_tier)

        elif crossover_down:
            long_pos = next((p for p in positions if p.direction == Direction.LONG), None)
            if long_pos is not None:
                close_tier = min(range(len(env.lot_tiers)),
                                 key=lambda i: abs(env.lot_tiers[i] - long_pos.lot_size))
                self._pending_action = self._sell(self._lot_tier)
                return self._buy(close_tier)    # toggle-close the long
            elif env._sim.n_positions == 0:
                return self._sell(self._lot_tier)

        return self._hold()