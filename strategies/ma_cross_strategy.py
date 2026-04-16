"""
strategies/ma_cross_strategy.py
-------------------------------
Classic moving-average crossover on raw close prices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from core.simulator import Direction
from .base import BaseStrategy

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


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

    def reset(self) -> None:
        pass

    def act(self, env: "TradingEnv") -> int:
        prices = self._prices_up_to_now(env)
        if len(prices) < self._slow:
            return self._hold()

        fast_ma = float(prices[-self._fast:].mean())
        slow_ma = float(prices[-self._slow:].mean())
        
        # The target state is to be LONG if fast > slow, and SHORT otherwise.
        target_dir = Direction.LONG if fast_ma > slow_ma else Direction.SHORT
        positions  = env._sim.positions

        # 1. Close any positions in the wrong direction
        wrong_pos = next((p for p in positions if p.direction != target_dir), None)
        if wrong_pos is not None:
            # Toggle-close the wrong position
            tier = min(range(len(env.lot_tiers)),
                       key=lambda i: abs(env.lot_tiers[i] - wrong_pos.lot_size))
            if wrong_pos.direction == Direction.LONG:
                return self._close_buy(tier)   # CLOSE the long
            else:
                return self._close_sell(tier)  # CLOSE the short

        # 2. If flat, open in the target direction
        if not positions:
            if target_dir == Direction.LONG:
                return self._buy(self._lot_tier)
            else:
                return self._sell(self._lot_tier)

        return self._hold()
