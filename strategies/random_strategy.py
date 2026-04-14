"""
strategies/random_strategy.py
-----------------------------
Uniformly random actions — the absolute floor baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .base import BaseStrategy

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


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
