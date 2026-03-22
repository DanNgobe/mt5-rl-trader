"""
strategies/base.py
------------------
Abstract base class for all hand-coded baseline strategies.

Every strategy receives the live TradingEnv and returns the same
MultiDiscrete([4, 3]) action array that the PPO agent returns:

    [direction_idx, lot_tier_idx]

    direction : 0=HOLD  1=BUY  2=SELL  3=CLOSE
    lot_tier  : 0=0.01  1=0.02  2=0.05

Strategies read from env.raw_close (un-normalised prices) and from
env._sim (open positions) — whatever they actually need.  They are
NOT forced through the preprocessed observation vector.

This makes them drop-in replacements for model.predict() in the
evaluation loop, and they are fully compatible with the visualiser.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


class BaseStrategy(ABC):
    """
    Abstract baseline strategy.

    Subclass and implement act().  Everything else is optional.
    """

    # Human-readable name used in logs and saved results
    name: str = "base"

    def reset(self) -> None:
        """
        Called at the start of every episode.

        Override if your strategy carries state across steps
        (e.g. an indicator warm-up buffer, position tracking).
        Default is a no-op.
        """

    @abstractmethod
    def act(self, env: "TradingEnv") -> np.ndarray:
        """
        Decide an action given the current environment state.

        Args:
            env: Live TradingEnv instance.  Read-only — do not call
                 env.step() or env.reset() from here.

        Returns:
            np.ndarray of shape (2,) and dtype int32:
                [direction_idx, lot_tier_idx]
        """

    # ------------------------------------------------------------------
    # Convenience helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _hold() -> np.ndarray:
        return np.array([0, 0], dtype=np.int32)   # HOLD, lot irrelevant

    @staticmethod
    def _buy(lot_tier: int = 0) -> np.ndarray:
        return np.array([1, lot_tier], dtype=np.int32)

    @staticmethod
    def _sell(lot_tier: int = 0) -> np.ndarray:
        return np.array([2, lot_tier], dtype=np.int32)

    @staticmethod
    def _close(lot_tier: int = 0) -> np.ndarray:
        return np.array([3, lot_tier], dtype=np.int32)

    def _prices_up_to_now(self, env: "TradingEnv") -> np.ndarray:
        """
        Return raw close prices from the start of the episode up to
        and including the current step.  Useful for indicator calc.
        """
        return env.raw_close[: env._step]