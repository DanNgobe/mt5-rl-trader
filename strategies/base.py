"""
strategies/base.py
------------------
BaseStrategy extends BaseAgent for hand-coded strategies.

Strategies get the same act(env) interface as PPOAgent, so they drop
straight into the shared Evaluator loop.  The only addition over
BaseAgent is _prices_up_to_now() — a convenience helper for indicator
calculations that need the raw price history.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from core.agent import BaseAgent

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


class BaseStrategy(BaseAgent):
    """
    Abstract base for hand-coded baseline strategies.

    Subclass and implement act().  reset() is a no-op by default —
    override it if your strategy carries state across steps.
    """

    name: str = "base_strategy"

    @abstractmethod
    def act(self, env: "TradingEnv") -> int:
        """Return a scalar action int for the current step."""

    def _prices_up_to_now(self, env: "TradingEnv") -> np.ndarray:
        """Raw close prices from episode start up to the current step."""
        return env.raw_close[: env._step]
