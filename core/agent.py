"""
core/agent.py
-------------
Abstract base class for every agent in the system — both RL models and
hand-coded strategies implement this interface, so the evaluation loop
works identically for all of them.

Interface contract
------------------
    act(env)  — return a scalar int action in [0, env.n_actions-1]:
                   0               = HOLD
                   1 + tier*2      = BUY  lot_tiers[tier]
                   2 + tier*2      = SELL lot_tiers[tier]

    reset()   — called at the start of every episode; default is a no-op
    load()    — optional; strategies don't need it
    save()    — optional; strategies don't need it
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


class BaseAgent(ABC):
    """Common interface for RL agents and hand-coded strategies."""

    name: str = "base_agent"

    def reset(self) -> None:
        """Called at the start of every episode.  Override if stateful."""

    @abstractmethod
    def act(self, env: "TradingEnv") -> int:
        """
        Returns:
            int in [0, env.n_actions-1] — Discrete action index.
        """

    def load(self, path: str) -> None:
        """Load agent state from disk.  No-op by default."""

    def save(self, path: str) -> None:
        """Persist agent state to disk.  No-op by default."""

    # ------------------------------------------------------------------
    # Shared action helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hold() -> int:
        return 0

    @staticmethod
    def _buy(lot_tier: int = 0) -> int:
        # lot_tier 0→action 1, 1→action 3, 2→action 5
        return 1 + lot_tier * 2

    @staticmethod
    def _sell(lot_tier: int = 0) -> int:
        # lot_tier 0→action 2, 1→action 4, 2→action 6
        return 2 + lot_tier * 2
