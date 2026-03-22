"""
core/agent.py
-------------
Abstract base class for every agent in the system — both RL models and
hand-coded strategies implement this interface, so the evaluation loop
works identically for all of them.

Interface contract
------------------
    act(env)  — given a live TradingEnv, return a (2,) int32 action array
                 [direction_idx, lot_tier_idx]  (same as MultiDiscrete([4,3]))

    reset()   — called at the start of every episode; default is a no-op
    load()    — optional; strategies don't need it
    save()    — optional; strategies don't need it

Why act(env) instead of act(obs)?
    Strategies need raw (un-normalised) prices for indicator calculations.
    Passing the full env gives them that without breaking the RL agent,
    which simply calls env._observation() internally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


class BaseAgent(ABC):
    """
    Common interface for RL agents and hand-coded strategies.

    Subclasses must implement act().  All other methods have sensible
    defaults so strategies don't need boilerplate load/save stubs.
    """

    #: Human-readable name used in logs and saved results.
    name: str = "base_agent"

    def reset(self) -> None:
        """Called at the start of every episode.  Override if stateful."""

    @abstractmethod
    def act(self, env: "TradingEnv") -> np.ndarray:
        """
        Decide an action given the current environment state.

        Args:
            env: Live TradingEnv instance (read-only — never call step/reset).

        Returns:
            np.ndarray of shape (2,) dtype int32: [direction_idx, lot_tier_idx]
        """

    def load(self, path: str) -> None:
        """Load agent state from disk.  No-op by default."""

    def save(self, path: str) -> None:
        """Persist agent state to disk.  No-op by default."""

    # ------------------------------------------------------------------
    # Shared action helpers — available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _hold() -> np.ndarray:
        return np.array([0, 0], dtype=np.int32)

    @staticmethod
    def _buy(lot_tier: int = 0) -> np.ndarray:
        return np.array([1, lot_tier], dtype=np.int32)

    @staticmethod
    def _sell(lot_tier: int = 0) -> np.ndarray:
        return np.array([2, lot_tier], dtype=np.int32)

    @staticmethod
    def _close_long(lot_tier: int = 0) -> np.ndarray:
        return np.array([3, lot_tier], dtype=np.int32)

    @staticmethod
    def _close_short(lot_tier: int = 0) -> np.ndarray:
        return np.array([4, lot_tier], dtype=np.int32)
