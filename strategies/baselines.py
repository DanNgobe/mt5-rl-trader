"""
strategies/baselines.py
-----------------------
Concrete baseline strategies for benchmarking the RL agent.

RandomStrategy
    Samples direction and lot tier uniformly at random each step.
    Sets the floor — the RL agent must beat this comfortably.

MACrossStrategy
    Classic moving-average crossover on raw close prices.
    - fast MA crosses above slow MA → BUY  (open long)
    - fast MA crosses below slow MA → SELL (open short)
    - Opposite crossover while a position is open → CLOSE then flip
    - No signal → HOLD

    Operates on env.raw_close directly (un-normalised prices).
    No SciPy / TA-Lib dependency — pure numpy rolling means.

Both strategies use lot_tier=0 (0.01 lots) throughout to keep
position sizing neutral and isolate the signal quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

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
        Random seed for reproducibility.  None = non-deterministic.
    lot_tier : int
        Fixed lot tier index (0=0.01, 1=0.02, 2=0.05).
        Default 0 keeps position sizes small and neutral.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = 42, lot_tier: int = 0):
        self._rng      = np.random.default_rng(seed)
        self._lot_tier = lot_tier

    def reset(self) -> None:
        # RNG state is NOT reset between episodes so successive episodes
        # are independent draws, not the same random sequence.
        pass

    def act(self, env: "TradingEnv") -> np.ndarray:
        direction = int(self._rng.integers(0, 5))   # 0-4 inclusive
        return np.array([direction, self._lot_tier], dtype=np.int32)


# ---------------------------------------------------------------------------
# Moving-average crossover
# ---------------------------------------------------------------------------

class MACrossStrategy(BaseStrategy):
    """
    Simple dual moving-average crossover strategy.

    Signal logic
    ------------
    - Requires at least `slow` bars of price history before acting.
      Returns HOLD during the warm-up period.
    - Golden cross (fast > slow, previously fast <= slow) → enter LONG:
        * If SHORT position open: CLOSE it first (next step will BUY).
        * If no position: BUY.
    - Death cross (fast < slow, previously fast >= slow) → enter SHORT:
        * If LONG position open: CLOSE it first (next step will SELL).
        * If no position: SELL.
    - No crossover: HOLD (let existing position run).

    Parameters
    ----------
    fast : int
        Fast MA period in bars (default 10).
    slow : int
        Slow MA period in bars (default 50).
    lot_tier : int
        Fixed lot tier index (0=0.01, 1=0.02, 2=0.05).
    """

    name = "ma_cross"

    def __init__(
        self,
        fast:     int = 10,
        slow:     int = 50,
        lot_tier: int = 0,
    ):
        if fast >= slow:
            raise ValueError(
                f"fast ({fast}) must be shorter than slow ({slow})."
            )
        self._fast     = fast
        self._slow     = slow
        self._lot_tier = lot_tier

        # Crossover state from the previous step
        self._prev_fast_above: Optional[bool] = None

        # Deferred action: when we need to close before flipping,
        # we emit CLOSE this step and store the flip for next step.
        self._pending_action: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_fast_above = None
        self._pending_action  = None

    def act(self, env: "TradingEnv") -> np.ndarray:
        # Emit deferred flip action from previous step (close-then-flip)
        if self._pending_action is not None:
            action                = self._pending_action
            self._pending_action  = None
            return action

        prices = self._prices_up_to_now(env)

        # Not enough history yet
        if len(prices) < self._slow:
            return self._hold()

        fast_ma = float(prices[-self._fast:].mean())
        slow_ma = float(prices[-self._slow:].mean())

        fast_above = fast_ma > slow_ma

        # First bar after warm-up — just record state, no signal yet
        if self._prev_fast_above is None:
            self._prev_fast_above = fast_above
            return self._hold()

        crossover_up   = fast_above and not self._prev_fast_above
        crossover_down = not fast_above and self._prev_fast_above

        self._prev_fast_above = fast_above

        n_positions = env._sim.n_positions
        positions   = env._sim.positions

        if crossover_up:
            has_short = any(p.direction.name == "SHORT" for p in positions)
            if has_short:
                self._pending_action = self._buy(self._lot_tier)
                return self._close_short(self._lot_tier)
            elif n_positions == 0:
                return self._buy(self._lot_tier)

        elif crossover_down:
            has_long = any(p.direction.name == "LONG" for p in positions)
            if has_long:
                self._pending_action = self._sell(self._lot_tier)
                return self._close_long(self._lot_tier)
            elif n_positions == 0:
                return self._sell(self._lot_tier)

        return self._hold()