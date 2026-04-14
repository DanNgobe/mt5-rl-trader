"""
tests/test_ma_cross.py
----------------------
Unit tests for MACrossStrategy.act().

Covers:
  - Hold when not enough history
  - Open LONG on bullish crossover
  - Open SHORT on bearish crossover
  - Close wrong-direction position before opening the new one
  - Hold when already in the correct direction
  - Toggle-close sends the right action code (same direction as the position)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import List

import numpy as np
import pytest

from core.simulator import Direction, Position
from strategies.ma_cross_strategy import MACrossStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(prices: np.ndarray, positions: List[Position]) -> SimpleNamespace:
    """
    Minimal fake TradingEnv that satisfies MACrossStrategy.act().

    The strategy only reads:
      - env.raw_close   (via _prices_up_to_now → env.raw_close[:env._step])
      - env._step
      - env._sim.positions
      - env.lot_tiers
    """
    step = len(prices)   # _step points one past the last available bar
    env  = SimpleNamespace(
        raw_close = prices,
        _step     = step,
        lot_tiers = [0.1, 0.2],
        _sim      = SimpleNamespace(positions=list(positions)),
    )
    return env


def _long_pos(lot_size: float = 0.1) -> Position:
    return Position(
        ticket      = 1,
        direction   = Direction.LONG,
        lot_size    = lot_size,
        entry_price = 1.1001,
        open_price  = 1.1000,
        open_step   = 0,
    )


def _short_pos(lot_size: float = 0.1) -> Position:
    return Position(
        ticket      = 2,
        direction   = Direction.SHORT,
        lot_size    = lot_size,
        entry_price = 1.0999,
        open_price  = 1.1000,
        open_step   = 0,
    )


def _ramp_up(n: int = 200) -> np.ndarray:
    """Monotonically increasing series — fast MA > slow MA after a warmup."""
    return np.linspace(1.0000, 1.2000, n)


def _ramp_down(n: int = 200) -> np.ndarray:
    """Monotonically decreasing series — fast MA < slow MA after a warmup."""
    return np.linspace(1.2000, 1.0000, n)


# ---------------------------------------------------------------------------
# Decode action helpers (mirrors TradingEnv._action_map with lot_tiers=[0.1,0.2])
# ---------------------------------------------------------------------------
#   0       = HOLD
#   1       = BUY  0.1  (tier 0)
#   2       = SELL 0.1  (tier 0)
#   3       = BUY  0.2  (tier 1)
#   4       = SELL 0.2  (tier 1)

HOLD      = 0
BUY_01    = 1
SELL_01   = 2
BUY_02    = 3
SELL_02   = 4


# ---------------------------------------------------------------------------
# Tests: insufficient history
# ---------------------------------------------------------------------------

class TestInsufficientHistory:

    def test_hold_when_fewer_bars_than_slow(self):
        """With < slow bars of history the strategy must return HOLD."""
        strat = MACrossStrategy(fast=10, slow=50)
        # Only 20 bars — not enough for the 50-bar slow MA
        prices = _ramp_up(20)
        env    = _make_env(prices, positions=[])

        action = strat.act(env)
        assert action == HOLD

    def test_hold_at_exactly_slow_minus_one(self):
        strat = MACrossStrategy(fast=5, slow=20)
        prices = _ramp_up(19)   # one short of slow
        env    = _make_env(prices, positions=[])
        assert strat.act(env) == HOLD

    def test_acts_at_exactly_slow_bars(self):
        """Exactly *slow* bars of history should be enough to generate a signal."""
        strat  = MACrossStrategy(fast=5, slow=20)
        prices = _ramp_up(20)   # exactly slow
        env    = _make_env(prices, positions=[])
        action = strat.act(env)
        assert action != HOLD


# ---------------------------------------------------------------------------
# Tests: open positions
# ---------------------------------------------------------------------------

class TestOpenPositions:

    def test_opens_long_on_bullish_crossover(self):
        """When fast > slow and flat, the strategy must issue BUY (tier 0)."""
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=0)
        prices = _ramp_up(200)
        env    = _make_env(prices, positions=[])

        action = strat.act(env)
        assert action == BUY_01, f"Expected BUY_01={BUY_01}, got {action}"

    def test_opens_short_on_bearish_crossover(self):
        """When fast < slow and flat, the strategy must issue SELL (tier 0)."""
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=0)
        prices = _ramp_down(200)
        env    = _make_env(prices, positions=[])

        action = strat.act(env)
        assert action == SELL_01, f"Expected SELL_01={SELL_01}, got {action}"

    def test_lot_tier_respected_on_open(self):
        """lot_tier=1 must produce BUY at tier 1 (lot 0.2), not tier 0."""
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=1)
        prices = _ramp_up(200)
        env    = _make_env(prices, positions=[])

        action = strat.act(env)
        assert action == BUY_02, f"Expected BUY_02={BUY_02}, got {action}"


# ---------------------------------------------------------------------------
# Tests: hold when already in correct direction
# ---------------------------------------------------------------------------

class TestHoldWhenCorrect:

    def test_hold_when_long_and_bullish(self):
        """Already LONG in a bullish regime → HOLD (no pyramid)."""
        strat  = MACrossStrategy(fast=10, slow=50)
        prices = _ramp_up(200)
        env    = _make_env(prices, positions=[_long_pos()])

        action = strat.act(env)
        assert action == HOLD

    def test_hold_when_short_and_bearish(self):
        """Already SHORT in a bearish regime → HOLD."""
        strat  = MACrossStrategy(fast=10, slow=50)
        prices = _ramp_down(200)
        env    = _make_env(prices, positions=[_short_pos()])

        action = strat.act(env)
        assert action == HOLD


# ---------------------------------------------------------------------------
# Tests: closing the wrong-direction position
# ---------------------------------------------------------------------------

class TestCloseWrongDirection:

    def test_closes_long_when_bearish(self):
        """
        Holding a LONG in a bearish regime.
        The strategy must emit BUY (toggle-closes the LONG) before opening SHORT.

        In TradingEnv, BUY toggles: if a LONG exists → close it.
        So the strategy returns _buy(tier) to close the wrong position.
        """
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=0)
        prices = _ramp_down(200)
        env    = _make_env(prices, positions=[_long_pos(0.1)])

        action = strat.act(env)
        # The LONG lot_size=0.1 maps to tier 0 → close action = BUY_01
        assert action == BUY_01, (
            f"Expected BUY_01={BUY_01} to toggle-close the LONG, got {action}"
        )

    def test_closes_short_when_bullish(self):
        """
        Holding a SHORT in a bullish regime.
        The strategy must emit SELL (toggle-closes the SHORT).
        """
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=0)
        prices = _ramp_up(200)
        env    = _make_env(prices, positions=[_short_pos(0.1)])

        action = strat.act(env)
        # The SHORT lot_size=0.1 maps to tier 0 → close action = SELL_01
        assert action == SELL_01, (
            f"Expected SELL_01={SELL_01} to toggle-close the SHORT, got {action}"
        )

    def test_tier_matched_to_wrong_position_lot(self):
        """
        Position opened at lot_size=0.2 (tier 1).
        Close action must target tier 1, not tier 0.
        """
        strat  = MACrossStrategy(fast=10, slow=50, lot_tier=0)
        prices = _ramp_down(200)
        env    = _make_env(prices, positions=[_long_pos(0.2)])  # tier 1

        action = strat.act(env)
        # To close the LONG 0.2 → BUY at tier 1
        assert action == BUY_02, (
            f"Expected BUY_02={BUY_02} to toggle-close the 0.2-lot LONG, got {action}"
        )


# ---------------------------------------------------------------------------
# Tests: constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError, match="fast"):
            MACrossStrategy(fast=50, slow=10)

    def test_equal_fast_slow_raises(self):
        with pytest.raises(ValueError):
            MACrossStrategy(fast=20, slow=20)

    def test_valid_params_no_error(self):
        MACrossStrategy(fast=10, slow=50, lot_tier=0)


# ---------------------------------------------------------------------------
# Tests: reset is a no-op (stateless between episodes)
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_does_not_raise(self):
        strat = MACrossStrategy()
        strat.reset()   # should be a silent no-op

    def test_strategy_stateless_across_episodes(self):
        """Output should be determined only by current prices, not prior episode state."""
        strat  = MACrossStrategy(fast=10, slow=50)
        prices = _ramp_up(200)

        env_a  = _make_env(prices, positions=[])
        action_a = strat.act(env_a)

        strat.reset()
        env_b  = _make_env(prices, positions=[])
        action_b = strat.act(env_b)

        assert action_a == action_b
