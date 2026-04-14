"""
tests/test_visualiser_markers.py
---------------------------------
Verify that EpisodeVisualiser records trade markers at the raw market price
(Position.open_price / ClosedTrade.close_price) rather than at the fill price
that includes spread + slippage.

These tests exercise the marker-recording logic in EpisodeVisualiser.update()
without opening a matplotlib window by patching the figure-build step.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.simulator import (
    ClosedTrade,
    Direction,
    Position,
    SymbolSpec,
    TradeSimulator,
)
from core.visualiser import EpisodeVisualiser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPREAD = 0.00020   # 2 pip spread
SLIP   = 0.00010   # 1 pip slippage


def _make_spec() -> SymbolSpec:
    return SymbolSpec(
        name          = "EURUSD",
        pip_value     = 0.0001,
        pip_location  = 4,
        contract_size = 100_000,
        spread_pips   = 2.0,
        min_lot       = 0.01,
        max_lot       = 10.0,
        margin_rate   = 0.01,
    )


def _make_position(
    direction: Direction,
    open_price: float,
    lot_size: float = 0.1,
    open_step: int = 5,
) -> Position:
    """Return a Position whose entry_price deliberately differs from open_price."""
    fill = open_price + SPREAD + SLIP if direction == Direction.LONG \
        else open_price - SPREAD - SLIP
    return Position(
        ticket      = 1,
        direction   = direction,
        lot_size    = lot_size,
        entry_price = fill,       # fill price — includes spread+slippage
        open_price  = open_price, # raw market price — what the chart shows
        spread_paid = SPREAD,
        slippage    = SLIP,
        open_step   = open_step,
    )


def _make_closed_trade(
    direction: Direction,
    open_price: float,
    close_price: float,
    open_step: int = 5,
    lot_size: float = 0.1,
) -> ClosedTrade:
    entry_fill = open_price  + SPREAD + SLIP if direction == Direction.LONG \
        else open_price  - SPREAD - SLIP
    exit_fill  = close_price - SPREAD - SLIP if direction == Direction.LONG \
        else close_price + SPREAD + SLIP

    pnl_dir = 1 if direction == Direction.LONG else -1
    pnl     = (close_price - open_price) * pnl_dir * lot_size * 100_000

    return ClosedTrade(
        ticket      = 1,
        direction   = direction,
        lot_size    = lot_size,
        entry_price = entry_fill,
        exit_price  = exit_fill,
        pnl         = pnl,
        spread_paid = SPREAD * 2,
        slippage    = SLIP   * 2,
        mfe_pnl     = pnl,
        mae_pnl     = 0.0,
        open_step   = open_step,
        open_price  = open_price,
        close_price = close_price,
    )


def _make_env(
    raw_close: np.ndarray,
    step: int,
    positions: list[Position],
    trades: list[ClosedTrade],
    balance: float = 10_000.0,
) -> SimpleNamespace:
    """Minimal mock env that satisfies EpisodeVisualiser.update()."""
    spec = _make_spec()

    sim = SimpleNamespace(
        _positions       = positions,
        _positions_count = len(positions),
        total_unrealized_pnl = lambda p: 0.0,
        n_positions      = len(positions),
    )
    sim.position_state_vector = lambda price, n_slots: np.zeros(n_slots * 5, dtype=np.float32)

    env = SimpleNamespace(
        _step            = step,
        raw_close        = raw_close,
        _balance         = balance,
        _episode_trades  = trades,
        _sim             = sim,
        initial_balance  = balance,
        lot_tiers        = [0.1],
        n_slots          = 2,
        spec             = SimpleNamespace(name="EURUSD"),
    )
    env._current_price = lambda: float(raw_close[min(step, len(raw_close) - 1)])
    return env


def _make_vis() -> EpisodeVisualiser:
    """Return a Visualiser with matplotlib patched out entirely."""
    vis = EpisodeVisualiser(window=120, pause=0)
    # Prevent any matplotlib calls
    vis._fig    = MagicMock()
    vis._axes   = [MagicMock() for _ in range(5)]
    vis._plt    = MagicMock()
    vis._mcolors   = MagicMock()
    vis._mpatches  = MagicMock()
    vis._Line2D    = MagicMock()
    vis._initial_balance = 10_000.0
    vis._symbol_name     = "EURUSD"
    vis._action_colours  = {0: "#4a5068"}
    vis._action_labels   = {0: "HOLD"}
    # Suppress _redraw — we only care about the data-recording part of update()
    vis._redraw = MagicMock()
    return vis


# ---------------------------------------------------------------------------
# Tests: open markers
# ---------------------------------------------------------------------------

class TestOpenMarkers:

    def test_long_open_marker_uses_open_price(self):
        """Buy marker y-value must equal Position.open_price, not entry_price."""
        raw_close  = np.linspace(1.1000, 1.1100, 20)
        open_price = raw_close[9]   # price at bar 9

        pos = _make_position(Direction.LONG, open_price, open_step=9)
        assert pos.open_price  == pytest.approx(open_price)
        assert pos.entry_price != pytest.approx(open_price)  # fill is offset

        env = _make_env(raw_close, step=10, positions=[pos], trades=[])
        vis = _make_vis()

        vis.update(env, reward=0.0, action=0)

        assert len(vis._buy_markers) == 1
        step_rec, price_rec = vis._buy_markers[0]
        assert price_rec == pytest.approx(open_price), (
            f"Buy marker at {price_rec} — expected raw open_price {open_price}, "
            f"not fill price {pos.entry_price}"
        )

    def test_short_open_marker_uses_open_price(self):
        """Sell marker y-value must equal Position.open_price, not entry_price."""
        raw_close  = np.linspace(1.1100, 1.1000, 20)
        open_price = raw_close[4]

        pos = _make_position(Direction.SHORT, open_price, open_step=4)
        env = _make_env(raw_close, step=5, positions=[pos], trades=[])
        vis = _make_vis()

        vis.update(env, reward=0.0, action=0)

        assert len(vis._sell_markers) == 1
        _, price_rec = vis._sell_markers[0]
        assert price_rec == pytest.approx(open_price), (
            f"Sell marker at {price_rec} — expected raw open_price {open_price}, "
            f"not fill price {pos.entry_price}"
        )

    def test_open_marker_x_matches_step(self):
        """Buy marker x-value must be env._step - 1 (the bar the action executed on)."""
        raw_close  = np.linspace(1.1000, 1.1100, 20)
        pos = _make_position(Direction.LONG, raw_close[7], open_step=7)
        env = _make_env(raw_close, step=8, positions=[pos], trades=[])
        vis = _make_vis()

        vis.update(env, reward=0.0, action=0)

        step_rec, _ = vis._buy_markers[0]
        assert step_rec == 7   # env._step - 1

    def test_no_duplicate_marker_on_subsequent_update(self):
        """Calling update() again with the same position must not add another marker."""
        raw_close = np.linspace(1.1000, 1.1100, 20)
        pos = _make_position(Direction.LONG, raw_close[7], open_step=7)
        env = _make_env(raw_close, step=8, positions=[pos], trades=[])
        vis = _make_vis()

        vis.update(env, reward=0.0, action=0)
        # Second call — position count unchanged
        env._step = 9
        vis.update(env, reward=0.0, action=0)

        assert len(vis._buy_markers) == 1


# ---------------------------------------------------------------------------
# Tests: close markers
# ---------------------------------------------------------------------------

class TestCloseMarkers:

    def test_close_marker_uses_close_price(self):
        """Close (diamond) marker y-value must equal ClosedTrade.close_price."""
        raw_close   = np.linspace(1.1000, 1.1100, 20)
        open_price  = raw_close[5]
        close_price = raw_close[12]

        trade = _make_closed_trade(Direction.LONG, open_price, close_price, open_step=5)
        assert trade.close_price  == pytest.approx(close_price)
        assert trade.exit_price   != pytest.approx(close_price)

        env = _make_env(raw_close, step=13, positions=[], trades=[trade])
        vis = _make_vis()

        vis.update(env, reward=0.0, action=0)

        assert len(vis._close_markers) == 1
        _, price_rec = vis._close_markers[0]
        assert price_rec == pytest.approx(close_price), (
            f"Close marker at {price_rec} — expected raw close_price {close_price}, "
            f"not fill exit_price {trade.exit_price}"
        )

    def test_trade_line_uses_raw_prices(self):
        """Closed trade dotted line must start and end at raw (non-fill) prices."""
        raw_close   = np.linspace(1.1000, 1.1100, 20)
        open_price  = raw_close[3]
        close_price = raw_close[11]

        trade = _make_closed_trade(Direction.LONG, open_price, close_price, open_step=3)
        env   = _make_env(raw_close, step=12, positions=[], trades=[trade])
        vis   = _make_vis()

        vis.update(env, reward=0.0, action=0)

        assert len(vis._closed_trade_lines) == 1
        os, ep, cs, xp, direction = vis._closed_trade_lines[0]
        assert ep == pytest.approx(open_price),  "Trade line start Y must be raw open_price"
        assert xp == pytest.approx(close_price), "Trade line end Y must be raw close_price"


# ---------------------------------------------------------------------------
# Tests: simulator integration — open_price propagation
# ---------------------------------------------------------------------------

class TestSimulatorOpenPrice:

    def _make_sim(self) -> TradeSimulator:
        return TradeSimulator(
            symbol_spec    = _make_spec(),
            lot_tiers      = [0.1],
            slippage_prob  = 0.0,   # deterministic: no slippage
            slippage_range = (0.0, 0.0),
        )

    def test_position_open_price_stored(self):
        """open_position() must store the raw market price in Position.open_price."""
        sim    = self._make_sim()
        market = 1.10500
        result = sim.open_position(market, Direction.LONG, 0.1, open_step=0)

        assert result.success
        pos = result.position
        assert pos.open_price  == pytest.approx(market), "Position.open_price must be raw market price"
        assert pos.entry_price != pytest.approx(market), "entry_price must differ (spread applied)"

    def test_closed_trade_open_price_propagated(self):
        """close_position() must carry Position.open_price → ClosedTrade.open_price."""
        sim    = self._make_sim()
        market = 1.10500
        sim.open_position(market, Direction.LONG, 0.1, open_step=0)

        close_market = 1.10700
        result = sim.close_position(close_market, Direction.LONG, 0.1)

        assert result.success
        trade = result.trade
        assert trade.open_price  == pytest.approx(market),       "ClosedTrade.open_price lost"
        assert trade.close_price == pytest.approx(close_market), "ClosedTrade.close_price must be raw close market"
        assert trade.exit_price  != pytest.approx(close_market), "exit_price must differ (spread applied)"

    def test_spread_still_applied_to_pnl(self):
        """Fixing the visualiser must not affect PnL calculation (fill prices still used)."""
        sim = self._make_sim()
        # Use a known spread to make the maths exact
        spec   = sim.spec
        market_open  = 1.10000
        market_close = 1.10100   # 10 pip move

        sim.open_position(market_open, Direction.LONG, 0.1, open_step=0)
        result = sim.close_position(market_close, Direction.LONG, 0.1)

        trade       = result.trade
        half_spread = spec.spread_price / 2.0
        # PnL = (exit_fill - entry_fill) * lots * contract
        #      = ((close - half_spread) - (open + half_spread)) * lots * contract
        expected_pnl = (
            (market_close - half_spread) - (market_open + half_spread)
        ) * 0.1 * spec.contract_size
        assert trade.pnl == pytest.approx(expected_pnl, rel=1e-6)
