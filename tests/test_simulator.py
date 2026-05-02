"""
tests/test_simulator.py
-----------------------
Unit tests for TradeSimulator and supporting data classes.
"""

import numpy as np
import pytest
from core.simulator import (
    TradeSimulator,
    SymbolSpec,
    Direction,
    Position,
    ClosedTrade,
)

@pytest.fixture
def symbol_spec():
    return SymbolSpec(
        name          = "EURUSD",
        pip_value     = 0.0001,
        pip_location  = 4,
        contract_size = 100_000,
        spread_pips   = 2.0,
        min_lot       = 0.01,
        max_lot       = 100.0,
        margin_rate   = 0.01,
    )

@pytest.fixture
def sim(symbol_spec):
    return TradeSimulator(
        symbol_spec    = symbol_spec,
        lot_tiers      = [0.1, 0.2, 0.5],
        slippage_prob  = 0.0,  # Disable for deterministic tests
    )

def test_open_position_long(sim):
    market_price = 1.1000
    res = sim.open_position(market_price, Direction.LONG, 0.1, open_step=10)
    
    assert res.success is True
    assert len(sim.positions) == 1
    
    pos = sim.positions[0]
    # fill = market_price + (spread / 2) = 1.1000 + 0.0001 = 1.1001
    assert pos.direction == Direction.LONG
    assert abs(pos.entry_price - 1.1001) < 1e-9
    assert pos.lot_size == 0.1
    assert pos.open_step == 10

def test_open_position_short(sim):
    market_price = 1.1000
    res = sim.open_position(market_price, Direction.SHORT, 0.1)
    
    # fill = market_price - (spread / 2) = 1.1000 - 0.0001 = 1.0999
    assert abs(sim.positions[0].entry_price - 1.0999) < 1e-9

def test_close_position_pnl(sim):
    sim.open_position(1.1000, Direction.LONG, 1.0) # entry 1.1001
    
    # Market at 1.1010. Close bid (exit) = 1.1010 - 0.0001 = 1.1009
    # PnL = (1.1009 - 1.1001) * 1.0 * 100,000 = 0.0008 * 100,000 = 80.0
    res = sim.close_position(1.1010, Direction.LONG, 1.0)
    
    assert res.success is True
    assert res.trade is not None
    assert abs(res.trade.pnl - 80.0) < 1e-7
    assert len(sim.positions) == 0
    assert abs(sim.cumulative_pnl - 80.0) < 1e-7

def test_close_position_not_found(sim):
    sim.open_position(1.1000, Direction.LONG, 0.1)
    # Wrong direction
    res = sim.close_position(1.1000, Direction.SHORT, 0.1)
    assert res.success is False
    assert res.invalid is True
    
    # Wrong lot size
    res = sim.close_position(1.1000, Direction.LONG, 0.2)
    assert res.success is False
    assert res.invalid is True

def test_unrealized_pnl(sim):
    sim.open_position(1.1000, Direction.LONG, 1.0) # entry 1.1001
    
    # Market at 1.1005. Unrealized PnL = (1.1005 - 1.1001) * 100,000 = 40.0
    upnl = sim.total_unrealized_pnl(1.1005)
    assert abs(upnl - 40.0) < 1e-7

def test_excursion_tracking(sim):
    sim.open_position(1.1000, Direction.LONG, 1.0) # entry 1.1001
    pos = sim._positions[0]
    
    sim.update_excursions(1.1010) # Favourite move
    assert abs(pos.mfe_price - 1.1010) < 1e-9
    
    sim.update_excursions(1.0990) # Adverse move
    assert abs(pos.mae_price - 1.0990) < 1e-9
    
    # mfe_pnl should be (1.1010 - 1.1001) * 100,000 = 90.0
    # mae_pnl should be (1.0990 - 1.1001) * 100,000 = -110.0
    res = sim.close_position(1.1000, Direction.LONG, 1.0)
    assert abs(res.trade.mfe_pnl - 90.0) < 1e-7
    assert abs(res.trade.mae_pnl - -110.0) < 1e-7

def test_close_all(sim):
    sim.open_position(1.1000, Direction.LONG, 0.1)
    sim.open_position(1.1000, Direction.SHORT, 0.2)
    assert len(sim.positions) == 2
    
    trades = sim.close_all(1.1000)
    assert len(trades) == 2
    assert len(sim.positions) == 0
    assert all(t.forced for t in trades)

def test_slippage_application(symbol_spec):
    # Enable slippage for this test
    sim = TradeSimulator(
        symbol_spec    = symbol_spec,
        slippage_prob  = 1.0,  # Always slip
        slippage_range = (0.0010, 0.0010), # Fixed slips
    )
    
    market_price = 1.1000
    # Long: fill = price + half_spread + slip = 1.1000 + 0.0001 + 0.0010 = 1.1011
    res = sim.open_position(market_price, Direction.LONG, 0.1)
    assert abs(sim.positions[0].entry_price - 1.1011) < 1e-9
    assert abs(sim.positions[0].slippage - 0.0010) < 1e-9
