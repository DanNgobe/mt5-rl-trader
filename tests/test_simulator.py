"""
Unit tests for the trade simulator.
"""

import pytest
from env.simulator import (
    TradeSimulator,
    PositionType,
    Position,
    Trade,
    ActionType,
)


class TestPosition:
    """Tests for the Position dataclass."""
    
    def test_unrealized_pnl_long_profit(self):
        """Test P&L calculation for profitable long position."""
        pos = Position(
            type=PositionType.LONG,
            entry_price=1.1000,
            size=1.0,
        )
        pnl = pos.unrealized_pnl(1.1050)
        assert pnl == 500.0  # (1.1050 - 1.1000) * 1.0 * 100_000
    
    def test_unrealized_pnl_long_loss(self):
        """Test P&L calculation for losing long position."""
        pos = Position(
            type=PositionType.LONG,
            entry_price=1.1000,
            size=1.0,
        )
        pnl = pos.unrealized_pnl(1.0950)
        assert pnl == -500.0
    
    def test_unrealized_pnl_short_profit(self):
        """Test P&L calculation for profitable short position."""
        pos = Position(
            type=PositionType.SHORT,
            entry_price=1.1000,
            size=1.0,
        )
        pnl = pos.unrealized_pnl(1.0950)
        assert pnl == 500.0
    
    def test_unrealized_pnl_short_loss(self):
        """Test P&L calculation for losing short position."""
        pos = Position(
            type=PositionType.SHORT,
            entry_price=1.1000,
            size=1.0,
        )
        pnl = pos.unrealized_pnl(1.1050)
        assert pnl == -500.0
    
    def test_stop_loss_check_long(self):
        """Test stop-loss breach detection for long position."""
        pos = Position(
            type=PositionType.LONG,
            entry_price=1.1000,
            size=1.0,
            stop_loss=1.0950,
        )
        assert pos.check_stop_loss(1.0949) is True
        assert pos.check_stop_loss(1.0950) is True
        assert pos.check_stop_loss(1.0951) is False
    
    def test_stop_loss_check_short(self):
        """Test stop-loss breach detection for short position."""
        pos = Position(
            type=PositionType.SHORT,
            entry_price=1.1000,
            size=1.0,
            stop_loss=1.0950,
        )
        assert pos.check_stop_loss(1.0951) is True
        assert pos.check_stop_loss(1.0950) is True
        assert pos.check_stop_loss(1.0949) is False
    
    def test_take_profit_check_long(self):
        """Test take-profit breach detection for long position."""
        pos = Position(
            type=PositionType.LONG,
            entry_price=1.1000,
            size=1.0,
            take_profit=1.1050,
        )
        assert pos.check_take_profit(1.1051) is True
        assert pos.check_take_profit(1.1050) is True
        assert pos.check_take_profit(1.1049) is False
    
    def test_take_profit_check_short(self):
        """Test take-profit breach detection for short position."""
        pos = Position(
            type=PositionType.SHORT,
            entry_price=1.1000,
            size=1.0,
            take_profit=1.0950,
        )
        assert pos.check_take_profit(1.0949) is True
        assert pos.check_take_profit(1.0950) is True
        assert pos.check_take_profit(1.0951) is False


class TestTradeSimulator:
    """Tests for the TradeSimulator class."""
    
    def test_open_long_position(self):
        """Test opening a long position."""
        sim = TradeSimulator(spread=0.0001, slippage_prob=0.0)
        pos = sim.open_position(
            market_price=1.1000,
            position_type=PositionType.LONG,
            size=1.0,
        )
        
        assert pos is not None
        assert pos.type == PositionType.LONG
        assert pos.size == 1.0
        # Entry price should include spread (half on entry)
        assert pos.entry_price == 1.1000 + 0.00005
    
    def test_open_short_position(self):
        """Test opening a short position."""
        sim = TradeSimulator(spread=0.0001, slippage_prob=0.0)
        pos = sim.open_position(
            market_price=1.1000,
            position_type=PositionType.SHORT,
            size=1.0,
        )
        
        assert pos is not None
        assert pos.type == PositionType.SHORT
        assert pos.size == 1.0
        # Entry price should include spread (half on entry)
        assert pos.entry_price == 1.1000 - 0.00005
    
    def test_cannot_open_second_position(self):
        """Test that opening a second position fails."""
        sim = TradeSimulator()
        sim.open_position(1.1000, PositionType.LONG)
        result = sim.open_position(1.1000, PositionType.SHORT)
        assert result is None
    
    def test_close_long_position(self):
        """Test closing a long position."""
        sim = TradeSimulator(spread=0.0001, slippage_prob=0.0)
        sim.open_position(1.1000, PositionType.LONG, size=1.0)
        
        trade = sim.close_position(1.1050)
        
        assert trade is not None
        assert trade.position_type == PositionType.LONG
        assert trade.entry_price == 1.1000 + 0.00005
        assert trade.exit_price == 1.1050 - 0.00005
        # P&L: (exit - entry) * size * 100_000
        expected_pnl = (1.1050 - 0.00005 - (1.1000 + 0.00005)) * 1.0 * 100_000
        assert trade.pnl == pytest.approx(expected_pnl)
    
    def test_close_short_position(self):
        """Test closing a short position."""
        sim = TradeSimulator(spread=0.0001, slippage_prob=0.0)
        sim.open_position(1.1000, PositionType.SHORT, size=1.0)
        
        trade = sim.close_position(1.0950)
        
        assert trade is not None
        assert trade.position_type == PositionType.SHORT
        # P&L: (entry - exit) * size * 100_000
        expected_pnl = ((1.1000 - 0.00005) - (1.0950 + 0.00005)) * 1.0 * 100_000
        assert trade.pnl == pytest.approx(expected_pnl)
    
    def test_cumulative_pnl(self):
        """Test cumulative P&L tracking."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        
        sim.open_position(1.1000, PositionType.LONG, size=1.0)
        sim.close_position(1.1050)
        
        sim.open_position(1.1050, PositionType.SHORT, size=1.0)
        sim.close_position(1.1000)
        
        assert sim.cumulative_pnl == 1000.0  # 500 + 500
    
    def test_stop_loss_breach_long(self):
        """Test stop-loss breach detection for long position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(
            1.1000,
            PositionType.LONG,
            size=1.0,
            stop_loss=1.0950,
        )
        
        # No breach yet
        trade = sim.check_breaches(1.0951)
        assert trade is None
        
        # Breach!
        trade = sim.check_breaches(1.0949)
        assert trade is not None
        assert trade.exit_reason == "stop_loss"
        assert sim.has_position is False
    
    def test_stop_loss_breach_short(self):
        """Test stop-loss breach detection for short position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(
            1.1000,
            PositionType.SHORT,
            size=1.0,
            stop_loss=1.0950,
        )
        
        # Breach!
        trade = sim.check_breaches(1.0951)
        assert trade is not None
        assert trade.exit_reason == "stop_loss"
    
    def test_take_profit_breach_long(self):
        """Test take-profit breach detection for long position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(
            1.1000,
            PositionType.LONG,
            size=1.0,
            take_profit=1.1050,
        )
        
        # Breach!
        trade = sim.check_breaches(1.1051)
        assert trade is not None
        assert trade.exit_reason == "take_profit"
    
    def test_take_profit_breach_short(self):
        """Test take-profit breach detection for short position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(
            1.1000,
            PositionType.SHORT,
            size=1.0,
            take_profit=1.0950,
        )
        
        # Breach!
        trade = sim.check_breaches(1.0949)
        assert trade is not None
        assert trade.exit_reason == "take_profit"
    
    def test_reverse_position(self):
        """Test reversing a position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(1.1000, PositionType.LONG, size=1.0)
        
        closed_trade, new_position = sim.reverse_position(
            1.1020,
            stop_loss=1.1070,
            take_profit=1.0970,
        )
        
        assert closed_trade is not None
        assert closed_trade.exit_reason == "reverse"
        assert new_position is not None
        assert new_position.type == PositionType.SHORT
        assert new_position.stop_loss == 1.1070
        assert new_position.take_profit == 1.0970
    
    def test_get_state_no_position(self):
        """Test state when no position is open."""
        sim = TradeSimulator()
        state = sim.get_state(1.1000)
        
        assert state["has_position"] is False
        assert state["position_type"] == 0
        assert state["unrealized_pnl"] == 0.0
    
    def test_get_state_with_position(self):
        """Test state with an open position."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        sim.open_position(1.1000, PositionType.LONG, size=1.0)
        
        state = sim.get_state(1.1020)
        
        assert state["has_position"] is True
        assert state["position_type"] == 1
        assert state["unrealized_pnl"] == 200.0
        assert state["entry_price"] == 1.1000
    
    def test_reset(self):
        """Test resetting the simulator."""
        sim = TradeSimulator()
        sim.open_position(1.1000, PositionType.LONG)
        sim.close_position(1.1050)
        
        sim.reset()
        
        assert sim.has_position is False
        assert len(sim.trade_history) == 1  # History preserved
        assert sim.cumulative_pnl == 500.0  # P&L preserved
    
    def test_reset_full(self):
        """Test full reset of the simulator."""
        sim = TradeSimulator()
        sim.open_position(1.1000, PositionType.LONG)
        sim.close_position(1.1050)
        
        sim.reset_full()
        
        assert sim.has_position is False
        assert len(sim.trade_history) == 0
        assert sim.cumulative_pnl == 0.0
    
    def test_trade_history(self):
        """Test trade history tracking."""
        sim = TradeSimulator(spread=0.0, slippage_prob=0.0)
        
        sim.open_position(1.1000, PositionType.LONG)
        sim.close_position(1.1050)
        
        sim.open_position(1.1050, PositionType.SHORT)
        sim.close_position(1.1000)
        
        assert len(sim.trade_history) == 2
        assert sim.trade_history[0].position_type == PositionType.LONG
        assert sim.trade_history[1].position_type == PositionType.SHORT


class TestActionType:
    """Tests for ActionType enum."""
    
    def test_action_values(self):
        """Test action enum values."""
        assert ActionType.HOLD.value == 0
        assert ActionType.BUY.value == 1
        assert ActionType.SELL.value == 2
