"""
Trade simulator for realistic order execution in forex trading.

Models slippage, spread, stop-loss/take-profit breaches, and position lifecycle.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class PositionType(Enum):
    """Type of trading position."""
    LONG = auto()
    SHORT = auto()


class ActionType(Enum):
    """Available trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class Position:
    """Represents an open trading position."""
    type: PositionType
    entry_price: float
    size: float  # Lot size
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    slippage: float = 0.0
    spread: float = 0.0
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L based on current price."""
        if self.type == PositionType.LONG:
            return (current_price - self.entry_price) * self.size * 100_000
        else:  # SHORT
            return (self.entry_price - current_price) * self.size * 100_000
    
    def check_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss has been breached."""
        if self.stop_loss is None:
            return False
        if self.type == PositionType.LONG:
            return current_price <= self.stop_loss
        else:  # SHORT
            return current_price >= self.stop_loss
    
    def check_take_profit(self, current_price: float) -> bool:
        """Check if take-profit has been breached."""
        if self.take_profit is None:
            return False
        if self.type == PositionType.LONG:
            return current_price >= self.take_profit
        else:  # SHORT
            return current_price <= self.take_profit


@dataclass
class Trade:
    """Record of a completed trade."""
    position_type: PositionType
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    slippage: float
    spread: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'close', 'reverse'


class TradeSimulator:
    """
    Simulates realistic forex trade execution with slippage, spread,
    and stop-loss/take-profit breach detection.
    """
    
    def __init__(
        self,
        spread: float = 0.0001,
        slippage_prob: float = 0.3,
        slippage_range: tuple[float, float] = (0.00001, 0.0005),
    ):
        """
        Initialize the trade simulator.
        
        Args:
            spread: Fixed spread in price units (e.g., 0.0001 for 1 pip on EURUSD)
            slippage_prob: Probability of slippage occurring on order execution
            slippage_range: Min and max slippage when it occurs
        """
        self.spread = spread
        self.slippage_prob = slippage_prob
        self.slippage_range = slippage_range
        
        self._position: Optional[Position] = None
        self._trade_history: list[Trade] = []
        self._cumulative_pnl: float = 0.0
    
    @property
    def position(self) -> Optional[Position]:
        """Get current open position."""
        return self._position
    
    @property
    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self._position is not None
    
    @property
    def cumulative_pnl(self) -> float:
        """Get cumulative P&L from all closed trades."""
        return self._cumulative_pnl
    
    @property
    def trade_history(self) -> list[Trade]:
        """Get history of all completed trades."""
        return self._trade_history
    
    def _calculate_slippage(self, price: float, position_type: PositionType) -> float:
        """
        Calculate slippage for order execution.
        
        Slippage is always adverse to the position:
        - LONG: entry price is higher than expected
        - SHORT: entry price is lower than expected
        """
        import random
        
        if random.random() > self.slippage_prob:
            return 0.0
        
        slippage = random.uniform(*self.slippage_range)
        return slippage  # Applied adversarially in execute_order
    
    def _get_execution_price(
        self,
        market_price: float,
        position_type: PositionType,
    ) -> tuple[float, float, float]:
        """
        Get actual execution price including spread and slippage.
        
        Returns:
            Tuple of (execution_price, spread_cost, slippage_cost)
        """
        slippage = self._calculate_slippage(market_price, position_type)
        
        if position_type == PositionType.LONG:
            # Buy at ask price (market_price + spread/2 + slippage)
            execution_price = market_price + self.spread / 2 + slippage
        else:  # SHORT
            # Sell at bid price (market_price - spread/2 - slippage)
            execution_price = market_price - self.spread / 2 - slippage
        
        spread_cost = self.spread / 2
        return execution_price, spread_cost, slippage
    
    def _get_exit_price(
        self,
        market_price: float,
        position_type: PositionType,
    ) -> tuple[float, float, float]:
        """
        Get exit execution price including spread and slippage.
        
        Returns:
            Tuple of (execution_price, spread_cost, slippage_cost)
        """
        slippage = self._calculate_slippage(market_price, position_type)
        
        if position_type == PositionType.LONG:
            # Close long at bid price (market_price - spread/2 - slippage)
            execution_price = market_price - self.spread / 2 - slippage
        else:  # SHORT
            # Close short at ask price (market_price + spread/2 + slippage)
            execution_price = market_price + self.spread / 2 + slippage
        
        spread_cost = self.spread / 2
        return execution_price, spread_cost, slippage
    
    def open_position(
        self,
        market_price: float,
        position_type: PositionType,
        size: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Open a new trading position.
        
        Args:
            market_price: Current market price
            position_type: LONG or SHORT
            size: Position size in lots
            stop_loss: Optional stop-loss price level
            take_profit: Optional take-profit price level
            
        Returns:
            The opened Position, or None if a position is already open
        """
        if self._position is not None:
            return None
        
        execution_price, spread_cost, slippage = self._get_execution_price(
            market_price, position_type
        )
        
        self._position = Position(
            type=position_type,
            entry_price=execution_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            slippage=slippage,
            spread=spread_cost,
        )
        
        return self._position
    
    def close_position(
        self,
        market_price: float,
        reason: str = "close",
    ) -> Optional[Trade]:
        """
        Close the current position at market price.
        
        Args:
            market_price: Current market price
            reason: Reason for closing ('close', 'stop_loss', 'take_profit', 'reverse')
            
        Returns:
            Trade record, or None if no position is open
        """
        if self._position is None:
            return None
        
        position = self._position
        exit_price, spread_cost, slippage = self._get_exit_price(
            market_price, position.type
        )
        
        # Calculate P&L
        if position.type == PositionType.LONG:
            pnl = (exit_price - position.entry_price) * position.size * 100_000
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.size * 100_000
        
        total_slippage = position.slippage + slippage
        total_spread = position.spread + spread_cost
        
        trade = Trade(
            position_type=position.type,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            pnl=pnl,
            slippage=total_slippage,
            spread=total_spread,
            exit_reason=reason,
        )
        
        self._trade_history.append(trade)
        self._cumulative_pnl += pnl
        self._position = None
        
        return trade
    
    def reverse_position(
        self,
        market_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> tuple[Optional[Trade], Optional[Position]]:
        """
        Reverse the current position (close and open opposite position).
        
        Args:
            market_price: Current market price
            stop_loss: Optional stop-loss for new position
            take_profit: Optional take-profit for new position
            
        Returns:
            Tuple of (closed Trade record, new Position)
        """
        if self._position is None:
            return None, None
        
        # Close current position
        closed_trade = self.close_position(market_price, reason="reverse")
        
        # Open opposite position
        new_type = (
            PositionType.SHORT
            if self._position is None or closed_trade is None
            else (
                PositionType.LONG
                if closed_trade.position_type == PositionType.SHORT
                else PositionType.SHORT
            )
        )
        
        # Need to re-check position state after close
        new_position = self.open_position(
            market_price=market_price,
            position_type=new_type,
            size=closed_trade.size if closed_trade else 1.0,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        
        return closed_trade, new_position
    
    def check_breaches(self, current_price: float) -> Optional[Trade]:
        """
        Check if stop-loss or take-profit has been breached and close if so.
        
        Args:
            current_price: Current market price to check against
            
        Returns:
            Trade record if a breach occurred and position was closed, None otherwise
        """
        if self._position is None:
            return None
        
        if self._position.check_stop_loss(current_price):
            return self.close_position(current_price, reason="stop_loss")
        
        if self._position.check_take_profit(current_price):
            return self.close_position(current_price, reason="take_profit")
        
        return None
    
    def get_state(self, current_price: float) -> dict:
        """
        Get current simulator state for observation.
        
        Args:
            current_price: Current market price
            
        Returns:
            Dictionary with current state information
        """
        if self._position is None:
            return {
                "has_position": False,
                "position_type": 0,
                "unrealized_pnl": 0.0,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
            }
        
        position = self._position
        return {
            "has_position": True,
            "position_type": 1 if position.type == PositionType.LONG else -1,
            "unrealized_pnl": position.unrealized_pnl(current_price),
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss or 0.0,
            "take_profit": position.take_profit or 0.0,
        }
    
    def reset(self) -> None:
        """Reset simulator state (clear position but keep trade history)."""
        self._position = None
    
    def reset_full(self) -> None:
        """Full reset (clear position and trade history)."""
        self._position = None
        self._trade_history = []
        self._cumulative_pnl = 0.0
