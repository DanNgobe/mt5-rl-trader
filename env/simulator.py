"""
Trade simulator for realistic order execution in forex trading.

This module simulates realistic trade execution with slippage, spread,
stop-loss/take-profit breaches, and position lifecycle management.
Supports multiple concurrent trades.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class OrderType(Enum):
    """Type of trading order."""
    BUY = auto()
    SELL = auto()


class PositionStatus(Enum):
    """Status of a trading position."""
    OPEN = auto()
    CLOSED = auto()
    STOPPED = auto()
    TAKE_PROFIT = auto()


@dataclass
class Position:
    """Represents a single trading position."""
    id: int
    symbol: str
    order_type: OrderType
    entry_price: float
    entry_spread: float
    entry_slippage: float
    lot_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0

    def unrealized_pnl(self, current_bid: float, current_ask: float) -> float:
        """
        Calculate unrealized P&L based on current prices.

        Args:
            current_bid: Current bid price
            current_ask: Current ask price

        Returns:
            Unrealized P&L in account currency
        """
        if self.status != PositionStatus.OPEN:
            return 0.0

        if self.order_type == OrderType.BUY:
            # Long position: profit when price goes up, exit at bid
            price_diff = current_bid - self.entry_price
        else:
            # Short position: profit when price goes down, exit at ask
            price_diff = self.entry_price - current_ask

        # P&L = price difference * lot size * 100000 (standard lot multiplier)
        return price_diff * self.lot_size * 100000

    def close(self, exit_price: float, exit_time: int, reason: PositionStatus,
              commission: float = 0.0, swap: float = 0.0) -> None:
        """
        Close the position.

        Args:
            exit_price: Price at which position was closed
            exit_time: Bar index when position was closed
            reason: Reason for closing (CLOSED, STOPPED, TAKE_PROFIT)
            commission: Commission charged
            swap: Swap/rollover fee
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = reason
        self.commission = commission
        self.swap = swap

        if self.order_type == OrderType.BUY:
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price

        self.pnl = price_diff * self.lot_size * 100000 - commission - swap


@dataclass
class SimConfig:
    """Configuration for the trade simulator."""
    # Spread in pips (will be converted to price)
    spread_pips: float = 1.0
    # Slippage in pips (random component)
    slippage_pips: float = 0.5
    # Commission per lot (in account currency)
    commission_per_lot: float = 7.0
    # Swap per lot per day (in account currency)
    swap_per_lot: float = -0.5
    # Pip value multiplier (depends on symbol)
    pip_value: float = 0.0001
    # Maximum number of concurrent positions
    max_positions: int = 5
    # Minimum lot size
    min_lot: float = 0.01
    # Maximum lot size
    max_lot: float = 10.0
    # Lot step increment
    lot_step: float = 0.01


class TradeSimulator:
    """
    Simulates realistic forex trade execution.

    Features:
    - Slippage and spread modeling
    - Stop-loss / take-profit breach detection
    - Position lifecycle management
    - Multiple concurrent trades support
    """

    def __init__(self, config: Optional[SimConfig] = None):
        """
        Initialize the trade simulator.

        Args:
            config: Simulator configuration. Uses defaults if not provided.
        """
        self.config = config or SimConfig()
        self.positions: list[Position] = []
        self._next_position_id = 1
        self._current_bar_index = 0
        self._equity_curve: list[float] = []
        self._balance = 10000.0  # Starting balance
        self._daily_swap: float = 0.0

    def reset(self, initial_balance: float = 10000.0) -> None:
        """
        Reset the simulator to initial state.

        Args:
            initial_balance: Starting balance for simulation
        """
        self.positions = []
        self._next_position_id = 1
        self._current_bar_index = 0
        self._equity_curve = []
        self._balance = initial_balance
        self._daily_swap = 0.0

    def _apply_slippage(self, price: float, order_type: OrderType) -> float:
        """
        Apply slippage to order execution price.

        Args:
            price: Base price
            order_type: Order type (BUY or SELL)

        Returns:
            Price with slippage applied
        """
        import random
        slippage = random.uniform(0, self.config.slippage_pips) * self.config.pip_value
        if order_type == OrderType.BUY:
            return price + slippage
        else:
            return price - slippage

    def _get_spread_prices(self, mid_price: float) -> tuple[float, float]:
        """
        Calculate bid and ask prices with spread.

        Args:
            mid_price: Mid market price

        Returns:
            Tuple of (bid, ask) prices
        """
        half_spread = (self.config.spread_pips * self.config.pip_value) / 2
        bid = mid_price - half_spread
        ask = mid_price + half_spread
        return bid, ask

    def _calculate_commission(self, lot_size: float) -> float:
        """Calculate commission for a trade."""
        return self.config.commission_per_lot * lot_size

    def _calculate_swap(self, lot_size: float, bars_held: int) -> float:
        """
        Calculate swap/rollover fee.

        Args:
            lot_size: Position lot size
            bars_held: Number of bars held (assumes H1 = 1 hour)
        """
        # Assuming H1 bars: 24 bars per day
        days = bars_held / 24.0
        return self.config.swap_per_lot * lot_size * days

    def open_position(
        self,
        symbol: str,
        order_type: OrderType,
        lot_size: float,
        mid_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Open a new trading position.

        Args:
            symbol: Forex pair symbol
            order_type: BUY or SELL
            lot_size: Position size in lots
            mid_price: Current mid market price
            stop_loss: Optional stop-loss price
            take_profit: Optional take-profit price

        Returns:
            Position object if opened successfully, None otherwise
        """
        # Validate lot size
        lot_size = round(lot_size / self.config.lot_step) * self.config.lot_step
        lot_size = max(self.config.min_lot, min(self.config.max_lot, lot_size))

        # Check max positions limit
        open_positions = sum(1 for p in self.positions if p.status == PositionStatus.OPEN)
        if open_positions >= self.config.max_positions:
            return None

        # Get spread prices
        bid, ask = self._get_spread_prices(mid_price)

        # Apply slippage and determine entry price
        if order_type == OrderType.BUY:
            entry_price = self._apply_slippage(ask, order_type)
            entry_spread = ask - bid
        else:
            entry_price = self._apply_slippage(bid, order_type)
            entry_spread = ask - bid

        # Calculate initial commission
        commission = self._calculate_commission(lot_size)

        # Create position
        position = Position(
            id=self._next_position_id,
            symbol=symbol,
            order_type=order_type,
            entry_price=entry_price,
            entry_spread=entry_spread,
            entry_slippage=entry_price - (ask if order_type == OrderType.BUY else bid),
            lot_size=lot_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            commission=commission,
        )

        self.positions.append(position)
        self._next_position_id += 1

        return position

    def close_position(self, position_id: int, mid_price: float) -> Optional[float]:
        """
        Manually close a position at current market price.

        Args:
            position_id: ID of position to close
            mid_price: Current mid market price

        Returns:
            P&L of closed position, or None if position not found
        """
        position = self.get_open_position(position_id)
        if position is None:
            return None

        bid, ask = self._get_spread_prices(mid_price)

        # Determine exit price based on order type
        if position.order_type == OrderType.BUY:
            exit_price = bid  # Close long at bid
        else:
            exit_price = ask  # Close short at ask

        # Calculate bars held and swap
        bars_held = self._current_bar_index - (position.exit_time or self._current_bar_index)
        swap = self._calculate_swap(position.lot_size, max(1, bars_held))

        # Close position
        position.close(exit_price, self._current_bar_index, PositionStatus.CLOSED,
                      commission=position.commission, swap=swap)

        return position.pnl

    def get_open_position(self, position_id: int) -> Optional[Position]:
        """Get an open position by ID."""
        for pos in self.positions:
            if pos.id == position_id and pos.status == PositionStatus.OPEN:
                return pos
        return None

    def get_open_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """
        Get all open positions, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open positions
        """
        positions = [p for p in self.positions if p.status == PositionStatus.OPEN]
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def check_stop_loss_take_profit(self, bar_high: float, bar_low: float,
                                    bar_index: int) -> list[Position]:
        """
        Check and close positions that hit SL or TP.

        Args:
            bar_high: High price of current bar
            bar_low: Low price of current bar
            bar_index: Current bar index

        Returns:
            List of positions that were closed
        """
        closed_positions = []

        for position in self.get_open_positions():
            should_close = False
            close_reason = PositionStatus.CLOSED
            close_price = None

            if position.order_type == OrderType.BUY:
                # Long position
                if position.stop_loss is not None and bar_low <= position.stop_loss:
                    should_close = True
                    close_reason = PositionStatus.STOPPED
                    close_price = position.stop_loss
                elif position.take_profit is not None and bar_high >= position.take_profit:
                    should_close = True
                    close_reason = PositionStatus.TAKE_PROFIT
                    close_price = position.take_profit
            else:
                # Short position
                if position.stop_loss is not None and bar_high >= position.stop_loss:
                    should_close = True
                    close_reason = PositionStatus.STOPPED
                    close_price = position.stop_loss
                elif position.take_profit is not None and bar_low <= position.take_profit:
                    should_close = True
                    close_reason = PositionStatus.TAKE_PROFIT
                    close_price = position.take_profit

            if should_close and close_price is not None:
                bars_held = bar_index - (position.exit_time or bar_index)
                swap = self._calculate_swap(position.lot_size, max(1, bars_held))
                position.close(close_price, bar_index, close_reason,
                              commission=position.commission, swap=swap)
                closed_positions.append(position)

        return closed_positions

    def update_unrealized_pnl(self, bar_high: float, bar_low: float,
                              bar_close: float) -> float:
        """
        Update unrealized P&L for all open positions.

        Args:
            bar_high: High price of current bar
            bar_low: Low price of current bar
            bar_close: Close price of current bar

        Returns:
            Total unrealized P&L
        """
        bid, ask = self._get_spread_prices(bar_close)
        total_unrealized = 0.0

        for position in self.get_open_positions():
            total_unrealized += position.unrealized_pnl(bid, ask)

        return total_unrealized

    def get_equity(self, bar_high: float, bar_low: float, bar_close: float) -> float:
        """
        Calculate current equity (balance + unrealized P&L).

        Args:
            bar_high: High price of current bar
            bar_low: Low price of current bar
            bar_close: Close price of current bar

        Returns:
            Current equity
        """
        unrealized = self.update_unrealized_pnl(bar_high, bar_low, bar_close)
        return self._balance + unrealized

    def step(self, bar_open: float, bar_high: float, bar_low: float,
             bar_close: float, bar_index: int) -> dict:
        """
        Process a single bar in the simulation.

        Args:
            bar_open: Open price
            bar_high: High price
            bar_low: Low price
            bar_close: Close price
            bar_index: Current bar index

        Returns:
            Dictionary with simulation state
        """
        self._current_bar_index = bar_index

        # Check for SL/TP hits
        closed = self.check_stop_loss_take_profit(bar_high, bar_low, bar_index)

        # Update closed positions' P&L to balance
        for position in closed:
            self._balance += position.pnl

        # Calculate current equity
        equity = self.get_equity(bar_high, bar_low, bar_close)
        self._equity_curve.append(equity)

        return {
            'bar_index': bar_index,
            'open_positions': len(self.get_open_positions()),
            'closed_this_bar': len(closed),
            'equity': equity,
            'balance': self._balance,
            'unrealized_pnl': self.update_unrealized_pnl(bar_high, bar_low, bar_close),
        }

    def get_statistics(self) -> dict:
        """
        Calculate trading statistics.

        Returns:
            Dictionary with trading statistics
        """
        closed_positions = [p for p in self.positions if p.status != PositionStatus.OPEN]

        if not closed_positions:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
            }

        winning = [p for p in closed_positions if p.pnl > 0]
        losing = [p for p in closed_positions if p.pnl <= 0]

        total_pnl = sum(p.pnl for p in closed_positions)
        gross_profit = sum(p.pnl for p in winning)
        gross_loss = abs(sum(p.pnl for p in losing))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate max drawdown from equity curve
        max_drawdown = 0.0
        peak = self._balance
        for equity in self._equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_trades': len(closed_positions),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(closed_positions) * 100 if closed_positions else 0,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'stopped_out': sum(1 for p in closed_positions if p.status == PositionStatus.STOPPED),
            'take_profit_hit': sum(1 for p in closed_positions if p.status == PositionStatus.TAKE_PROFIT),
            'manually_closed': sum(1 for p in closed_positions if p.status == PositionStatus.CLOSED),
        }

    def get_positions_summary(self) -> list[dict]:
        """Get summary of all positions."""
        return [
            {
                'id': p.id,
                'symbol': p.symbol,
                'type': p.order_type.name,
                'entry_price': p.entry_price,
                'exit_price': p.exit_price,
                'lot_size': p.lot_size,
                'stop_loss': p.stop_loss,
                'take_profit': p.take_profit,
                'status': p.status.name,
                'pnl': p.pnl,
            }
            for p in self.positions
        ]
