"""
Trade simulator modelling MT5 hedging account behaviour.

Supports multiple simultaneous positions per symbol, each with its own
ticket. No SL/TP — the agent is responsible for all exit decisions.
Spread and slippage are applied per-symbol using the symbols config.
"""

import logging
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Direction(IntEnum):
    LONG  = 1
    SHORT = -1


class Action(IntEnum):
    HOLD  = 0
    BUY   = 1
    SELL  = 2
    CLOSE = 3


LOT_TIERS: list[float] = [0.01, 0.02, 0.05]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SymbolSpec:
    """Per-symbol trading specification loaded from symbols config."""
    name: str
    pip_value: float        # e.g. 0.0001 for EURUSD, 0.01 for USDJPY
    pip_location: int       # decimal places for a pip
    contract_size: int      # units per lot, typically 100_000
    spread_pips: float      # typical spread in pips
    min_lot: float
    max_lot: float
    margin_rate: float      # e.g. 0.01 for 1:100 leverage

    @property
    def spread_price(self) -> float:
        """Spread expressed as a price delta."""
        return self.spread_pips * self.pip_value


@dataclass
class Position:
    """
    A single open hedging position, analogous to one MT5 ticket.

    ticket      — unique integer ID for this position
    direction   — LONG or SHORT
    lot_size    — position size in lots
    entry_price — actual fill price after spread/slippage
    open_price  — raw market price at the time of opening (for reference)
    """
    ticket:      int
    direction:   Direction
    lot_size:    float
    entry_price: float
    open_price:  float
    spread_paid: float = 0.0
    slippage:    float = 0.0

    def unrealized_pnl(self, current_price: float, contract_size: int) -> float:
        """
        Unrealized P&L in account currency units.

        For a standard lot (100_000 units):
            P&L = (current - entry) * lots * contract_size   [LONG]
            P&L = (entry - current) * lots * contract_size   [SHORT]
        """
        price_diff = (
            (current_price - self.entry_price)
            if self.direction == Direction.LONG
            else (self.entry_price - current_price)
        )
        return price_diff * self.lot_size * contract_size


@dataclass
class ClosedTrade:
    """Record of a completed trade."""
    ticket:      int
    direction:   Direction
    lot_size:    float
    entry_price: float
    exit_price:  float
    pnl:         float
    spread_paid: float
    slippage:    float


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class TradeSimulator:
    """
    Simulates MT5 hedging account order execution.

    Key behaviours
    --------------
    - Multiple positions can be open simultaneously (hedging mode).
    - BUY/SELL always opens a new position (never merges with existing).
    - CLOSE targets the oldest open position matching the requested lot tier.
      e.g. CLOSE + 0.1 lots → closes the oldest 0.1-lot position regardless
      of direction, matching the hybrid action semantics agreed with the user.
    - Spread is applied at fill time; slippage is probabilistic and adverse.
    - No SL/TP logic — the agent decides all exits.
    """

    def __init__(
        self,
        symbol_spec: SymbolSpec,
        max_positions: int = 3,
        slippage_prob: float = 0.3,
        slippage_range: tuple[float, float] = (0.00001, 0.0005),
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            symbol_spec:    Instrument specification (spread, pip value, etc.)
            max_positions:  Hard cap on simultaneous open positions.
            slippage_prob:  Probability that slippage occurs on any fill.
            slippage_range: (min, max) slippage in price units when it fires.
            rng:            NumPy random generator for reproducibility.
                            If None, a fresh default_rng() is used.
        """
        self.spec           = symbol_spec
        self.max_positions  = max_positions
        self.slippage_prob  = slippage_prob
        self.slippage_range = slippage_range
        self._rng           = rng if rng is not None else np.random.default_rng()

        self._positions:     list[Position]    = []
        self._closed_trades: list[ClosedTrade] = []
        self._next_ticket:   int               = 1
        self._cumulative_pnl: float            = 0.0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    @property
    def has_positions(self) -> bool:
        return len(self._positions) > 0

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_slippage(self) -> float:
        """Return an adverse slippage amount (0.0 if no slippage this fill)."""
        if self._rng.random() > self.slippage_prob:
            return 0.0
        lo, hi = self.slippage_range
        return float(self._rng.uniform(lo, hi))

    def _fill_price_open(self, market_price: float, direction: Direction) -> tuple[float, float, float]:
        """
        Compute the actual fill price when opening a position.

        LONG  → pay the ask  (market + half-spread + slippage)
        SHORT → receive bid  (market - half-spread - slippage)

        Returns (fill_price, half_spread, slippage).
        """
        half_spread = self.spec.spread_price / 2.0
        slip        = self._sample_slippage()

        if direction == Direction.LONG:
            fill = market_price + half_spread + slip
        else:
            fill = market_price - half_spread - slip

        return fill, half_spread, slip

    def _fill_price_close(self, market_price: float, direction: Direction) -> tuple[float, float, float]:
        """
        Compute the actual fill price when closing a position.

        Closing a LONG  → sell at bid  (market - half-spread - slippage)
        Closing a SHORT → buy  at ask  (market + half-spread + slippage)

        Returns (fill_price, half_spread, slippage).
        """
        half_spread = self.spec.spread_price / 2.0
        slip        = self._sample_slippage()

        if direction == Direction.LONG:
            fill = market_price - half_spread - slip
        else:
            fill = market_price + half_spread + slip

        return fill, half_spread, slip

    # ------------------------------------------------------------------
    # Core order operations
    # ------------------------------------------------------------------

    def open_position(
        self,
        market_price: float,
        direction: Direction,
        lot_size: float,
    ) -> Optional[Position]:
        """
        Open a new position at market price.

        Returns the new Position, or None if the position cap is already reached.
        """
        if self.n_positions >= self.max_positions:
            logger.debug(
                "Cannot open position: at max_positions cap (%d).", self.max_positions
            )
            return None

        fill, spread_paid, slippage = self._fill_price_open(market_price, direction)

        pos = Position(
            ticket      = self._next_ticket,
            direction   = direction,
            lot_size    = lot_size,
            entry_price = fill,
            open_price  = market_price,
            spread_paid = spread_paid,
            slippage    = slippage,
        )
        self._positions.append(pos)
        self._next_ticket += 1

        logger.debug(
            "Opened %s ticket=%d lot=%.2f entry=%.5f (spread=%.5f slip=%.5f)",
            direction.name, pos.ticket, lot_size, fill, spread_paid, slippage,
        )
        return pos

    def close_position(
        self,
        market_price: float,
        lot_size: float,
    ) -> Optional[ClosedTrade]:
        """
        Close the oldest open position whose lot_size matches the requested tier.

        This implements the agreed hybrid-action close semantics:
            (CLOSE, lot_tier) → close the oldest position opened with that lot size.

        Returns the ClosedTrade record, or None if no matching position exists.
        """
        # Find the oldest matching position (positions are appended in order)
        target: Optional[Position] = None
        for pos in self._positions:
            if abs(pos.lot_size - lot_size) < 1e-9:
                target = pos
                break

        if target is None:
            logger.debug(
                "CLOSE %.2f lots: no matching open position found.", lot_size
            )
            return None

        fill, spread_paid, slippage = self._fill_price_close(market_price, target.direction)

        # P&L: price difference × lots × contract size
        price_diff = (
            (fill - target.entry_price)
            if target.direction == Direction.LONG
            else (target.entry_price - fill)
        )
        pnl = price_diff * target.lot_size * self.spec.contract_size

        trade = ClosedTrade(
            ticket      = target.ticket,
            direction   = target.direction,
            lot_size    = target.lot_size,
            entry_price = target.entry_price,
            exit_price  = fill,
            pnl         = pnl,
            spread_paid = target.spread_paid + spread_paid,
            slippage    = target.slippage    + slippage,
        )

        self._positions.remove(target)
        self._closed_trades.append(trade)
        self._cumulative_pnl += pnl

        logger.debug(
            "Closed ticket=%d lot=%.2f exit=%.5f pnl=%.2f",
            trade.ticket, trade.lot_size, fill, pnl,
        )
        return trade

    def close_all(self, market_price: float) -> list[ClosedTrade]:
        """Close every open position. Used at episode end."""
        trades = []
        # Iterate over a copy since close_position mutates self._positions
        for pos in list(self._positions):
            trade = self.close_position(market_price, pos.lot_size)
            if trade is not None:
                trades.append(trade)
        return trades

    # ------------------------------------------------------------------
    # State query
    # ------------------------------------------------------------------

    def total_unrealized_pnl(self, current_price: float) -> float:
        """Sum of unrealized P&L across all open positions."""
        return sum(
            p.unrealized_pnl(current_price, self.spec.contract_size)
            for p in self._positions
        )

    def position_state_vector(
        self,
        current_price: float,
        max_positions: int,
    ) -> np.ndarray:
        """
        Return a fixed-length float32 vector representing all open positions.

        Each position slot is encoded as:
            [is_filled, direction, lot_size, entry_price_delta, unrealized_pnl_norm]

        Unfilled slots are zero-padded.  Length = max_positions * 5.
        """
        slot_size = 5
        vec = np.zeros(max_positions * slot_size, dtype=np.float32)

        for i, pos in enumerate(self._positions[:max_positions]):
            base = i * slot_size
            vec[base + 0] = 1.0                                              # is_filled
            vec[base + 1] = float(pos.direction)                             # +1 or -1
            vec[base + 2] = pos.lot_size                                     # raw lots
            vec[base + 3] = (pos.entry_price - current_price) / current_price  # rel delta
            vec[base + 4] = pos.unrealized_pnl(current_price, self.spec.contract_size)

        return vec

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset — clears positions, trade history, and cumulative P&L."""
        self._positions      = []
        self._closed_trades  = []
        self._cumulative_pnl = 0.0
        self._next_ticket    = 1
        logger.debug("Simulator reset.")