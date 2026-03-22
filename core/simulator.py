"""
Trade simulator modelling MT5 hedging account behaviour.

Multiple simultaneous positions per symbol (hedging mode), each with its
own ticket. No SL/TP — the agent is responsible for all exit decisions.
Spread and slippage are applied per-symbol via SymbolSpec.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Direction(IntEnum):
    LONG  = 1
    SHORT = -1


class Action(IntEnum):
    HOLD        = 0
    BUY         = 1
    SELL        = 2
    CLOSE_LONG  = 3
    CLOSE_SHORT = 4


LOT_TIERS: list[float] = [0.01, 0.02, 0.05]


@dataclass
class SymbolSpec:
    """Per-symbol trading specification."""
    name:          str
    pip_value:     float   # 0.0001 for EURUSD, 0.01 for USDJPY
    pip_location:  int
    contract_size: int     # units per lot, typically 100_000
    spread_pips:   float
    min_lot:       float
    max_lot:       float
    margin_rate:   float   # 0.01 = 1:100 leverage

    @property
    def spread_price(self) -> float:
        return self.spread_pips * self.pip_value


@dataclass
class Position:
    """A single open hedging position (one MT5 ticket)."""
    ticket:      int
    direction:   Direction
    lot_size:    float
    entry_price: float  # actual fill price after spread/slippage
    open_price:  float  # raw market price at open (reference only)
    spread_paid: float = 0.0
    slippage:    float = 0.0
    # Maximum favourable / adverse excursion prices (updated each step)
    mfe_price:   float = 0.0  # best price reached in trade's favour
    mae_price:   float = 0.0  # worst price reached against the trade

    def __post_init__(self):
        # Initialise excursion prices to entry so first update is correct
        self.mfe_price = self.entry_price
        self.mae_price = self.entry_price

    def unrealized_pnl(self, current_price: float, contract_size: int) -> float:
        price_diff = (
            (current_price - self.entry_price)
            if self.direction == Direction.LONG
            else (self.entry_price - current_price)
        )
        return price_diff * self.lot_size * contract_size

    def update_excursions(self, current_price: float) -> None:
        """Track the best and worst price seen during this trade's lifetime."""
        if self.direction == Direction.LONG:
            self.mfe_price = max(self.mfe_price, current_price)
            self.mae_price = min(self.mae_price, current_price)
        else:
            self.mfe_price = min(self.mfe_price, current_price)
            self.mae_price = max(self.mae_price, current_price)


@dataclass
class ClosedTrade:
    """Record of a completed trade, including excursion data."""
    ticket:      int
    direction:   Direction
    lot_size:    float
    entry_price: float
    exit_price:  float
    pnl:         float
    spread_paid: float
    slippage:    float
    mfe_pnl:     float  # P&L at the most favourable price reached
    mae_pnl:     float  # P&L at the most adverse price reached

    def to_dict(self) -> dict:
        return {
            "ticket":      self.ticket,
            "direction":   self.direction.name,
            "lot_size":    self.lot_size,
            "entry_price": self.entry_price,
            "exit_price":  self.exit_price,
            "pnl":         self.pnl,
            "spread_paid": self.spread_paid,
            "slippage":    self.slippage,
            "mfe_pnl":     self.mfe_pnl,
            "mae_pnl":     self.mae_pnl,
        }


@dataclass
class OrderResult:
    """
    Result of an open or close attempt.

    success  — the order was executed
    invalid  — the action was illegal (triggers a penalty in the env)
    reason   — human-readable explanation when invalid or failed
    position — set on successful open
    trade    — set on successful close
    """
    success:  bool
    invalid:  bool               = False
    reason:   str                = ""
    position: Optional[Position]    = None
    trade:    Optional[ClosedTrade] = None


class TradeSimulator:
    """
    Simulates MT5 hedging account order execution.

    BUY/SELL always opens a new position. CLOSE targets the oldest open
    position matching the requested lot tier. Spread is applied at fill
    time; slippage is probabilistic and always adverse.
    """

    def __init__(
        self,
        symbol_spec:    SymbolSpec,
        max_positions:  int   = 3,
        slippage_prob:  float = 0.3,
        slippage_range: tuple[float, float] = (0.00001, 0.0005),
        rng:            Optional[np.random.Generator] = None,
    ):
        self.spec           = symbol_spec
        self.max_positions  = max_positions
        self.slippage_prob  = slippage_prob
        self.slippage_range = slippage_range
        self._rng           = rng if rng is not None else np.random.default_rng()

        self._positions:      list[Position]    = []
        self._closed_trades:  list[ClosedTrade] = []
        self._next_ticket:    int               = 1
        self._cumulative_pnl: float             = 0.0

    @property
    def positions(self) -> list[Position]:
        return list(self._positions)

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    @property
    def has_positions(self) -> bool:
        return bool(self._positions)

    @property
    def cumulative_pnl(self) -> float:
        return self._cumulative_pnl

    @property
    def closed_trades(self) -> list[ClosedTrade]:
        return list(self._closed_trades)

    def update_excursions(self, current_price: float) -> None:
        """Update MFE/MAE on every open position. Call once per env step."""
        for pos in self._positions:
            pos.update_excursions(current_price)

    def _sample_slippage(self) -> float:
        if self._rng.random() > self.slippage_prob:
            return 0.0
        lo, hi = self.slippage_range
        return float(self._rng.uniform(lo, hi))

    def _fill_open(self, market_price: float, direction: Direction) -> tuple[float, float, float]:
        half_spread = self.spec.spread_price / 2.0
        slip        = self._sample_slippage()
        fill = (market_price + half_spread + slip) if direction == Direction.LONG \
               else (market_price - half_spread - slip)
        return fill, half_spread, slip

    def _fill_close(self, market_price: float, direction: Direction) -> tuple[float, float, float]:
        half_spread = self.spec.spread_price / 2.0
        slip        = self._sample_slippage()
        fill = (market_price - half_spread - slip) if direction == Direction.LONG \
               else (market_price + half_spread + slip)
        return fill, half_spread, slip

    def _pnl(self, entry: float, exit_: float, direction: Direction, lot_size: float) -> float:
        price_diff = (exit_ - entry) if direction == Direction.LONG else (entry - exit_)
        return price_diff * lot_size * self.spec.contract_size

    def open_position(self, market_price: float, direction: Direction, lot_size: float) -> OrderResult:
        """
        Open a new position at market price.

        Returns OrderResult(invalid=True) if at max_positions cap.
        """
        if self.n_positions >= self.max_positions:
            logger.debug("Cannot open: at max_positions cap (%d).", self.max_positions)
            return OrderResult(success=False, invalid=True, reason="max_positions_reached")

        fill, spread_paid, slippage = self._fill_open(market_price, direction)
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
        logger.debug("Opened %s ticket=%d lot=%.2f entry=%.5f", direction.name, pos.ticket, lot_size, fill)
        return OrderResult(success=True, position=pos)

    def close_position(self, market_price: float, direction: Direction, lot_size: float) -> OrderResult:
        """
        Close the oldest open position matching both direction and lot_size.

        Returns OrderResult(invalid=True) if no matching position exists.
        """
        target = next(
            (p for p in self._positions
             if p.direction == direction and abs(p.lot_size - lot_size) < 1e-9),
            None,
        )
        if target is None:
            logger.debug("CLOSE_%s %.2f lots: no matching position.", direction.name, lot_size)
            return OrderResult(success=False, invalid=True, reason="no_matching_lot")

        fill, spread_paid, slippage = self._fill_close(market_price, target.direction)

        mfe_pnl = self._pnl(target.entry_price, target.mfe_price, target.direction, target.lot_size)
        mae_pnl = self._pnl(target.entry_price, target.mae_price, target.direction, target.lot_size)
        pnl     = self._pnl(target.entry_price, fill,             target.direction, target.lot_size)

        trade = ClosedTrade(
            ticket      = target.ticket,
            direction   = target.direction,
            lot_size    = target.lot_size,
            entry_price = target.entry_price,
            exit_price  = fill,
            pnl         = pnl,
            spread_paid = target.spread_paid + spread_paid,
            slippage    = target.slippage    + slippage,
            mfe_pnl     = mfe_pnl,
            mae_pnl     = mae_pnl,
        )
        self._positions.remove(target)
        self._closed_trades.append(trade)
        self._cumulative_pnl += pnl
        logger.debug("Closed ticket=%d lot=%.2f exit=%.5f pnl=%.2f", trade.ticket, trade.lot_size, fill, pnl)
        return OrderResult(success=True, trade=trade)

    def close_all(self, market_price: float) -> list[ClosedTrade]:
        """Close every open position. Used at episode end."""
        trades = []
        for pos in list(self._positions):
            result = self.close_position(market_price, pos.direction, pos.lot_size)
            if result.trade is not None:
                trades.append(result.trade)
        return trades

    def total_unrealized_pnl(self, current_price: float) -> float:
        return sum(p.unrealized_pnl(current_price, self.spec.contract_size) for p in self._positions)

    def position_state_vector(self, current_price: float, max_positions: int) -> np.ndarray:
        """
        Fixed-length float32 vector of all open positions.
        Each slot: [is_filled, direction, lot_size, entry_price_delta, unrealized_pnl]
        Unfilled slots are zero-padded. Length = max_positions * 5.
        """
        vec = np.zeros(max_positions * 5, dtype=np.float32)
        for i, pos in enumerate(self._positions[:max_positions]):
            base = i * 5
            vec[base]     = 1.0
            vec[base + 1] = float(pos.direction)
            vec[base + 2] = pos.lot_size
            vec[base + 3] = (pos.entry_price - current_price) / current_price
            vec[base + 4] = pos.unrealized_pnl(current_price, self.spec.contract_size)
        return vec

    def reset(self) -> None:
        self._positions      = []
        self._closed_trades  = []
        self._cumulative_pnl = 0.0
        self._next_ticket    = 1
