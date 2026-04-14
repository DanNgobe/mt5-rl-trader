"""
strategies/martingale_strategy.py
---------------------------------
Averaging-Down Martingale Baseline Strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.simulator import Direction
from .base import BaseStrategy

if TYPE_CHECKING:
    from env.trading_env import TradingEnv


class MartingaleBaseline(BaseStrategy):
    """
    Averaging-down Martingale baseline strategy for Imitation Learning.

    Behaviour
    ---------
    1. **Flat** → open an anchor position at tier 0 (smallest lot size).

    2. **Holding** → the **anchor position's PnL** (oldest ticket) drives
       escalation in fixed loss_threshold steps:

       | Anchor PnL          | Desired positions open |
       |---------------------|------------------------|
       | > loss_threshold    | 1  (just the anchor)   |
       | <= 1× threshold     | 2  (add tier 1)        |
       | <= 2× threshold     | 3  (add tier 2)        |
       | <= 3× threshold     | 4  (add tier 3) …etc.  |

       If a new tier is needed but all tiers are already occupied, HOLD.
       Positions are *never* closed individually on a loss.

    3. **Aggregate PnL >= profit_threshold** → CLOSE ALL positions.

    The anchor-driven model prevents runaway stacking: each new position
    requires the *original* trade to have moved a full ``loss_threshold``
    interval further against you.

    Parameters
    ----------
    loss_threshold : float
        Anchor PnL interval (account currency) that triggers each new
        averaging tier.  Must be negative.  Default -2.0.
    profit_threshold : float
        Aggregate PnL above which all positions are closed.
        Must be positive.  Default 2.0.
    direction : Direction
        Which direction to trade (default Direction.LONG).
    """

    name = "martingale"

    def __init__(
        self,
        loss_threshold:   float     = -2.0,
        profit_threshold: float     = 2.0,
        direction:        Direction = Direction.LONG,
    ):
        if loss_threshold >= 0:
            raise ValueError("loss_threshold must be negative.")
        if profit_threshold <= 0:
            raise ValueError("profit_threshold must be positive.")

        self._loss_threshold   = loss_threshold
        self._profit_threshold = profit_threshold
        self._direction        = direction

    def reset(self) -> None:
        pass   # stateless — all state is derived from env positions

    def act(self, env: "TradingEnv") -> int:
        price     = env._current_price()
        positions = [p for p in env._sim.positions if p.direction == self._direction]

        # ── Flat: open anchor position at tier 0 ────────────────────────────
        if not positions:
            if self._direction == Direction.LONG:
                return self._buy(0)
            else:
                return self._sell(0)

        # ── Aggregate PnL → close everything when profitable ─────────────────
        total_pnl = sum(
            p.unrealized_pnl(price, env.spec.contract_size)
            for p in positions
        )
        if total_pnl >= self._profit_threshold:
            return env.n_actions - 1   # CLOSE_ALL

        # ── Anchor position: the oldest open position (lowest ticket) ─────────
        # Its PnL drives the escalation — not the aggregate.
        anchor     = min(positions, key=lambda p: p.ticket)
        anchor_pnl = anchor.unrealized_pnl(price, env.spec.contract_size)

        # How many positions *should* be open based on anchor's loss depth?
        #   depth 0 (loss < |threshold|)       → 1  position
        #   depth 1 (loss >= 1x |threshold|)   → 2  positions
        #   depth 2 (loss >= 2x |threshold|)   → 3  positions  … etc.
        depth = int(abs(anchor_pnl) / abs(self._loss_threshold))
        desired_count = min(1 + depth, len(env.lot_tiers))

        # ── Add the next tier if we're below the desired stack size ──────────
        if len(positions) < desired_count:
            open_tiers = {
                min(
                    range(len(env.lot_tiers)),
                    key=lambda i: abs(env.lot_tiers[i] - p.lot_size),
                )
                for p in positions
            }
            next_tier = next(
                (t for t in range(len(env.lot_tiers)) if t not in open_tiers),
                None,
            )
            if next_tier is not None:
                if self._direction == Direction.LONG:
                    return self._buy(next_tier)
                else:
                    return self._sell(next_tier)

        return self._hold()

