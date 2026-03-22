"""
core/metrics.py
---------------
Shared performance metric calculations and result persistence.

Previously split across agents/evaluate.py and strategies/evaluate_strategy.py
with near-identical implementations.  A single definition here guarantees
RL agent and baseline results are always computed identically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def calculate_metrics(
    trades:          list[dict],
    equity_curve:    np.ndarray,
    initial_balance: float,
    bars_per_year:   int = 6048,
) -> dict:
    """
    Compute comprehensive backtest performance metrics.

    Args:
        trades:          List of closed-trade dicts (must contain 'pnl' key).
        equity_curve:    Per-step equity values as a 1-D array.
        initial_balance: Starting account balance.
        bars_per_year:   Annualisation factor for Sharpe ratio.
                         H1=6048, H4=1512, D1=252, W1=52.

    Returns:
        Dict of scalar metrics.
    """
    empty = {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
        "final_balance": initial_balance, "total_return_pct": 0.0,
        "calmar_ratio": 0.0, "expectancy": 0.0,
    }
    if not trades:
        return empty

    pnls         = np.array([t["pnl"] for t in trades])
    winning_pnls = pnls[pnls > 0]
    losing_pnls  = pnls[pnls < 0]
    n_trades     = len(pnls)
    n_wins       = len(winning_pnls)
    n_losses     = len(losing_pnls)

    gross_profit  = float(winning_pnls.sum()) if n_wins   > 0 else 0.0
    gross_loss    = abs(float(losing_pnls.sum())) if n_losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_balance    = float(equity_curve[-1]) if len(equity_curve) > 0 else initial_balance
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100.0

    peak       = np.maximum.accumulate(equity_curve)
    dd_abs     = peak - equity_curve
    max_dd     = float(dd_abs.max())
    max_dd_pct = float((dd_abs / np.where(peak > 0, peak, 1)).max() * 100.0)

    if len(equity_curve) > 1:
        step_returns = np.diff(equity_curve) / np.where(
            equity_curve[:-1] != 0, equity_curve[:-1], 1
        )
        std    = step_returns.std()
        sharpe = float(np.sqrt(bars_per_year) * step_returns.mean() / std) if std > 0 else 0.0
    else:
        sharpe = 0.0

    win_rate   = n_wins / n_trades
    avg_win    = float(winning_pnls.mean()) if n_wins   > 0 else 0.0
    avg_loss   = float(losing_pnls.mean())  if n_losses > 0 else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    calmar     = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0

    return {
        "total_trades":     n_trades,
        "winning_trades":   n_wins,
        "losing_trades":    n_losses,
        "win_rate":         win_rate,
        "total_pnl":        float(pnls.sum()),
        "avg_pnl":          float(pnls.mean()),
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "profit_factor":    profit_factor,
        "sharpe_ratio":     sharpe,
        "max_drawdown":     max_dd,
        "max_drawdown_pct": max_dd_pct,
        "final_balance":    final_balance,
        "total_return_pct": total_return_pct,
        "calmar_ratio":     calmar,
        "expectancy":       expectancy,
    }


def print_metrics(m: dict, label: str = "RESULTS") -> None:
    """Log a formatted metrics summary."""
    log.info("=" * 52)
    log.info(label)
    log.info("=" * 52)
    log.info("Trades        : %d  (W:%d / L:%d)  win_rate=%.1f%%",
             m["total_trades"], m["winning_trades"], m["losing_trades"],
             m["win_rate"] * 100)
    log.info("Total P&L     : $%.2f  (avg $%.2f/trade)", m["total_pnl"], m["avg_pnl"])
    log.info("Avg win       : $%.2f    Avg loss: $%.2f", m["avg_win"], m["avg_loss"])
    log.info("Profit factor : %.2f", m["profit_factor"])
    log.info("Expectancy    : $%.2f/trade", m["expectancy"])
    log.info("Final balance : $%.2f  (%.2f%% return)", m["final_balance"], m["total_return_pct"])
    log.info("Sharpe ratio  : %.2f", m["sharpe_ratio"])
    log.info("Max drawdown  : $%.2f (%.2f%%)", m["max_drawdown"], m["max_drawdown_pct"])
    log.info("Calmar ratio  : %.2f", m["calmar_ratio"])


def save_results(
    metrics:           dict,
    trades:            list[dict],
    equity_curves:     list[np.ndarray],
    episode_summaries: list[dict],
    agent_label:       str,
    data_path:         str,
    n_episodes:        int,
    results_dir:       str,
) -> None:
    """Persist metrics, trades CSV, equity curve, and summary JSON."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)

    mf = base / f"metrics_{ts}.json"
    with open(mf, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics   → %s", mf)

    if trades:
        tf = base / f"trades_{ts}.csv"
        pd.DataFrame(trades).to_csv(tf, index=False)
        log.info("Trades    → %s", tf)

    if equity_curves:
        ef = base / f"equity_curve_{ts}.npy"
        np.save(ef, equity_curves[-1])
        log.info("Equity    → %s", ef)

    summary = {
        "timestamp":   ts,
        "agent":       agent_label,
        "data_path":   str(data_path),
        "n_episodes":  n_episodes,
        "metrics":     metrics,
        "episodes": [
            {
                "episode":       i + 1,
                "total_trades":  s["total_trades"],
                "total_pnl":     s["total_pnl"],
                "win_rate":      s["win_rate"],
                "max_drawdown":  s["max_drawdown"],
                "final_balance": s["final_balance"],
            }
            for i, s in enumerate(episode_summaries)
        ],
    }
    sf = base / f"summary_{ts}.json"
    with open(sf, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary   → %s", sf)
