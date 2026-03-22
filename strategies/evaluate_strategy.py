"""
strategies/evaluate_strategy.py
--------------------------------
Evaluation loop for hand-coded baseline strategies.

Mirrors agents/evaluate.py exactly — same metrics, same visualiser
support, same saved outputs — but calls strategy.act(env) instead of
model.predict(obs).  This makes results directly comparable.

Reuses calculate_metrics() and _save_results() from agents/evaluate.py
to guarantee metric definitions stay in sync.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.evaluate import calculate_metrics, _save_results
from env.preprocessor import load_csv, load_npy, preprocess
from env.simulator import SymbolSpec
from env.trading_env import TradingEnv
from strategies.base import BaseStrategy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol spec loader
# ---------------------------------------------------------------------------

def _load_symbol_spec(symbols_config_path: str, symbol: str) -> SymbolSpec:
    symbol = symbol.upper().split("_")[0]
    path   = Path(symbols_config_path)

    spec_raw = None
    if path.exists():
        with open(path) as f:
            import yaml as _yaml
            cfg = _yaml.safe_load(f)
        spec_raw = cfg.get("symbols", {}).get(symbol)

    if spec_raw is None:
        log.warning("Symbol %s not found — using EURUSD defaults.", symbol)
        spec_raw = {
            "pip_value": 0.0001, "pip_location": 4, "contract_size": 100_000,
            "typical_spread_pips": 1.0, "min_lot": 0.01,
            "max_lot": 100.0, "margin_requirement": 0.01,
        }

    return SymbolSpec(
        name          = symbol,
        pip_value     = float(spec_raw["pip_value"]),
        pip_location  = int(spec_raw["pip_location"]),
        contract_size = int(spec_raw["contract_size"]),
        spread_pips   = float(spec_raw.get("typical_spread_pips", 1.0)),
        min_lot       = float(spec_raw["min_lot"]),
        max_lot       = float(spec_raw["max_lot"]),
        margin_rate   = float(spec_raw.get("margin_requirement", 0.01)),
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def evaluate_strategy(
    strategy:     BaseStrategy,
    data_path:    str,
    config_path:  str   = "config/config.yaml",
    n_episodes:   int   = 1,
    save_results: bool  = True,
    results_dir:  str   = "data/evaluation",
    visualise:    bool  = False,
    vis_window:   int   = 120,
    vis_pause:    float = 0.01,
) -> dict:
    """
    Run a BaseStrategy for n_episodes and return aggregated metrics.

    Args:
        strategy:     Any BaseStrategy subclass instance.
        data_path:    Path to OHLCV CSV or .npy file.
        config_path:  Path to config/config.yaml.
        n_episodes:   Number of full episodes to run.
        save_results: Write metrics/trades/summary JSON to results_dir.
        results_dir:  Output directory for saved results.
        visualise:    Show live per-step dashboard (requires matplotlib).
        vis_window:   Price bars visible in the price panel.
        vis_pause:    Seconds to pause after each redraw.

    Returns:
        Dict with keys: metrics, trades, equity_curves, episode_summaries.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_cfg         = config["environment"]
    symbols_config  = config.get("symbols_config", "config/symbols.yaml")
    initial_balance = float(env_cfg["initial_balance"])

    log.info("[%s] Loading data from %s", strategy.name, data_path)
    path = Path(data_path)

    if path.suffix.lower() == ".npy":
        raw_data = load_npy(path)
    else:
        raw_data, _ = load_csv(path)

    processed = preprocess(raw_data)
    raw_close = raw_data[:, 3].astype(np.float64)
    log.info("Preprocessed shape: %s", processed.shape)

    symbol      = path.stem.upper().split("_")[0]
    symbol_spec = _load_symbol_spec(symbols_config, symbol)

    def _make_env():
        return TradingEnv(
            data            = processed,
            raw_close       = raw_close,
            symbol_spec     = symbol_spec,
            window_size     = env_cfg["window_size"],
            initial_balance = initial_balance,
            max_positions   = env_cfg.get("max_positions", 3),
            slippage_prob   = env_cfg["slippage_prob"],
            slippage_range  = tuple(env_cfg["slippage_range"]),
            render_mode     = None,
        )

    vec_env   = DummyVecEnv([_make_env])
    inner_env: TradingEnv = vec_env.envs[0]

    vis = None
    if visualise:
        from env.visualiser import EpisodeVisualiser
        vis = EpisodeVisualiser(window=vis_window, pause=vis_pause)
        log.info(
            "Visualiser enabled  (window=%d  pause=%.3fs)",
            vis_window, vis_pause,
        )

    log.info(
        "[%s] Running %d episode(s) on %s",
        strategy.name, n_episodes, symbol,
    )

    all_trades:        list[dict]       = []
    all_equity_curves: list[np.ndarray] = []
    episode_summaries: list[dict]       = []

    for ep in range(n_episodes):
        vec_env.reset()
        strategy.reset()

        if vis is not None:
            vis.reset()

        done         = False
        equity_curve = [inner_env._balance]
        prev_equity  = inner_env._balance

        while not done:
            # Strategy decides from live env state
            action = strategy.act(inner_env)

            obs, rewards, dones, infos = vec_env.step(action.reshape(1, -1))
            done = bool(dones[0])
            info = infos[0]

            current_price = inner_env._current_raw_price()
            equity        = (
                inner_env._balance
                + inner_env._sim.total_unrealized_pnl(current_price)
            )
            step_reward = (equity - prev_equity) / initial_balance
            prev_equity = equity

            # Pass action so the strip colours the correct action
            if vis is not None:
                vis.update(inner_env, step_reward, action=action)

            equity_curve.append(equity)

            ct = info.get("closed_trade")
            if ct is not None:
                all_trades.append({
                    "episode":     ep + 1,
                    "strategy":    strategy.name,
                    "pnl":         float(ct.pnl),
                    "lot_size":    float(ct.lot_size),
                    "entry_price": float(ct.entry_price),
                    "exit_price":  float(ct.exit_price),
                    "direction":   ct.direction.name,
                    "spread_paid": float(ct.spread_paid),
                    "slippage":    float(ct.slippage),
                })

            if done:
                stats = inner_env.episode_stats()
                episode_summaries.append(stats)

                if vis is not None:
                    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = (
                        Path(results_dir)
                        / f"vis_{strategy.name}_ep{ep + 1}_{ts}.png"
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    vis.save(str(save_path))

                log.info(
                    "[%s] Episode %d/%d  trades=%d  pnl=$%.2f  "
                    "win_rate=%.1f%%  balance=$%.2f",
                    strategy.name, ep + 1, n_episodes,
                    stats["total_trades"],
                    stats["total_pnl"],
                    stats["win_rate"] * 100,
                    stats["final_balance"],
                )

        all_equity_curves.append(np.array(equity_curve, dtype=np.float64))

    if vis is not None:
        vis.close()

    vec_env.close()

    full_equity = np.concatenate(all_equity_curves)
    metrics     = calculate_metrics(all_trades, full_equity, initial_balance)

    _log_metrics(strategy.name, metrics)

    if save_results:
        _save_results(
            metrics, all_trades, all_equity_curves, episode_summaries,
            model_path  = f"strategy:{strategy.name}",
            data_path   = data_path,
            n_episodes  = n_episodes,
            results_dir = results_dir,
        )

    return {
        "metrics":           metrics,
        "trades":            all_trades,
        "equity_curves":     all_equity_curves,
        "episode_summaries": episode_summaries,
    }


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_metrics(strategy_name: str, m: dict) -> None:
    log.info("=" * 52)
    log.info("BASELINE RESULTS — %s", strategy_name.upper())
    log.info("=" * 52)
    log.info("Trades        : %d  (W:%d / L:%d)  win_rate=%.1f%%",
             m["total_trades"], m["winning_trades"], m["losing_trades"],
             m["win_rate"] * 100)
    log.info("Total P&L     : $%.2f  (avg $%.2f/trade)", m["total_pnl"], m["avg_pnl"])
    log.info("Profit factor : %.2f", m["profit_factor"])
    log.info("Expectancy    : $%.2f/trade", m["expectancy"])
    log.info("Final balance : $%.2f  (%.2f%% return)",
             m["final_balance"], m["total_return_pct"])
    log.info("Sharpe ratio  : %.2f", m["sharpe_ratio"])
    log.info("Max drawdown  : $%.2f (%.2f%%)", m["max_drawdown"], m["max_drawdown_pct"])
    log.info("Calmar ratio  : %.2f", m["calmar_ratio"])