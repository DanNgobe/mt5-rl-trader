"""
Evaluation script for the forex RL agent.

Runs the trained model on a single data file for N episodes and reports
performance metrics: Sharpe, drawdown, win rate, profit factor, etc.

calculate_metrics() and _save_results() are intentionally public so
strategies/evaluate_strategy.py can import them — keeping metric
definitions in one place and guaranteeing RL vs baseline results
are always directly comparable.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.preprocessor import load_csv, load_npy, preprocess
from env.simulator import SymbolSpec
from env.trading_env import TradingEnv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_symbol_spec(symbols_config_path: str, symbol: str) -> SymbolSpec:
    """Build a SymbolSpec from symbols.yaml, falling back to EURUSD defaults."""
    symbol = symbol.upper().split("_")[0]
    path   = Path(symbols_config_path)

    spec_raw = None
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        spec_raw = cfg.get("symbols", {}).get(symbol)

    if spec_raw is None:
        log.warning("Symbol %s not in %s — using EURUSD defaults.", symbol, symbols_config_path)
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
# Metrics  (imported by strategies/evaluate_strategy.py — keep public)
# ---------------------------------------------------------------------------

def calculate_metrics(
    trades:          list[dict],
    equity_curve:    np.ndarray,
    initial_balance: float,
) -> dict:
    """Compute comprehensive backtest metrics."""
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
        step_returns = np.diff(equity_curve) / np.where(equity_curve[:-1] != 0, equity_curve[:-1], 1)
        std    = step_returns.std()
        sharpe = float(np.sqrt(6048) * step_returns.mean() / std) if std > 0 else 0.0
    else:
        sharpe = 0.0

    calmar     = total_return_pct / max_dd_pct if max_dd_pct > 0 else 0.0
    win_rate   = n_wins / n_trades
    avg_win    = float(winning_pnls.mean()) if n_wins   > 0 else 0.0
    avg_loss   = float(losing_pnls.mean())  if n_losses > 0 else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

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


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: PPO, output_path: str, obs_dim: int) -> bool:
    """Export the trained SB3 policy to ONNX for MT5 deployment."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        log.error("PyTorch not installed — cannot export ONNX.")
        return False

    try:
        policy = model.policy
        policy.eval()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class PolicyExport(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                features     = self.p.extract_features(obs)
                latent_pi, _ = self.p.mlp_extractor(features)
                return self.p.action_net(latent_pi)

        wrapper = PolicyExport(policy)
        wrapper.eval()

        dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
        tmp_path  = str(output_path) + ".tmp.onnx"

        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy_obs, tmp_path,
                input_names         = ["observation"],
                output_names        = ["action_logits"],
                opset_version       = 12,
                do_constant_folding = True,
                dynamo              = False,
            )

        try:
            import onnx
            model_proto = onnx.load(tmp_path)
            onnx.checker.check_model(model_proto)
            onnx.save(model_proto, str(output_path))
            import os; os.remove(tmp_path)
            log.info("ONNX model validated and saved (opset %d)",
                     model_proto.opset_import[0].version)
        except ImportError:
            import os; os.replace(tmp_path, str(output_path))
            log.warning("onnx package not installed — skipping validation.")

        log.info("ONNX exported → %s", output_path)
        return True

    except Exception as exc:
        log.error("ONNX export failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    model_path:   str,
    data_path:    str,
    config_path:  str   = "config/config.yaml",
    n_episodes:   int   = 10,
    render:       bool  = False,
    save_results: bool  = True,
    results_dir:  str   = "data/evaluation",
    export_onnx_: bool  = False,
    onnx_path:    str   = "models/model.onnx",
    visualise:    bool  = False,
    vis_window:   int   = 120,
    vis_pause:    float = 0.01,
) -> dict:
    """
    Run the trained model for n_episodes and return aggregated metrics.
    Optionally shows a live per-step visualiser dashboard.
    Optionally exports the policy to ONNX for MT5.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_cfg         = config["environment"]
    symbols_config  = config.get("symbols_config", "config/symbols.yaml")
    initial_balance = float(env_cfg["initial_balance"])

    log.info("Loading data from %s", data_path)
    path = Path(data_path)

    if path.suffix.lower() == ".npy":
        raw_data = load_npy(path)
    else:
        raw_data, _ = load_csv(path)

    processed = preprocess(raw_data)
    raw_close = raw_data[:, 3].astype(np.float64)
    log.info("Preprocessed shape: %s", processed.shape)

    symbol      = path.stem.upper().split("_")[0]
    symbol_spec = load_symbol_spec(symbols_config, symbol)
    log.info("Symbol spec: %s (spread=%.1f pips)", symbol_spec.name, symbol_spec.spread_pips)

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
            render_mode     = "human" if render else None,
        )

    vec_env = DummyVecEnv([_make_env])

    log.info("Loading model from %s", model_path)
    model = PPO.load(model_path, env=vec_env)

    if export_onnx_:
        obs_dim = vec_env.observation_space.shape[0]
        export_onnx(model, onnx_path, obs_dim)

    vis = None
    if visualise:
        from env.visualiser import EpisodeVisualiser
        vis = EpisodeVisualiser(window=vis_window, pause=vis_pause)
        log.info("Visualiser enabled  (window=%d  pause=%.3fs)", vis_window, vis_pause)

    log.info("Running %d evaluation episode(s)...", n_episodes)

    all_trades:        list[dict]       = []
    all_equity_curves: list[np.ndarray] = []
    episode_summaries: list[dict]       = []

    inner_env: TradingEnv = vec_env.envs[0]

    for ep in range(n_episodes):
        obs  = vec_env.reset()
        done = False
        equity_curve = [inner_env._balance]

        if vis is not None:
            vis.reset()

        while not done:
            action, _                  = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done                       = bool(dones[0])
            info                       = infos[0]

            # Pass the raw action array so the strip knows what was sent
            if vis is not None:
                vis.update(inner_env, float(rewards[0]), action=action[0])

            current_price = inner_env._current_raw_price()
            equity        = (
                inner_env._balance
                + inner_env._sim.total_unrealized_pnl(current_price)
            )
            equity_curve.append(equity)

            ct = info.get("closed_trade")
            if ct is not None:
                all_trades.append({
                    "episode":     ep + 1,
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
                    save_path = Path(results_dir) / f"vis_ep{ep + 1}_{ts}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    vis.save(str(save_path))

                log.info(
                    "Episode %d/%d  trades=%d  pnl=$%.2f  "
                    "win_rate=%.1f%%  balance=$%.2f",
                    ep + 1, n_episodes,
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
    _print_metrics(metrics)

    if save_results:
        _save_results(
            metrics, all_trades, all_equity_curves, episode_summaries,
            model_path, data_path, n_episodes, results_dir,
        )

    return {
        "metrics":           metrics,
        "trades":            all_trades,
        "equity_curves":     all_equity_curves,
        "episode_summaries": episode_summaries,
    }


# ---------------------------------------------------------------------------
# Output helpers  (imported by strategies/evaluate_strategy.py — keep public)
# ---------------------------------------------------------------------------

def _print_metrics(m: dict) -> None:
    log.info("=" * 52)
    log.info("EVALUATION RESULTS")
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


def _save_results(
    metrics:           dict,
    trades:            list[dict],
    equity_curves:     list[np.ndarray],
    episode_summaries: list[dict],
    model_path:        str,
    data_path:         str,
    n_episodes:        int,
    results_dir:       str,
) -> None:
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

    ef = base / f"equity_curve_{ts}.npy"
    np.save(ef, equity_curves[-1])
    log.info("Equity    → %s", ef)

    summary = {
        "timestamp":  ts,
        "model_path": str(model_path),
        "data_path":  str(data_path),
        "n_episodes": n_episodes,
        "metrics":    metrics,
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