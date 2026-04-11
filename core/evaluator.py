"""
core/evaluator.py
-----------------
Single evaluation loop that works for any BaseAgent — PPO model,
random strategy, MA-cross, or anything else that implements act(env).

Previously this logic was duplicated across agents/evaluate.py and
strategies/evaluate_strategy.py.  One class, one loop.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from core.agent import BaseAgent
from core.config import load_config, load_symbol_spec
from core.metrics import calculate_metrics, print_metrics, save_results
from env.preprocessor import build_obs_arrays, load_csv, load_npy, preprocess
from env.trading_env import TradingEnv

try:
    from sb3_contrib.common.wrappers import ActionMasker
    _HAS_MASKER = True
except ImportError:
    _HAS_MASKER = False

log = logging.getLogger(__name__)


class Evaluator:
    """
    Runs any BaseAgent for N episodes and returns aggregated metrics.

    Usage
    -----
        evaluator = Evaluator(config_path="config/config.yaml")
        results   = evaluator.run(agent, data_path="data/raw/EURUSD.csv")
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config      = load_config(config_path)
        self.env_cfg     = self.config["environment"]
        self.obs_cfg     = self.config["observation"]
        self.symbols_cfg = self.config.get("symbols_config", "config/symbols.yaml")
        self._reward_cfg = self.config.get("reward", {})
        self.bars_per_year = int(
            self.config.get("data", {}).get("bars_per_year", 6048)
        )

    def run(
        self,
        agent:        BaseAgent,
        data_path:    str,
        n_episodes:   int   = 10,
        save:         bool  = True,
        results_dir:  str   = "data/evaluation",
        visualise:    bool  = False,
        vis_window:   int   = 120,
        vis_pause:    float = 0.01,
    ) -> dict:
        """
        Evaluate agent for n_episodes and return aggregated metrics.

        Args:
            agent:       Any BaseAgent instance.
            data_path:   Path to OHLCV CSV or .npy file.
            n_episodes:  Number of full episodes to run.
            save:        Write results to results_dir.
            results_dir: Output directory.
            visualise:   Show live per-step dashboard (evaluation only).
            vis_window:  Price bars visible in the dashboard.
            vis_pause:   Seconds to pause after each redraw.

        Returns:
            Dict with keys: metrics, trades, equity_curves, episode_summaries.
        """
        initial_balance = float(self.env_cfg["initial_balance"])

        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------
        path = Path(data_path)
        raw_data, dt_index = load_npy(path) if path.suffix.lower() == ".npy" else load_csv(path)
        processed = preprocess(raw_data)
        raw_close = raw_data[:, 3].astype(np.float64)
        obs_arrays = build_obs_arrays(raw_data, self.obs_cfg, dt_index)

        symbol      = path.stem.upper().split("_")[0]
        symbol_spec = load_symbol_spec(self.symbols_cfg, symbol)
        log.info(
            "[%s] Evaluating on %s  (spread=%.1f pips)",
            agent.name, symbol, symbol_spec.spread_pips,
        )

        # ------------------------------------------------------------------
        # Build env
        # ------------------------------------------------------------------
        def _make_env():
            env = TradingEnv(
                obs_arrays             = obs_arrays,
                raw_close              = raw_close,
                symbol_spec            = symbol_spec,
                obs_config             = self.obs_cfg,
                initial_balance        = initial_balance,
                lot_tiers              = self.env_cfg.get("lot_tiers", [0.1, 0.2, 0.5]),
                slippage_prob          = self.env_cfg["slippage_prob"],
                slippage_range         = tuple(self.env_cfg["slippage_range"]),
                holding_cost_per_lot   = self._reward_cfg.get("holding_cost_per_lot", 0.0001),
                flat_penalty_per_step  = self._reward_cfg.get("flat_penalty_per_step", 0.0),
                spread_cost_scale      = self._reward_cfg.get("spread_cost_scale", 2.0),
                reward_mode            = self._reward_cfg.get("reward_mode", "sparse"),
                portfolio_offset_factor = self._reward_cfg.get("portfolio_offset_factor", 0.0),
                volatility_penalty_multiplier = self._reward_cfg.get("volatility_penalty_multiplier", 0.0),
                drawdown_penalty       = self._reward_cfg.get("drawdown_penalty", 5.0),
                max_drawdown_pct       = self.env_cfg.get("max_drawdown_pct", 0.5),
                render_mode            = None,
            )
            if _HAS_MASKER:
                env = ActionMasker(env, lambda e: e.action_masks())
            return env

        inner_env_unwrapped = _make_env()
        vec_env = DummyVecEnv([lambda: inner_env_unwrapped])
        
        n_stack = self.env_cfg.get("frame_stack", 1)
        if n_stack > 1:
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)

        inner_env: TradingEnv = inner_env_unwrapped.env if _HAS_MASKER else inner_env_unwrapped

        # ------------------------------------------------------------------
        # Optional visualiser
        # ------------------------------------------------------------------
        vis = None
        if visualise:
            from core.visualiser import EpisodeVisualiser
            vis = EpisodeVisualiser(window=vis_window, pause=vis_pause)
            log.info("Visualiser enabled (window=%d  pause=%.3fs)", vis_window, vis_pause)

        # ------------------------------------------------------------------
        # Episode loop
        # ------------------------------------------------------------------
        log.info("[%s] Running %d episode(s)...", agent.name, n_episodes)

        all_trades:        list[dict]       = []
        all_equity_curves: list[np.ndarray] = []
        episode_summaries: list[dict]       = []

        for ep in range(n_episodes):
            obs = vec_env.reset()
            obs = obs[0]
            agent.reset()

            if vis is not None:
                vis.reset()

            done         = False
            equity_curve = [inner_env._balance]

            while not done:
                action = agent.act(inner_env, obs)

                obs, vec_reward, vec_done, vec_info = vec_env.step([action])
                obs = obs[0]
                reward = vec_reward[0]
                done = vec_done[0]
                info = vec_info[0]

                terminal_info = info.get("terminal_info", info)

                equity = terminal_info.get("equity", info.get("equity", inner_env._balance))

                if vis is not None and not done:
                    vis.update(inner_env, float(reward), action=action)

                equity_curve.append(equity)

                if done:
                    stats = terminal_info.get("episode_stats") or info.get("episode_stats")
                    if stats:
                        episode_summaries.append(stats)

                    eps_trades = terminal_info.get("episode_trades") or info.get("episode_trades", [])
                    for td_obj in eps_trades:
                        td = td_obj.to_dict()
                        td.update({"episode": ep + 1, "agent": agent.name})
                        all_trades.append(td)

                    if vis is not None:
                        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = str(agent.name).replace(":", "_").replace("\\", "_").replace("/", "_").replace(".zip", "")
                        save_path = Path(results_dir) / f"vis_{safe_name}_ep{ep+1}_{ts}.png"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        vis.save(str(save_path))

                    log.info(
                        "[%s] ep %d/%d  trades=%d  pnl=$%.2f  "
                        "win_rate=%.1f%%  balance=$%.2f",
                        agent.name, ep + 1, n_episodes,
                        stats["total_trades"], stats["total_pnl"],
                        stats["win_rate"] * 100, stats["final_balance"],
                    )

            all_equity_curves.append(np.array(equity_curve, dtype=np.float64))

        if vis is not None:
            vis.close()
        vec_env.close()

        # ------------------------------------------------------------------
        # Metrics + output
        # ------------------------------------------------------------------
        full_equity = np.concatenate(all_equity_curves)
        metrics     = calculate_metrics(
            all_trades, full_equity, initial_balance, self.bars_per_year
        )
        print_metrics(metrics, label=f"RESULTS — {agent.name.upper()}")

        if save:
            save_results(
                metrics, all_trades, all_equity_curves, episode_summaries,
                agent_label = agent.name,
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
