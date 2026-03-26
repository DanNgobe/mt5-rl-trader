"""
Training script for the forex RL agent using Stable-Baselines3 PPO.

Expects:
    data/raw/EURUSD.csv, USDJPY.csv, ...  (or a single file)

Each file is loaded and preprocessed independently then the agent is
trained across all of them via SubprocVecEnv — one environment per symbol.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from core.config import load_config, load_symbol_spec
from env.preprocessor import load_symbol_files
from env.trading_env import TradingEnv

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _get_action_masks(env) -> np.ndarray:
    """Top-level callable for ActionMasker — picklable for SubprocVecEnv on Windows."""
    return env.action_masks()


def _make_env_fn(
    obs_arrays:  dict,
    raw_close:   np.ndarray,
    symbol_spec,
    env_config:  dict,
    reward_cfg:  dict,
    obs_cfg:     dict,
    seed:        int,
):
    def _init():
        env = TradingEnv(
            obs_arrays             = obs_arrays,
            raw_close              = raw_close,
            symbol_spec            = symbol_spec,
            obs_config             = obs_cfg,
            initial_balance        = env_config["initial_balance"],
            lot_tiers              = env_config.get("lot_tiers", [0.1, 0.2, 0.5]),
            slippage_prob          = env_config["slippage_prob"],
            slippage_range         = tuple(env_config["slippage_range"]),
            holding_cost_per_lot   = reward_cfg.get("holding_cost_per_lot", 0.0001),
            flat_penalty_per_step  = reward_cfg.get("flat_penalty_per_step", 0.0),
            spread_cost_scale      = reward_cfg.get("spread_cost_scale", 2.0),
            portfolio_offset_factor = reward_cfg.get("portfolio_offset_factor", 0.0),
            max_drawdown_pct       = env_config.get("max_drawdown_pct", 0.5),
            episode_length         = env_config.get("episode_length"),
            random_start           = env_config.get("random_start", False),
            render_mode            = None,
        )
        env = ActionMasker(env, _get_action_masks)
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TradingMetricsCallback(BaseCallback):
    """
    Logs trading-specific metrics to TensorBoard at the end of each rollout.

    Reads episode stats from the `episode_stats` key injected into the info
    dict by TradingEnv on termination — works with both DummyVecEnv and
    SubprocVecEnv because info dicts are serialised back to the main process.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_stats: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            stats = info.get("episode_stats")
            if stats is not None:
                self._episode_stats.append(stats)
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_stats:
            return
        pnls      = [s["total_pnl"]   for s in self._episode_stats]
        win_rates = [s["win_rate"]     for s in self._episode_stats]
        trades    = [s["total_trades"] for s in self._episode_stats]
        dds       = [s["max_drawdown"] for s in self._episode_stats]
        forced    = [s["forced_trades"] for s in self._episode_stats]
        self.logger.record("trading/mean_episode_pnl",       float(np.mean(pnls)))
        self.logger.record("trading/mean_win_rate",           float(np.mean(win_rates)))
        self.logger.record("trading/mean_trades_per_episode", float(np.mean(trades)))
        self.logger.record("trading/mean_max_drawdown",       float(np.mean(dds)))
        self.logger.record("trading/mean_forced_closes",      float(np.mean(forced)))
        self._episode_stats.clear()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: MaskablePPO, output_path: str, obs_dim: int) -> bool:
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

        from pathlib import Path as _Path
        out = _Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        class _PolicyExport(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.p = p
            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                features     = self.p.extract_features(obs)
                latent_pi, _ = self.p.mlp_extractor(features)
                return self.p.action_net(latent_pi)

        wrapper   = _PolicyExport(policy)
        wrapper.eval()
        dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
        tmp_path  = str(out) + ".tmp.onnx"

        with torch.no_grad():
            torch.onnx.export(
                wrapper, dummy_obs, tmp_path,
                input_names=["observation"], output_names=["action_logits"],
                opset_version=12, do_constant_folding=True, dynamo=False,
            )

        try:
            import onnx, os
            proto = onnx.load(tmp_path)
            onnx.checker.check_model(proto)
            onnx.save(proto, str(out))
            os.remove(tmp_path)
            log.info("ONNX validated and saved → %s", out)
        except ImportError:
            import os
            os.replace(tmp_path, str(out))
            log.warning("onnx package not installed — skipping validation.")

        return True
    except Exception as exc:
        log.error("ONNX export failed: %s", exc, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(
    config_path:     str                  = "config/config.yaml",
    data_dir:        Optional[str]        = None,
    symbols:         Optional[list[str]]  = None,
    model_path:      Optional[str]        = None,
    output_dir:      str                  = "models",
    total_timesteps: Optional[int]        = None,
    seed:            int                  = 42,
) -> MaskablePPO:
    config      = load_config(config_path)
    env_cfg     = config["environment"]
    train_cfg   = config["training"]
    agent_cfg   = config["agent"]
    reward_cfg  = config.get("reward", {})
    obs_cfg     = config.get("observation", {})
    symbols_cfg = config.get("symbols_config", "config/symbols.yaml")

    if data_dir is None:
        data_dir = config["data"]["raw_data_dir"]
    if total_timesteps is None:
        total_timesteps = train_cfg["total_timesteps"]

    log.info("Loading symbol data from %s (symbols=%s)", data_dir, symbols or "all")
    symbol_data = load_symbol_files(data_dir, obs_cfg=obs_cfg, symbols=symbols)
    if not symbol_data:
        raise ValueError(f"No data loaded from {data_dir}")
    log.info("Loaded %d symbol(s): %s", len(symbol_data), list(symbol_data.keys()))

    env_fns      = []
    eval_env_fns = []

    for i, (symbol, payload) in enumerate(symbol_data.items()):
        spec = load_symbol_spec(symbols_cfg, symbol)
        env_fns.append(_make_env_fn(
            payload["obs_arrays"], payload["raw_close"],
            spec, env_cfg, reward_cfg, obs_cfg, seed + i,
        ))
        eval_env_fns.append(_make_env_fn(
            payload["obs_arrays"], payload["raw_close"],
            spec, env_cfg, reward_cfg, obs_cfg, seed + 10_000 + i,
        ))

    VecEnvCls = DummyVecEnv if len(env_fns) == 1 else SubprocVecEnv
    vec_env   = VecEnvCls(env_fns)
    eval_env  = VecEnvCls(eval_env_fns)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = Path(train_cfg["log_dir"]) / f"ppo_{timestamp}"
    model_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = {}
    if "policy_kwargs" in agent_cfg:
        policy_kwargs["net_arch"] = agent_cfg["policy_kwargs"]["net_arch"]

    if model_path is not None:
        log.info("Continuing training from %s", model_path)
        model = MaskablePPO.load(model_path, env=vec_env)
    else:
        log.info("Building new MaskablePPO model.")
        model = MaskablePPO(
            policy              = agent_cfg["policy"],
            env                 = vec_env,
            learning_rate       = agent_cfg["learning_rate"],
            n_steps             = agent_cfg["n_steps"],
            batch_size          = agent_cfg["batch_size"],
            n_epochs            = agent_cfg["n_epochs"],
            gamma               = agent_cfg["gamma"],
            gae_lambda          = agent_cfg["gae_lambda"],
            clip_range          = agent_cfg["clip_range"],
            ent_coef            = agent_cfg["ent_coef"],
            vf_coef             = agent_cfg["vf_coef"],
            max_grad_norm       = agent_cfg["max_grad_norm"],
            clip_range_vf       = agent_cfg.get("clip_range_vf"),
            normalize_advantage = agent_cfg.get("normalize_advantage", True),
            target_kl           = agent_cfg.get("target_kl"),
            policy_kwargs       = policy_kwargs,
            seed                = seed,
            verbose             = train_cfg["verbose"],
            tensorboard_log     = str(log_dir),
            device              = "cpu",
        )

    callbacks = [
        CheckpointCallback(
            save_freq   = max(train_cfg["save_freq"] // len(env_fns), 1),
            save_path   = str(model_dir),
            name_prefix = "ppo_trading",
            verbose     = 1,
        ),
        TradingMetricsCallback(verbose=1),
    ]

    if train_cfg.get("eval_freq", 0) > 0:
        callbacks.append(
            MaskableEvalCallback(
                eval_env,
                best_model_save_path = str(model_dir),
                log_path             = str(log_dir),
                eval_freq            = max(train_cfg["eval_freq"] // len(env_fns), 1),
                n_eval_episodes      = train_cfg["n_eval_episodes"],
                deterministic        = True,
                verbose              = 1,
            )
        )

    log.info(
        "Training for %d timesteps across %d env(s). Logs → %s",
        total_timesteps, len(env_fns), log_dir,
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="PPO")

    final_path = model_dir / "ppo_trading_final.zip"
    model.save(str(final_path))
    log.info("Final model saved → %s", final_path)

    vec_env.close()
    eval_env.close()
    return model
