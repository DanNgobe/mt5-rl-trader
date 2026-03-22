"""
Training script for the forex RL agent using Stable-Baselines3 PPO.

Expects:
    data/raw/EURUSD.csv, USDJPY.csv, ...  (or a single file)

Each file is loaded and preprocessed independently (log returns + per-file
volume z-score) then the agent is trained across all of them via
SubprocVecEnv — one environment per symbol.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.preprocessor import load_symbol_files, preprocess, load_and_preprocess
from env.trading_env import TradingEnv
from env.simulator import SymbolSpec

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbols config loader
# ---------------------------------------------------------------------------

def load_symbol_spec(symbols_config_path: str, symbol: str) -> SymbolSpec:
    """
    Build a SymbolSpec from the symbols YAML config for a given symbol.

    Falls back to EURUSD defaults if the symbol is not found.
    """
    # Strip timeframe suffix (e.g., EURUSD_H1 -> EURUSD)
    symbol_base = symbol.upper().split("_")[0]
    
    path = Path(symbols_config_path)
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        spec_raw = cfg.get("symbols", {}).get(symbol_base)
    else:
        spec_raw = None

    if spec_raw is None:
        log.warning(
            "Symbol %s not found in %s — using EURUSD defaults.",
            symbol_base, symbols_config_path,
        )
        spec_raw = {
            "pip_value": 0.0001,
            "pip_location": 4,
            "contract_size": 100_000,
            "typical_spread_pips": 1.0,
            "min_lot": 0.01,
            "max_lot": 100.0,
            "margin_requirement": 0.01,
        }

    return SymbolSpec(
        name          = symbol.upper(),
        pip_value     = float(spec_raw["pip_value"]),
        pip_location  = int(spec_raw["pip_location"]),
        contract_size = int(spec_raw["contract_size"]),
        spread_pips   = float(spec_raw["typical_spread_pips"]),
        min_lot       = float(spec_raw["min_lot"]),
        max_lot       = float(spec_raw["max_lot"]),
        margin_rate   = float(spec_raw["margin_requirement"]),
    )


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env_fn(
    data:        np.ndarray,
    raw_close:   np.ndarray,
    symbol_spec: SymbolSpec,
    env_config:  dict,
    seed:        int,
):
    """Return a callable that creates a TradingEnv (required by VecEnv)."""
    def _init():
        env = TradingEnv(
            data            = data,
            raw_close       = raw_close,
            symbol_spec     = symbol_spec,
            window_size     = env_config["window_size"],
            initial_balance = env_config["initial_balance"],
            max_positions   = env_config.get("max_positions", 3),
            slippage_prob   = env_config["slippage_prob"],
            slippage_range  = tuple(env_config["slippage_range"]),
            render_mode     = None,
        )
        env.reset(seed=seed)
        return env
    return _init


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class TradingMetricsCallback(BaseCallback):
    """
    Logs trading-specific metrics to TensorBoard at the end of each rollout.

    Metrics logged:
        trading/mean_episode_pnl
        trading/mean_win_rate
        trading/mean_trades_per_episode
        trading/mean_max_drawdown
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_stats: list[dict] = []

    def _on_step(self) -> bool:
        # Collect episode stats whenever an episode finishes
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 stores episode info under the "episode" key when it ends
            if "episode" in info:
                # Pull stats directly from the env if accessible
                env = self.training_env
                # unwrap DummyVecEnv / SubprocVecEnv
                if hasattr(env, "envs"):
                    for e in env.envs:
                        if hasattr(e, "episode_stats"):
                            self._episode_stats.append(e.episode_stats())
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_stats:
            return

        pnls      = [s["total_pnl"]    for s in self._episode_stats]
        win_rates = [s["win_rate"]      for s in self._episode_stats]
        trades    = [s["total_trades"]  for s in self._episode_stats]
        dds       = [s["max_drawdown"]  for s in self._episode_stats]

        self.logger.record("trading/mean_episode_pnl",       float(np.mean(pnls)))
        self.logger.record("trading/mean_win_rate",           float(np.mean(win_rates)))
        self.logger.record("trading/mean_trades_per_episode", float(np.mean(trades)))
        self.logger.record("trading/mean_max_drawdown",       float(np.mean(dds)))

        self._episode_stats.clear()


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(
    config_path:     str            = "config/config.yaml",
    data_dir:        Optional[str]  = None,
    symbols:         Optional[list[str]] = None,
    model_path:      Optional[str]  = None,
    output_dir:      str            = "models",
    total_timesteps: Optional[int]  = None,
    seed:            int            = 42,
) -> PPO:
    """
    Load data, build environments, and train the PPO agent.

    Args:
        config_path:     Path to config/config.yaml.
        data_dir:        Directory containing SYMBOL.csv files.
                         Overrides config if provided.
        symbols:         Whitelist of symbols to train on.
                         If None, all files in data_dir are used.
        model_path:      Existing model to continue training from.
        output_dir:      Directory for checkpoints and final model.
        total_timesteps: Overrides config value if provided.
        seed:            Master random seed.
    """
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    log.info("Loading config from %s", config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env_cfg      = config["environment"]
    train_cfg    = config["training"]
    agent_cfg    = config["agent"]

    if data_dir is None:
        data_dir = config["data"]["raw_data_dir"]
    if total_timesteps is None:
        total_timesteps = train_cfg["total_timesteps"]

    symbols_config = config.get("symbols_config", "config/symbols.yaml")

    # ------------------------------------------------------------------
    # Load and preprocess all symbol files
    # ------------------------------------------------------------------
    log.info("Loading symbol data from %s (symbols=%s)", data_dir, symbols or "all")
    symbol_data = load_symbol_files(data_dir, symbols=symbols)

    if not symbol_data:
        raise ValueError(f"No data loaded from {data_dir}")

    log.info("Loaded %d symbol(s): %s", len(symbol_data), list(symbol_data.keys()))

    # ------------------------------------------------------------------
    # Build one env per symbol for vectorised training
    # ------------------------------------------------------------------
    env_fns = []
    eval_env_fns = []

    for i, (symbol, processed) in enumerate(symbol_data.items()):
        # Load the raw close prices alongside the processed data
        raw_path  = Path(data_dir) / f"{symbol}.csv"
        if not raw_path.exists():
            raw_path = Path(data_dir) / f"{symbol}.npy"

        if raw_path.suffix == ".csv":
            import pandas as pd
            raw_close = pd.read_csv(raw_path)["close"].to_numpy(dtype=np.float64)
        else:
            raw_close = np.load(raw_path)[:, 3]  # column 3 = close

        spec = load_symbol_spec(symbols_config, symbol)

        env_fns.append(make_env_fn(processed, raw_close, spec, env_cfg, seed + i))
        # Separate seed offset for eval envs
        eval_env_fns.append(make_env_fn(processed, raw_close, spec, env_cfg, seed + 10_000 + i))

    # Use SubprocVecEnv for multiple symbols, DummyVecEnv for one
    if len(env_fns) == 1:
        vec_env  = DummyVecEnv(env_fns)
        eval_env = DummyVecEnv(eval_env_fns)
    else:
        vec_env  = SubprocVecEnv(env_fns)
        eval_env = SubprocVecEnv(eval_env_fns)

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = Path(train_cfg["log_dir"]) / f"ppo_{timestamp}"
    model_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build or load PPO model
    # ------------------------------------------------------------------
    policy_kwargs = {}
    if "policy_kwargs" in agent_cfg:
        policy_kwargs["net_arch"] = agent_cfg["policy_kwargs"]["net_arch"]

    if model_path is not None:
        log.info("Continuing training from %s", model_path)
        model = PPO.load(model_path, env=vec_env)
    else:
        log.info("Building new PPO model.")
        model = PPO(
            policy             = agent_cfg["policy"],
            env                = vec_env,
            learning_rate      = agent_cfg["learning_rate"],
            n_steps            = agent_cfg["n_steps"],
            batch_size         = agent_cfg["batch_size"],
            n_epochs           = agent_cfg["n_epochs"],
            gamma              = agent_cfg["gamma"],
            gae_lambda         = agent_cfg["gae_lambda"],
            clip_range         = agent_cfg["clip_range"],
            ent_coef           = agent_cfg["ent_coef"],
            vf_coef            = agent_cfg["vf_coef"],
            max_grad_norm      = agent_cfg["max_grad_norm"],
            clip_range_vf      = agent_cfg.get("clip_range_vf"),
            normalize_advantage= agent_cfg.get("normalize_advantage", True),
            target_kl          = agent_cfg.get("target_kl"),
            policy_kwargs      = policy_kwargs,
            seed               = seed,
            verbose            = train_cfg["verbose"],
            tensorboard_log    = str(log_dir),
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
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
            EvalCallback(
                eval_env,
                best_model_save_path = str(model_dir),
                log_path             = str(log_dir),
                eval_freq            = max(train_cfg["eval_freq"] // len(env_fns), 1),
                n_eval_episodes      = train_cfg["n_eval_episodes"],
                deterministic        = True,
                verbose              = 1,
            )
        )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log.info(
        "Training for %d timesteps across %d environment(s). Logs → %s",
        total_timesteps, len(env_fns), log_dir,
    )

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        tb_log_name     = "PPO",
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    final_path = model_dir / "ppo_trading_final.zip"
    model.save(str(final_path))
    log.info("Final model saved to %s", final_path)

    vec_env.close()
    eval_env.close()

    return model