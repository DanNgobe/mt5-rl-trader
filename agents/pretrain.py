"""
agents/pretrain.py
------------------
Pre-trains a MaskablePPO agent using Behavioral Cloning (Imitation Learning).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from imitation.algorithms import bc
from imitation.data.types import Transitions

from core.config import load_config, load_symbol_spec
from env.preprocessor import load_symbol_files
from env.trading_env import TradingEnv

log = logging.getLogger(__name__)

def _get_action_masks(env):
    return env.action_masks()

def collect_expert_transitions(expert, vec_env, num_episodes: int) -> Transitions:
    """Collects expert transitions directly from the underlying environment."""
    env = vec_env.envs[0]
    inner_env = env.env if hasattr(env, "env") else env
    
    all_obs = []
    all_acts = []
    all_infos = []

    log.info("Collecting expert data for %d episodes...", num_episodes)
    for ep in range(num_episodes):
        obs = vec_env.reset()
        expert.reset()
        
        done = False
        while not done:
            action = expert.act(inner_env)
            
            all_obs.append(obs[0])
            all_acts.append(action)
            all_infos.append({})
            
            obs, reward, done_array, infos = vec_env.step([action])
            done = done_array[0]
            
    log.info("Collected %d transition samples.", len(all_obs))
            
    return Transitions(
        obs=np.array(all_obs),
        acts=np.array(all_acts),
        infos=np.array(all_infos),
        next_obs=np.zeros_like(all_obs),  # Not strictly needed for BC
        dones=np.zeros(len(all_obs), dtype=bool)
    )

def pretrain(
    expert_name: str,
    data_dir: str,
    symbols: Optional[list[str]],
    config_path: str,
    output_path: str,
    epochs: int,
    seed: int = 42,
):
    config = load_config(config_path)
    env_cfg = config["environment"]
    agent_cfg = config["agent"]
    reward_cfg = config.get("reward", {})
    obs_cfg = config.get("observation", {})
    symbols_cfg = config.get("symbols_config", "config/symbols.yaml")

    if expert_name == "martingale":
        from strategies.martingale_strategy import MartingaleBaseline
        expert = MartingaleBaseline()
    elif expert_name == "ma_cross":
        from strategies.ma_cross_strategy import MACrossStrategy
        expert = MACrossStrategy()
    else:
        raise ValueError(f"Unknown expert: {expert_name}")

    log.info("Loading symbol data from %s (symbols=%s)", data_dir, symbols or "all")
    symbol_data = load_symbol_files(data_dir, obs_cfg=obs_cfg, symbols=symbols)
    if not symbol_data:
        raise ValueError(f"No data loaded from {data_dir}")

    # Use first symbol for pretraining env
    symbol = list(symbol_data.keys())[0]
    payload = symbol_data[symbol]
    spec = load_symbol_spec(symbols_cfg, symbol)

    def _make_env():
        env = TradingEnv(
            obs_arrays             = payload["obs_arrays"],
            raw_close              = payload["raw_close"],
            symbol_spec            = spec,
            obs_config             = obs_cfg,
            initial_balance        = env_cfg["initial_balance"],
            lot_tiers              = env_cfg.get("lot_tiers", [0.1, 0.2, 0.5]),
            slippage_prob          = env_cfg["slippage_prob"],
            slippage_range         = tuple(env_cfg["slippage_range"]),
            holding_cost_per_lot   = reward_cfg.get("holding_cost_per_lot", 0.0001),
            flat_penalty_per_step  = reward_cfg.get("flat_penalty_per_step", 0.0),
            spread_cost_scale      = reward_cfg.get("spread_cost_scale", 2.0),
            reward_mode            = reward_cfg.get("reward_mode", "sparse"),
            portfolio_offset_factor = reward_cfg.get("portfolio_offset_factor", 0.0),
            volatility_penalty_multiplier = reward_cfg.get("volatility_penalty_multiplier", 0.0),
            drawdown_penalty       = reward_cfg.get("drawdown_penalty", 5.0),
            max_drawdown_pct       = env_cfg.get("max_drawdown_pct", 0.5),
            episode_length         = env_cfg.get("episode_length"),
            random_start           = True,
            max_positions          = env_cfg.get("max_positions"),
        )
        env = ActionMasker(env, _get_action_masks)
        env.reset(seed=seed)
        return env

    vec_env = DummyVecEnv([_make_env])

    # 1. Collect trajectories using expert
    transitions = collect_expert_transitions(expert, vec_env, num_episodes=10)

    # 2. Build MaskablePPO model
    log.info("Building new MaskablePPO model for pretraining.")
    policy_kwargs = {}
    if "policy_kwargs" in agent_cfg:
        policy_kwargs["net_arch"] = agent_cfg["policy_kwargs"]["net_arch"]

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
        device              = "cpu",
    )

    # 3. Use BC to train policy network
    log.info("Starting Behavioral Cloning training with %d epochs...", epochs)
    from imitation.util import logger as imit_logger
    custom_logger = imit_logger.configure(str(Path(output_path).parent / "bc_log"), ["stdout"])
    rng = np.random.default_rng(seed)
    
    bc_trainer = bc.BC(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        demonstrations=transitions,
        policy=model.policy,
        custom_logger=custom_logger,
        batch_size=agent_cfg["batch_size"],
        rng=rng,
    )
    
    bc_trainer.train(n_epochs=epochs)

    # 4. Save the model
    log.info("BC training complete. Saving model to %s", output_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    vec_env.close()
