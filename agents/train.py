"""
Training script for forex RL agent using Stable-Baselines3 PPO.

Loads configuration from YAML, prepares data, and trains the agent
with TensorBoard logging and model checkpointing.
"""

import argparse
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
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

from env.preprocessor import FeatureProcessor
from env.trading_env import TradingEnv


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading-specific metrics during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_trades: list[int] = []
    
    def _on_step(self) -> bool:
        """Called at each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (before update)."""
        # Log rollout statistics
        if len(self.episode_rewards) > 0:
            self.logger.record("trading/avg_episode_reward", np.mean(self.episode_rewards))
            self.logger.record("trading/avg_episode_length", np.mean(self.episode_lengths))
            self.logger.record("trading/avg_trades_per_episode", np.mean(self.episode_trades))
        
        # Reset for next rollout
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trades = []
    
    def _on_episode_end(self) -> None:
        """Called at the end of each episode."""
        # Extract episode info from environment
        if hasattr(self.training_env, "envs") and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            if hasattr(env, "get_episode_stats"):
                stats = env.get_episode_stats()
                self.episode_rewards.append(stats.get("total_pnl", 0))
                self.episode_lengths.append(self.num_timesteps)
                self.episode_trades.append(stats.get("total_trades", 0))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> np.ndarray:
    """
    Load OHLCV data from numpy file.
    
    Args:
        data_path: Path to data file (.npy or .npz)
        
    Returns:
        OHLCV data array
    """
    path = Path(data_path)
    
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        return data["data"] if "data" in data else data[list(data.files)[0]]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def create_env(
    data: np.ndarray,
    config: dict,
    seed: Optional[int] = None,
) -> TradingEnv:
    """
    Create trading environment from configuration.
    
    Args:
        data: OHLCV data array
        config: Environment configuration
        seed: Random seed
        
    Returns:
        TradingEnv instance
    """
    env_config = config["environment"]
    
    return TradingEnv(
        data=data,
        window_size=env_config["window_size"],
        initial_balance=env_config["initial_balance"],
        spread=env_config["spread"],
        slippage_prob=env_config["slippage_prob"],
        slippage_range=tuple(env_config["slippage_range"]),
        render_mode=None,
    )


def build_ppo_model(
    env,
    config: dict,
    log_dir: str,
    seed: Optional[int] = None,
) -> PPO:
    """
    Build PPO model from configuration.
    
    Args:
        env: Trading environment
        config: Full configuration dict
        log_dir: Directory for TensorBoard logs
        seed: Random seed
        
    Returns:
        PPO model
    """
    agent_config = config["agent"]
    training_config = config["training"]
    
    # Build policy kwargs
    policy_kwargs = {}
    if "policy_kwargs" in agent_config:
        policy_kwargs["net_arch"] = agent_config["policy_kwargs"]["net_arch"]
    
    # Create model
    model = PPO(
        policy=agent_config["policy"],
        env=env,
        learning_rate=agent_config["learning_rate"],
        n_steps=agent_config["n_steps"],
        batch_size=agent_config["batch_size"],
        n_epochs=agent_config["n_epochs"],
        gamma=agent_config["gamma"],
        gae_lambda=agent_config["gae_lambda"],
        clip_range=agent_config["clip_range"],
        ent_coef=agent_config["ent_coef"],
        vf_coef=agent_config["vf_coef"],
        max_grad_norm=agent_config["max_grad_norm"],
        clip_range_vf=agent_config.get("clip_range_vf"),
        normalize_advantage=agent_config.get("normalize_advantage", True),
        target_kl=agent_config.get("target_kl"),
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=training_config["verbose"],
        tensorboard_log=log_dir,
    )
    
    # Configure logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    return model


def train(
    config_path: str = "config/config.yaml",
    data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    output_dir: str = "models",
    total_timesteps: Optional[int] = None,
    seed: int = 42,
) -> PPO:
    """
    Main training function.
    
    Args:
        config_path: Path to configuration YAML
        data_path: Path to OHLCV data file
        model_path: Path to existing model to continue training
        output_dir: Directory to save trained model
        total_timesteps: Override total timesteps from config
        seed: Random seed
        
    Returns:
        Trained PPO model
    """
    # Load configuration
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Set defaults
    if data_path is None:
        data_path = config["data"]["raw_data_dir"]
    
    training_config = config["training"]
    if total_timesteps is None:
        total_timesteps = training_config["total_timesteps"]
    
    # Set random seeds
    np.random.seed(seed)
    
    # Load data
    print(f"Loading data from {data_path}")
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    preproc_config = config["preprocessing"]
    processor = FeatureProcessor(
        price_method=preproc_config["price_method"],
        volume_method=preproc_config["volume_method"],
        add_technical_indicators=preproc_config["add_technical_indicators"],
    )
    processed_data = processor.fit_transform(data)
    print(f"Processed data shape: {processed_data.shape}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(training_config["log_dir"]) / f"ppo_{timestamp}"
    model_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = create_env(processed_data, config, seed=seed)
    env = DummyVecEnv([lambda: env])
    
    # Optionally wrap with VecNormalize for reward normalization
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Build or load model
    if model_path is not None:
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("Building new PPO model...")
        model = build_ppo_model(env, config, str(log_dir), seed=seed)
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["save_freq"],
        save_path=str(model_dir),
        name_prefix="ppo_trading",
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Custom trading callback
    trading_callback = TradingCallback(verbose=1)
    callbacks.append(trading_callback)
    
    # Eval callback (optional)
    if training_config.get("eval_freq", 0) > 0:
        eval_env = create_env(processed_data, config, seed=seed + 1000)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=training_config["eval_freq"],
            n_eval_episodes=training_config["n_eval_episodes"],
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)
    
    # Train model
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Checkpoints will be saved to: {model_dir}")
    print("-" * 50)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="PPO",
    )
    
    # Save final model
    final_model_path = model_dir / "ppo_trading_final.zip"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save preprocessor for inference
    preprocessor_path = model_dir / "preprocessor.npy"
    np.save(preprocessor_path, {
        "price_reference": processor.price_scaler.reference_price,
        "price_mean": processor.price_scaler.mean,
        "price_std": processor.price_scaler.std,
        "volume_min": processor.volume_scaler.min_vol,
        "volume_max": processor.volume_scaler.max_vol,
        "volume_mean": processor.volume_scaler.mean_vol,
        "volume_std": processor.volume_scaler.std_vol,
    })
    print(f"Preprocessor saved to: {preprocessor_path}")
    
    return model


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train forex RL agent with PPO")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to OHLCV data file (overrides config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to existing model for continued training",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_path=args.data,
        model_path=args.model,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
