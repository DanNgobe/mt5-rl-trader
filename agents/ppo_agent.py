"""
agents/ppo_agent.py
-------------------
PPOAgent wraps a MaskablePPO model behind the BaseAgent interface so it
can be dropped into the shared Evaluator loop alongside hand-coded strategies.

act(env) builds the observation from the live env, fetches the current
action mask, calls model.predict() with the mask, and returns the action.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from core.agent import BaseAgent

if TYPE_CHECKING:
    from env.trading_env import TradingEnv

log = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    MaskablePPO model wrapped as a BaseAgent.

    Parameters
    ----------
    model_path : str or None
        Path to a saved .zip model.  If None the model must be set
        manually via agent.model before calling act().
    """

    name = "ppo"

    def __init__(self, model_path: Optional[str] = None):
        self.model:    Optional[MaskablePPO] = None
        self.norm_env: Optional[VecNormalize] = None
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, env: "TradingEnv") -> int:
        if self.model is None:
            raise RuntimeError("PPOAgent has no model loaded. Call load() first.")
        
        obs          = env._observation()
        action_masks = env.action_masks()

        # Apply normalization if stats were loaded
        if self.norm_env is not None:
            obs = self.norm_env.normalize_obs(obs)

        action, _    = self.model.predict(
            obs[np.newaxis, :],
            action_masks=action_masks[np.newaxis, :],
            deterministic=True,
        )
        return int(action[0])

    def load(self, path: str) -> None:
        from pathlib import Path
        p = Path(path)
        log.info("Loading MaskablePPO model from %s", path)
        self.model = MaskablePPO.load(path, device="cpu")
        self.name  = f"ppo:{path}"

        # Try to find VecNormalize stats
        # Check parent dir for vec_normalize.pkl or vec_normalize_best.pkl
        norm_files = ["vec_normalize.pkl", "vec_normalize_best.pkl"]
        for f in norm_files:
            norm_path = p.parent / f
            if norm_path.exists():
                log.info("Loading VecNormalize stats from %s", norm_path)
                # We need a dummy env to load VecNormalize. lambda: None is no longer accepted.
                import gymnasium as gym
                class MinimalEnv(gym.Env):
                    def __init__(self, obs_space, act_space):
                        self.observation_space = obs_space
                        self.action_space = act_space
                
                dummy_venv = DummyVecEnv([lambda: MinimalEnv(self.model.observation_space, self.model.action_space)])
                self.norm_env = VecNormalize.load(str(norm_path), venv=dummy_venv)
                self.norm_env.training = False
                self.norm_env.norm_reward = False
                break

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        log.info("MaskablePPO model saved → %s", path)

    # ------------------------------------------------------------------
    # Training helper — attach a vec env before training
    # ------------------------------------------------------------------

    def set_env(self, vec_env: DummyVecEnv) -> None:
        """Attach (or re-attach) a vec env to the underlying SB3 model."""
        if self.model is not None:
            self.model.set_env(vec_env)
