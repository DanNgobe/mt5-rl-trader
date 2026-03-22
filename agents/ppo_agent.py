"""
agents/ppo_agent.py
-------------------
PPOAgent wraps a Stable-Baselines3 PPO model behind the BaseAgent
interface so it can be dropped into the shared Evaluator loop alongside
hand-coded strategies.

act(env) builds the observation from the live env (identical to what
the env returns from step()), calls model.predict(), and returns the
action array.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.agent import BaseAgent

if TYPE_CHECKING:
    from env.trading_env import TradingEnv

log = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    Stable-Baselines3 PPO model wrapped as a BaseAgent.

    Parameters
    ----------
    model_path : str or None
        Path to a saved .zip model.  If None the model must be set
        manually via agent.model before calling act().
    """

    name = "ppo"

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[PPO] = None
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, env: "TradingEnv") -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PPOAgent has no model loaded. Call load() first.")
        obs   = env._observation()
        action, _ = self.model.predict(obs[np.newaxis, :], deterministic=True)
        return np.asarray(action[0], dtype=np.int32)

    def load(self, path: str) -> None:
        log.info("Loading PPO model from %s", path)
        self.model = PPO.load(path, device="cpu")
        self.name  = f"ppo:{path}"

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        log.info("PPO model saved → %s", path)

    # ------------------------------------------------------------------
    # Training helper — attach a vec env before training
    # ------------------------------------------------------------------

    def set_env(self, vec_env: DummyVecEnv) -> None:
        """Attach (or re-attach) a vec env to the underlying SB3 model."""
        if self.model is not None:
            self.model.set_env(vec_env)
