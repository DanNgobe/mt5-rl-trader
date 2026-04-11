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
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.agent import BaseAgent

if TYPE_CHECKING:
    from env.trading_env import TradingEnv

log = logging.getLogger(__name__)


class PPOAgent(BaseAgent):
    """
    RecurrentPPO model wrapped as a BaseAgent.

    Parameters
    ----------
    model_path : str or None
        Path to a saved .zip model.  If None the model must be set
        manually via agent.model before calling act().
    """

    name = "ppo"

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[RecurrentPPO] = None
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)
        if model_path is not None:
            self.load(model_path)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def act(self, env: "TradingEnv") -> int:
        if self.model is None:
            raise RuntimeError("PPOAgent has no model loaded. Call load() first.")
        obs = env._observation()
        
        action, self.lstm_states = self.model.predict(
            obs[np.newaxis, :],
            state=self.lstm_states,
            episode_start=self.episode_starts,
            deterministic=True,
        )
        self.episode_starts = np.zeros((1,), dtype=bool)
        return int(action[0])

    def load(self, path: str) -> None:
        log.info("Loading RecurrentPPO model from %s", path)
        self.model = RecurrentPPO.load(path, device="cpu")
        self.name  = f"ppo:{path}"

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        self.model.save(path)
        log.info("RecurrentPPO model saved → %s", path)

    # ------------------------------------------------------------------
    # Training helper — attach a vec env before training
    # ------------------------------------------------------------------

    def set_env(self, vec_env: DummyVecEnv) -> None:
        """Attach (or re-attach) a vec env to the underlying SB3 model."""
        if self.model is not None:
            self.model.set_env(vec_env)
