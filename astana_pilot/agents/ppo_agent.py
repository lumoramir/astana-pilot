"""Thin wrapper around Stable-Baselines3 PPO."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .config import PPOConfig


class PPOAgent:
    """PPO agent with a small, explicit public interface."""

    def __init__(self, model: PPO) -> None:
        self.model = model

    @classmethod
    def create(
        cls,
        env: VecEnv,
        config: Optional[PPOConfig] = None,
        tensorboard_log: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        device: str = "auto",
    ) -> "PPOAgent":
        """Build a fresh agent for ``env``."""
        cfg = config or PPOConfig()
        model = PPO(
            "MlpPolicy", env, seed=seed, device=device, verbose=1,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
            **cfg.to_kwargs(),
        )
        return cls(model)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "PPOAgent":
        """Load a saved agent (``.zip``)."""
        return cls(PPO.load(str(path), device=device))

    def attach_env(self, env: VecEnv) -> None:
        """Re-attach an environment to a loaded model (to continue training)."""
        self.model.set_env(env)

    def learn(self, total_timesteps: int, callback=None) -> None:
        """Train the policy."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback,
                         progress_bar=False, reset_num_timesteps=False)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action for a single observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: Union[str, Path]) -> None:
        """Save the agent to ``path``."""
        self.model.save(str(path))
