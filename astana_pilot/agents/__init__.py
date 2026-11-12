"""RL agent (Stable-Baselines3 PPO). Imports are lazy so that torch-free helpers (``agents.layout``,
``agents.config``) can be used and tested without torch / stable-baselines3 installed."""
from __future__ import annotations

__all__ = ["PPOAgent", "PPOConfig"]


def __getattr__(name):
    if name == "PPOConfig":
        from .config import PPOConfig
        return PPOConfig
    if name == "PPOAgent":
        from .ppo_agent import PPOAgent
        return PPOAgent
    raise AttributeError(name)
