"""Policy network for the 351-number observation: a circular 1-D convolution over the radar rays + an MLP over
the rest, so that "something closing in at 30 degrees to the left" is recognised at every angle with shared
weights (the old plain MLP had to learn each of the 48 rays separately).

Used as ``features_extractor_class`` of the SB3 ``MlpPolicy``; the PPO actor / critic heads (``net_arch``) sit on top.
"""
from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from astana_pilot.agents.layout import ray_and_rest_indices


class CityExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, n_rays: int = 48, features_dim: int = 384,
                 ray_channels: int = 32, rest_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        obs_dim = int(observation_space.shape[0])
        ray_idx, rest_idx = ray_and_rest_indices(n_rays, obs_dim)
        self.register_buffer("ray_idx", th.as_tensor(ray_idx, dtype=th.long))
        self.register_buffer("rest_idx", th.as_tensor(rest_idx, dtype=th.long))
        c = ray_channels
        self.rays = nn.Sequential(
            nn.Conv1d(4, c, 5, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(c, 2 * c, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(2 * c, 2 * c, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            flat = self.rays(th.zeros(1, 4, n_rays)).shape[1]
        self.ray_fc = nn.Sequential(nn.Linear(flat, 192), nn.ReLU())
        self.rest = nn.Sequential(nn.Linear(len(rest_idx), rest_dim), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(192 + rest_dim, features_dim), nn.ReLU())

    def forward(self, obs: th.Tensor) -> th.Tensor:
        rays = obs[:, self.ray_idx]                       # (B, 4, n_rays)
        rest = obs[:, self.rest_idx]                      # (B, rest)
        return self.out(th.cat([self.ray_fc(self.rays(rays)), self.rest(rest)], dim=1))
