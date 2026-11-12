"""Which observation entries are radar rays (torch-free, so it can be unit-tested)."""
from __future__ import annotations

import numpy as np

from astana_pilot.env.config import EnvConfig
from astana_pilot.env.driving_env import obs_layout


def ray_and_rest_indices(n_rays: int, obs_dim: int) -> "tuple[np.ndarray, np.ndarray]":
    """Index arrays: ``(4, n_rays)`` for [distance, closing speed, pedestrian flag, vehicle flag] and the remainder."""
    lay = obs_layout(EnvConfig(n_rays=n_rays))
    rays = np.stack([np.arange(lay[k].start, lay[k].stop) for k in ("lidar", "lidar_v", "lidar_ped", "lidar_veh")])
    mask = np.ones(obs_dim, dtype=bool)
    mask[rays.ravel()] = False
    return rays, np.nonzero(mask)[0]
