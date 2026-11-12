"""Factories for parallel (``SubprocVecEnv``) environments."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Sequence

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize

from astana_pilot.env import DrivingEnv, EnvConfig


def make_env_fn(rank: int, seed: int, config: Optional[EnvConfig] = None) -> Callable[[], DrivingEnv]:
    """Return a picklable thunk that builds one seeded environment."""

    def _init() -> DrivingEnv:
        e = DrivingEnv(config)
        e.reset(seed=seed + rank)
        return e

    return _init


def make_vec_env(n_envs: int, seed: int = 0, config: Optional[EnvConfig] = None,
                 maps: Optional[Sequence[str]] = None, difficulty: Optional[float] = None,
                 vecnormalize_path: Optional[str] = None) -> VecEnv:
    """Create ``n_envs`` environments; subprocesses when ``n_envs > 1``.

    ``maps``: one map spec per city (``"synthetic:3"``, a ``.pkl`` ...); worker ``i`` gets ``maps[i % len(maps)]``,
    so one training run sees several different cities (against memorising a single map).
    Wrapped in ``VecMonitor`` and reward-only ``VecNormalize`` (observations are already in [-1, 1]).
    """
    base = config or EnvConfig()
    cfgs = [replace(base, map_path=maps[i % len(maps)]) if maps else base for i in range(n_envs)]
    fns = [make_env_fn(i, seed, cfgs[i]) for i in range(n_envs)]
    vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    if difficulty is not None:
        vec.env_method("set_difficulty", difficulty)
    vec = VecMonitor(vec)
    if vecnormalize_path:                                   # resume: keep the reward scaling the critic was trained with
        vn = VecNormalize.load(vecnormalize_path, vec)
        vn.training, vn.norm_reward = True, True
        return vn
    return VecNormalize(vec, norm_obs=False, norm_reward=True, gamma=0.995)
