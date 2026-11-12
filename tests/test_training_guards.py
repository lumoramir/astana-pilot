"""Regression tests for the failures that wasted a night of training (see docs/TRAINING_NOTES.md)."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _fake_model():
    nn = torch.nn
    pol = SimpleNamespace(features_extractor=nn.Linear(2, 2), mlp_extractor=SimpleNamespace(policy_net=nn.Linear(2, 2), value_net=nn.Linear(2, 2)),
                          action_net=nn.Linear(2, 2), log_std=nn.Parameter(torch.zeros(2)))
    return SimpleNamespace(policy=pol, num_timesteps=0)


def test_critic_warmup_freezes_then_releases_the_actor():
    from astana_pilot.training.callbacks import CriticWarmupCallback
    cb = CriticWarmupCallback(rollouts=2)
    cb.model = _fake_model()
    cb._on_training_start()
    p = cb.model.policy
    assert not p.action_net.weight.requires_grad and not p.log_std.requires_grad
    assert p.mlp_extractor.value_net.weight.requires_grad                  # the critic keeps learning
    cb._on_rollout_end()
    assert not p.action_net.weight.requires_grad
    cb._on_rollout_end()
    assert p.action_net.weight.requires_grad and p.log_std.requires_grad


def test_critic_warmup_is_skipped_on_resume():
    from astana_pilot.training.callbacks import CriticWarmupCallback
    cb = CriticWarmupCallback(rollouts=2)
    cb.model = _fake_model()
    cb.model.num_timesteps = 500_000
    cb._on_training_start()
    assert cb.model.policy.action_net.weight.requires_grad


def test_finetune_keeps_exploration_noise_small():
    from astana_pilot.agents.config import PPOConfig
    assert PPOConfig().finetune().log_std_init <= -3.5


def test_no_vehicle_stays_frozen_at_its_route_end():
    """A car that cannot continue its route must leave (it used to freeze for ever and block the agent)."""
    from astana_pilot.env import DrivingEnv, EnvConfig
    env = DrivingEnv(EnvConfig(map_path="synthetic"))
    env.set_difficulty(1.0)
    env.reset(seed=1)
    w = env.world
    car = next((a for a in w.actors if a.alive and a.path is not None and a.kind == "car"), None)
    assert car is not None
    car.nodes = []                                           # no continuation possible
    car.s, car.v = car.path.length - 1.0, 0.0
    for _ in range(90):                                      # 9 s
        w.step(0.1, env._ego_view())
        if not car.alive:
            break
    assert not car.alive
