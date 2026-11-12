"""Behaviour cloning of the expert into the PPO policy (+ critic warm-up). Needs torch.

1. ``fit_policy``: regress the policy mean onto the expert action (MSE; steering and pedal weighted equally).
2. ``fit_critic``: regress the value head onto discounted returns of the demonstrations, with rewards scaled
   exactly like ``VecNormalize`` will scale them (running std of the discounted return). Without this the
   random critic would produce garbage advantages in the first PPO updates and wreck the cloned policy.
   Returns ``(mean, var, count)`` of the return statistics so that training can start with the same scaling.
3. ``dagger``: optional rounds - roll out the *learner*, label its states with the expert, retrain.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch as th

from astana_pilot.training.returns import GAMMA, normalized_returns  # noqa: F401  (re-exported)


def policy_mean(policy, obs: th.Tensor) -> th.Tensor:
    """Differentiable mean action of an SB3 ``ActorCriticPolicy`` (works with and without gSDE)."""
    dist = policy.get_distribution(obs)
    return dist.distribution.mean


def _batches(n: int, bs: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def fit_policy(policy, obs: np.ndarray, act: np.ndarray, device: str, epochs: int = 15, batch_size: int = 1024,
               lr: float = 3e-4, val_frac: float = 0.05, seed: int = 0, log: Callable[[str], None] = print) -> Dict:
    """Behaviour cloning of the policy mean. Returns the final train / validation MSE."""
    rng = np.random.default_rng(seed)
    n = len(act)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    va, tr = perm[:n_val], perm[n_val:]
    X = th.as_tensor(obs, device="cpu")
    Y = th.as_tensor(act, device="cpu")
    params = list(policy.features_extractor.parameters()) + list(policy.mlp_extractor.policy_net.parameters()) \
        + list(policy.action_net.parameters())
    opt = th.optim.Adam(params, lr=lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * (len(tr) // batch_size + 1)), eta_min=lr * 0.05)
    policy.set_training_mode(True)
    last = {}
    for ep in range(epochs):
        t0, tot, cnt = time.time(), 0.0, 0
        for b in _batches(len(tr), batch_size, rng):
            ib = tr[b]
            xb = X[ib].float().to(device)
            yb = Y[ib].to(device)
            loss = th.nn.functional.mse_loss(policy_mean(policy, xb), yb)
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot, cnt = tot + float(loss.detach()) * len(ib), cnt + len(ib)
        with th.no_grad():
            xv, yv = X[va].float().to(device), Y[va].to(device)
            pv = policy_mean(policy, xv)
            vmse = float(th.nn.functional.mse_loss(pv, yv))
            per = ((pv - yv) ** 2).mean(0).cpu().numpy()
        last = {"train_mse": tot / max(cnt, 1), "val_mse": vmse, "val_mse_steer": float(per[0]),
                "val_mse_pedal": float(per[1])}
        log(f"  [bc] epoch {ep + 1}/{epochs}  train {last['train_mse']:.4f}  val {vmse:.4f} "
            f"(steer {per[0]:.4f}, pedal {per[1]:.4f})  {time.time() - t0:.0f}s")
    return last


def fit_critic(policy, obs: np.ndarray, targets: np.ndarray, device: str, epochs: int = 8, batch_size: int = 1024,
               lr: float = 3e-4, seed: int = 0, log: Callable[[str], None] = print) -> float:
    """Regress the value head onto ``targets`` (the feature extractor is frozen so the cloned policy is untouched)."""
    rng = np.random.default_rng(seed)
    X, Y = th.as_tensor(obs), th.as_tensor(targets)
    params = list(policy.mlp_extractor.value_net.parameters()) + list(policy.value_net.parameters())
    opt = th.optim.Adam(params, lr=lr)
    last = 0.0
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for b in _batches(len(Y), batch_size, rng):
            xb, yb = X[b].float().to(device), Y[b].to(device)
            with th.no_grad():
                feats = policy.extract_features(xb)
            v = policy.value_net(policy.mlp_extractor.forward_critic(feats)).flatten()
            loss = th.nn.functional.mse_loss(v, yb)
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot, cnt = tot + float(loss.detach()) * len(b), cnt + len(b)
        last = tot / max(cnt, 1)
        log(f"  [critic] epoch {ep + 1}/{epochs}  mse {last:.4f}")
    return last


def warm_start(policy, demos: Dict[str, np.ndarray], device: str, bc_epochs: int = 15, critic_epochs: int = 8,
               log: Callable[[str], None] = print) -> Tuple[float, float, float]:
    """BC + critic warm-up on ``demos``. Returns the return statistics for ``VecNormalize.ret_rms``."""
    obs = demos["obs"].astype(np.float32)
    log(f"[bc] cloning {len(obs):,} samples (obs {obs.shape[1]}) on {device}")
    fit_policy(policy, obs, demos["act"], device, epochs=bc_epochs, log=log)
    tgt, stats = normalized_returns(demos["rew"], demos["term"], demos["ep_id"])
    fit_critic(policy, obs, tgt, device, epochs=critic_epochs, log=log)
    policy.set_training_mode(False)
    return stats
