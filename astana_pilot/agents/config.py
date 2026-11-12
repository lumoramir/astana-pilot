"""PPO hyper-parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List


class LinearSchedule:
    """Picklable linear learning-rate decay: ``lr * (final_frac + (1 - final_frac) * progress_remaining)``."""

    def __init__(self, lr: float, final_frac: float = 0.1) -> None:
        self.lr, self.final_frac = lr, final_frac

    def __call__(self, progress_remaining: float) -> float:
        return self.lr * (self.final_frac + (1.0 - self.final_frac) * progress_remaining)


def fit_batch_size(rollout: int, target: int) -> int:
    """Largest power-of-two fraction of ``target`` that divides the rollout size (``n_envs * n_steps``)."""
    b = target
    while b > 64 and rollout % b:
        b //= 2
    return b if rollout % b == 0 else rollout


@dataclass(frozen=True)
class PPOConfig:
    """Hyper-parameters passed to ``stable_baselines3.PPO``.

    The observation has 351 numbers (48 rays x 4 channels + ego / route / rules / tracked objects), so the policy is
    ``CityExtractor`` (circular conv over the rays + MLP over the rest, 384 features) followed by [256, 256] heads.
    """

    learning_rate: float = 3e-4
    lr_final_frac: float = 0.1       # lr decays linearly to 10 % -> fine, smooth control at the end
    n_steps: int = 1024              # per environment, per rollout
    batch_size: int = 4096           # big minibatches: stable gradients for a 351-dim heterogeneous input
    n_epochs: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0            # gSDE already explores; entropy bonus would only add jitter
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # stop an update early if the policy moved too far (protects the BC start)
    # gSDE (Raffin et al. 2021): exploration noise is a state-dependent function resampled every
    # `sde_sample_freq` steps -> temporally correlated -> no high-frequency steering jitter.
    use_sde: bool = True
    sde_sample_freq: int = 4
    log_std_init: float = -2.0       # initial exploration noise ~0.14 of the action range (small: start from BC)
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    features_dim: int = 384
    n_rays: int = 48

    @classmethod
    def for_device(cls, device: str) -> "PPOConfig":
        """Preset for the device (CPU: smaller net and minibatches are faster)."""
        if device.startswith("cuda"):
            return cls()
        return cls(batch_size=1024, net_arch=[128, 128], features_dim=192, n_epochs=6)

    def finetune(self) -> "PPOConfig":
        """Gentler settings for PPO that starts from a behaviour-cloned policy."""
        # log_std -4: measured on a cloned policy - exploration noise of std 0.135 (log_std -2) crashes it within ~20 steps
        # (PPO then only ever sees crashes and destroys the policy); std 0.018 keeps it driving ~300+ steps.
        return replace(self, learning_rate=3e-5, n_epochs=4, clip_range=0.12, target_kl=0.02, log_std_init=-4.0)

    def to_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments for the SB3 PPO constructor."""
        from astana_pilot.agents.networks import CityExtractor

        kw = asdict(self)
        net_arch = kw.pop("net_arch")
        features_dim, n_rays, log_std = kw.pop("features_dim"), kw.pop("n_rays"), kw.pop("log_std_init")
        kw["learning_rate"] = LinearSchedule(kw.pop("learning_rate"), kw.pop("lr_final_frac"))
        kw["policy_kwargs"] = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            features_extractor_class=CityExtractor,
            features_extractor_kwargs=dict(n_rays=n_rays, features_dim=features_dim),
            log_std_init=log_std,
        )
        return kw
