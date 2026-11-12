"""SB3 callbacks: TensorBoard metrics, adaptive curriculum, resumable autosave, wall-clock budget."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from astana_pilot.training.metrics import COLLISION_KINDS, VIOLATION_KINDS, EpisodeAggregator


class EpisodeStatsCallback(BaseCallback):
    """Rolling per-episode metrics -> TensorBoard (``custom/...``).

    success, collision_rate and collision/<kind> (wall / parked / car / bus / pedestrian), stall / lost / timeout,
    viol/<kind> and viol_any, aeb_frac, unneeded_slow_frac, speeding_frac, wrong_side_frac, speed_kmh,
    steer_jerk, lane_dev_m, progress and the mean reward components (rew/<part>).
    """

    def __init__(self, window: int = 300, log_every: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.agg = EpisodeAggregator(window)
        self.log_every = log_every
        self.total_episodes = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "ep_steps" in info:
                self.agg.add(info)
                self.total_episodes += 1
        if len(self.agg) >= 20 and self.n_calls % self.log_every == 0:
            for k, v in self.agg.summary().items():
                if k != "episodes":
                    self.logger.record(f"custom/{k}", v)
            self.logger.record("custom/episodes_total", self.total_episodes)
        return True

    def recent(self) -> dict:
        return self.agg.summary()


class CriticWarmupCallback(BaseCallback):
    """Freeze the cloned policy for the first ``rollouts`` PPO iterations so that only the critic learns.

    Measured failure without it: the critic was fitted on expert demonstrations, so on the learner's own (worse) states
    its advantages are noise, and the first PPO updates (KL 0.3, 44 % clipped) destroyed the cloned policy within ~30k steps
    (success 0.18 -> 0.00).  With the actor frozen the critic first learns the value of the policy that actually drives.
    """

    def __init__(self, rollouts: int = 6, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.rollouts, self.seen, self._frozen = rollouts, 0, []

    def _actor_params(self):
        p = self.model.policy
        return list(p.features_extractor.parameters()) + list(p.mlp_extractor.policy_net.parameters()) \
            + list(p.action_net.parameters()) + [p.log_std]

    def _on_training_start(self) -> None:
        if self.rollouts > 0 and self.model.num_timesteps < 1000:          # a fresh run only (not after a resume)
            self._frozen = self._actor_params()
            for q in self._frozen:
                q.requires_grad_(False)
            print(f"[warmup] policy frozen for {self.rollouts} PPO iterations: the critic adapts first", flush=True)

    def _on_rollout_end(self) -> None:
        if self._frozen:
            self.seen += 1
            if self.seen >= self.rollouts:
                for q in self._frozen:
                    q.requires_grad_(True)
                self._frozen = []
                print("[warmup] policy unfrozen: PPO now improves the driver", flush=True)

    def _on_step(self) -> bool:
        return True


class _LRBox:
    """Picklable learning-rate 'schedule' whose value is steered by :class:`StabilityCallback`."""

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def __call__(self, progress_remaining: float) -> float:
        return self.lr


class StabilityCallback(BaseCallback):
    """Keeps PPO from destroying a good policy (measured: success 0.78 -> 0.00 within two updates, KL 0.25).

    * KL-adaptive learning rate: after every update, lr is divided by 1.5 if the policy moved more than 2x ``target_kl``
      and multiplied by 1.5 if it moved less than half of it. With the tiny exploration noise of a cloned policy a fixed
      lr of 1e-4 is far too big; this finds the right size on its own.
    * Best-policy memory and roll-back: the weights with the best (difficulty-weighted) rolling success are kept
      (``best_model.zip``); if success collapses below ``drop`` x the best, the best weights come back, the optimiser state
      is cleared and lr is halved.
    """

    def __init__(self, stats: "EpisodeStatsCallback", curriculum: Optional["AdaptiveCurriculumCallback"], run_dir: Path,
                 lr0: float = 3e-5, lr_min: float = 2e-6, lr_max: float = 1e-4, target_kl: float = 0.01,
                 min_episodes: int = 300, drop: float = 0.4, min_best: float = 0.25, save_every: int = 30_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stats, self.cur, self.run_dir = stats, curriculum, Path(run_dir)
        self.lr0, self.lr_min, self.lr_max, self.target_kl = lr0, lr_min, lr_max, target_kl
        self.min_episodes, self.drop, self.min_best, self.save_every = min_episodes, drop, min_best, save_every
        self.box = _LRBox(lr0)
        self.best_score, self.best_success, self.best_state, self._best_step = -1.0, 0.0, None, 0
        self._ep_gate = 0                                    # no judgement before this many episodes (after a roll-back)
        self.rollbacks = 0

    def _on_training_start(self) -> None:
        self.model.lr_schedule = self.box
        self._ep_gate = self.stats.total_episodes + self.min_episodes

    def _unfrozen(self) -> bool:
        return bool(self.model.policy.log_std.requires_grad)

    def _on_rollout_start(self) -> None:
        if not self._unfrozen():
            return
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is not None and kl > 0.0:
            if kl > 2.0 * self.target_kl:
                self.box.lr = max(self.lr_min, self.box.lr / 1.5)
            elif kl < 0.5 * self.target_kl:
                self.box.lr = min(self.lr_max, self.box.lr * 1.5)
        self.logger.record("custom/lr", self.box.lr)
        if self.stats.total_episodes < self._ep_gate or len(self.stats.agg) < self.min_episodes:
            return
        s = self.stats.recent().get("success")
        if s is None:
            return
        diff = self.cur.level if self.cur else 1.0
        score = s * (0.4 + 0.6 * diff)
        if (self.best_state is not None and self.best_success >= self.min_best and s < self.drop * self.best_success):
            import torch as th
            self.model.policy.load_state_dict({k: v.to(self.model.device) for k, v in self.best_state.items()})
            self.model.policy.optimizer.state.clear()
            self.box.lr = max(self.lr_min, self.box.lr * 0.5)
            self._ep_gate = self.stats.total_episodes + self.min_episodes
            self.rollbacks += 1
            print(f"[stability] success fell to {s:.2f} (best {self.best_success:.2f}): best weights restored, "
                  f"lr -> {self.box.lr:.1e}  (roll-back #{self.rollbacks})", flush=True)
            del th
            return
        if score > self.best_score + 0.01 and self.num_timesteps - self._best_step >= self.save_every:
            self.best_score, self.best_success, self._best_step = score, s, self.num_timesteps
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()}
            try:
                self.model.save(str(self.run_dir / "best_model"))
            except Exception:
                pass
            self.logger.record("custom/best_score", self.best_score)

    def _on_step(self) -> bool:
        return True


class AdaptiveCurriculumCallback(BaseCallback):
    """Difficulty follows skill: +``up`` when the recent success rate >= ``up_thr``, -``down`` when < ``down_thr``.

    Checked every ``check_every`` env steps with at least ``min_episodes`` new episodes since the last change.
    Safety net: after ``force_full_after`` steps the difficulty is at least ``start + ramp`` so the agent always
    reaches full traffic. State is saved in ``state_path`` (restored on resume).
    """

    def __init__(self, stats: EpisodeStatsCallback, state_path: Optional[Path] = None, start: float = 0.2,
                 up: float = 0.1, down: float = 0.05, up_thr: float = 0.80, down_thr: float = 0.45,
                 check_every: int = 50_000, min_episodes: int = 120, force_full_after: int = 5_000_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stats, self.state_path = stats, state_path
        self.level, self.start, self.up, self.down = start, start, up, down
        self.up_thr, self.down_thr = up_thr, down_thr
        self.check_every, self.min_episodes, self.force_full_after = check_every, min_episodes, force_full_after
        self._last_check, self._last_eps = 0, 0
        if state_path and Path(state_path).exists():
            try:
                self.level = float(json.loads(Path(state_path).read_text())["difficulty"])
            except Exception:
                pass

    def _apply(self) -> None:
        self.training_env.env_method("set_difficulty", self.level)
        self.logger.record("custom/difficulty", self.level)
        if self.state_path:
            Path(self.state_path).write_text(json.dumps({"difficulty": self.level}))

    def _on_training_start(self) -> None:
        self._last_check = self.num_timesteps
        self._last_eps = self.stats.total_episodes
        self._apply()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_every:
            return True
        self._last_check = self.num_timesteps
        new_eps = self.stats.total_episodes - self._last_eps
        floor = min(1.0, self.start + (1.0 - self.start) * self.num_timesteps / max(self.force_full_after, 1))
        s = self.stats.recent().get("success")
        new = self.level
        if s is not None and new_eps >= self.min_episodes:
            if s >= self.up_thr:
                new = min(1.0, self.level + self.up)
            elif s < self.down_thr:
                new = max(self.start, self.level - self.down)
            self._last_eps = self.stats.total_episodes
        new = max(new, floor)
        if abs(new - self.level) > 1e-9:
            self.level = new
            self._apply()
        return True


class AutosaveCallback(BaseCallback):
    """Resumable state: ``latest.zip`` + ``vecnormalize_latest.pkl`` (+ numbered checkpoints) every ``every_minutes``.

    Files are written to a temporary name and renamed, so a power cut never leaves a half-written checkpoint.
    """

    def __init__(self, run_dir: Path, every_minutes: float = 15.0, numbered_every: int = 1_000_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir, self.every = Path(run_dir), every_minutes * 60.0
        self.numbered_every = numbered_every
        self._t = time.time()
        self._next_numbered = 0

    def _save(self, numbered: bool = False) -> None:
        ck = self.run_dir / "checkpoints"
        ck.mkdir(parents=True, exist_ok=True)
        tmp = ck / "tmp_latest.zip"
        self.model.save(str(tmp))
        os.replace(tmp, ck / "latest.zip")
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(ck / "tmp_vn.pkl"))
            os.replace(ck / "tmp_vn.pkl", ck / "vecnormalize_latest.pkl")
        if numbered:
            self.model.save(str(ck / f"ppo_{self.num_timesteps}_steps"))

    def _on_training_start(self) -> None:
        self._next_numbered = (self.num_timesteps // self.numbered_every + 1) * self.numbered_every

    def _on_step(self) -> bool:
        numbered = self.num_timesteps >= self._next_numbered
        if numbered or time.time() - self._t >= self.every:
            self._save(numbered)
            self._t = time.time()
            if numbered:
                self._next_numbered += self.numbered_every
        return True

    def _on_training_end(self) -> None:
        self._save()


class TimeBudgetCallback(BaseCallback):
    """Stop training cleanly after ``hours`` of wall-clock time (``None`` = no limit)."""

    def __init__(self, hours: Optional[float], verbose: int = 0) -> None:
        super().__init__(verbose)
        self.deadline = time.time() + hours * 3600.0 if hours else None

    def _on_step(self) -> bool:
        if self.deadline and time.time() > self.deadline:
            print("[budget] time budget used up - stopping and saving", flush=True)
            return False
        return True


class ProgressPrintCallback(BaseCallback):
    """One readable status line every ``every_s`` seconds (visible in the console and in ``train.log``)."""

    def __init__(self, stats: EpisodeStatsCallback, curriculum: Optional[AdaptiveCurriculumCallback],
                 every_s: float = 120.0, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.stats, self.cur, self.every_s = stats, curriculum, every_s
        self._t0 = self._t = time.time()
        self._steps0 = 0

    def _on_training_start(self) -> None:
        self._steps0 = self.num_timesteps

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._t < self.every_s:
            return True
        self._t = now
        s = self.stats.recent()
        sps = (self.num_timesteps - self._steps0) / max(now - self._t0, 1e-9)
        cols = ", ".join(f"{k}={s[k]:.2f}" for k in ("success", "collision_rate", "viol_any", "aeb_frac",
                                                      "unneeded_slow_frac") if k in s)
        lv = f" diff={self.cur.level:.2f}" if self.cur else ""
        print(f"[train] {self.num_timesteps:>11,} steps  {sps:,.0f} sps  {(now - self._t0) / 3600:.2f} h{lv}  {cols}",
              flush=True)
        return True
