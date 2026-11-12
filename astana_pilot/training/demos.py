"""Collect (observation, expert action) demonstrations with ``HeuristicPolicy`` - no torch needed.

* ``noise``: Ornstein-Uhlenbeck noise is added to the *executed* action while the stored label stays the clean expert
  action (DART). The demonstrations then also cover the states the learner will drift into.
* ``learner`` + ``beta`` (DAgger): the learner (a saved PPO model) drives with probability ``1 - beta`` per episode
  and the expert only labels - fixes the covariate shift of plain behaviour cloning.
* By default only episodes that the expert completed without collision are kept (``keep_failed=False``).

    python -m astana_pilot.training.demos --episodes 300 --out runs/x/demos.npz --maps synthetic:0,synthetic:1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from astana_pilot.env import DrivingEnv, EnvConfig
from astana_pilot.viz.policies import HeuristicPolicy

_ENVS: Dict[str, Tuple[DrivingEnv, HeuristicPolicy]] = {}
_LEARNER = {"path": None, "policy": None}


def _get_env(map_path: str) -> Tuple[DrivingEnv, HeuristicPolicy]:
    if map_path not in _ENVS:
        cfg = EnvConfig(map_path=map_path)
        _ENVS[map_path] = (DrivingEnv(cfg), HeuristicPolicy(cfg))
    return _ENVS[map_path]


def run_episode(map_path: str, seed: int, difficulty: float, noise: float = 0.0,
                learner_path: Optional[str] = None, beta: float = 1.0) -> Dict[str, np.ndarray]:
    """One episode. Returns arrays ``obs, act, rew, term`` plus ``info`` of the last step."""
    env, expert = _get_env(map_path)
    rng = np.random.default_rng(seed + 7919)
    env.set_difficulty(difficulty)
    learner = None
    if learner_path and rng.random() > beta:
        if _LEARNER["path"] != learner_path:
            from astana_pilot.viz.policies import ModelPolicy
            _LEARNER.update(path=learner_path, policy=ModelPolicy(learner_path))
        learner = _LEARNER["policy"]
    obs, _ = env.reset(seed=seed)
    expert.reset()
    ou = np.zeros(2)
    O, A, R, T = [], [], [], []
    done, info = False, {}
    while not done:
        a_exp = expert.act(obs)
        ou = 0.9 * ou + noise * rng.standard_normal(2) * np.array([1.0, 1.2])
        # DART + DAgger: the executed action (learner's or expert's) is perturbed, the stored label stays the clean expert action,
        # so the clone sees the states it drifts into and learns how the expert would get back to the lane
        base = np.asarray(learner.act(obs), dtype=np.float32) if learner is not None else a_exp
        a_run = np.clip(base + ou, -1.0, 1.0).astype(np.float32)
        O.append(obs.astype(np.float16))
        A.append(a_exp)
        obs, r, term, trunc, info = env.step(a_run)
        R.append(r)
        T.append(term)
        done = term or trunc
    return {"obs": np.asarray(O), "act": np.asarray(A, dtype=np.float32), "rew": np.asarray(R, dtype=np.float32),
            "term": np.asarray(T, dtype=bool), "success": bool(info.get("is_success", False)),
            "collision": bool(info.get("collision", False)), "learner": learner is not None,
            "violations": float(info.get("ep_violations", 0.0))}


def _init_worker(uses_torch: bool) -> None:
    """Workers that run the learner network must not touch the GPU (the trainer owns it) and stay single-threaded."""
    if uses_torch:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("OMP_NUM_THREADS", "1")


def _job(args):
    return run_episode(*args)


def collect(n_episodes: int, maps: Sequence[str], seed: int = 0, n_workers: Optional[int] = None,
            noise: float = 0.03, difficulty: Tuple[float, float] = (0.3, 1.0), keep_failed: bool = False,
            learner_path: Optional[str] = None, beta: float = 1.0, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Collect ``n_episodes`` over ``maps`` (round-robin) with random difficulty in ``difficulty``."""
    rng = np.random.default_rng(seed)
    jobs = [(maps[i % len(maps)], seed * 100_003 + i, float(rng.uniform(*difficulty)),
             noise if rng.random() < 0.4 else 0.0, learner_path, beta) for i in range(n_episodes)]
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    if learner_path:
        # every worker imports torch + SB3 (~2 GB of DLLs on Windows): too many of them fail with WinError 1114
        n_workers = min(n_workers, 5)
    t0 = time.time()
    results: List[Dict] = []
    if n_workers == 1:
        for i, j in enumerate(jobs):
            results.append(run_episode(*j))
    else:
        with mp.get_context("spawn").Pool(n_workers, initializer=_init_worker, initargs=(bool(learner_path),)) as pool:
            for i, r in enumerate(pool.imap(_job, jobs, chunksize=1)):
                results.append(r)
                if verbose and (i + 1) % 25 == 0:
                    print(f"  [demos] {i + 1}/{n_episodes} episodes, {time.time() - t0:.0f}s", flush=True)
    kept = [r for r in results if keep_failed or r["learner"] or (r["success"] and not r["collision"])]
    if not kept:
        raise RuntimeError("no usable demonstration episodes were collected")
    ep_id = np.concatenate([np.full(len(r["act"]), k, dtype=np.int32) for k, r in enumerate(kept)])
    out = {
        "obs": np.concatenate([r["obs"] for r in kept]),
        "act": np.concatenate([r["act"] for r in kept]),
        "rew": np.concatenate([r["rew"] for r in kept]),
        "term": np.concatenate([r["term"] for r in kept]),
        "ep_id": ep_id,
    }
    if verbose:
        ok = np.mean([r["success"] for r in results])
        print(f"[demos] {len(kept)}/{len(results)} episodes kept, {len(out['act']):,} samples; "
              f"expert success {ok:.1%}; {time.time() - t0:.0f}s", flush=True)
    return out


def save(path: str, data: Dict[str, np.ndarray]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def merge(*sets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Concatenate several demo sets (episode ids are re-numbered)."""
    out, off = {k: [] for k in sets[0]}, 0
    for s in sets:
        for k, v in s.items():
            out[k].append(v + off if k == "ep_id" else v)
        off += int(s["ep_id"].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--maps", type=str, default="synthetic:0,synthetic:1,synthetic:2,synthetic:3")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--noise", type=float, default=0.03)
    p.add_argument("--keep-failed", action="store_true")
    a = p.parse_args()
    save(a.out, collect(a.episodes, a.maps.split(","), a.seed, a.workers, a.noise, keep_failed=a.keep_failed))
    print("saved", a.out)


if __name__ == "__main__":
    main()
