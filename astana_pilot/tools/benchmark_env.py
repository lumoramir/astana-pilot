"""Measure raw environment throughput on THIS machine (steps/second) and project training time.

    python -m astana_pilot.tools.benchmark_env --map assets/maps/astana_expo.pkl --n-envs 12
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from astana_pilot.env import DrivingEnv, EnvConfig


def bench(cfg: EnvConfig, steps: int) -> float:
    env = DrivingEnv(cfg)
    env.set_difficulty(0.7)
    env.reset(seed=0)
    t0 = time.perf_counter()
    for _ in range(steps):
        _, _, te, tr, _ = env.step(env.action_space.sample())
        if te or tr:
            env.reset()
    return steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--n-envs", type=int, default=12)
    a = ap.parse_args()
    sps = bench(EnvConfig(map_path=a.map), a.steps)
    print(f"single env: {sps:,.0f} steps/s (random actions, 1 core)")
    print(f"ideal x{a.n_envs} envs: {sps * a.n_envs:,.0f} steps/s  (real PPO fps is far lower: IPC + GPU policy calls + updates)")
    for frac in (0.1, 0.2):
        fps = sps * a.n_envs * frac
        print(f"  at {frac:.0%}: 3M steps = {3e6 / fps / 3600:.1f} h,  5M = {5e6 / fps / 3600:.1f} h,"
              f"  20M = {2e7 / fps / 3600:.1f} h")


if __name__ == "__main__":
    main()
