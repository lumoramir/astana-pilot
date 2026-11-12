"""Self-test of the training stack (about 1 minute). Run this FIRST on the GPU machine:

    python -m astana_pilot.tools.check_install.py

Checks, one by one: imports + CUDA, environment, policy network, behaviour cloning (policy + critic), a few PPO
updates, save / load, the evaluation worker. Prints PASS / FAIL for each step and exits with code 1 on the first failure.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def step(name):
    def deco(fn):
        def run(ctx):
            t = time.time()
            try:
                fn(ctx)
            except Exception:
                print(f"[FAIL] {name}\n{traceback.format_exc()}")
                raise SystemExit(1)
            print(f"[PASS] {name}  ({time.time() - t:.1f}s)", flush=True)
        run.__name__ = fn.__name__
        return run
    return deco


@step("imports, torch, CUDA")
def t_imports(ctx):
    import gymnasium, numpy, stable_baselines3, torch
    print(f"       torch {torch.__version__}  sb3 {stable_baselines3.__version__}  gymnasium {gymnasium.__version__}")
    print(f"       CUDA available: {torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else
             "  <- training would run on CPU; see docs/PART3_TRAINING.md (torch with CUDA)"))
    ctx["device"] = "cuda" if torch.cuda.is_available() else "cpu"


@step("environment (synthetic city, 200 steps)")
def t_env(ctx):
    from astana_pilot.env import DrivingEnv, EnvConfig
    e = DrivingEnv(EnvConfig(map_path="synthetic:1"))
    o, _ = e.reset(seed=0)
    assert o.shape == e.observation_space.shape, o.shape
    for _ in range(200):
        o, r, te, tr, i = e.step(e.action_space.sample())
        if te or tr:
            e.reset()
    ctx["obs_dim"] = o.shape[0]
    print(f"       observation size {o.shape[0]}")


@step("policy network + PPO construction + one rollout/update")
def t_ppo(ctx):
    from dataclasses import replace
    from astana_pilot.agents import PPOAgent, PPOConfig
    from astana_pilot.training.vec_env import make_vec_env
    cfg = replace(PPOConfig.for_device(ctx["device"]), n_steps=128, batch_size=128, n_epochs=2)
    vec = make_vec_env(2, 0, maps=["synthetic:0", "synthetic:1"])
    agent = PPOAgent.create(vec, cfg, device=ctx["device"], seed=0)
    n = sum(p.numel() for p in agent.model.policy.parameters())
    print(f"       policy parameters: {n:,}")
    agent.model.learn(256, progress_bar=False)
    ctx["agent"], ctx["vec"] = agent, vec


@step("behaviour cloning (policy + critic) on 8 expert episodes")
def t_bc(ctx):
    import numpy as np
    from astana_pilot.training import bc, demos
    d = demos.collect(8, ["synthetic:0", "synthetic:1"], seed=3, n_workers=1, difficulty=(0.2, 0.4),
                      keep_failed=True, verbose=False)
    a = ctx["agent"]
    before = bc.fit_policy(a.model.policy, d["obs"].astype(np.float32), d["act"], ctx["device"], epochs=1, batch_size=256)
    after = bc.fit_policy(a.model.policy, d["obs"].astype(np.float32), d["act"], ctx["device"], epochs=4, batch_size=256)
    tgt, stats = bc.normalized_returns(d["rew"], d["term"], d["ep_id"])
    bc.fit_critic(a.model.policy, d["obs"].astype(np.float32), tgt, ctx["device"], epochs=1, batch_size=256)
    assert after["train_mse"] < before["train_mse"] * 1.05, (before, after)
    print(f"       samples {len(d['act'])}, train mse {before['train_mse']:.4f} -> {after['train_mse']:.4f}")


@step("save / load / ModelPolicy / evaluation worker")
def t_io(ctx):
    import numpy as np
    from astana_pilot.training.evaluate import _worker
    from astana_pilot.viz.policies import ModelPolicy
    path = Path("runs/_check_install")
    path.mkdir(parents=True, exist_ok=True)
    ctx["agent"].save(path / "m")
    pol = ModelPolicy(str(path / "m.zip"))
    act = pol.act(np.zeros(ctx["obs_dim"], dtype=np.float32))
    assert act.shape == (2,), act.shape
    row = _worker((str(path / "m.zip"), "synthetic:0", 1, 0.3))
    print(f"       model episode outcome: {row['outcome']}")
    ctx["vec"].close()


def main():
    ctx = {}
    for fn in (t_imports, t_env, t_ppo, t_bc, t_io):
        fn(ctx)
    print("\nALL CHECKS PASSED - the training stack works on this machine.")


if __name__ == "__main__":
    main()
