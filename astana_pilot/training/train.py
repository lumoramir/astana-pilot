"""Train the driving policy: behaviour-cloning warm-up, then PPO. Resumable, safe to leave running overnight.

    python -m astana_pilot.training.train --run-name night1 --hours 8            # everything, ~8 h wall clock
    python -m astana_pilot.training.train --run-name night1 --hours 8            # run it again after a crash/power cut: it RESUMES
    python -m astana_pilot.training.train --run-name test --timesteps 100000 --demos-episodes 40 --bc-epochs 3   # short test
    python -m astana_pilot.training.train --run-name night3 --init-from runs/night2/best_model.zip --hours 8   # fine-tune

What happens
  1. (first start only) expert demonstrations on several cities -> behaviour cloning of the policy + critic warm-up
     (+ optional DAgger rounds). Skip with --no-bc. Demos are cached in runs/<name>/demos.npz.
  2. PPO with `n_envs` parallel cities, adaptive curriculum (difficulty follows the success rate),
     TensorBoard metrics (`tensorboard --logdir runs`), autosave every 15 min to runs/<name>/checkpoints/latest.zip.
  3. It stops when `--timesteps` (target TOTAL, including earlier runs) or `--hours` is reached; final_model.zip is saved.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from astana_pilot.agents import PPOAgent, PPOConfig
from astana_pilot.agents.config import fit_batch_size
from astana_pilot.env import EnvConfig

DEFAULT_MAPS = "synthetic:0,synthetic:1,synthetic:2,synthetic:3"      # evaluation uses other cities (synthetic:100+)


def default_n_envs() -> int:
    """Parallel envs: leave 2 logical cores free, multiple of 4, between 4 and 16."""
    n = max(4, min(16, (os.cpu_count() or 8) - 2))
    return max(4, n - (n % 4))


def resolve_device(requested: str) -> str:
    """Turn ``auto`` / ``cuda`` / ``cpu`` into a usable device and print what we got."""
    import torch

    cuda_ok = torch.cuda.is_available()
    if requested.startswith("cuda") and not cuda_ok:
        raise SystemExit(
            "CUDA was requested but this PyTorch cannot see a GPU.\n"
            f"  installed torch: {torch.__version__} (CUDA build: {torch.version.cuda})\n"
            "  Windows pip installs a CPU-only build by default. Fix:\n"
            "    pip uninstall -y torch\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    device = "cuda" if (requested == "auto" and cuda_ok) else ("cpu" if requested == "auto" else requested)
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        torch.set_float32_matmul_precision("high")        # TF32 on RTX cards
        print(f"[device] CUDA: {props.name}, {props.total_memory / 2**30:.1f} GB, "
              f"torch {torch.__version__} / CUDA {torch.version.cuda}", flush=True)
    else:
        print(f"[device] CPU (torch {torch.__version__}, CUDA build: {torch.version.cuda}).", flush=True)
    return device


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-name", type=str, default="night1")
    p.add_argument("--timesteps", type=int, default=100_000_000, help="TARGET total env steps (incl. earlier runs)")
    p.add_argument("--hours", type=float, default=None, help="stop after this much wall-clock time")
    p.add_argument("--maps", type=str, default=None,
                   help="comma-separated cities: 'synthetic:K' or baked .pkl maps; env i gets maps[i %% len]\n"
                        f"(default: {DEFAULT_MAPS}, or the cities of the --init-from run)")
    p.add_argument("--init-from", type=str, default=None,
                   help="fine-tune an already trained network (e.g. runs/night2/best_model.zip) in a NEW run:\n"
                        "no behaviour cloning, reward scaling and cities taken over from that run")
    p.add_argument("--start-difficulty", type=float, default=None, help="curriculum start (default 0.2, fine-tune 0.3)")
    p.add_argument("--rule-penalty-scale", type=float, default=1.0,
                   help="multiply the one-off penalties for red light / yellow / stop / give-way / zebra violations\n"
                        "(night2 measured 37%% of trips with a violation at 1.0; 2.5 is the fine-tune setting)")
    p.add_argument("--checkpoint-every", type=int, default=1_000_000,
                   help="numbered checkpoint every N steps (the points of `astana progress`)")
    p.add_argument("--n-envs", type=int, default=None, help="default: from CPU cores (kept on resume)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    # behaviour cloning
    p.add_argument("--no-bc", action="store_true", help="PPO from scratch (no expert warm-up)")
    p.add_argument("--demos", type=str, default=None, help="reuse a demos .npz instead of collecting")
    p.add_argument("--demos-episodes", type=int, default=300)
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--critic-epochs", type=int, default=8)
    p.add_argument("--dagger-rounds", type=int, default=2, help="learner-drives / expert-labels rounds after BC")
    # PPO
    p.add_argument("--curriculum", choices=["adaptive", "off"], default="adaptive")
    p.add_argument("--checkpoint-minutes", type=float, default=15.0)
    p.add_argument("--no-sde", action="store_true", help="plain Gaussian exploration instead of gSDE")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup-rollouts", type=int, default=6, help="PPO iterations with the cloned policy frozen (critic warm-up)")
    return p.parse_args(argv)


def run_bc_stage(agent: PPOAgent, args: argparse.Namespace, run_dir: Path, maps: List[str], device: str):
    """Collect (or load) demos and clone them into ``agent``; returns the return statistics for VecNormalize."""
    from astana_pilot.training import bc, demos

    src = Path(args.demos) if args.demos else run_dir / "demos.npz"
    path = run_dir / "demos.npz"                     # never overwrite a demo file that came from another run
    if src.exists():
        data = demos.load(str(src))
        print(f"[bc] loaded {len(data['act']):,} demo samples from {src}", flush=True)
    else:
        print(f"[bc] collecting {args.demos_episodes} expert episodes on {len(maps)} cities ...", flush=True)
        data = demos.collect(args.demos_episodes, maps, seed=args.seed + 1)
        demos.save(str(path), data)
    stats = bc.warm_start(agent.model.policy, data, device, args.bc_epochs, args.critic_epochs)
    for r in range(args.dagger_rounds):
        tmp = run_dir / "bc_stage.zip"
        agent.save(tmp)
        print(f"[dagger] round {r + 1}/{args.dagger_rounds}: learner drives, expert labels", flush=True)
        extra = demos.collect(max(20, args.demos_episodes // 3), maps, seed=args.seed + 50 + r, noise=0.08,
                              learner_path=str(tmp), beta=0.0, keep_failed=True)
        data = demos.merge(data, extra)
        demos.save(str(path), data)
        stats = bc.warm_start(agent.model.policy, data, device, max(3, args.bc_epochs // 3), args.critic_epochs)
    agent.save(run_dir / "bc_model")
    print(f"[bc] saved {run_dir / 'bc_model.zip'}", flush=True)
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    run_dir = Path("runs") / args.run_name
    ck = run_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.json"
    latest = ck / "latest.zip"
    resume = latest.exists() and not args.fresh
    old_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    init = None
    if args.init_from and not resume:
        init = Path(args.init_from)
        if not init.exists():
            raise SystemExit(f"--init-from: {init} not found")
    init_from = str(init) if init else (old_state.get("init_from") if resume else None)
    src_state = {}
    if init is not None and (init.parent / "state.json").exists():
        src_state = json.loads((init.parent / "state.json").read_text())
    maps = (args.maps or ",".join(src_state.get("maps", [])) or DEFAULT_MAPS).split(",")
    n_envs = args.n_envs or src_state.get("n_envs") or default_n_envs()
    if resume and old_state and args.n_envs is None:
        n_envs, maps = old_state.get("n_envs", n_envs), old_state.get("maps", maps)

    cfg = PPOConfig.for_device(device)
    if args.no_sde:
        cfg = replace(cfg, use_sde=False, ent_coef=0.005)
    use_bc = not args.no_bc and not resume and init is None
    if use_bc:
        cfg = cfg.finetune()
    cfg = replace(cfg, batch_size=args.batch_size or fit_batch_size(n_envs * cfg.n_steps, cfg.batch_size))
    env_cfg = EnvConfig()
    if args.rule_penalty_scale != 1.0:
        k, w = args.rule_penalty_scale, env_cfg.reward
        env_cfg = replace(env_cfg, reward=replace(w, red_light=w.red_light * k, yellow_light=w.yellow_light * k,
                                                  stop_sign=w.stop_sign * k, yield_sign=w.yield_sign * k, zebra=w.zebra * k))
        print(f"[reward] rule penalties x{k}: red light {env_cfg.reward.red_light:.0f}, stop {env_cfg.reward.stop_sign:.0f}, "
              f"zebra {env_cfg.reward.zebra:.0f} (goal bonus {w.goal_bonus:.0f})", flush=True)
    state_path.write_text(json.dumps({"n_envs": n_envs, "maps": maps, "seed": args.seed, "init_from": init_from,
                                      "rule_penalty_scale": args.rule_penalty_scale}))
    print(f"[config] run={args.run_name} maps={maps} n_envs={n_envs} batch={cfg.batch_size} net={cfg.net_arch}"
          f" features={cfg.features_dim} sde={cfg.use_sde} lr={cfg.learning_rate} resume={resume} init_from={init_from}",
          flush=True)

    from astana_pilot.training.callbacks import (AdaptiveCurriculumCallback, AutosaveCallback, CriticWarmupCallback,
                                    EpisodeStatsCallback, ProgressPrintCallback, StabilityCallback, TimeBudgetCallback)
    from astana_pilot.training.vec_env import make_vec_env
    from stable_baselines3.common.callbacks import CallbackList

    stats = None
    pre_agent = None
    if use_bc:           # demos are collected BEFORE the training workers start (they use the same CPU cores)
        from astana_pilot.training.vec_env import make_vec_env as _mk
        probe = _mk(1, args.seed, EnvConfig(map_path=maps[0]), maps=maps)
        pre_agent = PPOAgent.create(probe, cfg, tensorboard_log=run_dir / "tb", seed=args.seed, device=device)
        stats = run_bc_stage(pre_agent, args, run_dir, maps, device)
        probe.close()

    vn_path = ck / "vecnormalize_latest.pkl"
    if resume:
        vn_use = str(vn_path) if vn_path.exists() else None
    elif init is not None:                       # the critic of the source run was trained with ITS reward scaling
        cands = [init.parent / "final_vecnormalize.pkl", init.parent / "checkpoints" / "vecnormalize_latest.pkl"]
        vn_use = next((str(c) for c in cands if c.exists()), None)
    else:
        vn_use = None
    vec_env = make_vec_env(n_envs, args.seed, env_cfg, maps=maps, vecnormalize_path=vn_use)
    if resume:
        agent = PPOAgent.load(latest, device=device)
        agent.attach_env(vec_env)
        print(f"[resume] {latest}  (already trained: {agent.model.num_timesteps:,} steps)", flush=True)
    elif init is not None:
        from stable_baselines3 import PPO
        agent = PPOAgent(PPO.load(str(init), env=vec_env, device=device))     # env given: n_envs may differ
        print(f"[init] fine-tuning {init}  (it had {agent.model.num_timesteps:,} steps; reward scaling: {vn_use})", flush=True)
    else:
        agent = PPOAgent.create(vec_env, cfg, tensorboard_log=run_dir / "tb", seed=args.seed, device=device)
        if pre_agent is not None:                  # take over the cloned weights (the rollout buffer needs n_envs)
            agent.model.policy.load_state_dict(pre_agent.model.policy.state_dict())
            del pre_agent
            if stats is not None:                  # the critic was fitted on rewards scaled by these statistics
                rms = vec_env.ret_rms
                rms.mean, rms.var, rms.count = stats[0], stats[1], max(stats[2], 1.0)

    remaining = args.timesteps - agent.model.num_timesteps
    if remaining <= 0:
        print("[train] target already reached - nothing to do", flush=True)
    else:
        es = EpisodeStatsCallback()
        start = args.start_difficulty if args.start_difficulty is not None else (0.3 if init_from else 0.2)
        cur = AdaptiveCurriculumCallback(es, run_dir / "curriculum.json", start=start) if args.curriculum == "adaptive" else None
        callbacks = [es] + ([cur] if cur else []) + ([CriticWarmupCallback(args.warmup_rollouts)] if use_bc else []) + [
            StabilityCallback(es, cur, run_dir),
            AutosaveCallback(run_dir, args.checkpoint_minutes, numbered_every=args.checkpoint_every), TimeBudgetCallback(args.hours),
            ProgressPrintCallback(es, cur)]
        if not cur:
            vec_env.env_method("set_difficulty", 1.0)
        try:
            agent.model.tensorboard_log = str(run_dir / "tb")
            agent.learn(remaining, CallbackList(callbacks))
        finally:
            agent.save(run_dir / "final_model")
            vec_env.save(str(run_dir / "final_vecnormalize.pkl"))
            vec_env.close()
        print(f"[done] saved {run_dir / 'final_model.zip'}  ({agent.model.num_timesteps:,} steps)", flush=True)


if __name__ == "__main__":   # required for SubprocVecEnv on Windows/macOS
    main()
