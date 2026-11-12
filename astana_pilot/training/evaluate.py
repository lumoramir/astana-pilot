"""Evaluate a trained model against the expert (HeuristicPolicy) on known AND unknown cities.

    python -m astana_pilot.training.evaluate --model runs/night1/final_model.zip --episodes 60
    python -m astana_pilot.training.evaluate --baseline-only --episodes 60                      # expert only
    python -m astana_pilot.training.evaluate --model m.zip --unseen-maps assets/maps/other_city.pkl

* "seen" cities = the ones used in training; "unseen" = never trained on (``synthetic:100+``: other size and street
  spacing, other random layout jitter / parked cars / traffic). The same episode seeds are used for every policy, so
  the comparison is paired. A large drop seen -> unseen means the network memorised the map.
* Output: console tables + ``report.md`` + ``results.json`` + ``episodes.csv`` in ``--out``.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from astana_pilot.training.metrics import COLLISION_KINDS, VIOLATION_KINDS, EpisodeAggregator, wilson

DEFAULT_SEEN = "synthetic:0,synthetic:1,synthetic:2,synthetic:3"
DEFAULT_UNSEEN = "synthetic:100,synthetic:101,synthetic:102,synthetic:103"
EVAL_SEED0 = 10_000_000          # never used by training (seeds there are small)

_CACHE: Dict = {}


def _worker(job: Tuple[str, str, int, float]) -> Dict:
    """Run one episode of policy spec ``spec`` ("heuristic" or a model path) on ``map_path``."""
    spec, map_path, seed, difficulty = job
    from astana_pilot.env import DrivingEnv, EnvConfig
    key = ("env", map_path)
    if key not in _CACHE:
        _CACHE[key] = DrivingEnv(EnvConfig(map_path=map_path))
    env = _CACHE[key]
    pkey = ("pol", spec, map_path)
    if pkey not in _CACHE:
        if spec == "heuristic":
            from astana_pilot.viz.policies import HeuristicPolicy
            _CACHE[pkey] = HeuristicPolicy(env.cfg)
        else:
            from astana_pilot.viz.policies import ModelPolicy
            _CACHE[pkey] = ModelPolicy(spec)
    pol = _CACHE[pkey]
    env.set_difficulty(difficulty)
    obs, _ = env.reset(seed=seed)
    if hasattr(pol, "reset"):
        pol.reset()
    done, info = False, {}
    while not done:
        obs, _, term, trunc, info = env.step(pol.act(obs))
        done = term or trunc
    agg = EpisodeAggregator()
    row = agg.add(info)
    row.update(spec=spec, map=map_path, seed=seed)
    return row


def run_eval(specs: Dict[str, str], groups: Dict[str, Sequence[str]], episodes: int, difficulty: float = 1.0,
             workers: Optional[int] = None, seed0: int = EVAL_SEED0) -> Dict[Tuple[str, str], EpisodeAggregator]:
    """Evaluate every policy in ``specs`` ({label: spec}) on every group of maps ({group: [maps]})."""
    jobs, tags = [], []
    for g, maps in groups.items():
        for i in range(episodes):
            m = maps[i % len(maps)]
            for label, spec in specs.items():
                jobs.append((spec, m, seed0 + i, difficulty))
                tags.append((label, g))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    uses_torch = any(s != "heuristic" for s in specs.values())
    if uses_torch:                                   # each worker imports torch + SB3: cap it (Windows DLL memory)
        workers = min(workers, 5)
    t0 = time.time()
    if workers == 1:
        rows = [_worker(j) for j in jobs]
    else:
        from astana_pilot.training.demos import _init_worker
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(uses_torch,)) as pool:
            rows = []
            for k, r in enumerate(pool.imap(_worker, jobs, chunksize=2)):
                rows.append(r)
                if (k + 1) % 50 == 0:
                    print(f"  [eval] {k + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    res: Dict[Tuple[str, str], EpisodeAggregator] = {}
    for tag, row in zip(tags, rows):
        res.setdefault(tag, EpisodeAggregator()).rows.append(row)
    return res


# ----------------------------------------------------------------------------- tables
def _md(headers: List[str], rows: List[List[str]]) -> str:
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda c: "| " + " | ".join(x.ljust(w[i]) for i, x in enumerate(c)) + " |"
    return "\n".join([line(headers), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def make_report(res: Dict[Tuple[str, str], EpisodeAggregator], episodes: int, difficulty: float) -> str:
    keys = sorted(res, key=lambda t: (t[1], t[0]))
    summ = {k: res[k].summary() for k in keys}
    out = [f"# Evaluation: {episodes} episodes per policy and city group, difficulty {difficulty}\n"]
    rows = []
    for k in keys:
        s, (lo, hi) = summ[k], res[k].success_ci()
        rows.append([k[0], k[1], f"{int(s['episodes'])}", f"{_pct(s['success'])} [{_pct(lo)}..{_pct(hi)}]",
                     _pct(s["collision_rate"]), _pct(s["stall"]), _pct(s["lost"]), _pct(s["timeout"])])
    out += ["## Outcome (success with 95 % Wilson interval)\n",
            _md(["policy", "cities", "n", "success", "collision", "stall", "lost", "timeout"], rows), ""]
    rows = [[k[0], k[1]] + [_pct(summ[k][f"collision/{c}"]) for c in COLLISION_KINDS] for k in keys]
    out += ["## What was hit (share of all episodes)\n", _md(["policy", "cities"] + list(COLLISION_KINDS), rows), ""]
    rows = [[k[0], k[1]] + [f"{summ[k][f'viol/{v}']:.3f}" for v in VIOLATION_KINDS]
            + [_pct(summ[k]["viol_any"]), _pct(summ[k]["speeding_frac"]), _pct(summ[k]["wrong_side_frac"])]
            for k in keys]
    out += ["## Rule violations (events per episode; viol_any = episodes with >= 1; speeding / wrong side = time share)\n",
            _md(["policy", "cities"] + list(VIOLATION_KINDS) + ["viol_any", "speeding", "wrong side"], rows), ""]
    rows = [[k[0], k[1], _pct(summ[k]["aeb_frac"]), _pct(summ[k]["unneeded_slow_frac"]),
             f"{summ[k]['speed_kmh']:.1f}", f"{summ[k]['steer_jerk']:.4f}", f"{summ[k]['lane_dev_m']:.2f}",
             _pct(summ[k]["progress"])] for k in keys]
    out += ["## Driving manners\n", _md(["policy", "cities", "AEB time", "needless slow", "speed km/h",
                                        "steer jerk", "lane dev m", "route done"], rows), ""]
    labels = sorted({k[0] for k in keys})
    groups = sorted({k[1] for k in keys})
    if len(groups) > 1 and "seen" in groups and "unseen" in groups:
        out.append("## Memorising check (seen -> unseen cities)\n")
        rows = []
        for lb in labels:
            a, b = summ[(lb, "seen")]["success"], summ[(lb, "unseen")]["success"]
            rows.append([lb, _pct(a), _pct(b), f"{100 * (b - a):+.1f} pp"])
        out += [_md(["policy", "success seen", "success unseen", "change"], rows), "",
                "A drop of more than ~10 pp (beyond the intervals above) would mean the map was memorised.", ""]
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", type=str, default=None, help="PPO .zip to evaluate")
    p.add_argument("--baseline-only", action="store_true", help="evaluate only the HeuristicPolicy")
    p.add_argument("--no-baseline", action="store_true", help="do not evaluate the HeuristicPolicy")
    p.add_argument("--episodes", type=int, default=60, help="per policy and city group")
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--seen-maps", type=str, default=DEFAULT_SEEN)
    p.add_argument("--unseen-maps", type=str, default=DEFAULT_UNSEEN)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="output directory (default: next to the model)")
    a = p.parse_args(argv)
    if not a.model and not a.baseline_only:
        p.error("give --model or --baseline-only")
    specs: Dict[str, str] = {}
    if a.model:
        specs["model"] = a.model
    if not a.no_baseline:
        specs["expert"] = "heuristic"
    groups = {"seen": a.seen_maps.split(","), "unseen": a.unseen_maps.split(",")}
    res = run_eval(specs, groups, a.episodes, a.difficulty, a.workers)
    report = make_report(res, a.episodes, a.difficulty)
    print("\n" + report)
    out = Path(a.out) if a.out else (Path(a.model).parent / "eval" if a.model else Path("runs/eval_baseline"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({f"{k[0]}/{k[1]}": v.summary() for k, v in res.items()}, indent=1))
    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "cities", "map", "seed", "outcome", "kind", "spawn", "speed_kmh", "violations"])
        for (lb, g), agg in res.items():
            for r in agg.rows:
                w.writerow([lb, g, r["map"], r["seed"], r["outcome"], r["kind"], r["spawn"],
                            f"{r['speed_kmh']:.1f}", int(sum(r["violations"].values()))])
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
