"""How the network improved, step by step: every saved checkpoint of a run driven on the SAME fixed routes.

    astana progress night3                       -> runs/night3/progress.png + progress.md
    astana progress night3 --episodes 40         -> more routes per checkpoint (narrower error bars)
    astana progress night3 --max-checkpoints 8   -> faster: evaluate only 8 evenly spaced checkpoints

Unlike the TensorBoard curves (``astana plot``: noisy, measured on whatever random routes and difficulty the training
happened to see), this is a controlled experiment: identical seeds, full traffic (difficulty 1.0), the same cities for
every checkpoint, plus the rule-based expert as the reference line. Results are cached in ``runs/<run>/progress/`` so a
second call only evaluates new checkpoints.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from astana_pilot.training.metrics import wilson

BG, PANEL, GRID, TEXT, DIM = "#0b0e14", "#121722", "#222a38", "#e8edf5", "#8b96a8"
SEEN_C, UNSEEN_C, EXPERT_C = "#6ee0ff", "#fbbf24", "#9aa6c0"


def model_steps(path: Path) -> int:
    try:
        with zipfile.ZipFile(path) as z:
            return int(json.loads(z.read("data")).get("num_timesteps", 0))
    except Exception:
        m = re.search(r"(\d+)_steps", path.name)
        return int(m.group(1)) if m else 0


def checkpoints(run_dir: Path, include_source: bool = True) -> List[Tuple[str, Path, int]]:
    """(label, path, env steps) of every policy snapshot of a run, in training order."""
    out: List[Tuple[str, Path, int]] = []
    st = run_dir / "state.json"
    src = json.loads(st.read_text()).get("init_from") if st.exists() else None
    if include_source and src and Path(src).exists():
        out.append(("start (" + Path(src).parent.name + ")", Path(src), model_steps(Path(src))))
    if (run_dir / "bc_model.zip").exists():
        out.append(("behaviour cloning", run_dir / "bc_model.zip", 0))
    for p in sorted((run_dir / "checkpoints").glob("ppo_*_steps.zip"), key=model_steps):
        out.append((f"{model_steps(p) / 1e6:.1f} M", p, model_steps(p)))
    for name in ("best_model.zip", "final_model.zip"):
        p = run_dir / name
        if p.exists():
            out.append((name.split("_")[0], p, model_steps(p)))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", help="run name (folder in runs/)")
    ap.add_argument("--episodes", type=int, default=30, help="routes per checkpoint and city group")
    ap.add_argument("--seen-maps", default=None, help="default: the cities the run trained on")
    ap.add_argument("--unseen-maps", default="synthetic:100,synthetic:101,synthetic:102,synthetic:103")
    ap.add_argument("--max-checkpoints", type=int, default=14)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    run_dir = Path("runs") / a.run
    if not run_dir.exists():
        raise SystemExit(f"no such run: {run_dir}")
    st = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
    seen = a.seen_maps.split(",") if a.seen_maps else st.get("maps", ["synthetic:0", "synthetic:1"])
    groups = {"seen": seen, "unseen": a.unseen_maps.split(",")}

    cps = checkpoints(run_dir)
    best_final = [c for c in cps if c[0] in ("best", "final")]
    timeline = [c for c in cps if c[0] not in ("best", "final")]
    if len(timeline) > a.max_checkpoints:                      # keep first and last, evenly spaced in between
        idx = sorted({round(i * (len(timeline) - 1) / (a.max_checkpoints - 1)) for i in range(a.max_checkpoints)})
        timeline = [timeline[i] for i in idx]
    todo = timeline + best_final
    if not todo:
        raise SystemExit(f"no checkpoints in {run_dir} yet (they are written every 1 M steps)")

    cache_dir = run_dir / "progress"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "results.json"
    cache: Dict[str, Dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = lambda p: f"{p.resolve()}|{p.stat().st_mtime:.0f}|{a.episodes}|{','.join(seen)}|{a.unseen_maps}"

    specs = {str(p): str(p) for _, p, _ in todo if key(p) not in cache}
    if "expert" not in cache or cache["expert"].get("episodes") != a.episodes:
        specs["expert"] = "heuristic"
    if specs:
        from astana_pilot.training.evaluate import run_eval
        print(f"[progress] evaluating {len(specs)} policies x {a.episodes} routes x {len(groups)} city groups ...", flush=True)
        res = run_eval(specs, groups, a.episodes, 1.0, a.workers)
        for (label, g), agg in res.items():
            s = agg.summary()
            k = sum(1 for r in agg.rows if r["outcome"] == "success")
            s["ci"] = wilson(k, len(agg.rows))
            ck = "expert" if label == "expert" else key(Path(label))
            cache.setdefault(ck, {"episodes": a.episodes})[g] = s
        cache_path.write_text(json.dumps(cache, indent=1))

    rows = []
    for lab, p, steps in todo:
        r = cache.get(key(p), {})
        rows.append((lab, steps, r.get("seen", {}), r.get("unseen", {})))
    exp = cache.get("expert", {})

    # ---------------------------------------------------------------- table
    md = [f"# How `{a.run}` learned - every checkpoint on the same {a.episodes} routes per city group\n",
          f"seen cities: {', '.join(seen)}  \nunseen cities: {a.unseen_maps}  \nfull traffic (difficulty 1.0), identical seeds\n",
          "| checkpoint | env steps | success seen | success unseen | collisions seen | rule-violation episodes seen |",
          "|---|---|---|---|---|---|"]
    for lab, steps, s, u in rows + [("expert (rule-based)", 0, exp.get("seen", {}), exp.get("unseen", {}))]:
        f = lambda d, k: f"{100 * d[k]:.0f} %" if k in d else "-"
        md.append(f"| {lab} | {steps:,} | {f(s, 'success')} | {f(u, 'success')} | {f(s, 'collision_rate')} | {f(s, 'viol_any')} |")
    out_png = Path(a.out) if a.out else run_dir / "progress.png"
    (run_dir / "progress.md").write_text("\n".join(md) + f"\n\n![progress]({out_png.name})\n", encoding="utf-8")
    print("\n".join(md))

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    line = [r for r in rows if r[0] not in ("best", "final")]
    xs = [r[1] / 1e6 for r in line]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=130)
    fig.patch.set_facecolor(BG)
    panels = (("Trips completed (success)", "success", True), ("Collisions", "collision_rate", False),
              ("Trips with a rule violation", "viol_any", False))
    for ax, (title, k, ci) in zip(axes, panels):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, lw=0.7)
        ax.tick_params(colors=DIM)
        ax.set_title(title, color=TEXT, loc="left", fontsize=13, fontweight="bold")
        for g, col, name in (("seen", SEEN_C, "known cities"), ("unseen", UNSEEN_C, "never-seen cities")):
            ys = [r[2 if g == "seen" else 3].get(k) for r in line]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), "-o", color=col, lw=2.4, ms=5, label=name)
            if ci:
                lo = [r[2 if g == "seen" else 3].get("ci", (y, y))[0] for r, y in zip(line, ys) if y is not None]
                hi = [r[2 if g == "seen" else 3].get("ci", (y, y))[1] for r, y in zip(line, ys) if y is not None]
                ax.fill_between([p[0] for p in pts], lo, hi, color=col, alpha=0.12)
            ev = exp.get(g, {}).get(k)
            if ev is not None:
                ax.axhline(ev, color=col, ls="--", lw=1.2, alpha=0.7)
        ev = exp.get("seen", {}).get(k)
        if ev is not None:
            ax.text(0.99, ev, "  expert", color=EXPERT_C, fontsize=9, va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("training experience, million simulation steps", color=DIM)
        leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="best")
        leg.get_frame().set_alpha(0.85)
    fig.suptitle(f"Astana Pilot - how the network learned ({a.run}): each point = one saved network, "
                 f"{a.episodes} identical routes per group, full traffic; dashed = rule-based expert",
                 color=TEXT, fontsize=13.5, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, facecolor=BG)
    plt.close(fig)
    print(f"\n[progress] {out_png}  +  {run_dir / 'progress.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
