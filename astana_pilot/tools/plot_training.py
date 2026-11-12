"""Training dashboard: one clean figure from the TensorBoard logs of a run (or several runs, overlaid).

    astana plot night1                      -> runs/night1/training_curves.png
    astana plot night1 baseline_run         -> overlay two runs
    astana plot night1 --watch              -> refresh every minute while the training is running
    astana plot night1 --smooth 0.95        -> stronger smoothing (0 = raw)

Panels: success, collisions (by kind), rule violations, needless braking + emergency-brake share, speed, curriculum
difficulty, reward, policy health (entropy, KL, explained variance). Raw values are drawn faintly behind the smoothed line.
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BG, PANEL, GRID, TEXT, DIM = "#0b0e14", "#121722", "#222a38", "#e8edf5", "#8b96a8"
PALETTE = ["#6ee0ff", "#fbbf24", "#fb7185", "#a78bfa", "#4ade80", "#f472b6", "#60a5fa", "#f59e0b"]


def load_run(run: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """All scalar series of the newest TensorBoard log of ``runs/<run>`` -> {tag: (steps, values)}."""
    from tensorboard.backend.event_processing import event_accumulator as ea
    def _num(p: str) -> int:                                # PPO_2 before PPO_10
        tail = Path(p).name.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    dirs = sorted(glob.glob(f"runs/{run}/tb/*"), key=_num)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for d in dirs:                                         # a resumed run has several PPO_n folders: concatenate by step
        acc = ea.EventAccumulator(d, size_guidance={ea.SCALARS: 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            ev = acc.Scalars(tag)
            s = np.array([e.step for e in ev], dtype=float)
            v = np.array([e.value for e in ev], dtype=float)
            if tag in out and len(s):
                s0, v0 = out[tag]
                keep = s0 < s[0]                            # after a resume the new run re-does steps since the last checkpoint
                s, v = np.concatenate([s0[keep], s]), np.concatenate([v0[keep], v])
            out[tag] = (s, v)
    return out


def ema(v: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average with TensorBoard-style debiasing."""
    if alpha <= 0 or len(v) < 3:
        return v
    out = np.empty_like(v)
    last, w = 0.0, 0.0
    for i, x in enumerate(v):
        last = last * alpha + (1 - alpha) * x
        w = w * alpha + (1 - alpha)
        out[i] = last / w
    return out


def render(runs: Sequence[str], out_path: str, smooth: float = 0.9) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {r: load_run(r) for r in runs}
    if not any(data.values()):
        raise SystemExit(f"no TensorBoard logs found for {', '.join(runs)} (looked in runs/<name>/tb)")

    panels: List[Tuple[str, List[Tuple[str, str]], str, Optional[Tuple[float, float]]]] = [
        ("Success rate", [("custom/success", "trips completed"), ("custom/progress", "share of route driven")], "%", (0, 1)),
        ("Collisions", [("custom/collision_rate", "all"), ("custom/collision/pedestrian", "pedestrian"), ("custom/collision/car", "car"),
                        ("custom/collision/wall", "wall")], "%", (0, None)),
        ("Rule violations", [("custom/viol_any", "episodes with a violation"), ("custom/viol/red_light", "red light"),
                             ("custom/viol/zebra", "zebra"), ("custom/viol/stop_sign", "stop sign")], "/ep", (0, None)),
        ("Manners", [("custom/unneeded_slow_frac", "needless slow"), ("custom/aeb_frac", "emergency brake"),
                     ("custom/speeding_frac", "speeding")], "%", (0, 1)),
        ("Speed", [("custom/speed_kmh", "km/h")], "", None),
        ("Curriculum difficulty", [("custom/difficulty", "difficulty")], "", (0, 1.02)),
        ("Reward", [("rollout/ep_rew_mean", "mean episode reward")], "", None),
        ("Policy health", [("train/explained_variance", "explained variance"), ("train/approx_kl", "KL x 10"),
                           ("train/entropy_loss", "entropy loss / 10")], "", None),
    ]
    plt.rcParams.update({"font.family": "Segoe UI" if "Segoe UI" in {f.name for f in matplotlib.font_manager.fontManager.ttflist} else "DejaVu Sans"})
    fig, axes = plt.subplots(2, 4, figsize=(21, 9.6), dpi=130)
    fig.patch.set_facecolor(BG)
    last_summary = []
    for ax, (title, series, unit, ylim) in zip(axes.ravel(), panels):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(color=GRID, lw=0.7)
        ax.tick_params(colors=DIM, labelsize=9)
        ax.set_title(title, color=TEXT, loc="left", fontsize=12.5, fontweight="bold", pad=10)
        plotted = False
        ci = 0
        for ri, run in enumerate(runs):
            for tag, label in series:
                if tag not in data[run]:
                    continue
                s, v = data[run][tag]
                if tag == "train/approx_kl":
                    v = v * 10
                if tag == "train/entropy_loss":
                    v = v / 10
                col = PALETTE[ci % len(PALETTE)]
                ci += 1
                x = s / 1e6
                ax.plot(x, v, color=col, alpha=0.16, lw=1)
                y = ema(v, smooth)
                lab = label if len(runs) == 1 else f"{run}: {label}"
                ax.plot(x, y, color=col, lw=2.2, label=lab)
                ax.scatter([x[-1]], [y[-1]], color=col, s=26, zorder=5)
                plotted = True
                if tag == "custom/success" and ri == len(runs) - 1:
                    last_summary.append(f"success {100 * y[-1]:.0f}%")
                if title == "Collisions" and label == "all" and ri == len(runs) - 1:
                    last_summary.append(f"collisions {100 * y[-1]:.1f}%")
                if title == "Rule violations" and label.startswith("episodes") and ri == len(runs) - 1:
                    last_summary.append(f"violations {100 * y[-1]:.1f}%")
        if not plotted:
            ax.text(0.5, 0.5, "no data yet", color=DIM, ha="center", va="center", transform=ax.transAxes)
        else:
            leg = ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5, loc="best")
            leg.get_frame().set_alpha(0.85)
        if ylim:
            ax.set_ylim(*ylim) if ylim[1] is not None else ax.set_ylim(bottom=ylim[0])
        if unit == "%":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("million environment steps", color=DIM, fontsize=9)
    steps = max((float(s[-1]) for d in data.values() for s, _ in d.values()), default=0.0)
    fig.suptitle(f"Astana Pilot — training  ·  {', '.join(runs)}  ·  {steps / 1e6:.2f} M steps"
                 + ("   |   " + "   ".join(last_summary) if last_summary else ""), color=TEXT, fontsize=15, fontweight="bold", x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names (folders in runs/)")
    ap.add_argument("--smooth", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    ap.add_argument("--watch", action="store_true", help="redraw every 60 s")
    ap.add_argument("--open", action="store_true", help="open the picture when done")
    a = ap.parse_args(argv)
    out = a.out or f"runs/{a.runs[0]}/training_curves.png"
    while True:
        render(a.runs, out, a.smooth)
        print(f"[plot] {out}  ({time.strftime('%H:%M:%S')})", flush=True)
        if a.open:
            import os
            os.startfile(out)                                   # type: ignore[attr-defined]
            a.open = False
        if not a.watch:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
