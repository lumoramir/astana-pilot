"""Top-down snapshot of the living world around the car (cars, buses, pedestrians, lights, lidar).

    python -m astana_pilot.tools.preview_world --map synthetic --seed 5 --steps 150 --out assets/part2_world.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
from matplotlib.patches import Circle, Polygon                         # noqa: E402

from astana_pilot.env import DrivingEnv, EnvConfig                                  # noqa: E402
from astana_pilot.viz.policies import HeuristicPolicy                     # noqa: E402

COL = {"car": "#4aa3ff", "bus": "#ffb020", "parked": "#8a93a6", "ped": "#ff4d6d"}


def render(env: DrivingEnv, out: str, radius: float = 70.0) -> None:
    st = env.get_render_state()
    v = st.vehicle
    fig, ax = plt.subplots(figsize=(9, 9), dpi=130)
    fig.patch.set_facecolor("#0b0f1e"); ax.set_facecolor("#0b0f1e")
    city = st.city
    H, W = city.grid.shape
    ext = (city.x0, city.x0 + (W - 1) * city.res, city.y0, city.y0 + (H - 1) * city.res)
    ax.imshow(np.where(city.grid > 0, 1.0, 0.0), origin="lower", extent=ext, cmap="gray", vmin=-0.6, vmax=1.6)
    ax.plot(st.ref.xy[:, 0], st.ref.xy[:, 1], color="#3cff8c", lw=1.2, alpha=0.8)
    for c in city.infra.crosswalks:
        ax.add_patch(Circle(c.xy, 3.0, fc="none", ec="white", lw=0.8))
    for ctl in city.infra.controls:
        if ctl.kind == "light":
            s = city.infra.light_state(ctl.node, ctl.group, st.world_time)
            ax.plot(*ctl.xy, "o", ms=6, color=("#22c55e", "#facc15", "#ef4444")[s])
        else:
            ax.plot(*ctl.xy, "s", ms=4, color="#facc15" if ctl.kind == "yield" else "#ef4444")
    for bs in env.world.bus_stops:
        ax.plot(*bs.kerb, "P", ms=5, color="#ffb020")
    for a in st.actors:
        if not a.alive:
            continue
        if a.kind == "ped":
            ax.add_patch(Circle((a.x, a.y), 0.9, fc=COL["ped"] if a.relevant else "#a78bfa", ec="none"))
            continue
        ct, sn = math.cos(a.theta), math.sin(a.theta)
        hl, hw = a.length / 2, a.width / 2
        pts = [(a.x + ct * dx - sn * dy, a.y + sn * dx + ct * dy) for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
        ax.add_patch(Polygon(pts, fc=COL[a.kind], ec="white", lw=0.5))
    ct, sn = math.cos(v.theta), math.sin(v.theta)
    pts = [(v.x + ct * dx - sn * dy, v.y + sn * dx + ct * dy) for dx, dy in ((2.25, 0.95), (2.25, -0.95), (-2.25, -0.95), (-2.25, 0.95))]
    ax.add_patch(Polygon(pts, fc="#00f0ff", ec="white", lw=0.8))
    for d, ang in zip(st.lidar, st.lidar_angles):
        ax.plot([v.x, v.x + d * math.cos(ang)], [v.y, v.y + d * math.sin(ang)], color="#00f0ff", lw=0.3, alpha=0.5)
    ax.set_xlim(v.x - radius, v.x + radius); ax.set_ylim(v.y - radius, v.y + radius); ax.set_aspect("equal")
    p = st.perception
    ax.set_title(f"cyan = agent, blue = cars, orange = buses, grey = parked, red = pedestrians, violet = decor walkers   "
                 f"t={st.world_time:.0f}s  v={v.v:.1f} m/s  gap={p.gap:.0f} m", color="white", fontsize=8)
    ax.tick_params(colors="white")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="synthetic")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", default="assets/part2_world.png")
    a = ap.parse_args()
    cfg = EnvConfig(map_path=a.map, recovery_prob=0.0)
    env = DrivingEnv(cfg); env.set_difficulty(1.0)
    obs, _ = env.reset(seed=a.seed)
    pol = HeuristicPolicy(cfg)
    for _ in range(a.steps):
        obs, _, te, tr, _ = env.step(pol.act(obs))
        if te or tr:
            break
    render(env, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
