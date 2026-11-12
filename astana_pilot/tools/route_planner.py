"""Navigator window: choose A and B on the real map, see the alternatives the planner compared, launch the 3-D drive.

    astana route --map assets/maps/astana_expo.pkl --model runs/night2/best_model.zip
        left click  = start A        right click (or a second left click) = goal B
        ENTER       = drive it in 3-D (the neural network drives, the navigator only gives the route)
        R           = random A -> B       C = clear       Q / ESC = close
    astana route --map ... --random --seed 7 --out route.png       # no window: print the plan, save a picture
    astana route --map ... --from "Nur Alem" --to "Kabanbay" --out route.png
    astana route --map ... --list                                  # landmarks / streets you can type, with node ids
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List, Optional

import numpy as np

from astana_pilot.env import EnvConfig
from astana_pilot.env import navigator as nav
from astana_pilot.env.city_map import load_city_map

BG, ROAD, BUILD, TEXT, DIM = "#0b0e14", "#5c6884", "#1d2536", "#e8edf5", "#8b96a8"
CHOSEN, ALT, A_COL, B_COL = "#2578ff", "#9aa6c0", "#2fd67a", "#ff4f5a"


def _draw_city(ax, city) -> None:
    from matplotlib.collections import LineCollection, PolyCollection
    ax.set_facecolor(BG)
    if city.raw.buildings:
        ax.add_collection(PolyCollection([r for r, _ in city.raw.buildings], facecolors=BUILD, edgecolors="none", zorder=1))
    seen, segs, widths = set(), [], []
    for i, e in enumerate(city.raw.edges):
        key = (min(e.u, e.v), max(e.u, e.v), len(e.xy))
        if key in seen:
            continue
        seen.add(key)
        segs.append(e.xy)
        widths.append(0.5 + 0.18 * city.edge_hw[i])
    ax.add_collection(LineCollection(segs, colors=ROAD, linewidths=widths, zorder=2, capstyle="round"))
    for lm in getattr(city.raw, "landmarks", []) or []:
        x, y = lm["xy"]
        ax.scatter([x], [y], s=40, marker="*", color="#fbbf24", zorder=6)
        ax.text(x, y + 25, lm.get("name", ""), color="#fbbf24", fontsize=8, ha="center", zorder=6)
    x0, y0, x1, y1 = city.bounds
    ax.set_xlim(x0 - 30, x1 + 30)
    ax.set_ylim(y0 - 30, y1 + 30)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_plan(ax, city, plan, artists: List) -> str:
    for a in artists:
        a.remove()
    artists.clear()
    if plan is None:
        return ""
    for i, o in enumerate(plan.options):
        if i != plan.chosen:
            artists += ax.plot(o.ref.xy[:, 0], o.ref.xy[:, 1], color=ALT, lw=4.0, alpha=0.75, zorder=4, solid_capstyle="round")
    b = plan.best
    artists += ax.plot(b.ref.xy[:, 0], b.ref.xy[:, 1], color=CHOSEN, lw=5.5, zorder=5, solid_capstyle="round")
    for (x, y), c, lab in ((b.ref.xy[0], A_COL, "A"), (b.ref.xy[-1], B_COL, "B")):
        artists.append(ax.scatter([x], [y], s=260, color=c, edgecolors="white", linewidths=2, zorder=7))
        artists.append(ax.text(x, y, lab, color="white", fontsize=10, fontweight="bold", ha="center", va="center", zorder=8))
    lines = [f"A: {plan.start_name}   →   B: {plan.goal_name}"]
    for i, o in enumerate(plan.options):
        tag = "✓ fastest - chosen" if i == plan.chosen else f"+{(o.eta_s - b.eta_s) / 60:.1f} min"
        dot = "●" if i == plan.chosen else "○"
        lines.append(f"{dot} route {i + 1}:  {o.length / 1000:.2f} km   {o.eta_s / 60:.1f} min   "
                     f"{o.lights} signals   {o.crossings} zebras   [{tag}]")
    lines.append(f"compared {plan.considered} candidate routes, {plan.rejected} rejected (duplicate / not drivable)")
    return "\n".join(lines)


def _figure(city, title: str):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 11), facecolor=BG)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.86))
    _draw_city(ax, city)
    head = fig.text(0.015, 0.975, title, color=TEXT, fontsize=15, fontweight="bold", va="top")
    info = fig.text(0.015, 0.935, "", color=DIM, fontsize=10, va="top", family="monospace")
    return fig, ax, head, info


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--model", default=None, help="network for the 3-D drive (default: rule-based expert)")
    ap.add_argument("--from", dest="start", default=None, help="node id | 'x,y' | landmark / street name")
    ap.add_argument("--to", dest="goal", default=None)
    ap.add_argument("--random", action="store_true", help="random A -> B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route-km", default="0.6-1.5")
    ap.add_argument("--alternatives", type=int, default=3)
    ap.add_argument("--out", default=None, help="save the picture (no window)")
    ap.add_argument("--list", action="store_true", help="print landmarks / streets you can use as --from / --to")
    ap.add_argument("--quality", default="auto")
    a = ap.parse_args(argv)

    city = load_city_map(a.map)
    cfg = EnvConfig(map_path=a.map)
    name = city.raw.name.replace("_", " ").title()
    if a.list:
        print(f"{name}: {len(city.G)} graph nodes. Places (use the name or the node id):")
        for nm, n, (x, y) in nav.list_places(city):
            print(f"  node {n:>6}   ({x:7.0f}, {y:7.0f})   {nm}")
        return 0
    km = [float(v) for v in a.route_km.replace(",", "-").split("-")]
    rng = np.random.default_rng(a.seed)

    def make_plan(sa, sb):
        if sa is not None and sb is not None:
            (na, la), (nb, lb) = nav.resolve_point(city, sa, "start"), nav.resolve_point(city, sb, "goal")
            return nav.plan_route(city, cfg, na, nb, a.alternatives, start_name=la, goal_name=lb)
        return nav.random_plan(city, cfg, rng, km[0] * 1000, km[-1] * 1000, a.alternatives)

    plan = make_plan(a.start, a.goal) if (a.random or a.start or a.goal or a.out) else None
    if plan is not None:
        print(plan.summary())
        print("  turn by turn: " + "; ".join(f"{m.s:.0f} m {m.text()}" for m in plan.best.maneuvers[:8]))
        print(f"\n  drive it:  astana view --map {a.map} --start-node {plan.start} --goal-node {plan.goal}"
              + (f" --model {a.model}" if a.model else ""))
    elif a.start or a.goal:
        print("No drivable route between these points.")
        return 1

    import matplotlib
    if a.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax, head, info = _figure(city, f"Astana Pilot navigator  ·  {name}")
    artists: List = []
    if a.out:
        info.set_text(_draw_plan(ax, city, plan, artists))
        fig.savefig(a.out, facecolor=BG, dpi=130)
        print(f"[route] picture saved: {a.out}")
        return 0

    state = {"A": plan.start if plan else None, "B": plan.goal if plan else None, "plan": plan, "marks": []}
    hint = "left click = A (start)   right click = B (finish)   ENTER = drive in 3-D   R = random   C = clear   Q = quit"

    def redraw() -> None:
        for m in state["marks"]:
            m.remove()
        state["marks"].clear()
        txt = _draw_plan(ax, city, state["plan"], artists)
        if state["plan"] is None:
            for key, col in (("A", A_COL), ("B", B_COL)):
                if state[key] is not None:
                    x, y = nav.node_xy(city, state[key])
                    state["marks"].append(ax.scatter([x], [y], s=260, color=col, edgecolors="white", linewidths=2, zorder=7))
                    state["marks"].append(ax.text(x, y, key, color="white", fontsize=10, fontweight="bold", ha="center",
                                                  va="center", zorder=8))
            txt = "Choose the start A and the goal B on the map."
        info.set_text(txt + "\n" + hint)
        fig.canvas.draw_idle()

    def replan() -> None:
        state["plan"] = None
        if state["A"] is not None and state["B"] is not None:
            p = nav.plan_route(city, cfg, state["A"], state["B"], a.alternatives)
            if p is None:
                info.set_text("No drivable route between these points (one-way streets?). Pick other points.\n" + hint)
                fig.canvas.draw_idle()
                state["B"] = None
                redraw()
                return
            state["plan"] = p
            print(p.summary(), flush=True)
        redraw()

    def on_click(ev) -> None:
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if ev.button == 3 or (ev.button == 1 and state["A"] is not None and state["B"] is None):
            state["B"] = nav.nearest_node(city, ev.xdata, ev.ydata, "goal")
        elif ev.button == 1:
            state["A"], state["B"] = nav.nearest_node(city, ev.xdata, ev.ydata, "start"), None
        replan()

    def on_key(ev) -> None:
        if ev.key in ("q", "escape"):
            plt.close(fig)
        elif ev.key == "c":
            state.update(A=None, B=None, plan=None)
            redraw()
        elif ev.key == "r":
            p = nav.random_plan(city, cfg, rng, km[0] * 1000, km[-1] * 1000, a.alternatives)
            state.update(A=p.start if p else None, B=p.goal if p else None, plan=p)
            if p:
                print(p.summary(), flush=True)
            redraw()
        elif ev.key == "enter" and state["plan"] is not None:
            p = state["plan"]
            cmd = [sys.executable, "-m", "astana_pilot", "view", "--map", a.map, "--start-node", str(p.start),
                   "--goal-node", str(p.goal), "--quality", a.quality] + (["--model", a.model] if a.model else [])
            print("[route] launching:", " ".join(cmd[2:]), flush=True)
            subprocess.Popen(cmd)
            info.set_text("3-D drive started in a new window (first start: ~10 s to build the scene).\n" + hint)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    try:
        fig.canvas.manager.set_window_title("Astana Pilot - navigator")
    except Exception:
        pass
    plt.rcParams["keymap.quit"] = []
    redraw()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
