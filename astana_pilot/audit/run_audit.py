"""Road-safety audit of a city district with the driving agent as a *probe* (surrogate safety measures).

    astana audit --map assets/maps/astana_expo.pkl --episodes 120
    astana audit --map assets/maps/astana_expo.pkl --model runs/night1/final_model.zip --variants "baseline;limit0.8;green30"

Idea. Real crash statistics need years of data and, by definition, people getting hurt. Traffic engineers therefore
study *surrogate safety measures* (SSMs): near-misses that are observable long before a crash - emergency braking,
time-to-collision below a threshold, hard braking, a pedestrian passing very close, red-light conflicts.  The simulator
already produces all of them for a driver that is steered by the same rules everywhere, so we can:

1. drive thousands of random trips through a district built from OpenStreetMap (a few minutes on a laptop),
2. log every near-miss with its position and normalise by exposure (metres driven through that place),
3. rank the hot-spots (nearest junction / street, lat-lon for GIS), draw a risk heat-map,
4. compare *what-if* variants of the infrastructure (speed limits, signal green time) on identical trips.

It does not predict crashes - it shows *where an autonomous (or careful human) driver is under the most stress*, which
is where an engineer should look first. See docs/IMPACT.md for what it can and cannot claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EVENT_WEIGHT = {"collision": 10.0, "aeb": 3.0, "ttc": 2.0, "red_light": 4.0, "yellow_light": 2.0, "stop_sign": 4.0, "yield": 3.0,
                "zebra": 5.0, "ped_close": 3.0, "hard_brake": 1.0, "phantom_brake": 0.7}
EVENTS = tuple(EVENT_WEIGHT)
CELL = 25.0                       # [m] grid cell of the hot-spot map


# ------------------------------------------------------------------------------------------ variants
def apply_variant(city, name: str, env=None) -> str:
    """Mutate the (process-local, cached) city for a what-if scenario. Returns a human label."""
    name = name.strip()
    if name in ("", "baseline"):
        return "baseline"
    if name.startswith("limit"):                       # limit0.8 -> every speed limit x 0.8
        k = float(name[5:])
        city.edge_vlim = [v * k for v in city.edge_vlim]
        for s in city.infra.signs:
            s.value *= k
        return f"speed limits x{k:g}"
    if name.startswith("green"):                       # green30 -> 30 s of green per phase
        g = float(name[5:])
        city.infra.params = replace(city.infra.params, light_green=g)
        return f"signal green {g:g} s"
    if name in ("aggr", "calm"):                       # driving culture of the OTHER road users (needs the env)
        if env is not None:
            wp = env.world.wp
            env.world.wp = replace(wp, speed_factor=(1.0, 1.3), idm_headway=(0.6, 1.1), assertive_prob=0.45, jaywalk_prob=0.7) \
                if name == "aggr" else replace(wp, speed_factor=(0.75, 0.95), idm_headway=(1.6, 2.4), assertive_prob=0.05, jaywalk_prob=0.1)
        return "aggressive road users" if name == "aggr" else "calm road users"
    raise ValueError(f"unknown variant '{name}' (use baseline, limit<k>, green<seconds>, aggr, calm)")


# ------------------------------------------------------------------------------------------ one episode
def _episode(job: Tuple[str, str, str, int, float]) -> Dict:
    from ..env import DrivingEnv, EnvConfig
    from ..viz.policies import HeuristicPolicy, ModelPolicy

    map_path, policy_spec, variant, seed, difficulty = job
    env = DrivingEnv(EnvConfig(map_path=map_path))
    label = apply_variant(env.city, variant, env)
    env.set_difficulty(difficulty)
    pol = HeuristicPolicy(env.cfg) if policy_spec == "expert" else ModelPolicy(policy_spec)
    obs, _ = env.reset(seed=seed)
    if hasattr(pol, "reset"):
        pol.reset()
    dt = env.cfg.dt
    ev: List[Tuple[str, float, float, float]] = []          # (type, x, y, t)
    expo: List[Tuple[float, float, float]] = []              # (x, y, metres)
    last = defaultdict(lambda: -99.0)
    prev_aeb, slow_run, prev_v = False, 0, 0.0
    prev_slow = 0
    done, info, t = False, {}, 0.0
    while not done:
        a = pol.act(obs)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        t += dt
        v = env.vehicle
        p = env._perc
        x, y = float(v.x), float(v.y)
        expo.append((x, y, abs(v.v) * dt))

        def emit(kind: str, cool: float = 3.0) -> None:
            if t - last[kind] >= cool:
                last[kind] = t
                ev.append((kind, x, y, t))

        if info.get("aeb") and not prev_aeb:
            emit("aeb", 1.0)
        prev_aeb = bool(info.get("aeb"))
        if p is not None and v.v > 1.0 and p.t_conflict < 1.5:
            emit("ttc")
        for k, val in info.get("violations", {}).items():
            if k in EVENT_WEIGHT and val > 0:
                emit(k, 0.5)
        if (v.v - prev_v) / dt < -4.5:
            emit("hard_brake")
        prev_v = v.v
        if p is not None and v.v > 2.0 and len(p.circles):
            ped = p.circles[p.circle_kind == 4]
            if len(ped):
                d = np.hypot(ped[:, 0] - x, ped[:, 1] - y) - ped[:, 2] - 1.0
                if float(d.min()) < 1.2:
                    emit("ped_close")
        slow = env._unneeded_slow
        slow_run = slow_run + 1 if slow > prev_slow else 0     # consecutive steps of crawling for no reason
        prev_slow = slow
        if slow_run >= 10:
            emit("phantom_brake", 6.0)
    if info.get("collision"):
        ev.append(("collision", float(env.vehicle.x), float(env.vehicle.y), t))
    return {"seed": seed, "variant": variant, "label": label, "success": bool(info.get("is_success")),
            "collision": bool(info.get("collision")), "km": float(sum(m for _, _, m in expo) / 1000.0),
            "time": t, "events": ev, "expo": np.array(expo, dtype=np.float32)}


# ------------------------------------------------------------------------------------------ aggregation
class Grid:
    def __init__(self, bounds, cell: float = CELL) -> None:
        x0, y0, x1, y1 = bounds
        self.x0, self.y0, self.cell = x0 - 20, y0 - 20, cell
        self.nx, self.ny = int((x1 - x0 + 40) / cell) + 1, int((y1 - y0 + 40) / cell) + 1
        self.ev = {k: np.zeros((self.ny, self.nx)) for k in EVENTS}
        self.expo = np.zeros((self.ny, self.nx))

    def idx(self, x, y):
        return (np.clip(((np.asarray(y) - self.y0) / self.cell).astype(int), 0, self.ny - 1),
                np.clip(((np.asarray(x) - self.x0) / self.cell).astype(int), 0, self.nx - 1))

    def add(self, res: Dict) -> None:
        for kind, x, y, _ in res["events"]:
            iy, ix = self.idx(x, y)
            self.ev[kind][iy, ix] += 1
        e = res["expo"]
        if len(e):
            iy, ix = self.idx(e[:, 0], e[:, 1])
            np.add.at(self.expo, (iy, ix), e[:, 2])

    def score(self) -> np.ndarray:
        return sum(EVENT_WEIGHT[k] * self.ev[k] for k in EVENTS)


def hotspots(grid: Grid, city, top: int = 15, min_exposure: float = 300.0) -> List[Dict]:
    """Rank cells by weighted events per km driven (cells with little exposure are ignored), merge neighbours."""
    from scipy.ndimage import maximum_filter
    sc = grid.score()
    nz = grid.expo[grid.expo > 0]
    if len(nz):                                  # adaptive: ignore the least-visited half of the driven cells
        min_exposure = min(min_exposure, max(60.0, float(np.percentile(nz, 50))))
    rate = np.where(grid.expo >= min_exposure, sc / np.maximum(grid.expo / 1000.0, 1e-6), 0.0)
    rate = np.where(sc >= 2.0, rate, 0.0)
    loc_max = (rate == maximum_filter(rate, size=5)) & (rate > 0)
    cand = np.argwhere(loc_max)
    cand = sorted(cand, key=lambda ij: -rate[ij[0], ij[1]])[:top]
    raw = city.raw
    nodes = np.array([raw.nodes[n] for n in city.G.nodes]) if len(city.G) else np.zeros((0, 2))
    ids = list(city.G.nodes)
    out = []
    from ..tools.osm_builder import Projector
    proj = Projector(*raw.center_latlon) if raw.center_latlon != (0.0, 0.0) else None
    for rank, (iy, ix) in enumerate(cand, 1):
        x, y = grid.x0 + (ix + 0.5) * grid.cell, grid.y0 + (iy + 0.5) * grid.cell
        # nearest street name and junction
        best, name = 1e9, ""
        for e in raw.edges:
            if not e.name:
                continue
            d = float(np.min(np.hypot(e.xy[:, 0] - x, e.xy[:, 1] - y)))
            if d < best:
                best, name = d, e.name
        jn = ""
        if len(nodes):
            d = np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)
            k = int(d.argmin())
            deg = len(set(city.G.successors(ids[k])) | set(city.G.predecessors(ids[k])))
            if d[k] < 35 and deg >= 3:
                jn = f"junction {ids[k]} ({'signalised' if ids[k] in city.infra.signals else 'priority'})"
        lat = lon = None
        if proj is not None:
            la, lo = proj.inv(x, y)
            lat, lon = float(la), float(lo)
        out.append({"rank": rank, "x": float(x), "y": float(y), "lat": lat, "lon": lon, "street": name, "where": jn,
                    "score": float(sc[iy, ix]), "per_km": float(rate[iy, ix]), "exposure_m": float(grid.expo[iy, ix]),
                    **{k: int(grid.ev[k][iy, ix]) for k in EVENTS}})
    return out


def draw_heatmap(grid: Grid, city, spots: List[Dict], path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from scipy.ndimage import gaussian_filter

    sc = gaussian_filter(grid.score(), 1.2)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=130)
    fig.patch.set_facecolor("#0b0f1e")
    ax.set_facecolor("#0b0f1e")
    for e in city.raw.edges:
        ax.plot(e.xy[:, 0], e.xy[:, 1], color="#4b5870", lw=0.5, zorder=1)
    if city.raw.buildings:
        ax.add_collection(PolyCollection([b[0] for b in city.raw.buildings], facecolors="#1a2440", edgecolors="none", zorder=0))
    ext = (grid.x0, grid.x0 + grid.nx * grid.cell, grid.y0, grid.y0 + grid.ny * grid.cell)
    masked = np.ma.masked_less(sc, 0.05)
    im = ax.imshow(masked, origin="lower", extent=ext, cmap="inferno", alpha=0.85, zorder=2, interpolation="bilinear",
                   norm=matplotlib.colors.PowerNorm(0.6, vmin=0.05, vmax=max(float(sc.max()), 0.1)))
    for s in spots[:10]:
        ax.plot(s["x"], s["y"], "o", ms=15, mfc="none", mec="#38dcff", mew=2, zorder=3)
        ax.annotate(str(s["rank"]), (s["x"], s["y"]), color="white", ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title(title, color="white")
    ax.tick_params(colors="#8aa")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("weighted near-miss density", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.get_yticklabels(), color="white")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def geojson(spots: List[Dict]) -> Dict:
    feats = []
    for s in spots:
        if s["lat"] is None:
            continue
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                      "properties": {k: v for k, v in s.items() if k not in ("lat", "lon", "x", "y")}})
    return {"type": "FeatureCollection", "features": feats}


# ------------------------------------------------------------------------------------------ main
def run(map_path: str, policy: str, variants: Sequence[str], episodes: int, difficulty: float, workers: int, out: str,
        seed0: int = 20_000_000) -> Dict:
    from ..env.city_map import load_city_map
    city = load_city_map(map_path)
    results: Dict[str, List[Dict]] = {}
    labels: Dict[str, str] = {}
    t0 = time.time()
    for v in variants:
        jobs = [(map_path, policy, v, seed0 + i, difficulty) for i in range(episodes)]
        uses_torch = policy != "expert"
        if workers <= 1:
            res = [_episode(j) for j in jobs]
        else:
            from ..training.demos import _init_worker
            with mp.get_context("spawn").Pool(min(workers, 5) if uses_torch else workers, initializer=_init_worker,
                                              initargs=(uses_torch,)) as pool:
                res = pool.map(_episode, jobs, chunksize=2)
        results[v] = res
        labels[v] = res[0]["label"]
        print(f"[audit] variant '{v}' ({labels[v]}): {len(res)} trips, {sum(r['km'] for r in res):.1f} km, "
              f"{time.time() - t0:.0f} s", flush=True)

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict] = {}
    per_trip: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for v, res in results.items():
        km = sum(r["km"] for r in res)
        ev = defaultdict(int)
        for r in res:
            for k, *_ in r["events"]:
                ev[k] += 1
        wt = np.array([sum(EVENT_WEIGHT[k] for k, *_ in r["events"]) for r in res], dtype=float)
        kms = np.array([r["km"] for r in res], dtype=float)
        per_trip[v] = (wt, kms)
        summary[v] = {"label": labels[v], "trips": len(res), "km": km, "success": float(np.mean([r["success"] for r in res])),
                      "collisions": int(sum(r["collision"] for r in res)),
                      "mean_trip_min": float(np.mean([r["time"] for r in res]) / 60.0),
                      "events": {k: int(ev[k]) for k in EVENTS}, "events_per_100km": {k: 100.0 * ev[k] / max(km, 1e-6) for k in EVENTS},
                      "weighted_per_100km": float(wt.sum() * 100.0 / max(km, 1e-6))}
    # bootstrap over trips (same seeds in every variant -> paired): 95 % intervals of the weighted rate and of the change
    rng = np.random.default_rng(0)
    n = len(next(iter(per_trip.values()))[0])
    idx = rng.integers(0, n, size=(500, n))
    base_wt, base_km = per_trip[variants[0]]
    base_boot = base_wt[idx].sum(1) / np.maximum(base_km[idx].sum(1), 1e-9) * 100.0
    for v, (wt, kms) in per_trip.items():
        boot = wt[idx].sum(1) / np.maximum(kms[idx].sum(1), 1e-9) * 100.0
        summary[v]["weighted_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if v != variants[0]:
            d = boot - base_boot
            summary[v]["change_vs_baseline"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    base_key = variants[0]
    grid = Grid(city.bounds)
    for r in results[base_key]:
        grid.add(r)
    spots = hotspots(grid, city)
    title = f"Near-miss density - {city.raw.name} - {policy} driver, {len(results[base_key])} trips"
    draw_heatmap(grid, city, spots, str(outdir / "risk_heatmap.png"), title)
    with open(outdir / "hotspots.csv", "w", newline="", encoding="utf-8") as f:
        if spots:
            w = csv.DictWriter(f, fieldnames=list(spots[0].keys()))
            w.writeheader()
            w.writerows(spots)
    (outdir / "hotspots.geojson").write_text(json.dumps(geojson(spots), indent=1, ensure_ascii=False), encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    report = make_report(city.raw.name, policy, summary, spots, base_key)
    (outdir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[audit] wrote {outdir}/ (report.md, risk_heatmap.png, hotspots.csv, hotspots.geojson, summary.json)")
    return summary


def make_report(name: str, policy: str, summary: Dict[str, Dict], spots: List[Dict], base: str) -> str:
    lines = [f"# Road-safety audit: {name}", "",
             f"Probe driver: **{policy}**. Events are surrogate safety measures (near-misses), not crashes.", "",
             "## What-if comparison (identical trips)", "",
             "| variant | trips | km | success | collisions | mean trip (min) | weighted events / 100 km |", "|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        lo, hi = s["weighted_ci95"]
        chg = ""
        if "change_vs_baseline" in s:
            c = s["change_vs_baseline"]
            sig = "significant" if (c["ci95"][0] > 0 or c["ci95"][1] < 0) else "not significant"
            chg = f" (change {c['mean']:+.0f}, 95% CI {c['ci95'][0]:+.0f}..{c['ci95'][1]:+.0f}: {sig})"
        lines.append(f"| {s['label']} | {s['trips']} | {s['km']:.1f} | {100 * s['success']:.0f}% | {s['collisions']} | "
                     f"{s['mean_trip_min']:.1f} | {s['weighted_per_100km']:.0f} [{lo:.0f}..{hi:.0f}]{chg} |")
    lines += ["", "## Events per 100 km", "", "| variant | " + " | ".join(EVENTS) + " |", "|---|" + "---|" * len(EVENTS)]
    for v, s in summary.items():
        lines.append(f"| {s['label']} | " + " | ".join(f"{s['events_per_100km'][k]:.1f}" for k in EVENTS) + " |")
    lines += ["", f"## Top hot-spots ({summary[base]['label']})", "",
              "| # | street | location | events/km | AEB | TTC<1.5s | violations | ped. close | lat, lon |", "|---|---|---|---|---|---|---|---|---|"]
    for s in spots[:12]:
        viol = sum(s[k] for k in ("red_light", "yellow_light", "stop_sign", "yield", "zebra"))
        ll = f"{s['lat']:.5f}, {s['lon']:.5f}" if s["lat"] is not None else "-"
        lines.append(f"| {s['rank']} | {s['street'] or '-'} | {s['where'] or 'mid-block'} | {s['per_km']:.1f} | {s['aeb']} | "
                     f"{s['ttc']} | {viol} | {s['ped_close']} | {ll} |")
    lines += ["", "*Method:* weights collision 10, zebra / red-light / stop violations 4-5, AEB 3, pedestrian < 1.2 m 3, "
              "TTC < 1.5 s 2, hard braking 1, needless crawling 0.7; rate = weighted events per km driven through a "
              f"{CELL:.0f} m cell (cells with < 300 m of exposure ignored)."]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--model", default=None, help="PPO .zip (default: the rule-based expert)")
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--variants", default="baseline", help='";"-separated what-if scenarios: baseline;limit0.8;green30')
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or f"reports/audit_{Path(a.map).stem}"
    run(a.map, a.model or "expert", [v.strip() for v in a.variants.split(";")], a.episodes, a.difficulty, a.workers, out)


if __name__ == "__main__":
    main()
