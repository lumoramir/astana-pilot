"""Render a PNG overview of a baked map with sample routes - check the area BEFORE training.

    python -m astana_pilot.tools.preview_map --map assets/maps/astana_expo.pkl --out assets/map_preview.png
    python -m astana_pilot.tools.preview_map --map synthetic --out /tmp/synthetic.png
"""
from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
from matplotlib.collections import PolyCollection                      # noqa: E402

from astana_pilot.env.city_map import CityMap, load_city_map                        # noqa: E402
from astana_pilot.env.config import EnvConfig                                       # noqa: E402
from astana_pilot.env.route import build_reference_path                             # noqa: E402

# Known landmark (GPS from public sources) drawn on the preview to orient yourself.
LANDMARKS = {"MEGA Silk Way": (51.0889, 71.4072)}


def latlon_to_local(lat: float, lon: float, origin: tuple) -> tuple:
    """Equirectangular approximation (error < 1 m over a few km) for labelling only."""
    return ((lon - origin[1]) * math.cos(math.radians(origin[0])) * 111_320.0,
            (lat - origin[0]) * 110_540.0)


def render_preview(city: CityMap, out_path: str, n_routes: int = 4, seed: int = 1) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
    ax.set_facecolor("#0b0f1e")
    H, W = city.grid.shape
    ext = (city.x0, city.x0 + (W - 1) * city.res, city.y0, city.y0 + (H - 1) * city.res)
    ax.imshow(np.where(city.grid > 0, 1.0, 0.0), origin="lower", extent=ext, cmap="gray",
              vmin=-0.6, vmax=1.6, interpolation="nearest")
    if city.raw.buildings:
        ax.add_collection(PolyCollection([b[0] for b in city.raw.buildings], facecolors="#3a4b78",
                                         edgecolors="#6d84c4", linewidths=0.3, alpha=0.85))
    rng, cfg = np.random.default_rng(seed), EnvConfig(map_path="synthetic")
    colors = ["#00f0ff", "#ff00a0", "#ffdc00", "#3cff8c"]
    done = 0
    while done < n_routes:
        nodes = city.sample_route(rng, 300, 700)
        ref = build_reference_path(city, nodes, cfg) if nodes else None
        if ref is None:
            continue
        ax.plot(ref.xy[:, 0], ref.xy[:, 1], color=colors[done % 4], lw=2.0)
        ax.plot(*ref.xy[0], "o", color=colors[done % 4], ms=7)
        ax.plot(*ref.xy[-1], "*", color=colors[done % 4], ms=13)
        done += 1
    origin = city.raw.center_latlon
    if origin != (0.0, 0.0):
        for name, (la, lo) in LANDMARKS.items():
            x, y = latlon_to_local(la, lo, origin)
            ax.plot(x, y, "r^", ms=10)
            ax.annotate(name, (x, y), color="white", fontsize=9, xytext=(6, 6), textcoords="offset points")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_title(f"{city.raw.name}: {len(city.G)} junctions, {city.G.number_of_edges()} road links, "
                 f"{len(city.raw.buildings)} buildings   (white = drivable)", color="white")
    ax.tick_params(colors="white"); fig.patch.set_facecolor("#0b0f1e")
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--out", default="assets/map_preview.png")
    a = ap.parse_args()
    render_preview(load_city_map(a.map), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
