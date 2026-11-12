"""Download a real district from OpenStreetMap and bake it for the simulator (works for ANY city).

    python -m astana_pilot.tools.fetch_astana_map                       # default preset (see PRESETS)
    python -m astana_pilot.tools.fetch_astana_map --preset expo
    python -m astana_pilot.tools.fetch_astana_map --center 51.1285,71.4380 --half-size 1400 --name my_district
    python -m astana_pilot.tools.fetch_astana_map --list-presets

What comes out (``assets/maps/<name>.pkl``): directed road edges with lane counts / limits, bridge + tunnel height
profiles with ramps, traffic-signal junctions, mapped crossings, building footprints with heights, parks, water, trees,
landmarks, and one signed-distance field **per road level**. A PNG preview is written next to it.

Needs internet (free Overpass API; requests only - no osmnx). Responses are cached in ``assets/osm_cache``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .osm_builder import build_rawmap, describe, fetch_osm

# name -> (lat, lon, half_size_m, drive_side, description)
PRESETS = {
    "expo": (51.0895, 71.4150, 1400.0, "right",
             "EXPO-2017 / Nur Alem sphere + MEGA Silk Way: wide avenues, ring junctions, a flyover, service tunnels"),
    "nurzhol": (51.1285, 71.4380, 1400.0, "right",
                "Government quarter: Bayterek, Akorda, Nurzhol boulevard, Ishim embankment bridges"),
}
DEFAULT_PRESET = "expo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", type=str, default=None, help="named district (see --list-presets)")
    ap.add_argument("--list-presets", action="store_true")
    ap.add_argument("--center", type=str, default=None, help="lat,lon of the area centre (overrides the preset)")
    ap.add_argument("--half-size", type=float, default=None, help="half side of the square [m]")
    ap.add_argument("--name", type=str, default=None, help="map name (file name)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--res", type=float, default=0.5, help="raster resolution for the SDF build [m]")
    ap.add_argument("--curb-radius", type=float, default=2.5, help="kerb fillet radius at junctions [m]")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()

    if a.list_presets:
        for k, (la, lo, hs, side, desc) in PRESETS.items():
            print(f"  {k:16s} {la:.4f},{lo:.4f}  ±{hs:.0f} m  {side:5s}  {desc}")
        return

    preset = a.preset or (DEFAULT_PRESET if a.center is None else None)
    if preset is not None:
        lat, lon, half, side, _ = PRESETS[preset]
    else:
        lat, lon = (float(t) for t in a.center.split(","))
        half, side = 1400.0, "right"
    if a.center:
        lat, lon = (float(t) for t in a.center.split(","))
    half = a.half_size or half
    side = "right"                      # left-hand traffic (Hong Kong) is on the roadmap, see docs/IMPACT.md
    name = a.name or (preset or "custom").replace("-", "_")
    name = name if name.startswith(("astana", "hk")) or a.name else f"astana_{name}"
    out = Path(a.out or f"assets/maps/{name}.pkl")

    print(f"[1/5] downloading OpenStreetMap data around ({lat:.5f}, {lon:.5f}), ±{half:.0f} m ...")
    osm = fetch_osm(lat, lon, half, cache=not a.no_cache)
    print("[2/5] building the road graph (ways -> junctions -> height profiles -> ramps) ...")
    raw = build_rawmap(osm, (lat, lon), half, name)
    raw.drive_side = side
    info = describe(raw)
    print("      " + json.dumps(info, ensure_ascii=False))

    print("[3/5] rasterising roads and computing one signed-distance field per level ...")
    from ..env.city_map import CityMap
    t0 = time.time()
    city = CityMap.build(raw, res_build=a.res, curb_radius=a.curb_radius)
    print(f"      layers {city.layers}, grid {city.grid.shape[1]}x{city.grid.shape[0]} @ {city.res:.1f} m  ({time.time() - t0:.0f} s)")

    print("[4/5] saving ...")
    city.save(out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if not a.no_preview:
        print("[5/5] preview ...")
        from .preview_map import render_preview
        png = out.with_suffix(".png")
        render_preview(city, str(png))
        print(f"      {png}")


if __name__ == "__main__":
    main()
