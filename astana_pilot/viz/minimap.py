"""Raster mini-map of a whole city (north up) for the HUD - roads per level, buildings, parks, water."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..env.city_map import LAYER_BRIDGE, LAYER_TUNNEL, CityMap

GROUND = (16, 22, 34, 255)
PARK = (26, 54, 44, 255)
WATER = (24, 52, 86, 255)
BUILD = (40, 52, 76, 255)
ROAD = (112, 124, 146, 255)
FLY = (255, 168, 72, 255)
TUN = (96, 170, 255, 255)


def render_minimap(city: CityMap, mpp: float = 2.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """``mpp`` = metres per pixel. Returns (RGBA uint8 image, info with x0/y0/res of the image)."""
    x0, y0, x1, y1 = city.bounds
    pad = 40.0
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    W, H = int((x1 - x0) / mpp) + 1, int((y1 - y0) / mpp) + 1
    img = Image.new("RGBA", (W, H), GROUND)
    d = ImageDraw.Draw(img)

    def px(p):
        return ((p[0] - x0) / mpp, H - (p[1] - y0) / mpp)

    for ring in getattr(city.raw, "green", []) or []:
        d.polygon([px(p) for p in ring], fill=PARK)
    for ring in getattr(city.raw, "water", []) or []:
        d.polygon([px(p) for p in ring], fill=WATER)
    for ring, _ in city.raw.buildings:
        d.polygon([px(p) for p in ring], fill=BUILD)
    # roads: draw as thick polylines per level (ground first, then tunnels, then flyovers on top)
    order = [(0, ROAD), (LAYER_TUNNEL, TUN), (LAYER_BRIDGE, FLY)]
    for layer, col in order:
        seen = set()
        for i, e in enumerate(city.raw.edges):
            if layer not in city.edge_layers[i] or (layer != 0 and city.edge_layers[i] == {0}):
                continue
            if layer == 0 and city.edge_layers[i] != {0}:
                continue
            key = (min(e.u, e.v), max(e.u, e.v), len(e.xy))
            if key in seen:
                continue
            seen.add(key)
            w = max(2, int(2 * city.edge_hw[i] / mpp))
            d.line([px(p) for p in e.xy], fill=col, width=w, joint="curve")
    return np.asarray(img), {"x0": x0, "y0": y0, "res": mpp}
