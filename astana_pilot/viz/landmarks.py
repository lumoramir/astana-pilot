"""Hand-made 3-D models of Astana landmarks (procedural, a few hundred triangles each).

A landmark is placed where OpenStreetMap says it is (name match in ``raw.landmarks``) or, if the map window contains
the known GPS position, there. The generic OSM extrusion of that building is skipped (see ``build_buildings``).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.city_map import CityMap
from .mesh import MeshBuilder

# (lat, lon) fallbacks - public GPS positions of the well-known sites
GPS = {
    "nur_alem": (51.08935, 71.41625),
    "bayterek": (51.12830, 71.43052),
    "khan_shatyr": (51.13265, 71.40360),
    "akorda": (51.12525, 71.44527),
}


def _latlon_to_xy(city: CityMap, lat: float, lon: float) -> Optional[Tuple[float, float]]:
    la0, lo0 = city.raw.center_latlon
    if la0 == 0.0 and lo0 == 0.0:
        return None
    x = (lon - lo0) * math.cos(math.radians(la0)) * 111_320.0
    y = (lat - la0) * 111_132.0
    x0, y0, x1, y1 = city.bounds
    if not (x0 - 100 < x < x1 + 100 and y0 - 100 < y < y1 + 100):
        return None
    return float(x), float(y)


def _circular_buildings(city: CityMap) -> List[Tuple[float, np.ndarray, float]]:
    """(circularity, centre, radius) of big round footprints - the Nur Alem sphere is one."""
    out = []
    for ring, h in city.raw.buildings:
        r = np.asarray(ring)
        if len(r) < 8:
            continue
        a = abs(0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1])))
        p = float(np.sum(np.linalg.norm(np.diff(np.vstack([r, r[:1]]), axis=0), axis=1)))
        if a < 1200 or a > 9000 or p < 1:
            continue
        circ = 4 * math.pi * a / (p * p)
        if circ > 0.80:
            out.append((circ, r.mean(axis=0), math.sqrt(a / math.pi)))
    return out


# ------------------------------------------------------------------------------------------ models
def nur_alem(B: Dict[str, MeshBuilder], c: Tuple[float, float], radius: float = 38.0) -> None:
    """EXPO-2017 Kazakhstan pavilion: an 80 m glass sphere on a podium, with steel ribs."""
    cx, cz = c
    glass, solid = B["glass"], B["landmark"]
    # podium + plinth
    solid.cylinder((cx, 0.0, cz), radius * 0.92, 5.0, 40, (0.86, 0.86, 0.88, 1.0), top_color=(0.8, 0.8, 0.82, 1.0))
    solid.cylinder((cx, 5.0, cz), radius * 0.55, 8.0, 32, (0.75, 0.77, 0.80, 1.0), r_top=radius * 0.42)
    cy = 11.0 + radius
    # inner core and floors (what you see through the glass)
    solid.sphere((cx, cy, cz), radius * 0.30, 16, 10, (0.92, 0.94, 0.97, 1.0))
    for k in range(5):
        yy = cy - radius * 0.6 + k * radius * 0.3
        rr = math.sqrt(max(radius ** 2 * 0.80 ** 2 - (yy - cy) ** 2, 1.0))
        solid.cylinder((cx, yy, cz), rr * 0.9, 0.5, 28, (0.85, 0.88, 0.93, 1.0), caps=True)
    # glass shell (translucent, bluish)
    glass.sphere((cx, cy, cz), radius, 32, 20, (0.55, 0.75, 0.95, 0.30))
    # steel ribs: meridians + parallels as thin shell strips slightly outside the glass
    rib = (0.93, 0.94, 0.96, 1.0)
    for m in range(24):
        a0 = 2 * math.pi * m / 24
        pts = []
        for lat in np.linspace(-1.2, 1.2, 25):
            pts.append((cx + (radius + 0.25) * math.cos(lat) * math.cos(a0), cy + (radius + 0.25) * math.sin(lat),
                        cz + (radius + 0.25) * math.cos(lat) * math.sin(a0)))
        for p, q in zip(pts[:-1], pts[1:]):
            _strip(solid, p, q, 0.28, rib)
    for lat in np.linspace(-1.1, 1.1, 9):
        r = (radius + 0.25) * math.cos(lat)
        yy = cy + (radius + 0.25) * math.sin(lat)
        solid.cylinder((cx, yy - 0.12, cz), r, 0.24, 36, rib, caps=False)


def _strip(b: MeshBuilder, p, q, w: float, col) -> None:
    """Thin vertical-ish rib segment from p to q (two crossed quads)."""
    p, q = np.array(p), np.array(q)
    d = q - p
    side = np.cross(d, [0, 1, 0])
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side) * w / 2
    b.quad(p - side, p + side, q + side, q - side, col, two_sided=True)


def bayterek(B: Dict[str, MeshBuilder], c: Tuple[float, float]) -> None:
    """Baiterek: 97 m tower, golden sphere on top."""
    cx, cz = c
    s = B["landmark"]
    white = (0.93, 0.94, 0.96, 1.0)
    s.cylinder((cx, 0.0, cz), 14.0, 1.2, 28, (0.80, 0.80, 0.82, 1.0))
    # tapered trunk: three tulip-like branches approximated by a cone + 6 leaning pylons
    s.cylinder((cx, 1.2, cz), 9.0, 22.0, 24, white, r_top=4.5)
    s.cylinder((cx, 23.2, cz), 4.5, 30.0, 24, white, r_top=3.0)
    for k in range(6):
        a = 2 * math.pi * k / 6
        _strip(s, (cx + 9 * math.cos(a), 1.2, cz + 9 * math.sin(a)), (cx + 3 * math.cos(a), 46.0, cz + 3 * math.sin(a)), 1.6, (0.85, 0.87, 0.92, 1.0))
    s.cylinder((cx, 53.2, cz), 3.0, 10.0, 20, white, r_top=2.2)
    s.sphere((cx, 76.0, cz), 11.0, 28, 16, (0.98, 0.80, 0.30, 1.0))               # the golden egg
    for k in range(10):                                                            # sphere ribs
        a = 2 * math.pi * k / 10
        pts = [(cx + 11.15 * math.cos(l) * math.cos(a), 76.0 + 11.15 * math.sin(l), cz + 11.15 * math.cos(l) * math.sin(a))
               for l in np.linspace(-1.3, 1.3, 13)]
        for p, q in zip(pts[:-1], pts[1:]):
            _strip(s, p, q, 0.3, (0.95, 0.9, 0.7, 1.0))
    s.cylinder((cx, 87.0, cz), 0.5, 10.0, 8, (0.9, 0.9, 0.92, 1.0), r_top=0.05)    # spire


def khan_shatyr(B: Dict[str, MeshBuilder], c: Tuple[float, float]) -> None:
    """Khan Shatyr: a 150 m translucent tent."""
    cx, cz = c
    g, s = B["glass"], B["landmark"]
    s.cylinder((cx, 0.0, cz), 95.0, 6.0, 40, (0.85, 0.85, 0.88, 1.0), caps=True)       # podium (the oval base is huge)
    segs = 40
    # tent profile: concave curve from base radius 95 to apex at 150 m
    prof = [(95.0, 6.0), (80.0, 28.0), (60.0, 55.0), (40.0, 85.0), (22.0, 115.0), (8.0, 140.0), (1.0, 150.0)]
    for (r0, y0), (r1, y1) in zip(prof[:-1], prof[1:]):
        g.cylinder((cx, y0, cz), r0, y1 - y0, segs, (0.92, 0.95, 1.0, 0.55), r_top=r1, caps=False)
    for k in range(segs // 2):                                                         # masts / cables
        a = 2 * math.pi * k / (segs // 2)
        for (r0, y0), (r1, y1) in zip(prof[:-1], prof[1:]):
            _strip(s, (cx + r0 * math.cos(a), y0 + 0.2, cz + r0 * math.sin(a)), (cx + r1 * math.cos(a), y1 + 0.2, cz + r1 * math.sin(a)),
                   0.5, (0.85, 0.88, 0.92, 1.0))
    s.cylinder((cx, 148.0, cz), 0.6, 22.0, 8, (0.8, 0.8, 0.85, 1.0), r_top=0.1)


def akorda(B: Dict[str, MeshBuilder], c: Tuple[float, float], yaw: float = 0.0) -> None:
    """Presidential palace: white block, colonnade, blue dome and golden spire."""
    cx, cz = c
    s = B["landmark"]
    white, gold, blue = (0.94, 0.94, 0.93, 1.0), (0.95, 0.78, 0.30, 1.0), (0.30, 0.55, 0.85, 1.0)
    s.box((cx, 11.0, cz), (70.0, 22.0, 34.0), yaw=yaw, color=white)
    for k in range(-6, 7):                                                             # columns
        dx = k * 5.0
        s.cylinder((cx + dx * math.cos(yaw) + 17.5 * math.sin(yaw), 0.0, cz - dx * math.sin(yaw) + 17.5 * math.cos(yaw)),
                   0.9, 20.0, 10, (0.97, 0.97, 0.96, 1.0), caps=False)
    s.cylinder((cx, 22.0, cz), 15.0, 3.0, 28, white)
    s.sphere((cx, 25.0, cz), 14.0, 28, 12, blue, squash=0.9, y_range=(0.0, 1.0))
    s.cylinder((cx, 36.0, cz), 0.8, 24.0, 8, gold, r_top=0.1)


BUILDERS = {"nur_alem": nur_alem, "bayterek": bayterek, "khan_shatyr": khan_shatyr, "akorda": akorda}
SITE_RADIUS = {"nur_alem": 60.0, "bayterek": 40.0, "khan_shatyr": 110.0, "akorda": 55.0}


def build_landmarks(city: CityMap, B: Dict[str, MeshBuilder]) -> int:
    for name in ("glass", "landmark"):
        B.setdefault(name, MeshBuilder())
    placed = 0
    done = set()
    sites = city.raw.landmark_sites                      # type: ignore[attr-defined]
    # 1) landmarks named in OpenStreetMap
    for m in getattr(city.raw, "landmarks", []) or []:
        k = m.get("kind")
        if k in BUILDERS and k not in done:
            BUILDERS[k](B, tuple(m["xy"]))
            sites.append((tuple(m["xy"]), SITE_RADIUS[k]))
            done.add(k)
            placed += 1
    # 2) the Nur Alem sphere = the most circular big footprint near the middle of the map (EXPO) if not found by name
    if "nur_alem" not in done:
        circ = _circular_buildings(city)
        if circ:
            circ.sort(key=lambda t: -t[0] * 1000 + np.hypot(*t[1]) * 0.1)
            c, ctr, r = circ[0]
            if np.hypot(*ctr) < 900:
                nur_alem(B, (float(ctr[0]), float(ctr[1])), radius=min(max(r, 25.0), 42.0))
                sites.append(((float(ctr[0]), float(ctr[1])), SITE_RADIUS["nur_alem"]))
                done.add("nur_alem")
                placed += 1
    # 3) GPS fallbacks for the other sites if they lie inside the map
    for k, (la, lo) in GPS.items():
        if k in done or (k == "nur_alem" and city.raw.name.startswith("hk")):
            continue
        p = _latlon_to_xy(city, la, lo)
        if p is not None:
            BUILDERS[k](B, p)
            sites.append((p, SITE_RADIUS[k]))
            done.add(k)
            placed += 1
    city.raw.landmark_placed = sorted(done)            # type: ignore[attr-defined]
    return placed
