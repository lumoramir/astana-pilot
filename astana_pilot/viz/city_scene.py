"""Static 3-D city from a :class:`CityMap` - pure NumPy (no graphics library), unit-testable.

Output is a :class:`SceneData`: a set of named *layers* (asphalt, sidewalks, markings, facades, trees, ...) each holding
tile-split meshes (so the renderer can frustum-cull) plus the info the viewer needs for animation (traffic-light heads).

Heights: every road edge carries ``z0 -> z1``. Ground roads sit at ``y = 0.05``, sidewalks at ``0.15`` (with a kerb face),
flyovers get girders / parapets / piers, tunnel trenches get retaining walls + a see-through roof and the terrain gets a
matching hole (the "cut-away" view that lets the committee see the car inside the tunnel).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..env.city_map import LANE_WIDTH, CityMap, _paint_capsules, polyline_length
from ..env.infrastructure import polyline_at, road_rank
from .mesh import Mesh, MeshBuilder, instances, shade, split_into_tiles
from .quality import Quality

Y_GROUND, Y_GREEN, Y_ROAD, Y_MARK, Y_WALK = 0.0, 0.02, 0.05, 0.085, 0.15
ASPHALT = (0.62, 0.62, 0.64, 1.0)          # multiplied with the asphalt texture
CONCRETE = (0.72, 0.72, 0.70, 1.0)
PAINT = (0.93, 0.93, 0.90, 1.0)
PAINT_Y = (0.95, 0.78, 0.12, 1.0)


@dataclass
class Layer:
    name: str
    meshes: List[Tuple[Tuple[int, int], Mesh]] = field(default_factory=list)
    texture: str = "white"                  # key of :mod:`textures` generator
    world_uv: float = 0.0                   # >0: texture coords = world xz / world_uv
    spec: float = 0.0
    glass: float = 0.0                      # texture alpha = window mask
    unlit: bool = False
    transparent: bool = False
    depth_offset: int = 0
    shadows: bool = True                    # casts shadows
    glow_night: float = 0.0                 # extra brightness at night (lamps)


@dataclass
class LightHead:
    """One traffic-light head: the three lamps are 3 groups of ``k`` vertices in the ``signals`` layer."""
    control: int                            # index into infra.controls
    first_vertex: int
    verts_per_lamp: int


@dataclass
class SceneData:
    layers: Dict[str, Layer] = field(default_factory=dict)
    lights: List[LightHead] = field(default_factory=list)
    light_mesh: Optional[Mesh] = None       # un-tiled mesh of all lamp discs (dynamic colours)
    bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    stats: Dict[str, int] = field(default_factory=dict)


# ====================================================================================== helpers
def _edge_z(xy: np.ndarray, z0: float, z1: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    return z0 + (z1 - z0) * (s / max(s[-1], 1e-9))


def _left_normals(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-vertex unit left normals and miter scale (1/cos of half the turning angle, clamped)."""
    d = np.diff(xy, axis=0)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nseg = np.column_stack([-d[:, 1], d[:, 0]])
    n = np.zeros_like(xy)
    n[0], n[-1] = nseg[0], nseg[-1]
    if len(xy) > 2:
        m = nseg[:-1] + nseg[1:]
        ml = np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
        n[1:-1] = m / ml
        cosh = np.clip(np.einsum("ij,ij->i", n[1:-1], nseg[:-1]), 0.35, 1.0)
        scale = np.ones(len(xy))
        scale[1:-1] = 1.0 / cosh
    else:
        scale = np.ones(len(xy))
    return n, scale


def _offset_line(xy: np.ndarray, z: np.ndarray, off: float, dy: float,
                 n: Optional[np.ndarray] = None, sc: Optional[np.ndarray] = None) -> np.ndarray:
    """3-D polyline ``off`` metres to the LEFT of the travel direction at height ``z + dy``."""
    if n is None:
        n, sc = _left_normals(xy)
    p = xy + n * (off * sc)[:, None]
    return np.column_stack([p[:, 0], z + dy, p[:, 1]])


def _resample(xy: np.ndarray, z: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Add vertices so that no segment is longer than ``step`` (keeps ribbons smooth on ramps)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < 1e-6:
        return xy, z
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    t = np.unique(np.r_[t, s])
    return (np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]), np.interp(t, s, z))


def _dedupe_edges(city: CityMap) -> List[int]:
    seen, out = set(), []
    for i, e in enumerate(city.raw.edges):
        key = (min(e.u, e.v), max(e.u, e.v), round(polyline_length(e.xy), 1), len(e.xy))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _hash_color(seed: int, palette) -> Tuple[float, float, float, float]:
    k = (seed * 2654435761) & 0xFFFFFFFF
    c = palette[k % len(palette)]
    return (*c, 1.0)


class _PolyMask:
    """Rasterised polygons (buildings) on a coarse grid - fast 'is this point inside a building' test."""

    def __init__(self, bounds, polys, res: float = 2.0) -> None:
        x0, y0, x1, y1 = bounds
        self.x0, self.y0, self.res = x0, y0, res
        self.W, self.H = int((x1 - x0) / res) + 2, int((y1 - y0) / res) + 2
        img = Image.new("L", (self.W, self.H), 0)
        d = ImageDraw.Draw(img)
        for ring in polys:
            pts = [((p[0] - x0) / res, (p[1] - y0) / res) for p in ring]
            if len(pts) >= 3:
                d.polygon(pts, fill=255)
        self.m = np.asarray(img) > 0

    def inside(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        ix = ((pts[:, 0] - self.x0) / self.res).astype(int)
        iy = ((pts[:, 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = self.m[iy[ok], ix[ok]]
        return out


# ====================================================================================== roads
def _lanes_per_dir(hw: float, twoway: bool) -> int:
    n = max(1, int(round(2.0 * hw / LANE_WIDTH)))
    return max(1, n // 2) if twoway else n


def _dash_quads(xy: np.ndarray, z: np.ndarray, lateral: float, width: float, start: float, end: float,
                on: float, off: float, y: float) -> np.ndarray:
    """Quads (K,4,3) of dashes of a line ``lateral`` m LEFT of the axis, between arc lengths start..end."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if end - start < 0.5:
        return np.zeros((0, 4, 3))
    n, sc = _left_normals(xy)
    if off <= 0.0:                                         # solid line: split every ~6 m for curvature
        a = np.arange(start, end, 6.0)
        b = np.minimum(a + 6.0, end)
    else:
        a = np.arange(start, end - on * 0.5, on + off)
        b = np.minimum(a + on, end)
    if len(a) == 0:
        return np.zeros((0, 4, 3))

    def at(t):
        px, py = np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])
        nx, ny = np.interp(t, s, n[:, 0]), np.interp(t, s, n[:, 1])
        l = np.maximum(np.hypot(nx, ny), 1e-9)
        return px, py, nx / l, ny / l, np.interp(t, s, z)

    ax_, ay_, anx, any_, az = at(a)
    bx_, by_, bnx, bny, bz = at(b)
    h = width / 2.0
    q = np.zeros((len(a), 4, 3))
    # left side first then right side so the quad is CCW seen from above (see MeshBuilder.ribbon)
    for k, (px, py, nx, ny, zz, side) in enumerate(((ax_, ay_, anx, any_, az, +1), (ax_, ay_, anx, any_, az, -1),
                                                    (bx_, by_, bnx, bny, bz, -1), (bx_, by_, bnx, bny, bz, +1))):
        q[:, k, 0] = px + nx * (lateral + side * h)
        q[:, k, 2] = py + ny * (lateral + side * h)
        q[:, k, 1] = zz + y
    return q


def build_roads(city: CityMap, B: Dict[str, MeshBuilder], hole_edges: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
    raw, infra = city.raw, city.infra
    G = city.G
    junction_deg = {n: len({*G.successors(n), *G.predecessors(n)}) for n in G.nodes}
    # zebra / stop-line positions per edge, so that dashes are interrupted around them
    cw_s: Dict[int, List[float]] = {}
    for cw in infra.crosswalks:
        for ei, s in cw.edges.items():
            cw_s.setdefault(ei, []).append(s)

    for i in _dedupe_edges(city):
        e = raw.edges[i]
        hw = city.edge_hw[i]
        two = city.edge_twoway[i]
        xy0 = e.xy
        z0 = _edge_z(xy0, e.z0, e.z1)
        xy, z = _resample(xy0, z0, 14.0 if abs(e.z1 - e.z0) > 0.05 else 1e9)
        if len(xy) < 2:
            continue
        n, sc = _left_normals(xy)
        yb = Y_ROAD
        L3 = _offset_line(xy, z, hw, yb, n, sc)
        R3 = _offset_line(xy, z, -hw, yb, n, sc)
        # asphalt: left/right in travel direction - left first => CCW from above
        B["road"].ribbon(L3, R3, ASPHALT)

        flat_ground = abs(e.z0) < 0.25 and abs(e.z1) < 0.25
        if flat_ground:                                    # round joints: fill the kerb corners at junctions
            for p in (xy[0], xy[-1]):
                k = 14
                a = np.linspace(0, 2 * math.pi, k, endpoint=False)
                ring = np.column_stack([p[0] + hw * np.cos(a), p[1] + hw * np.sin(a)])
                B["road"].polygon(ring, yb + float(z[0]), ASPHALT, _fan(k))

        # ---------------- markings
        L = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
        trim0 = (hw + 4.0) if junction_deg.get(e.u, 0) >= 3 else 1.0
        trim1 = (hw + 4.0) if junction_deg.get(e.v, 0) >= 3 else 1.0
        if L - trim0 - trim1 > 3.0 and e.highway not in ("service", "living_street") or (L > 30 and e.highway == "service"):
            a_, b_ = trim0, L - trim1
            lanes = _lanes_per_dir(hw, two)
            quads = []
            if two:
                if lanes >= 2:
                    quads.append((_dash_quads(xy, z, 0.18, 0.12, a_, b_, 1e9, 0.0, Y_MARK), PAINT_Y))
                    quads.append((_dash_quads(xy, z, -0.18, 0.12, a_, b_, 1e9, 0.0, Y_MARK), PAINT_Y))
                else:
                    quads.append((_dash_quads(xy, z, 0.0, 0.14, a_, b_, 3.0, 6.0, Y_MARK), PAINT))
                for k in range(1, lanes):
                    for sgn in (+1, -1):
                        quads.append((_dash_quads(xy, z, sgn * k * LANE_WIDTH, 0.12, a_, b_, 3.0, 6.0, Y_MARK), PAINT))
            else:
                for k in range(1, lanes):
                    off = (k - lanes / 2.0) * LANE_WIDTH
                    quads.append((_dash_quads(xy, z, off, 0.12, a_, b_, 3.0, 6.0, Y_MARK), PAINT))
            if e.highway not in ("service", "residential", "living_street", "unclassified"):
                for sgn in (+1, -1):
                    quads.append((_dash_quads(xy, z, sgn * (hw - 0.28), 0.14, a_, b_, 1e9, 0.0, Y_MARK), PAINT))
            for q, col in quads:
                if len(q):
                    B["mark"].quads(q, np.asarray(col, dtype=np.float32), None, np.tile([0, 1, 0], (len(q), 1)))

        # ---------------- structures: flyover girders / parapets / piers, tunnel trench + roof
        zmax, zmin = float(z.max()), float(z.min())
        if zmax > 0.4:
            _bridge_structure(B, xy, z, hw, n, sc)
        if zmin < -0.1:
            _tunnel_structure(B, xy, z, hw, n, sc)
            hole_edges.append((xy, z, hw))


def _fan(k: int):
    return [(0, j, j + 1) for j in range(1, k - 1)]


def _wall(builder: MeshBuilder, P_bottom: np.ndarray, P_top: np.ndarray, side: str, color) -> None:
    """Vertical strip along a 3-D polyline. ``side``: which side of the travel direction the viewer is on."""
    a0, b0 = P_bottom[:-1], P_bottom[1:]
    a1, b1 = P_top[:-1], P_top[1:]
    if side == "right":
        Q = np.stack([a0, b0, b1, a1], axis=1)
    else:
        Q = np.stack([b0, a0, a1, b1], axis=1)
    builder.quads(Q, np.asarray(color, dtype=np.float32))


def _bridge_structure(B, xy, z, hw, n, sc) -> None:
    c = B["concrete"]
    deck_t = 0.9
    for side, off in (("left", hw), ("right", -hw)):
        top = _offset_line(xy, z, off, Y_ROAD, n, sc)
        bot = top.copy()
        bot[:, 1] = np.maximum(0.0, z - deck_t)
        # outer girder face: viewer stands outside -> "left" side wall seen from the left of travel
        _wall(c, bot, top, side, shade(CONCRETE, 0.95))
        # parapet (outer + inner face + top)
        par = top.copy()
        par[:, 1] += 0.85
        _wall(c, top, par, side, shade(CONCRETE, 1.02))
        inner = _offset_line(xy, z, off * (1 - 0.25 / hw), Y_ROAD, n, sc)
        inner_top = inner.copy()
        inner_top[:, 1] += 0.85
        _wall(c, inner, inner_top, "right" if side == "left" else "left", shade(CONCRETE, 0.85))
        c.ribbon(par if side == "left" else inner_top, inner_top if side == "left" else par, shade(CONCRETE, 1.1))
    # underside (visible from below)
    L3, R3 = _offset_line(xy, z, hw, Y_ROAD - deck_t, n, sc), _offset_line(xy, z, -hw, Y_ROAD - deck_t, n, sc)
    keep = z > 0.8
    if keep.sum() >= 2:
        c.ribbon(R3[keep], L3[keep], shade(CONCRETE, 0.6), up=False)
    # piers every ~26 m where the deck is clearly above the ground
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    for st in np.arange(13.0, s[-1] - 6.0, 26.0):
        zz = float(np.interp(st, s, z))
        if zz < 2.2:
            continue
        px, py = float(np.interp(st, s, xy[:, 0])), float(np.interp(st, s, xy[:, 1]))
        k = min(int(np.searchsorted(s, st)), len(xy) - 1)
        nn = n[max(k - 1, 0)]
        cols = (0.0,) if hw < 5.5 else (-hw * 0.5, hw * 0.5)
        for o in cols:
            c.cylinder((px + nn[0] * o, 0.0, py + nn[1] * o), 0.75, zz - deck_t + Y_ROAD, 10,
                       shade(CONCRETE, 0.9), caps=False)
        # cap beam
        c.box((px, zz - deck_t + 0.2, py), (2.0 * hw * 0.9, 0.5, 1.4), yaw=math.atan2(nn[0], nn[1]) + math.pi / 2,
              color=shade(CONCRETE, 0.8))


def _tunnel_structure(B, xy, z, hw, n, sc) -> None:
    c = B["concrete"]
    below = z < -0.08
    if below.sum() < 2:
        return
    for side, off in (("left", hw + 0.15), ("right", -hw - 0.15)):
        top = _offset_line(xy, np.zeros_like(z), off, 0.0, n, sc)
        bot = _offset_line(xy, z, off, Y_ROAD, n, sc)
        # keep only the part below ground
        idx = np.where(below)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for r in runs:
            r = np.arange(max(r[0] - 1, 0), min(r[-1] + 2, len(z)))
            if len(r) < 2:
                continue
            tt, bb = top[r].copy(), bot[r].copy()
            tt[:, 1] = np.maximum(tt[:, 1], bb[:, 1])
            # the viewer is inside the trench: the wall faces the road centre
            _wall(c, bb, tt, "right" if side == "left" else "left", shade(CONCRETE, 0.8 if side == "left" else 0.7))
    # coping strips on the ground level, covering the jagged edge of the terrain hole
    for off0, off1 in ((hw + 0.1, hw + 1.5), (-hw - 1.5, -hw - 0.1)):
        a = _offset_line(xy, np.zeros_like(z), off1, 0.04, n, sc)
        b = _offset_line(xy, np.zeros_like(z), off0, 0.04, n, sc)
        idx = np.where(z < -0.08)[0]
        if len(idx) >= 2:
            lo, hi = max(idx[0] - 1, 0), min(idx[-1] + 2, len(z))
            c.ribbon(a[lo:hi], b[lo:hi], shade(CONCRETE, 1.15))
    deep = np.where(z < -2.6)[0]
    if len(deep) >= 2:                                     # see-through roof where the tunnel is fully covered
        lo, hi = max(deep[0] - 1, 0), min(deep[-1] + 2, len(z))
        L3 = _offset_line(xy, np.zeros_like(z), hw + 0.15, -0.04, n, sc)[lo:hi]
        R3 = _offset_line(xy, np.zeros_like(z), -hw - 0.15, -0.04, n, sc)[lo:hi]
        B["roof"].ribbon(L3, R3, (0.45, 0.62, 0.78, 0.20))
        # ceiling light strips
        mid = _offset_line(xy, np.zeros_like(z), 0.0, 0.0, n, sc)[lo:hi]
        mid[:, 1] = z[lo:hi] + 5.0
        l2, r2 = mid.copy(), mid.copy()
        nn = n[lo:hi]
        l2[:, 0] += nn[:, 0] * 0.15
        l2[:, 2] += nn[:, 1] * 0.15
        r2[:, 0] -= nn[:, 0] * 0.15
        r2[:, 2] -= nn[:, 1] * 0.15
        B["lamp"].ribbon(l2, r2, (1.0, 0.95, 0.8, 1.0), up=False)


def build_sidewalks(city: CityMap, B, mask: _PolyMask, width: float = 2.6) -> None:
    raw = city.raw
    for i in _dedupe_edges(city):
        e = raw.edges[i]
        if abs(e.z0) > 0.5 or abs(e.z1) > 0.5 or e.highway == "motorway":
            continue
        L = polyline_length(e.xy)
        if L < 6.0:
            continue
        hw = city.edge_hw[i]
        xy = e.xy
        z = _edge_z(xy, e.z0, e.z1)
        xy, z = _resample(xy, z, 2.8)
        n, sc = _left_normals(xy)
        for off0, off1 in ((hw, hw + width), (-hw - width, -hw)):
            for sgn in (1, -1):
                pass
            # strip made of cells ~2.8 m long
            a_in = _offset_line(xy, z * 0, off0, Y_WALK, n, sc)
            a_out = _offset_line(xy, z * 0, off1, Y_WALK, n, sc)
            mid = 0.5 * (a_in + a_out)
            cen = 0.5 * (mid[:-1] + mid[1:])
            c2 = np.column_stack([cen[:, 0], cen[:, 2]])
            sd = city.sdf(c2, 0)
            ok = (sd < -0.3) & ~mask.inside(c2)
            sd_in = city.sdf(0.5 * (a_in[:-1] + a_in[1:])[:, [0, 2]], 0)
            if not ok.any():
                continue
            P = np.stack([a_in[:-1], a_out[:-1], a_out[1:], a_in[1:]], axis=1)
            if off0 < 0:                                           # keep CCW seen from above for the right strip
                P = np.stack([a_out[:-1], a_in[:-1], a_in[1:], a_out[1:]], axis=1)
            P = P[ok]
            B["walk"].quads(P, np.asarray((0.80, 0.80, 0.78, 1.0), dtype=np.float32), None,
                            np.tile([0, 1, 0], (len(P), 1)))
            # kerb face on the road side
            inner = a_in if off0 > 0 else a_out
            lo = inner.copy()
            lo[:, 1] = Y_ROAD
            k = np.where(ok)[0]
            Q0 = np.stack([lo[k], lo[k + 1], inner[k + 1], inner[k]], axis=1)       # viewer on the road side
            if off0 > 0:                                           # road is on the right of the strip's travel direction
                Q0 = np.stack([lo[k], lo[k + 1], inner[k + 1], inner[k]], axis=1)
            else:
                Q0 = np.stack([lo[k + 1], lo[k], inner[k], inner[k + 1]], axis=1)
            B["walk"].quads(Q0, np.asarray((0.62, 0.62, 0.60, 1.0), dtype=np.float32))


def build_crosswalks_and_lines(city: CityMap, B) -> None:
    infra = city.infra
    edges = city.raw.edges
    # ---- zebras
    quads = []
    for cw in infra.crosswalks:
        h = cw.heading
        fwd = np.array([math.cos(h), math.sin(h)])
        left = np.array([-fwd[1], fwd[0]])
        ctr = np.array(cw.xy)
        depth = infra.params.crosswalk_width
        t = np.arange(-cw.half_len + 0.8, cw.half_len - 0.5, 1.0)
        z = 0.0
        ei = next(iter(cw.edges), None)
        if ei is not None:
            e = edges[ei]
            z = float(np.interp(cw.edges[ei], np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(e.xy, axis=0), axis=1))],
                                np.zeros(len(e.xy)))) + _edge_z(e.xy, e.z0, e.z1).mean() * 0
        for tt in t:
            c = ctr + left * tt
            a = c - fwd * depth / 2
            b = c + fwd * depth / 2
            w = 0.28 * left
            q = np.array([[a - w, a + w, b + w, b - w]])         # (1,4,2) CCW seen from above (left first)
            quads.append(np.column_stack([q[0][:, 0], np.full(4, z + Y_MARK), q[0][:, 1]]))
    if quads:
        Q = np.array(quads, dtype=np.float32)
        # ensure CCW from above: compute orientation by shoelace on (x,z) and flip where negative
        area = 0.5 * np.sum(Q[:, :, 0] * np.roll(Q[:, :, 2], -1, axis=1) - np.roll(Q[:, :, 0], -1, axis=1) * Q[:, :, 2], axis=1)
        Q[area < 0] = Q[area < 0][:, ::-1]
        B["mark"].quads(Q, np.asarray(PAINT, dtype=np.float32), None, np.tile([0, 1, 0], (len(Q), 1)))
    # ---- stop lines (lights / stop) and yield teeth
    sq, tq = [], []
    for c in infra.controls:
        i = c.edge
        hw, off = city.edge_hw[i], city.edge_off[i]
        h = c.heading
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        axis = np.array(c.xy) - right * off
        two = city.edge_twoway[i]
        lo, hi = (0.1 if two else -hw + 0.2), hw - 0.25
        e = edges[i]
        zc = float(_edge_z(e.xy, e.z0, e.z1).mean()) if abs(e.z0) + abs(e.z1) > 0.01 else 0.0
        zc = 0.0 if abs(zc) < 0.5 else zc
        if c.kind in ("light", "stop"):
            a, b = axis + right * lo, axis + right * hi
            w = 0.22 * fwd
            P = [a - w, a + w, b + w, b - w]
            sq.append(np.column_stack([np.array(P)[:, 0], np.full(4, zc + Y_MARK), np.array(P)[:, 1]]))
        else:                                                  # yield: row of triangles pointing at the driver
            k = max(1, int((hi - lo) / 0.9))
            for j in range(k):
                x0 = lo + j * (hi - lo) / k
                x1 = x0 + 0.8 * (hi - lo) / k
                base0, base1 = axis + right * x0, axis + right * x1
                tip = 0.5 * (base0 + base1) - fwd * 0.7
                tq.append(np.array([[base0[0], zc + Y_MARK, base0[1]], [base1[0], zc + Y_MARK, base1[1]],
                                    [tip[0], zc + Y_MARK, tip[1]]]))
    for lst in (sq,):
        if lst:
            Q = np.array(lst, dtype=np.float32)
            area = 0.5 * np.sum(Q[:, :, 0] * np.roll(Q[:, :, 2], -1, axis=1) - np.roll(Q[:, :, 0], -1, axis=1) * Q[:, :, 2], axis=1)
            Q[area < 0] = Q[area < 0][:, ::-1]
            B["mark"].quads(Q, np.asarray(PAINT, dtype=np.float32), None, np.tile([0, 1, 0], (len(Q), 1)))
    for T in tq:
        area = 0.5 * float(np.sum(T[:, 0] * np.roll(T[:, 2], -1) - np.roll(T[:, 0], -1) * T[:, 2]))
        t = [(0, 1, 2)] if area > 0 else [(0, 2, 1)]
        B["mark"].add(T, [0, 1, 0], np.asarray(PAINT, dtype=np.float32), None, t)


def build_terrain(city: CityMap, B, hole_edges, mask: _PolyMask, bounds, cell: float = 2.0, tile: float = 128.0) -> None:
    """Ground: coarse tiles, fine cells only where a tunnel trench cuts a hole."""
    x0, y0, x1, y1 = bounds
    x0, y0 = math.floor(x0 / tile) * tile, math.floor(y0 / tile) * tile
    x1, y1 = math.ceil(x1 / tile) * tile, math.ceil(y1 / tile) * tile
    W, H = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    hole = np.zeros((H + 1, W + 1), dtype=bool)
    for xy, z, hw in hole_edges:
        sel = z < -0.1
        idx = np.where(sel)[0]
        for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(r) < 1:
                continue
            r = np.arange(max(r[0] - 1, 0), min(r[-1] + 2, len(xy)))
            _paint_capsules(hole, xy[r], max(hw - 0.9, 0.5), x0, y0, cell)
    col = np.asarray((0.66, 0.69, 0.50, 1.0), dtype=np.float32)       # olive steppe: less beige "void"
    ty0, tx0 = 0, 0
    nt_x, nt_y = int(round((x1 - x0) / tile)), int(round((y1 - y0) / tile))
    up = np.array([0, 1, 0])
    cpt = int(tile / cell)
    for ty in range(nt_y):
        for tx in range(nt_x):
            sub = hole[ty * cpt:(ty + 1) * cpt + 1, tx * cpt:(tx + 1) * cpt + 1]
            xa, ya = x0 + tx * tile, y0 + ty * tile
            if not sub.any():
                B["ground"].quad((xa, Y_GROUND, ya), (xa + tile, Y_GROUND, ya), (xa + tile, Y_GROUND, ya + tile),
                                 (xa, Y_GROUND, ya + tile), col, normal=up)
                continue
            keep = ~(sub[:-1, :-1] & sub[1:, 1:])                       # drop cells that lie inside a hole
            iy, ix = np.nonzero(keep)
            if len(iy) == 0:
                continue
            xs, ys = xa + ix * cell, ya + iy * cell
            P = np.stack([np.stack([xs, np.full_like(xs, Y_GROUND), ys], 1), np.stack([xs + cell, np.full_like(xs, Y_GROUND), ys], 1),
                          np.stack([xs + cell, np.full_like(xs, Y_GROUND), ys + cell], 1),
                          np.stack([xs, np.full_like(xs, Y_GROUND), ys + cell], 1)], axis=1)
            B["ground"].quads(P, col, None, np.tile(up, (len(P), 1)))
    if hole_edges:                                     # dark soil under everything: visible only through tunnel cut-aways
        B["subsoil"].quad((x0 - 600, -6.9, y0 - 600), (x1 + 600, -6.9, y0 - 600), (x1 + 600, -6.9, y1 + 600),
                          (x0 - 600, -6.9, y1 + 600), (0.20, 0.16, 0.13, 1.0), normal=up)
    # outer frame to the horizon
    R = 6000.0
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    for (ax, ay, bx, by) in ((x0 - R, y0 - R, x1 + R, y0), (x0 - R, y1, x1 + R, y1 + R),
                             (x0 - R, y0, x0, y1), (x1, y0, x1 + R, y1)):
        B["ground"].quad((ax, Y_GROUND, ay), (bx, Y_GROUND, ay), (bx, Y_GROUND, by), (ax, Y_GROUND, by), col, normal=up)


# ====================================================================================== buildings
_PALETTE = [(0.92, 0.90, 0.84), (0.86, 0.80, 0.70), (0.78, 0.74, 0.68), (0.93, 0.93, 0.92), (0.80, 0.82, 0.85),
            (0.88, 0.76, 0.62), (0.72, 0.62, 0.55), (0.90, 0.86, 0.78), (0.70, 0.72, 0.76), (0.95, 0.92, 0.82)]
_GLASS_PALETTE = [(0.62, 0.78, 0.88), (0.55, 0.72, 0.80), (0.70, 0.80, 0.90), (0.50, 0.62, 0.72), (0.80, 0.86, 0.92)]
_NAMED = {"white": (0.95, 0.95, 0.93), "beige": (0.88, 0.80, 0.64), "yellow": (0.93, 0.85, 0.55), "grey": (0.7, 0.7, 0.72),
          "gray": (0.7, 0.7, 0.72), "brown": (0.6, 0.45, 0.35), "red": (0.75, 0.4, 0.35), "blue": (0.55, 0.68, 0.85),
          "green": (0.6, 0.78, 0.62), "orange": (0.9, 0.62, 0.4), "pink": (0.92, 0.75, 0.75), "black": (0.25, 0.25, 0.27)}


def _parse_colour(s: str) -> Optional[Tuple[float, float, float]]:
    if not s:
        return None
    s = s.strip().lower()
    if s in _NAMED:
        return _NAMED[s]
    if s.startswith("#") and len(s) == 7:
        try:
            return (int(s[1:3], 16) / 255, int(s[3:5], 16) / 255, int(s[5:7], 16) / 255)
        except ValueError:
            return None
    return None


def _facade_style(btype: str, h: float, area: float) -> str:
    if btype in ("office", "commercial", "hotel") or h > 55 or (btype == "yes" and h > 38):
        return "glass"
    if btype in ("retail", "supermarket", "garage", "garages", "parking", "industrial", "warehouse"):
        return "shop"
    if btype in ("public", "government", "civic", "university", "school", "hospital", "museum", "church", "mosque"):
        return "stone"
    return "panel"


def build_buildings(city: CityMap, B, quality: Quality) -> Dict[str, int]:
    raw = city.raw
    meta = getattr(raw, "building_meta", []) or []
    lm_xy = [np.array(xy) for xy, rad in getattr(raw, "landmark_sites", []) or []]
    lm_r = [rad for xy, rad in getattr(raw, "landmark_sites", []) or []]
    counts: Dict[str, int] = {}
    items = list(zip(raw.buildings, meta + [{}] * (len(raw.buildings) - len(meta))))
    items.sort(key=lambda it: -abs(_ring_area(it[0][0])))              # big ones first (they matter when capped)
    for k, ((ring, h), m) in enumerate(items[: quality.max_buildings]):
        ring = np.asarray(ring, dtype=np.float64)
        if len(ring) < 3:
            continue
        if any(np.hypot(*(ring.mean(axis=0) - c)) < max(25.0, 0.9 * r) for c, r in zip(lm_xy, lm_r)):
            continue                                                  # replaced by a hand-made landmark model
        area = abs(_ring_area(ring))
        style = _facade_style(m.get("type", "yes"), h, area)
        seed = int((ring[0, 0] * 73856093) % 1000003 + (ring[0, 1] * 19349663) % 999983)
        col = _parse_colour(m.get("colour", "")) if m else None
        base = (*col, 1.0) if col else (_hash_color(seed, _GLASS_PALETTE if style == "glass" else _PALETTE))
        a = (seed % 997) / 997.0                                      # per-building random seed (night windows)
        base = (base[0], base[1], base[2], a)
        floors = max(1, int(round((h - 1.0) / 3.3)))
        y1 = floors * 3.3 + 1.0 if h > 4 else h
        tile_w = 2.4 if style == "glass" else 3.3
        roof_col = shade((0.50, 0.50, 0.52), 1.0)
        key = f"fac_{style}"
        B[key].extrude(ring, -0.1, y1, base, (*roof_col[:3], a), _ear(ring), tile_w=tile_w, tile_h=3.3,
                       floors=floors if h > 4 else 1)
        counts[style] = counts.get(style, 0) + 1
    return counts


def _ring_area(r: np.ndarray) -> float:
    return 0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))


def _ear(ring: np.ndarray):
    from .mesh import ear_clip
    return ear_clip(ring)


# ====================================================================================== nature
def _tree_template(kind: int = 0) -> Mesh:
    b = MeshBuilder()
    b.cylinder((0, 0, 0), 0.16, 2.4, 6, (0.36, 0.26, 0.18, 1.0), caps=False)
    if kind == 0:                                                   # round deciduous crown
        b.sphere((0, 3.5, 0), 1.9, 9, 6, (0.30, 0.50, 0.24, 1.0), squash=1.15)
        b.sphere((0.7, 3.0, 0.3), 1.2, 8, 5, (0.36, 0.56, 0.28, 1.0))
    elif kind == 1:                                                 # conifer
        b.cylinder((0, 1.6, 0), 1.7, 3.2, 9, (0.17, 0.38, 0.22, 1.0), r_top=0.05, caps=False)
        b.cylinder((0, 3.6, 0), 1.2, 2.6, 9, (0.20, 0.42, 0.25, 1.0), r_top=0.05, caps=False)
    else:                                                           # poplar
        b.sphere((0, 4.5, 0), 1.0, 8, 6, (0.28, 0.48, 0.22, 1.0), squash=3.0)
    return b.build()


def build_trees(city: CityMap, B, mask: _PolyMask, quality: Quality, rng: np.random.Generator) -> int:
    raw = city.raw
    pts: List[np.ndarray] = []
    if len(getattr(raw, "trees", [])):
        pts.append(np.asarray(raw.trees, dtype=np.float64))
    # street trees along the sidewalk (outer edge of the walk), alternating sides
    for i in _dedupe_edges(city):
        e = raw.edges[i]
        if abs(e.z0) > 0.5 or abs(e.z1) > 0.5 or e.highway in ("motorway", "service"):
            continue
        L = polyline_length(e.xy)
        if L < 25.0:
            continue
        hw = city.edge_hw[i]
        step = 14.0 / max(quality.trees, 0.1)
        ss = np.arange(10.0, L - 10.0, step) + rng.uniform(-2, 2, size=len(np.arange(10.0, L - 10.0, step)))
        for side in (1, -1):
            for s in ss:
                if rng.random() > 0.55 * quality.trees:
                    continue
                p, h = polyline_at(e.xy, float(s))
                nrm = np.array([-math.sin(h), math.cos(h)])
                pts.append((p + nrm * side * (hw + 2.1)).reshape(1, 2))
    # park scatter
    for ring in getattr(raw, "green", []) or []:
        a = abs(_ring_area(ring))
        n = int(min(260, a / 140.0 * quality.trees))
        if n <= 0:
            continue
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        cand = rng.uniform(lo, hi, size=(n * 3, 2))
        from matplotlib.path import Path
        inside = Path(ring).contains_points(cand)
        pts.append(cand[inside][:n])
    # open land: clustered groves + lone trees, so wide empty plots between roads read as parkland, not as a void
    x0, y0, x1, y1 = city.bounds
    gap = 17.0 / max(math.sqrt(max(quality.trees, 0.1)), 0.3)
    gx, gy = np.meshgrid(np.arange(x0, x1, gap), np.arange(y0, y1, gap))
    G = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-0.5, 0.5, size=(gx.size, 2)) * gap
    ph = rng.uniform(0, 6.28, size=4)
    field = (np.sin(G[:, 0] / 260.0 + ph[0]) * np.sin(G[:, 1] / 310.0 + ph[1])
             + 0.6 * np.sin(G[:, 0] / 95.0 + G[:, 1] / 120.0 + ph[2]) + 0.3 * np.sin(G[:, 1] / 47.0 - G[:, 0] / 61.0 + ph[3]))
    keep_p = np.clip(0.10 + 0.42 * (field - 0.1), 0.0, 0.75)            # dense groves where the field is high, sparse elsewhere
    G = G[rng.random(len(G)) < keep_p]
    if len(G):
        G = G[city.sdf(G, 0) < -8.0]                                    # not close to a road
        pts.append(G[:int(5200 * quality.trees)])
    if not pts:
        return 0
    P = np.vstack(pts)
    sd0 = city.sdf(P, 0)
    ok = (sd0 < -1.2) & ~mask.inside(P)
    for L_ in city.layers:
        if L_ != 0:
            ok &= city.sdf(P, L_) < -1.0
    # keep clear of zebras and junction mouths
    if city.infra.crosswalks:
        cwxy = np.array([c.xy for c in city.infra.crosswalks])
        d = np.min(np.hypot(P[:, None, 0] - cwxy[None, :, 0], P[:, None, 1] - cwxy[None, :, 1]), axis=1)
        ok &= d > 9.0
    P = P[ok]
    if len(P) == 0:
        return 0
    kind = rng.choice([0, 1, 2], size=len(P), p=[0.62, 0.25, 0.13])
    for kd in (0, 1, 2):
        sel = P[kind == kd]
        if len(sel) == 0:
            continue
        T = _tree_template(kd)
        k = len(sel)
        pos = np.column_stack([sel[:, 0], np.full(k, Y_GREEN), sel[:, 1]])
        sc = rng.uniform(0.75, 1.35, size=k)
        tint = rng.uniform(0.82, 1.12, size=(k, 3))
        B["tree"].append_mesh(instances(T, pos, rng.uniform(0, 6.28, size=k), sc, tint))
    return int(len(P))


def build_green_water(city: CityMap, B) -> None:
    raw = city.raw
    for ring in getattr(raw, "green", []) or []:
        B["green"].polygon(ring, Y_GREEN, (0.52, 0.66, 0.38, 1.0), _ear(ring))
    for ring in getattr(raw, "water", []) or []:
        B["water"].polygon(ring, Y_GREEN + 0.01, (0.22, 0.38, 0.56, 1.0), _ear(ring))


# ====================================================================================== street furniture
def build_traffic_furniture(city: CityMap, B) -> Tuple[Mesh, List[LightHead]]:
    """Poles, light heads, stop / yield / speed signs and street lamps. Returns the lamp-disc mesh (dynamic)."""
    infra, edges = city.infra, city.raw.edges
    lamp_b = MeshBuilder()
    heads: List[LightHead] = []
    k_disc = 10
    for c in infra.controls:
        i = c.edge
        hw = city.edge_hw[i]
        h = c.heading
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        axis = np.array(c.xy) - right * city.edge_off[i]
        pole = axis + right * (hw + 0.9)
        if c.kind == "light":
            B["prop"].cylinder((pole[0], 0.0, pole[1]), 0.09, 4.6, 8, (0.18, 0.19, 0.20, 1.0))
            # head on the pole facing the approaching traffic (looking against ``fwd``)
            _light_head(B["prop"], lamp_b, heads, c.idx, pole, 3.9, fwd, k_disc)
            # overhead head on a short mast arm over the lane
            lane = np.array(c.xy)
            arm_end = pole + (lane - pole) * 0.9
            B["prop"].box(((pole[0] + arm_end[0]) / 2, 4.5, (pole[1] + arm_end[1]) / 2),
                          (0.07, 0.07, float(np.hypot(*(arm_end - pole)))), yaw=math.atan2(*(arm_end - pole)),
                          color=(0.18, 0.19, 0.20, 1.0))
            _light_head(B["prop"], lamp_b, heads, c.idx, arm_end, 4.1, fwd, k_disc)
        elif c.kind in ("stop", "yield"):
            B["prop"].cylinder((pole[0], 0.0, pole[1]), 0.05, 2.3, 6, (0.5, 0.5, 0.52, 1.0), caps=False)
            _sign_plate(B["sign_" + c.kind], pole, 2.3, fwd, 0.45)
    # speed-limit signs
    for s in infra.signs:
        if s.kind != "speed":
            continue
        fwd = np.array([math.cos(s.heading), math.sin(s.heading)])
        pos = np.array(s.xy)
        B["prop"].cylinder((pos[0], 0.0, pos[1]), 0.05, 2.3, 6, (0.5, 0.5, 0.52, 1.0), caps=False)
        kmh = int(round(s.value * 3.6 / 10.0) * 10)
        _sign_plate(B[f"sign_speed_{kmh}"], pos, 2.5, fwd, 0.6)
    return lamp_b.build(), heads


def _light_head(prop: MeshBuilder, lamp_b: MeshBuilder, heads: List[LightHead], idx: int, pos: np.ndarray,
                y: float, fwd: np.ndarray, k: int) -> None:
    yaw = math.atan2(-fwd[0], -fwd[1])                              # box front (+z) faces the approaching cars
    prop.box((pos[0], y, pos[1]), (0.34, 1.0, 0.28), yaw=yaw, color=(0.05, 0.05, 0.06, 1.0))
    base = len(lamp_b)
    front = -fwd
    for j, dy in enumerate((0.32, 0.0, -0.32)):                     # red, yellow, green from the top
        c = np.array([pos[0] + front[0] * 0.16, y + dy, pos[1] + front[1] * 0.16])
        # small disc facing ``front``
        a = np.linspace(0, 2 * math.pi, k, endpoint=False)
        right = np.array([front[1], -front[0]])
        ring = np.stack([c[0] + 0.115 * np.cos(a) * right[0], c[1] + 0.115 * np.sin(a),
                         c[2] + 0.115 * np.cos(a) * right[1]], axis=1)
        ctr = np.array([c])
        v = np.vstack([ctr, ring])
        t = [(0, 1 + (m + 1) % k, 1 + m) for m in range(k)]
        lamp_b.add(v, [front[0], 0, front[1]], np.array([0.15, 0.15, 0.15, 1.0], dtype=np.float32), None, t)
    heads.append(LightHead(idx, base, k + 1))


def _sign_plate(builder: MeshBuilder, pos: np.ndarray, y: float, fwd: np.ndarray, size: float) -> None:
    """Square plate facing the approaching traffic (textured by the layer's own texture)."""
    front = -fwd
    right = np.array([front[1], -front[0]])
    c = np.array([pos[0], y, pos[1]]) + np.array([front[0], 0, front[1]]) * 0.06
    h = size / 2
    corners = [c + np.array([-right[0], 0, -right[1]]) * h + [0, -h, 0], c + np.array([right[0], 0, right[1]]) * h + [0, -h, 0],
               c + np.array([right[0], 0, right[1]]) * h + [0, h, 0], c + np.array([-right[0], 0, -right[1]]) * h + [0, h, 0]]
    # ensure the quad faces ``front``: CCW on screen when viewed from the front
    n = np.array([front[0], 0, front[1]])
    Q = np.array([corners], dtype=np.float32)
    builder.quads(Q, np.asarray((1, 1, 1, 1), dtype=np.float32), np.array([[[1, 0], [0, 0], [0, 1], [1, 1]]]), n[None], two_sided=True)


def build_street_lamps(city: CityMap, B, mask: _PolyMask, quality: Quality, rng: np.random.Generator) -> int:
    raw = city.raw
    n = 0
    for i in _dedupe_edges(city):
        e = raw.edges[i]
        if abs(e.z0) > 0.5 or abs(e.z1) > 0.5 or road_rank(e.highway) < 2:
            continue
        L = polyline_length(e.xy)
        hw = city.edge_hw[i]
        for s in np.arange(30.0, L - 25.0, 42.0 / max(quality.props, 0.2)):
            p, h = polyline_at(e.xy, float(s))
            right = np.array([math.sin(h), -math.cos(h)])
            pos = p + right * (hw + 1.0)
            if city.sdf(pos[None], 0)[0] > -0.4 or mask.inside(pos[None])[0]:
                continue
            B["prop"].cylinder((pos[0], 0.0, pos[1]), 0.09, 8.2, 7, (0.30, 0.31, 0.33, 1.0), r_top=0.06)
            tip = pos - right * 1.9
            B["prop"].box(((pos[0] + tip[0]) / 2, 8.15, (pos[1] + tip[1]) / 2), (0.08, 0.08, 1.9),
                          yaw=math.atan2(*(tip - pos)), color=(0.3, 0.31, 0.33, 1.0))
            B["lamp"].box((tip[0], 8.05, tip[1]), (0.5, 0.12, 0.28), yaw=math.atan2(*(tip - pos)),
                          color=(1.0, 0.93, 0.75, 1.0), bottom=True)
            n += 1
    return n


# ====================================================================================== top level
def build_scene(city: CityMap, quality: Quality, seed: int = 3, landmarks: bool = True) -> SceneData:
    raw = city.raw
    rng = np.random.default_rng(seed)
    names = ["ground", "green", "water", "road", "walk", "mark", "concrete", "roof", "lamp", "prop", "tree",
             "fac_glass", "fac_panel", "fac_stone", "fac_shop", "sign_stop", "sign_yield", "glass", "landmark", "subsoil"]
    speeds = sorted({int(round(s.value * 3.6 / 10.0) * 10) for s in city.infra.signs})
    names += [f"sign_speed_{v}" for v in speeds]
    B: Dict[str, MeshBuilder] = {n: MeshBuilder() for n in names}

    x0, y0, x1, y1 = city.bounds
    pad = 60.0
    bounds = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    mask = _PolyMask(bounds, [r for r, _ in raw.buildings], 2.0)
    hole_edges: List[Tuple[np.ndarray, np.ndarray, float]] = []

    raw.landmark_sites = []                                       # type: ignore[attr-defined]
    n_landmarks = 0
    if landmarks:
        from .landmarks import build_landmarks
        n_landmarks = build_landmarks(city, B)
    build_roads(city, B, hole_edges)
    build_sidewalks(city, B, mask)
    build_crosswalks_and_lines(city, B)
    build_terrain(city, B, hole_edges, mask, bounds)
    build_green_water(city, B)
    stats = {"buildings": sum(build_buildings(city, B, quality).values())}
    stats["trees"] = build_trees(city, B, mask, quality, rng)
    light_mesh, heads = build_traffic_furniture(city, B)
    stats["street_lamps"] = build_street_lamps(city, B, mask, quality, rng)
    stats["landmarks"] = n_landmarks

    layers: Dict[str, Layer] = {}
    spec = {
        "ground": Layer("ground", texture="steppe", world_uv=18.0),
        "green": Layer("green", texture="grass", world_uv=14.0),
        "water": Layer("water", spec=0.9, shadows=False),
        "road": Layer("road", texture="asphalt", world_uv=7.0, spec=0.16),
        "walk": Layer("walk", texture="paving", world_uv=2.0, depth_offset=2),
        "mark": Layer("mark", spec=0.1, depth_offset=6, shadows=False),
        "concrete": Layer("concrete", texture="concrete", world_uv=4.0),
        "roof": Layer("roof", unlit=False, transparent=True, shadows=False),
        "lamp": Layer("lamp", unlit=True, glow_night=2.2, shadows=False),
        "prop": Layer("prop", spec=0.25),
        "tree": Layer("tree", shadows=quality.shadows >= 4096),
        "fac_glass": Layer("fac_glass", texture="facade_glass", glass=1.0, spec=0.35),
        "fac_panel": Layer("fac_panel", texture="facade_panel", glass=1.0, spec=0.2),
        "fac_stone": Layer("fac_stone", texture="facade_stone", glass=1.0, spec=0.2),
        "fac_shop": Layer("fac_shop", texture="facade_shop", glass=1.0, spec=0.2),
        "glass": Layer("glass", transparent=True, spec=0.9, shadows=False),
        "subsoil": Layer("subsoil", shadows=False),
        "landmark": Layer("landmark", spec=0.4),
        "sign_stop": Layer("sign_stop", texture="sign_stop", spec=0.2),
        "sign_yield": Layer("sign_yield", texture="sign_yield", spec=0.2),
    }
    for v in speeds:
        spec[f"sign_speed_{v}"] = Layer(f"sign_speed_{v}", texture=f"sign_speed_{v}", spec=0.2)
    for name, mb in B.items():
        lay = spec.get(name, Layer(name))
        m = mb.build()
        lay.meshes = split_into_tiles(m, 260.0) if not m.empty else []
        if name == "ground":
            lay.meshes = split_into_tiles(m, 600.0) if not m.empty else []
        layers[name] = lay
    return SceneData(layers, heads, light_mesh, bounds, stats)


SCENE_VERSION = 9         # bump when the geometry code changes: invalidates cached scenes


def build_scene_cached(city: CityMap, quality: Quality, map_path: str, landmarks: bool = True) -> SceneData:
    """``build_scene`` with an on-disk cache (assets/cache): start-up drops from ~8 s to ~1 s."""
    import hashlib
    import os
    import pickle
    from pathlib import Path
    if not os.path.exists(map_path):
        return build_scene(city, quality, landmarks=landmarks)
    st = os.stat(map_path)
    key = hashlib.sha1(f"{map_path}|{st.st_mtime_ns}|{st.st_size}|{quality.trees}|{quality.props}|{quality.max_buildings}|"
                       f"{landmarks}|{SCENE_VERSION}".encode()).hexdigest()[:14]
    path = Path("assets/cache") / f"{Path(map_path).stem}_{key}.scene"
    try:
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    sd = build_scene(city, quality, landmarks=landmarks)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(sd, f, protocol=4)
    except Exception:
        pass
    return sd
