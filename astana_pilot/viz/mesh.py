"""Pure-NumPy triangle-mesh toolkit for the 3-D viewer (no graphics library needed - unit-testable).

Coordinates are the *viewer's*: ``x`` east, ``y`` UP, ``z`` north (= simulation ``y``).  A simulation point
``(x, y)`` at height ``h`` becomes ``(x, h, y)``.  The viewer's coordinate system is left-handed (Ursina / Panda
``y-up-left``), so *looking from the front* a face is listed **clockwise** in ``(x, z)`` order for an up-facing
surface; this is hidden inside :meth:`MeshBuilder.quads` (give corners counter-clockwise as seen from the front
in the usual way - the builder fixes the winding with the face normal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Color = Tuple[float, float, float, float]


def rgba(c, a: float = 1.0) -> Color:
    """Accept ``(r, g, b)`` / ``(r, g, b, a)`` floats in 0..1 or ``'#rrggbb'`` strings."""
    if isinstance(c, str):
        s = c.lstrip("#")
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0, a)
    c = tuple(float(x) for x in c)
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else a)


def shade(c, k: float) -> Color:
    r = rgba(c)
    return (min(1.0, r[0] * k), min(1.0, r[1] * k), min(1.0, r[2] * k), r[3])


@dataclass
class Mesh:
    v: np.ndarray            # (N, 3) float32
    n: np.ndarray            # (N, 3) float32
    c: np.ndarray            # (N, 4) float32
    uv: np.ndarray           # (N, 2) float32
    t: np.ndarray            # (M, 3) int32

    @property
    def empty(self) -> bool:
        return len(self.t) == 0

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.v.min(axis=0), self.v.max(axis=0)

    def transformed(self, yaw: float = 0.0, pos=(0.0, 0.0, 0.0), scale=1.0) -> "Mesh":
        R = rot_y(yaw)
        v = (self.v * np.asarray(scale, dtype=np.float32)) @ R.T + np.asarray(pos, dtype=np.float32)
        n = self.n @ R.T
        return Mesh(v.astype(np.float32), n.astype(np.float32), self.c, self.uv, self.t)


def rot_y(yaw: float) -> np.ndarray:
    """Rotation about the up axis; ``yaw`` radians, positive turns +z toward +x (compass-style, clockwise from above)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def heading_to_yaw(theta: float) -> float:
    """Simulation heading (CCW from +x east) -> viewer yaw (CW from +z north): a car's local +z is its front."""
    return math.pi / 2.0 - theta


def _normalize(a: np.ndarray) -> np.ndarray:
    l = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(l, 1e-9)


class MeshBuilder:
    """Accumulates geometry; every method appends and returns nothing (call :meth:`build` at the end)."""

    def __init__(self) -> None:
        self._v: List[np.ndarray] = []
        self._n: List[np.ndarray] = []
        self._c: List[np.ndarray] = []
        self._uv: List[np.ndarray] = []
        self._t: List[np.ndarray] = []
        self._count = 0

    # ------------------------------------------------------------------ raw
    def add(self, v, n, c, uv, t) -> int:
        v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
        k = len(v)
        n = np.asarray(n, dtype=np.float32).reshape(-1, 3)
        c = np.asarray(c, dtype=np.float32).reshape(-1, 4)
        uv = np.zeros((k, 2), dtype=np.float32) if uv is None else np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        if len(n) == 1:
            n = np.repeat(n, k, axis=0)
        if len(c) == 1:
            c = np.repeat(c, k, axis=0)
        if len(uv) == 1:
            uv = np.repeat(uv, k, axis=0)
        assert len(n) == k and len(c) == k and len(uv) == k, (len(n), len(c), len(uv), k)
        base = self._count
        self._v.append(v)
        self._n.append(n)
        self._c.append(c)
        self._uv.append(uv)
        self._t.append(np.asarray(t, dtype=np.int32).reshape(-1, 3) + base)
        self._count += k
        return base

    def __len__(self) -> int:
        return self._count

    def append_mesh(self, m: Mesh) -> None:
        if not m.empty:
            self.add(m.v, m.n, m.c, m.uv, m.t)

    def build(self) -> Mesh:
        if not self._v:
            z3 = np.zeros((0, 3), dtype=np.float32)
            return Mesh(z3, z3, np.zeros((0, 4), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                        np.zeros((0, 3), dtype=np.int32))
        return Mesh(np.concatenate(self._v), np.concatenate(self._n), np.concatenate(self._c),
                    np.concatenate(self._uv), np.concatenate(self._t))

    # ------------------------------------------------------------------ quads
    def quads(self, P, color, uv=None, normal=None, two_sided: bool = False) -> None:
        """``P`` (K, 4, 3) corners, counter-clockwise **on screen** as seen from the front (e.g. an up-facing ground quad
        is listed (x-,z-), (x+,z-), (x+,z+), (x-,z+): CCW on a north-up map).

        ``color``: single colour or (K, 4). ``normal`` (K, 3) optional; computed from the corners otherwise.
        """
        P = np.asarray(P, dtype=np.float32).reshape(-1, 4, 3)
        K = len(P)
        if K == 0:
            return
        if normal is None:
            normal = _normalize(np.cross(P[:, 3] - P[:, 0], P[:, 1] - P[:, 0]))
        normal = np.broadcast_to(np.asarray(normal, dtype=np.float32), (K, 3))
        col = np.asarray(color, dtype=np.float32)
        col = np.broadcast_to(col[:, None, :], (K, 4, 4)) if col.ndim == 2 else np.broadcast_to(col.reshape(1, 1, 4), (K, 4, 4))
        if uv is None:
            uv = np.broadcast_to(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), (K, 4, 2))
        uv = np.asarray(uv, dtype=np.float32).reshape(K, 4, 2)
        nn = np.repeat(normal[:, None, :], 4, axis=1)
        idx = np.arange(K * 4).reshape(K, 4)
        tri = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]], axis=0)         # CCW on screen = front face
        self.add(P.reshape(-1, 3), nn.reshape(-1, 3), col.reshape(-1, 4), uv.reshape(-1, 2), tri)
        if two_sided:
            self.add(P.reshape(-1, 3), (-nn).reshape(-1, 3), col.reshape(-1, 4), uv.reshape(-1, 2),
                     np.concatenate([idx[:, [0, 2, 1]], idx[:, [0, 3, 2]]], axis=0))

    def quad(self, p0, p1, p2, p3, color, uv=None, normal=None, two_sided: bool = False) -> None:
        self.quads(np.array([[p0, p1, p2, p3]], dtype=np.float32), np.asarray(rgba(color), dtype=np.float32),
                   None if uv is None else np.asarray(uv)[None], None if normal is None else np.asarray(normal)[None],
                   two_sided)

    # ------------------------------------------------------------------ ribbons / polygons
    def ribbon(self, left: np.ndarray, right: np.ndarray, color, uv=None, up: bool = True) -> None:
        """Quad strip between two polylines (K, 3) each (top surface, normal up)."""
        left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
        k = len(left)
        if k < 2:
            return
        P = np.stack([left[:-1], right[:-1], right[1:], left[1:]], axis=1)
        if uv is not None:
            uv = np.asarray(uv, dtype=np.float32)
            U = np.stack([uv[0][:-1], uv[1][:-1], uv[1][1:], uv[0][1:]], axis=1)
        else:
            U = None
        n = np.zeros((k - 1, 3), dtype=np.float32)
        n[:, 1] = 1.0 if up else -1.0
        self.quads(P, np.asarray(rgba(color), dtype=np.float32), U, n)

    def polygon(self, ring_xz: np.ndarray, y: float, color, tris: Sequence[Tuple[int, int, int]],
                uv_scale: float = 0.0, normal_up: bool = True) -> None:
        """Horizontal polygon given as (n, 2) ``(x, z)`` ring and its triangulation (see ``ear_clip``)."""
        ring = np.asarray(ring_xz, dtype=np.float32)
        v = np.column_stack([ring[:, 0], np.full(len(ring), y, dtype=np.float32), ring[:, 1]])
        uv = ring * uv_scale if uv_scale else None
        t = np.asarray(tris, dtype=np.int32).reshape(-1, 3)           # ``ear_clip`` output: CCW seen from above
        if not normal_up:
            t = t[:, [0, 2, 1]]
        self.add(v, [0.0, 1.0 if normal_up else -1.0, 0.0], np.asarray(rgba(color), dtype=np.float32), uv, t)

    # ------------------------------------------------------------------ solids
    def box(self, center, size, yaw: float = 0.0, color=(1, 1, 1, 1), top_color=None, bottom: bool = True,
            uv_scale: float = 0.0) -> None:
        """Oriented box. ``size`` = (sx, sy, sz) in the box's own frame (x right, y up, z front)."""
        cx, cy, cz = center
        hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        R = rot_y(yaw)

        def P(*loc):
            return (np.array(loc, dtype=np.float32) @ R.T) + np.array([cx, cy, cz], dtype=np.float32)

        col = rgba(color)
        topc = rgba(top_color) if top_color is not None else col
        faces = [
            # (corners CCW from outside), colour
            ((P(-hx, -hy, hz), P(hx, -hy, hz), P(hx, hy, hz), P(-hx, hy, hz)), col),           # front +z
            ((P(hx, -hy, -hz), P(-hx, -hy, -hz), P(-hx, hy, -hz), P(hx, hy, -hz)), col),        # back
            ((P(hx, -hy, hz), P(hx, -hy, -hz), P(hx, hy, -hz), P(hx, hy, hz)), col),            # right +x
            ((P(-hx, -hy, -hz), P(-hx, -hy, hz), P(-hx, hy, hz), P(-hx, hy, -hz)), col),        # left
            ((P(-hx, hy, hz), P(hx, hy, hz), P(hx, hy, -hz), P(-hx, hy, -hz)), topc),           # top
        ]
        if bottom:
            faces.append(((P(-hx, -hy, -hz), P(hx, -hy, -hz), P(hx, -hy, hz), P(-hx, -hy, hz)), col))
        for corners, c in faces:
            Q = np.array(corners, dtype=np.float32)[::-1].copy()      # listed CW above -> CCW on screen from outside
            ctr = Q.mean(axis=0) - np.array([cx, cy, cz], dtype=np.float32)
            nrm = _normalize(np.cross(Q[3] - Q[0], Q[1] - Q[0]))
            if float(nrm @ ctr) < 0:                                  # make sure the normal points away from the centre
                nrm = -nrm
            uv = None
            if uv_scale:
                w = np.linalg.norm(Q[1] - Q[0]); h = np.linalg.norm(Q[3] - Q[0])
                uv = np.array([[0, 0], [w * uv_scale, 0], [w * uv_scale, h * uv_scale], [0, h * uv_scale]])
            self.quads(Q[None], np.asarray(c, dtype=np.float32), None if uv is None else uv[None], nrm[None])

    def cylinder(self, center, radius: float, height: float, segs: int = 12, color=(1, 1, 1, 1),
                 top_color=None, r_top: Optional[float] = None, caps: bool = True, smooth: bool = True) -> None:
        """Vertical (frustum) cylinder; ``center`` is the middle of the BASE (x, y, z)."""
        cx, cy, cz = center
        r0, r1 = radius, radius if r_top is None else r_top
        a = np.linspace(0.0, 2.0 * math.pi, segs, endpoint=False)
        ca, sa = np.cos(a), np.sin(a)
        base = np.column_stack([cx + r0 * ca, np.full(segs, cy), cz + r0 * sa])
        top = np.column_stack([cx + r1 * ca, np.full(segs, cy + height), cz + r1 * sa])
        slope = (r0 - r1) / max(height, 1e-6)
        nrm = _normalize(np.column_stack([ca, np.full(segs, slope), sa]))
        col = np.asarray(rgba(color), dtype=np.float32)
        nxt = (np.arange(segs) + 1) % segs
        P = np.stack([base, base[nxt], top[nxt], top], axis=1)
        if smooth:
            N = np.stack([nrm, nrm[nxt], nrm[nxt], nrm], axis=1)
        else:
            mid = _normalize(nrm + nrm[nxt])
            N = np.repeat(mid[:, None, :], 4, axis=1)
        idx = np.arange(segs * 4).reshape(segs, 4)
        tri = np.concatenate([idx[:, [0, 1, 2]], idx[:, [0, 2, 3]]], axis=0)
        self.add(P.reshape(-1, 3), N.reshape(-1, 3), col, None, tri)
        if caps:
            tc = np.asarray(rgba(top_color) if top_color is not None else rgba(color), dtype=np.float32)
            if r1 > 1e-4:
                v = np.vstack([[cx, cy + height, cz], top])
                t = np.array([[0, 1 + i, 1 + (i + 1) % segs] for i in range(segs)])
                self.add(v, [0, 1, 0], tc, None, t)
            if r0 > 1e-4:
                v = np.vstack([[cx, cy, cz], base])
                t = np.array([[0, 1 + (i + 1) % segs, 1 + i] for i in range(segs)])
                self.add(v, [0, -1, 0], col, None, t)

    def sphere(self, center, radius, segs: int = 14, rings: int = 8, color=(1, 1, 1, 1), squash: float = 1.0,
               y_range: Tuple[float, float] = (-1.0, 1.0)) -> None:
        """UV sphere (or spherical band: ``y_range`` clips by sin(latitude)); ``squash`` scales the vertical radius."""
        cx, cy, cz = center
        lat0, lat1 = math.asin(y_range[0]), math.asin(y_range[1])
        lat = np.linspace(lat0, lat1, rings + 1)
        lon = np.linspace(0.0, 2.0 * math.pi, segs + 1)
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        nx_, ny_, nz_ = np.cos(LA) * np.cos(LO), np.sin(LA), np.cos(LA) * np.sin(LO)
        radius = np.asarray(radius, dtype=np.float32)
        v = np.stack([cx + radius * nx_, cy + radius * squash * ny_, cz + radius * nz_], axis=-1).reshape(-1, 3)
        n = _normalize(np.stack([nx_, ny_ / max(squash, 1e-3), nz_], axis=-1).reshape(-1, 3))
        idx = np.arange((rings + 1) * (segs + 1)).reshape(rings + 1, segs + 1)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        uv = np.stack([LO / (2 * math.pi), (LA - lat0) / max(lat1 - lat0, 1e-6)], axis=-1).reshape(-1, 2)
        self.add(v, n, np.asarray(rgba(color), dtype=np.float32), uv, tri)

    # ------------------------------------------------------------------ extrusion (buildings)
    def extrude(self, ring_xz: np.ndarray, y0: float, y1: float, wall_color, roof_color, tris,
                tile_w: float = 3.4, tile_h: float = 3.3, shade_dir=(-0.5, 0.8), floors: Optional[int] = None) -> None:
        """Walls of a footprint (CCW in x,z) from y0 to y1 with facade UVs + a flat roof."""
        ring = np.asarray(ring_xz, dtype=np.float64)
        if _ring_area(ring) < 0:
            ring = ring[::-1]
        a = ring
        b = np.roll(ring, -1, axis=0)
        d = b - a
        L = np.hypot(d[:, 0], d[:, 1])
        keep = L > 0.05
        a, b, d, L = a[keep], b[keep], d[keep], L[keep]
        if len(a) == 0:
            return
        outward = np.column_stack([d[:, 1], -d[:, 0]]) / L[:, None]       # right of the walking direction (CCW ring)
        sd = np.asarray(shade_dir, dtype=np.float64)
        sd = sd / np.hypot(*sd)
        k = 0.58 + 0.42 * np.clip(outward @ sd, 0.0, 1.0)
        base = np.asarray(rgba(wall_color), dtype=np.float32)
        col = np.stack([base[:3] * kk for kk in k]).astype(np.float32)
        col = np.concatenate([col, np.ones((len(col), 1), dtype=np.float32)], axis=1)
        K = len(a)
        P = np.zeros((K, 4, 3), dtype=np.float32)
        P[:, 0, 0], P[:, 0, 2], P[:, 0, 1] = a[:, 0], a[:, 1], y0
        P[:, 1, 0], P[:, 1, 2], P[:, 1, 1] = b[:, 0], b[:, 1], y0
        P[:, 2, 0], P[:, 2, 2], P[:, 2, 1] = b[:, 0], b[:, 1], y1
        P[:, 3, 0], P[:, 3, 2], P[:, 3, 1] = a[:, 0], a[:, 1], y1
        nrm = np.zeros((K, 3), dtype=np.float32)
        nrm[:, 0], nrm[:, 2] = outward[:, 0], outward[:, 1]
        nbays = np.maximum(1.0, np.round(L / tile_w))
        fl = float(floors) if floors else max(1.0, round((y1 - y0) / tile_h))
        uv = np.zeros((K, 4, 2), dtype=np.float32)
        uv[:, 1, 0] = nbays
        uv[:, 2, 0] = nbays
        uv[:, 2, 1] = fl
        uv[:, 3, 1] = fl
        self.quads(P, col, uv, nrm)                       # the explicit normal decides what is outside
        self.polygon(ring[:, :2], y1, roof_color, ear_clip(ring))


def _ring_area(r: np.ndarray) -> float:
    return 0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))


def ear_clip(ring: np.ndarray) -> List[Tuple[int, int, int]]:
    """Triangulate a simple polygon (CCW) by ear clipping; falls back to a fan when stuck."""
    ring = np.asarray(ring, dtype=np.float64)
    if _ring_area(ring) < 0:                          # work on a CCW copy, report indices of the input ring
        order = list(range(len(ring) - 1, -1, -1))
    else:
        order = list(range(len(ring)))
    idx = order[:]
    tris: List[Tuple[int, int, int]] = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % n]
            a, b, c = ring[i0], ring[i1], ring[i2]
            if cross(a, b, c) <= 1e-9:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = ring[j]
                if cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12 and cross(c, a, p) >= -1e-12:
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    elif len(idx) > 3:
        tris.extend((idx[0], idx[j], idx[j + 1]) for j in range(1, len(idx) - 1))
    return tris


# ---------------------------------------------------------------------------------------- instancing
def instances(template: Mesh, pos: np.ndarray, yaw: np.ndarray, scale, tint: Optional[np.ndarray] = None) -> Mesh:
    """Copy ``template`` K times (vectorised). ``pos`` (K, 3), ``yaw`` (K,), ``scale`` scalar / (K,) / (K, 3)."""
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    K = len(pos)
    if K == 0:
        return MeshBuilder().build()
    yaw = np.broadcast_to(np.asarray(yaw, dtype=np.float32), (K,))
    sc = np.asarray(scale, dtype=np.float32)
    if sc.ndim == 0:
        sc = np.full((K, 1, 3), float(sc), dtype=np.float32)
    elif sc.ndim == 1:
        sc = np.repeat(sc.reshape(K, 1, 1), 3, axis=2)
    else:
        sc = sc.reshape(K, 1, 3)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((K, 3, 3), dtype=np.float32)
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    v = template.v[None, :, :] * sc                                   # (K, Nv, 3)
    v = np.einsum("kij,knj->kni", R, v) + pos[:, None, :]
    n = np.einsum("kij,nj->kni", R, template.n)
    nv = len(template.v)
    cols = np.broadcast_to(template.c[None], (K, nv, 4)).copy()
    if tint is not None:
        tint = np.asarray(tint, dtype=np.float32).reshape(K, 1, -1)
        cols[:, :, :3] *= tint[:, :, :3]
    t = template.t[None, :, :] + (np.arange(K, dtype=np.int32) * nv)[:, None, None]
    uv = np.broadcast_to(template.uv[None], (K, nv, 2))
    return Mesh(v.reshape(-1, 3), n.reshape(-1, 3), cols.reshape(-1, 4), np.ascontiguousarray(uv).reshape(-1, 2),
                t.reshape(-1, 3))


def split_into_tiles(mesh: Mesh, tile: float = 300.0) -> List[Tuple[Tuple[int, int], Mesh]]:
    """Partition by the triangle centroid so the renderer can frustum-cull chunks."""
    if mesh.empty:
        return []
    cen = mesh.v[mesh.t].mean(axis=1)
    key = np.stack([np.floor(cen[:, 0] / tile), np.floor(cen[:, 2] / tile)], axis=1).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    out = []
    for g, k in enumerate(uniq):
        tri = mesh.t[inv == g]
        used, remap = np.unique(tri, return_inverse=True)
        out.append(((int(k[0]), int(k[1])),
                    Mesh(mesh.v[used], mesh.n[used], mesh.c[used], mesh.uv[used], remap.reshape(-1, 3).astype(np.int32))))
    return out

