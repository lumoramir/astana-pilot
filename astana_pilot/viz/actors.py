"""Procedural models of the road users (cars, buses, pedestrians) and the node pool that animates them.

Models are lofted from a few cross-sections (cheap, smooth, no assets to download). The paint is stored white so a
``color_scale`` tints it per instance. Local axes: +z = front, +x = right, y = up, origin = centre of the footprint on
the road surface.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
from panda3d.core import NodePath, Vec4

from .gfx import mesh_to_node
from .mesh import Mesh, MeshBuilder, rgba

FADE_S = 0.45


def panda_heading(theta: float) -> float:
    """Panda ``H`` (degrees) that turns a model whose front is +z towards simulation heading ``theta`` (CCW from east).
    In the viewer's left-handed y-up frame Panda's H runs the other way round, hence the minus sign (verified numerically
    in tests/test_viz_core.py::test_heading_convention)."""
    return math.degrees(theta - math.pi / 2.0)


def _ease(f: float) -> float:
    # never exactly 0: a zero scale makes a singular transform and Panda asserts (has_mat) while rendering
    return max(0.02, f * f * (3.0 - 2.0 * f))


PAINT = (1.0, 1.0, 1.0, 1.0)
GLASS = (0.16, 0.22, 0.30, 1.0)
TIRE = (0.045, 0.045, 0.05, 1.0)
RIM = (0.62, 0.64, 0.68, 1.0)
CAR_COLORS = [(0.86, 0.86, 0.88), (0.12, 0.12, 0.14), (0.55, 0.57, 0.60), (0.70, 0.12, 0.12), (0.12, 0.28, 0.55),
              (0.92, 0.92, 0.90), (0.75, 0.75, 0.78), (0.18, 0.35, 0.25), (0.85, 0.70, 0.20), (0.35, 0.18, 0.12),
              (0.78, 0.80, 0.84), (0.20, 0.22, 0.26)]
SHIRTS = [(0.85, 0.2, 0.2), (0.2, 0.45, 0.8), (0.9, 0.75, 0.2), (0.25, 0.65, 0.4), (0.8, 0.8, 0.82), (0.3, 0.3, 0.35),
          (0.8, 0.45, 0.7), (0.95, 0.55, 0.2)]
SKIN = [(0.93, 0.76, 0.62), (0.80, 0.62, 0.48), (0.65, 0.46, 0.34), (0.96, 0.82, 0.70)]


def _ring(yb: float, yt: float, hw: float, kind: str) -> np.ndarray:
    """Cross-section outline (x, y) of one loft station."""
    if kind == "cabin":                                     # tapered greenhouse
        return np.array([(-hw, yb), (hw, yb), (hw * 0.82, yt), (-hw * 0.82, yt)])
    h = yt - yb
    return np.array([(-hw * 0.90, yb), (hw * 0.90, yb), (hw, yb + 0.30 * h), (hw * 0.96, yb + 0.80 * h),
                     (hw * 0.70, yt), (-hw * 0.70, yt), (-hw * 0.96, yb + 0.80 * h), (-hw, yb + 0.30 * h)])


def loft(b: MeshBuilder, sections: Sequence[Tuple[float, float, float, float]], color, cap: bool = True,
         top_color=None, kind: str = "body") -> None:
    """Loft along z. ``sections``: ``(z, y_bottom, y_top, half_width)`` rear -> front. Each station is a rounded
    8-gon (``body``) or a tapered 4-gon (``cabin``); faces are smooth-ish and the paint gets darker towards the sill."""
    S = np.asarray(sections, dtype=np.float32)
    n = len(S)
    rings = [_ring(float(s[1]), float(s[2]), float(s[3]), kind) for s in S]
    R = len(rings[0])
    pts = np.zeros((n, R, 3), dtype=np.float32)
    for i, (s, r) in enumerate(zip(S, rings)):
        pts[i, :, 0], pts[i, :, 1], pts[i, :, 2] = r[:, 0], r[:, 1], s[0]
    col = np.asarray(rgba(color), dtype=np.float32)
    ctr = pts.mean(axis=(0, 1))
    for k in range(R):
        k1 = (k + 1) % R
        a0, a1 = pts[:-1, k], pts[1:, k]
        b0, b1 = pts[:-1, k1], pts[1:, k1]
        Q = np.stack([a0, a1, b1, b0], axis=1)
        nrm = np.cross(Q[:, 3] - Q[:, 0], Q[:, 1] - Q[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        flip = ((Q.mean(axis=1) - ctr) * nrm).sum(axis=1) < 0
        P = Q.copy()
        P[flip] = Q[flip][:, ::-1]
        nrm[flip] *= -1.0
        face_col = col.copy()
        ymid = float(pts[:, [k, k1], 1].mean())
        yb_ = float(pts[:, :, 1].min())
        yt_ = float(pts[:, :, 1].max())
        face_col[:3] *= 0.62 + 0.38 * (ymid - yb_) / max(yt_ - yb_, 1e-6)
        b.quads(P, face_col, None, nrm)
    if cap:
        for idx, z_dir in ((0, -1.0), (n - 1, 1.0)):
            ring = pts[idx]
            c0 = ring.mean(axis=0)
            for k in range(R):
                tri = np.array([[c0, ring[k], ring[(k + 1) % R]]], dtype=np.float32)
                nn = np.array([[0.0, 0.0, z_dir]], dtype=np.float32)
                # triangle as a degenerate quad (repeat the last corner) keeps the quad() winding logic
                Q = np.array([[c0, ring[k], ring[(k + 1) % R], ring[(k + 1) % R]]], dtype=np.float32)
                cr = np.cross(Q[:, 3] - Q[:, 0], Q[:, 1] - Q[:, 0])
                if float((cr * nn).sum()) < 0:
                    Q = Q[:, ::-1]
                b.quads(Q, col, None, nn)


def wheel(b: MeshBuilder, x: float, y: float, z: float, r: float, w: float, segs: int = 14) -> None:
    """Horizontal cylinder (axis = x)."""
    a = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    for sgn in (1,):
        pass
    sub = MeshBuilder()
    sub.cylinder((0, -w / 2, 0), r, w, segs, TIRE, caps=False, smooth=True)
    for yy, flip in ((w / 2 + 0.004, False), (-w / 2 - 0.004, True)):           # rim discs on both faces
        a = np.linspace(0, 2 * math.pi, segs, endpoint=False)
        v = np.vstack([[0, yy, 0], np.column_stack([r * 0.68 * np.cos(a), np.full(segs, yy), r * 0.68 * np.sin(a)])])
        t = [(0, 1 + i, 1 + (i + 1) % segs) if not flip else (0, 1 + (i + 1) % segs, 1 + i) for i in range(segs)]
        sub.add(v, [0, -1.0 if flip else 1.0, 0], np.asarray(RIM, dtype=np.float32), None, t)
    m = sub.build()
    # rotate the vertical cylinder so its axis points along x: (x,y,z) -> (y, -x, z)
    v = m.v.copy()
    v = np.stack([v[:, 1], -v[:, 0], v[:, 2]], axis=1) + np.array([x, y, z], dtype=np.float32)
    n = np.stack([m.n[:, 1], -m.n[:, 0], m.n[:, 2]], axis=1)
    # the rotation is a proper rotation about z by -90 deg, so winding is preserved
    b.add(v, n, m.c, m.uv, m.t)


def car_models(length: float = 4.5, width: float = 1.9, suv: bool = False) -> Tuple[Mesh, Mesh, Mesh, Mesh]:
    """``(paint, glass, wheels, lights)`` - lights are unlit emissive quads (head front, tail rear)."""
    L, W = length, width
    hw = W / 2
    k = 1.0 + (0.12 if suv else 0.0)
    ride = 0.28 + (0.07 if suv else 0.0)
    paint, glass, wheels, lights = MeshBuilder(), MeshBuilder(), MeshBuilder(), MeshBuilder()
    # body: z fractions of L
    z = np.array([-0.5, -0.49, -0.46, -0.40, -0.30, 0.0, 0.20, 0.31, 0.40, 0.46, 0.49, 0.5]) * L
    yt = np.array([0.74, 0.86, 0.93, 0.97, 0.98, 0.98, 0.98, 0.96, 0.90, 0.80, 0.68, 0.60]) * k
    yb = np.full(len(z), ride)
    yb[0], yb[1], yb[-2], yb[-1] = ride + 0.12, ride + 0.05, ride + 0.03, ride + 0.07
    w = np.array([0.40, 0.45, 0.48, 0.50, 0.50, 0.50, 0.50, 0.50, 0.49, 0.47, 0.44, 0.40]) * W
    loft(paint, list(zip(z, yb, yt, w)), PAINT)
    # greenhouse (glass) + roof (paint)
    gz = np.array([-0.30, -0.255, -0.18, -0.02, 0.10, 0.20]) * L
    gb = np.array([0.96, 0.96, 0.96, 0.96, 0.96, 0.96]) * k
    gt = np.array([1.18, 1.38, 1.46, 1.44, 1.30, 1.02]) * k
    gw = np.array([0.43, 0.41, 0.40, 0.40, 0.41, 0.44]) * W
    loft(glass, list(zip(gz, gb, gt, gw)), GLASS, kind="cabin")
    rz = np.array([-0.17, -0.02]) * L
    loft(paint, [(rz[0], 1.43 * k, 1.47 * k, 0.37 * W), (rz[1], 1.41 * k, 1.455 * k, 0.37 * W)], PAINT, kind="cabin")
    # wheels
    r = 0.34 + (0.04 if suv else 0.0)
    for sx in (-1, 1):
        for zz in (0.31 * L, -0.30 * L):
            wheel(wheels, sx * (hw - 0.06), r, zz, r, 0.24)
    # head / tail lights
    for sx in (-1, 1):
        q = ((sx * hw * 0.80, 0.62 * k, L / 2 + 0.01), (sx * hw * 0.42, 0.62 * k, L / 2 + 0.01),
             (sx * hw * 0.42, 0.78 * k, L / 2 - 0.05), (sx * hw * 0.80, 0.78 * k, L / 2 - 0.12))
        lights.quad(*(q[::-1] if sx > 0 else q), (1.0, 0.97, 0.85, 1.0), normal=(0, 0, 1))
        q = ((sx * hw * 0.82, 0.74 * k, -L / 2 - 0.01), (sx * hw * 0.40, 0.74 * k, -L / 2 - 0.01),
             (sx * hw * 0.40, 0.86 * k, -L / 2 + 0.02), (sx * hw * 0.82, 0.86 * k, -L / 2 + 0.02))
        lights.quad(*(q if sx > 0 else q[::-1]), (0.85, 0.05, 0.05, 1.0), normal=(0, 0, -1))
    return paint.build(), glass.build(), wheels.build(), lights.build()


def bus_models(length: float = 12.0, width: float = 2.55) -> Tuple[Mesh, Mesh, Mesh, Mesh]:
    L, W = length, width
    hw = W / 2
    paint, glass, wheels, lights = MeshBuilder(), MeshBuilder(), MeshBuilder(), MeshBuilder()
    z = np.array([-0.5, -0.49, 0.49, 0.5]) * L
    loft(paint, list(zip(z, [0.35, 0.30, 0.30, 0.38], [3.15, 3.25, 3.25, 3.0], np.array([0.46, 0.5, 0.5, 0.47]) * W)), PAINT)
    # window band (slightly proud of the body)
    gz = np.array([-0.46, 0.46]) * L
    loft(glass, [(gz[0], 1.45, 2.65, 0.502 * W), (gz[1], 1.45, 2.65, 0.502 * W)], GLASS, cap=False)
    loft(glass, [(0.498 * L, 1.0, 2.75, 0.45 * W), (0.502 * L, 1.0, 2.75, 0.45 * W)], GLASS)    # windscreen
    for sx in (-1, 1):
        for zz in (0.32 * L, -0.30 * L, -0.40 * L):
            wheel(wheels, sx * (hw - 0.1), 0.52, zz, 0.52, 0.3, 16)
    for sx in (-1, 1):
        q = ((sx * hw * 0.8, 0.8, L / 2 + 0.02), (sx * hw * 0.5, 0.8, L / 2 + 0.02),
             (sx * hw * 0.5, 1.0, L / 2 + 0.02), (sx * hw * 0.8, 1.0, L / 2 + 0.02))
        lights.quad(*(q[::-1] if sx > 0 else q), (1.0, 0.97, 0.85, 1.0), normal=(0, 0, 1))
        q = ((sx * hw * 0.8, 0.9, -L / 2 - 0.02), (sx * hw * 0.5, 0.9, -L / 2 - 0.02),
             (sx * hw * 0.5, 1.15, -L / 2 - 0.02), (sx * hw * 0.8, 1.15, -L / 2 - 0.02))
        lights.quad(*(q if sx > 0 else q[::-1]), (0.85, 0.05, 0.05, 1.0), normal=(0, 0, -1))
    return paint.build(), glass.build(), wheels.build(), lights.build()


def pedestrian_models() -> Dict[str, Mesh]:
    """Parts in local coordinates, limbs hang from their pivot (so a rotation about x swings them)."""
    out: Dict[str, Mesh] = {}
    b = MeshBuilder()                                              # torso + shorts (tinted)
    b.box((0, 1.22, 0), (0.38, 0.56, 0.2), color=PAINT)
    out["torso"] = b.build()
    b = MeshBuilder()
    b.sphere((0, 1.66, 0), 0.115, 9, 6, (1, 1, 1, 1))              # head (tinted with skin)
    out["head"] = b.build()
    b = MeshBuilder()
    b.box((0, -0.43, 0), (0.15, 0.86, 0.17), color=(0.16, 0.18, 0.28, 1))   # leg hanging from y=0 pivot
    out["leg"] = b.build()
    b = MeshBuilder()
    b.box((0, -0.28, 0), (0.1, 0.56, 0.11), color=PAINT)           # arm hanging from y=0 pivot
    out["arm"] = b.build()
    return out


class ActorPool:
    """Creates / recycles NodePaths for road users keyed by actor id; smooths motion between 20 Hz sim steps."""

    def __init__(self, parent: NodePath, lit_shader, unlit_shader, white_tex, quality_max: int = 140) -> None:
        self.parent = parent.attach_new_node("actors")
        self.lit, self.unlit, self.white = lit_shader, unlit_shader, white_tex
        self.nodes: Dict[int, dict] = {}
        self._tpl: Dict[str, tuple] = {}
        self._tnp: Dict[str, NodePath] = {}
        self.max = quality_max
        self.ped = {k: None for k in ("torso", "head", "leg", "arm")}
        self._ped_meshes = pedestrian_models()

    # ------------------------------------------------------------ templates
    def _template(self, kind: str, length: float, width: float):
        key = f"{kind}:{length:.1f}:{width:.1f}"
        if key not in self._tpl:
            suv = kind == "car" and width > 1.95
            meshes = bus_models(length, width) if kind == "bus" else car_models(length, width, suv)
            self._tpl[key] = meshes
        return self._tpl[key]

    def _vehicle_template(self, kind: str, length: float, width: float) -> NodePath:
        key = f"{kind}:{length:.1f}:{width:.1f}"
        if key in self._tnp:
            return self._tnp[key]
        paint, glass, wheels, lights = self._template(kind, length, width)
        root = NodePath(f"tpl-{key}")
        body = mesh_to_node(paint, "paint", root)
        gl = mesh_to_node(glass, "glass", root)
        wh = mesh_to_node(wheels, "wheels", root)
        li = mesh_to_node(lights, "lights", root)
        for n in (body, gl, wh):
            n.set_shader(self.lit)
            n.set_texture(self.white)
            n.set_shader_input("spec_amt", 0.35)
            n.set_shader_input("world_uv", 0.0)
        gl.set_shader_input("spec_amt", 1.0)
        li.set_shader(self.unlit)
        li.set_texture(self.white)
        li.set_shader_input("glow", 1.0)
        bl = MeshBuilder()
        L, W = length, width
        for sx in (-1, 1):                              # brake lights: red glow quads at the rear (shown while braking)
            bl.box((sx * W * 0.34, 0.82 if kind != "bus" else 1.0, -L / 2), (W * 0.30, 0.10, 0.06), color=(1.0, 0.1, 0.08, 1.0))
        brake = mesh_to_node(bl.build(), "brake", root)
        brake.set_shader(self.unlit)
        brake.set_texture(self.white)
        brake.set_shader_input("glow", 2.2)
        self._tnp[key] = root
        return root

    def _make_vehicle(self, a) -> dict:
        tpl = self._vehicle_template(a.kind, a.length, a.width)
        root = tpl.copy_to(self.parent)                  # shares the vertex data: no re-upload, no hitch
        root.set_name(f"veh{a.id}")
        body = root.find("paint")
        rng = np.random.default_rng(a.id * 7919 + 13)
        col = (0.20, 0.62, 0.38) if a.kind == "bus" else CAR_COLORS[int(rng.integers(len(CAR_COLORS)))]
        body.set_color_scale(Vec4(*col, 1.0))
        brake = root.find("brake")
        brake.hide()
        return {"root": root, "brake": brake, "prev": None, "kind": a.kind}

    def _ped_template(self) -> NodePath:
        if "ped" in self._tnp:
            return self._tnp["ped"]
        root = NodePath("tpl-ped")
        for name in ("torso", "head"):
            mesh_to_node(self._ped_meshes[name], name, root)
        for name, (x, y) in (("legL", (-0.09, 0.82)), ("legR", (0.09, 0.82)), ("armL", (-0.25, 1.46)), ("armR", (0.25, 1.46))):
            piv = root.attach_new_node(name)
            piv.set_pos(x, y, 0)
            mesh_to_node(self._ped_meshes["leg" if name.startswith("leg") else "arm"], name + "m", piv)
        for n in root.find_all_matches("**/+GeomNode"):
            n.set_shader(self.lit)
            n.set_texture(self.white)
            n.set_shader_input("spec_amt", 0.1)
            n.set_shader_input("world_uv", 0.0)
        self._tnp["ped"] = root
        return root

    def _make_ped(self, a) -> dict:
        root = self._ped_template().copy_to(self.parent)
        root.set_name(f"ped{a.id}")
        rng = np.random.default_rng(a.id * 104729 + 5)
        shirt = SHIRTS[int(rng.integers(len(SHIRTS)))]
        skin = SKIN[int(rng.integers(len(SKIN)))]
        root.find("torso").set_color_scale(Vec4(*shirt, 1.0))
        root.find("head").set_color_scale(Vec4(*skin, 1.0))
        parts = {"legL": root.find("legL"), "legR": root.find("legR"), "armL": root.find("armL"), "armR": root.find("armR"),
                 "head": root.find("head"), "torso": root.find("torso")}
        for nm in ("armL", "armR"):
            parts[nm].find("**/+GeomNode").set_color_scale(Vec4(*shirt, 1.0))
        root.set_sx(float(rng.uniform(0.94, 1.06)))                          # every person a little different
        root.set_sy(root.get_sx())
        root.set_sz(float(rng.uniform(0.93, 1.08)))
        return {"root": root, "parts": parts, "prev": None, "kind": "ped", "phase": float(rng.uniform(0, 6.28)),
                "idle": float(rng.uniform(0, 100.0)), "tempo": float(rng.uniform(0.85, 1.2)), "base_sz": root.get_sz(),
                "base_sx": root.get_sx()}

    def prewarm(self, actors) -> None:
        """Build the shared templates for every kind/size that is around, so no frame pays for it later."""
        self._ped_template()
        for a in actors:
            if a.kind != "ped":
                self._vehicle_template(a.kind, a.length, a.width)
        for kind, L, W in (("car", 4.4, 1.8), ("bus", 12.0, 2.55), ("parked", 4.4, 1.8)):
            self._vehicle_template(kind, L, W)

    # ------------------------------------------------------------ per frame
    def update(self, actors, alpha: float, dt: float, hide_ids: Sequence[int] = ()) -> None:
        """``actors``: current snapshot (Actor list). ``alpha``: interpolation factor between the previous and the
        current sim state (0..1)."""
        live = set()
        n = 0
        for a in actors:
            if not a.alive or a.id in hide_ids:
                continue
            n += 1
            if n > self.max:
                break
            live.add(a.id)
            st = self.nodes.get(a.id)
            if st is None:
                st = self._make_ped(a) if a.kind == "ped" else self._make_vehicle(a)
                self.nodes[a.id] = st
            root = st["root"]
            root.show()
            st["fade"] = min(1.0, st.get("fade", 0.0) + dt / FADE_S)          # fade-in: appear smoothly, never pop
            root.set_scale(_ease(st["fade"]))
            px, py, pth, pz = st["prev"] if st["prev"] else (a.x, a.y, a.theta, a.z)
            x, y = px + (a.x - px) * alpha, py + (a.y - py) * alpha
            d = (a.theta - pth + math.pi) % (2 * math.pi) - math.pi
            th = pth + d * alpha
            z = pz + (a.z - pz) * alpha
            root.set_pos(x, z + (0.05 if a.kind != "ped" else 0.15), y)
            root.set_h(panda_heading(th))
            st["next"] = (a.x, a.y, a.theta, a.z)
            if st["kind"] == "ped":
                self._animate_ped(st, a, dt)
            elif st["kind"] in ("car", "bus", "parked"):
                if a.kind != "parked":
                    (st["brake"].show() if getattr(a, "braking", False) else st["brake"].hide())
        for k in list(self.nodes):
            if k in live:
                continue
            st = self.nodes[k]
            if st.get("fade", 0.0) > 0.0:                                     # fade-out where it stood
                st["fade"] = max(0.0, st["fade"] - dt / FADE_S)
                st["root"].set_scale(_ease(st["fade"]))
                if st["fade"] <= 0.0:
                    st["root"].hide()
            elif len(self.nodes) > 3 * self.max:
                self.nodes.pop(k)["root"].remove_node()

    def commit(self) -> None:
        """Call once per sim step *before* the next ``update``: current state becomes the previous one."""
        for st in self.nodes.values():
            if "next" in st:
                st["prev"] = st["next"]

    def _animate_ped(self, st: dict, a, dt: float) -> None:
        p = st["parts"]
        st["idle"] += dt
        t = st["idle"]
        moving = abs(a.v) > 0.03 or (a.state == "walk" and getattr(a, "speed", 0.0) > 0.03)
        spd = max(abs(a.v), 0.6 if a.state == "walk" else 0.0)
        # smooth blend between idle and walking so nobody freezes or snaps into a pose
        st["mv"] = st.get("mv", 0.0) + ((1.0 if moving else 0.0) - st.get("mv", 0.0)) * min(1.0, dt * 6.0)
        mv = st["mv"]
        st["phase"] += dt * (2.2 + 2.6 * min(spd, 1.8)) * st["tempo"] * mv * 1.6
        sw = math.sin(st["phase"]) * (24.0 + 16.0 * min(spd / 1.4, 1.0)) * mv
        breathe = math.sin(t * 1.7) * 0.012
        look = math.sin(t * 0.55) * 28.0 + math.sin(t * 1.3 + 2.0) * 10.0          # glances around
        shift = math.sin(t * 0.7) * 3.5 * (1.0 - mv)                                 # weight shift while waiting
        p["legL"].set_p(sw + shift)
        p["legR"].set_p(-sw - shift)
        p["armL"].set_p(-sw * 0.85 + math.sin(t * 0.9) * 3.0 * (1.0 - mv))
        p["armR"].set_p(sw * 0.85 - math.sin(t * 0.9 + 1.0) * 3.0 * (1.0 - mv))
        p["armL"].set_r(-2.0 - 3.0 * (1.0 - mv))
        p["armR"].set_r(2.0 + 3.0 * (1.0 - mv))
        bob = abs(math.sin(st["phase"])) * 0.035 * mv
        p["torso"].set_z(bob)
        p["head"].set_z(bob)
        p["torso"].set_sz(1.0 + breathe)
        p["torso"].set_r(math.sin(st["phase"]) * 2.5 * mv)                          # hip sway
        p["head"].set_h(look * (1.0 - 0.7 * mv))
