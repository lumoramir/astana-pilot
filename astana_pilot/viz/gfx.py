"""Panda3D bridge: fast conversion of :class:`Mesh` to a ``GeomNode``, textures, lights, fog, sky."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from panda3d.core import (AmbientLight, DirectionalLight, Geom, GeomNode, GeomTriangles, GeomVertexArrayFormat, GeomVertexData,
                          GeomVertexFormat, InternalName, NodePath, SamplerState, Texture,
                          Vec2, Vec3, Vec4)

from .mesh import Mesh

_FMT: Optional[GeomVertexFormat] = None


def _format() -> GeomVertexFormat:
    global _FMT
    if _FMT is None:
        arr = GeomVertexArrayFormat()
        arr.add_column(InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point)
        arr.add_column(InternalName.get_normal(), 3, Geom.NT_float32, Geom.C_normal)
        arr.add_column(InternalName.get_color(), 4, Geom.NT_float32, Geom.C_color)
        arr.add_column(InternalName.get_texcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        fmt = GeomVertexFormat()
        fmt.add_array(arr)
        _FMT = GeomVertexFormat.register_format(fmt)
    return _FMT


def mesh_to_geom(mesh: Mesh, name: str = "mesh", dynamic: bool = False) -> Geom:
    n = len(mesh.v)
    usage = Geom.UH_dynamic if dynamic else Geom.UH_static
    vdata = GeomVertexData(name, _format(), usage)
    vdata.unclean_set_num_rows(n)
    buf = np.empty((n, 12), dtype=np.float32)
    buf[:, 0:3], buf[:, 3:6], buf[:, 6:10], buf[:, 10:12] = mesh.v, mesh.n, mesh.c, mesh.uv
    vdata.modify_array(0).modify_handle().copy_data_from(buf.tobytes())
    prim = GeomTriangles(usage)
    prim.set_index_type(Geom.NT_uint32 if n > 65000 else Geom.NT_uint16)
    idx = mesh.t.astype(np.uint32 if n > 65000 else np.uint16).ravel()
    prim.modify_vertices().modify_handle().copy_data_from(idx.tobytes())
    geom = Geom(vdata)
    geom.add_primitive(prim)
    return geom


def mesh_to_node(mesh: Mesh, name: str = "mesh", parent: Optional[NodePath] = None, dynamic: bool = False) -> NodePath:
    gn = GeomNode(name)
    if not mesh.empty:
        gn.add_geom(mesh_to_geom(mesh, name, dynamic))
    np_ = NodePath(gn)
    if parent is not None:
        np_.reparent_to(parent)
    return np_


def update_vertex_colors(np_: NodePath, colors: np.ndarray) -> None:
    """Overwrite the per-vertex colours of a node built with :func:`mesh_to_node` (``dynamic=True`` advisable)."""
    gn = np_.node()
    vdata = gn.modify_geom(0).modify_vertex_data()
    handle = vdata.modify_array(0).modify_handle()
    n = vdata.get_num_rows()
    buf = np.frombuffer(handle.get_data(), dtype=np.float32).reshape(n, 12).copy()
    buf[:, 6:10] = colors
    handle.copy_data_from(buf.tobytes())


# -------------------------------------------------------------------------------------- textures
def texture_from_array(img: np.ndarray, name: str = "tex", repeat: bool = True, mipmaps: bool = True,
                       aniso: int = 8) -> Texture:
    """``img``: (H, W, 3|4) uint8 with row 0 at the TOP."""
    img = np.ascontiguousarray(np.flipud(img))
    h, w, ch = img.shape
    tex = Texture(name)
    tex.setup_2d_texture(w, h, Texture.T_unsigned_byte, Texture.F_rgba if ch == 4 else Texture.F_rgb)
    tex.set_ram_image_as(img.tobytes(), "RGBA" if ch == 4 else "RGB")
    if repeat:
        tex.set_wrap_u(SamplerState.WM_repeat)
        tex.set_wrap_v(SamplerState.WM_repeat)
    else:
        tex.set_wrap_u(SamplerState.WM_clamp)
        tex.set_wrap_v(SamplerState.WM_clamp)
    if mipmaps:
        tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
        tex.set_anisotropic_degree(aniso)
    else:
        tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


def update_texture(tex: Texture, img: np.ndarray) -> None:
    """Replace the pixels of an existing RGBA texture of the same size (HUD panels)."""
    # Panda keeps RGBA textures as BGRA in RAM: hand it that order directly (set_ram_image_as converts per pixel: 5x slower)
    tex.set_ram_image(np.ascontiguousarray(img[::-1, :, [2, 1, 0, 3]]).tobytes())


# -------------------------------------------------------------------------------------- lights / fog
class Atmosphere:
    """Sun + ambient light + fog + sky colour, switchable between presets (day / dusk / night / snow)."""

    PRESETS = {
        #            sun colour           ambient            fog / horizon       zenith             sun dir (from)
        "day": dict(sun=(1.00, 0.95, 0.85), amb=(0.42, 0.47, 0.56), fog=(0.74, 0.82, 0.90), zenith=(0.30, 0.52, 0.86),
                    sun_dir=(-0.45, -0.75, -0.55), fog_density=0.00030, glow=0.0, ground=(0.58, 0.60, 0.50)),
        "dusk": dict(sun=(1.00, 0.62, 0.34), amb=(0.34, 0.31, 0.42), fog=(0.88, 0.60, 0.48), zenith=(0.20, 0.25, 0.52),
                     sun_dir=(-0.80, -0.28, -0.35), fog_density=0.00075, glow=0.25, ground=(0.50, 0.47, 0.42)),
        "night": dict(sun=(0.34, 0.42, 0.72), amb=(0.15, 0.17, 0.30), fog=(0.05, 0.07, 0.14), zenith=(0.01, 0.02, 0.06),
                      sun_dir=(-0.30, -0.80, -0.50), fog_density=0.0009, glow=1.0, ground=(0.22, 0.24, 0.26)),
        "snow": dict(sun=(0.92, 0.95, 1.00), amb=(0.52, 0.56, 0.64), fog=(0.82, 0.86, 0.90), zenith=(0.58, 0.66, 0.76),
                     sun_dir=(-0.35, -0.80, -0.45), fog_density=0.0016, glow=0.0, ground=(0.93, 0.95, 0.98)),
    }

    def __init__(self, render: NodePath, shadows: int = 2048, shadow_size: float = 120.0) -> None:
        self.render = render
        self.name = "day"
        self.shadow_res = shadows
        self.shadow_size = shadow_size
        self.ambient = AmbientLight("amb")
        self.sun = DirectionalLight("sun")
        self.sun_np = render.attach_new_node(self.sun)
        self.amb_np = render.attach_new_node(self.ambient)
        render.set_light(self.sun_np)
        render.set_light(self.amb_np)
        if shadows > 0:
            self.sun.set_shadow_caster(True, shadows, shadows)
            lens = self.sun.get_lens()
            lens.set_film_size(shadow_size * 2.0, shadow_size * 2.0)
            lens.set_near_far(10.0, 700.0)
        self.sun_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        render.set_shader_input("shadows_on", 1.0 if shadows > 0 else 0.0)
        render.set_shader_input("head_on", 0.0)
        render.set_shader_input("head_pos", Vec3(0, 0, 0))
        render.set_shader_input("head_dir", Vec3(0, 0, 1))
        render.set_shader_input("spec_amt", 0.0)
        render.set_shader_input("glass_amt", 0.0)
        render.set_shader_input("world_uv", 0.0)
        render.set_shader_input("alpha_cut", 0.0)
        render.set_shader_input("use_alpha", 0.0)
        render.set_shader_input("cut_on", 0.0)
        render.set_shader_input("cut_apply", 0.0)
        render.set_shader_input("cut_c", Vec2(0, 0))
        render.set_shader_input("cut_r", 0.0)
        render.set_shader_input("use_fog", 1.0)
        render.set_shader_input("glow", 1.0)
        self.set("day")

    @property
    def preset(self) -> dict:
        return self.PRESETS[self.name]

    def set(self, name: str) -> None:
        p = self.PRESETS[name]
        self.name = name
        r = self.render
        self.sun.set_color(Vec4(*p["sun"], 1.0))
        self.ambient.set_color(Vec4(*p["amb"], 1.0))
        d = Vec3(*p["sun_dir"])
        d.normalize()
        self.sun_np.look_at(d)                      # light looks along ``d``
        self.sun_dir = np.array(p["sun_dir"], dtype=np.float32) / np.linalg.norm(p["sun_dir"])
        amb = np.array(p["amb"], dtype=np.float32)
        r.set_shader_input("sun_color", Vec3(*(np.array(p["sun"]) * (1.25 if name != "night" else 0.9))))
        r.set_shader_input("amb_sky", Vec3(*(amb * 1.15)))
        r.set_shader_input("amb_ground", Vec3(*(amb * 0.65 * np.array(p["ground"]) * 1.4)))
        r.set_shader_input("fog_color", Vec3(*p["fog"]))
        r.set_shader_input("fog_density", float(p["fog_density"]))
        r.set_shader_input("night", 1.0 if name == "night" else 0.0)

    def set_shadow_res(self, res: int) -> None:
        self.shadow_res = res
        if res > 0:
            self.sun.set_shadow_caster(True, res, res)
            self.render.set_shader_input("shadows_on", 1.0)
        else:
            self.sun.set_shadow_caster(False)
            self.render.set_shader_input("shadows_on", 0.0)

    def follow(self, x: float, y: float, z: float) -> None:
        """Keep the shadow frustum centred on the camera focus, snapped to the shadow-map texel grid."""
        if self.shadow_res <= 0:
            return
        texel = 2.0 * self.shadow_size / self.shadow_res
        x, z = round(x / texel) * texel, round(z / texel) * texel
        d = self.sun_dir
        self.sun_np.set_pos(x - d[0] * 300.0, y - d[1] * 300.0, z - d[2] * 300.0)


def sky_mesh(radius: float, horizon, zenith, glow: float = 0.0, sun_dir=None) -> Mesh:
    """Large inverted gradient dome (vertex colours): horizon -> zenith, optional warm sun glow."""
    segs, rings = 32, 14
    lat = np.linspace(-0.06, 1.0, rings + 1) * (math.pi / 2)
    lon = np.linspace(0, 2 * math.pi, segs + 1)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    v = np.stack([np.cos(LA) * np.cos(LO), np.sin(LA), np.cos(LA) * np.sin(LO)], axis=-1) * radius
    h = np.clip(np.sin(LA), 0.0, 1.0) ** 0.55
    hz, zn = np.asarray(horizon, dtype=np.float32), np.asarray(zenith, dtype=np.float32)
    col = hz[None, None, :] * (1 - h[..., None]) + zn[None, None, :] * h[..., None]
    if glow > 0 and sun_dir is not None:
        sd = -np.asarray(sun_dir, dtype=np.float32)
        sd = sd / np.linalg.norm(sd)
        dirv = v / radius
        k = np.clip((dirv @ sd - 0.6) / 0.4, 0.0, 1.0) ** 2 * glow
        col = col + k[..., None] * np.array([0.55, 0.30, 0.10], dtype=np.float32)
    col = np.clip(col, 0, 1)
    col4 = np.concatenate([col, np.ones(col.shape[:-1] + (1,), dtype=np.float32)], axis=-1).reshape(-1, 4)
    n = -v.reshape(-1, 3) / radius
    idx = np.arange((rings + 1) * (segs + 1)).reshape(rings + 1, segs + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return Mesh(v.reshape(-1, 3).astype(np.float32), n.astype(np.float32), col4.astype(np.float32),
                np.zeros((len(col4), 2), dtype=np.float32), tri.astype(np.int32))
