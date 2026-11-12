"""Turns a :class:`SceneData` into Panda3D nodes (shaders, textures, shadows); owns the sky and the atmosphere."""
from __future__ import annotations

import builtins
from typing import Dict, Optional

import numpy as np
from panda3d.core import (AntialiasAttrib, BitMask32, DepthOffsetAttrib, NodePath, Texture, TransparencyAttrib)

from . import textures as tx
from .city_scene import Layer, SceneData
from .gfx import Atmosphere, mesh_to_node, sky_mesh, texture_from_array, update_vertex_colors
from .quality import Quality
from .shaders import city_shader, unlit_shader

SHADOW_BIT = BitMask32.bit(1)


class TextureBank:
    """Lazily generated procedural textures by key."""

    def __init__(self, quality: Quality) -> None:
        self.q = quality
        self._c: Dict[str, Texture] = {}

    def get(self, key: str) -> Texture:
        if key not in self._c:
            self._c[key] = self._make(key)
        return self._c[key]

    def _make(self, key: str) -> Texture:
        if not self.q.textures and key not in ("white",) and not key.startswith("sign_"):
            if key.startswith("facade_"):                       # flat facade without window detail on low quality
                img = tx.facade(key.split("_", 1)[1], 32)
                return texture_from_array(img, key, True, True, 1)
            fn = {"asphalt": tx.asphalt, "grass": tx.grass, "steppe": tx.steppe, "paving": tx.paving,
                  "concrete": tx.concrete}.get(key)
            if fn is not None:                              # same average brightness as the real texture, one flat colour
                avg = fn(32).reshape(-1, 4).mean(axis=0).astype(np.uint8) if key != "steppe" else fn(32).reshape(-1, 4).mean(axis=0).astype(np.uint8)
                return texture_from_array(np.tile(avg, (4, 4, 1)), key)
            return texture_from_array(tx.white_pixel(), key)
        if key == "white":
            return texture_from_array(tx.white_pixel(), key)
        if key.startswith("facade_"):
            return texture_from_array(tx.facade(key.split("_", 1)[1], 128), key, True, True, 8)
        if key == "sign_stop":
            return texture_from_array(tx.stop_sign(), key, False, True, 4)
        if key == "sign_yield":
            return texture_from_array(tx.yield_sign(), key, False, True, 4)
        if key.startswith("sign_speed_"):
            return texture_from_array(tx.speed_sign(int(key.rsplit("_", 1)[1])), key, False, True, 4)
        fn = {"asphalt": tx.asphalt, "grass": tx.grass, "steppe": tx.steppe, "paving": tx.paving,
              "concrete": tx.concrete}[key]
        return texture_from_array(fn(), key, True, True, 8)


class StaticRenderer:
    """All static geometry of a city + sky + lights."""

    def __init__(self, scene: SceneData, quality: Quality, preset: str = "day", render: Optional[NodePath] = None) -> None:
        self.render = render or builtins.base.render  # type: ignore[attr-defined]
        self.q = quality
        self.scene = scene
        self.bank = TextureBank(quality)
        self.atm = Atmosphere(self.render, quality.shadows, quality.shadow_size)
        self.atm.set(preset)
        self.root = self.render.attach_new_node("astana-static")
        self.lit = city_shader()
        self.unlit = unlit_shader()
        self.layer_nodes: Dict[str, NodePath] = {}
        for name, layer in scene.layers.items():
            if layer.meshes:
                self.layer_nodes[name] = self._build_layer(layer)
        self.sky: Optional[NodePath] = None
        self._build_sky()
        self.lamps: Optional[NodePath] = None
        self.light_geom_colors: Optional[np.ndarray] = None
        self._build_signal_lamps()
        self.render.set_antialias(AntialiasAttrib.M_multisample if quality.msaa else AntialiasAttrib.M_none)

    # ---------------------------------------------------------------------- layers
    def _build_layer(self, layer: Layer) -> NodePath:
        ln = self.root.attach_new_node(layer.name)
        for tile, mesh in layer.meshes:
            mesh_to_node(mesh, f"{layer.name}_{tile[0]}_{tile[1]}", ln)
        ln.set_texture(self.bank.get(layer.texture))
        if layer.unlit:
            ln.set_shader(self.unlit)
            ln.set_shader_input("glow", 1.0)
            ln.set_light_off()
        else:
            ln.set_shader(self.lit)
            ln.set_shader_input("world_uv", float(layer.world_uv))
            ln.set_shader_input("spec_amt", float(layer.spec))
            ln.set_shader_input("glass_amt", float(layer.glass))
            ln.set_shader_input("alpha_cut", 1.0 if layer.name.startswith("sign_") else 0.0)
            ln.set_shader_input("use_alpha", 1.0 if layer.transparent else 0.0)
            ln.set_shader_input("cut_apply", 1.0 if layer.name in ("ground", "green", "water", "road", "walk", "mark", "prop") else 0.0)
        if layer.depth_offset:
            ln.set_attrib(DepthOffsetAttrib.make(layer.depth_offset))
        if layer.transparent:
            ln.set_transparency(TransparencyAttrib.M_alpha)
            ln.set_depth_write(False)
        if not layer.shadows:
            ln.hide(SHADOW_BIT)
        ln.set_two_sided(layer.name in ("lamp", "roof", "tree", "sign_stop", "sign_yield") or layer.name.startswith("sign"))
        return ln

    def _build_sky(self) -> None:
        p = self.atm.preset
        self.sky = mesh_to_node(sky_mesh(self.q.far * 0.9, p["fog"], p["zenith"], p["glow"], self.atm.sun_dir), "sky", self.render)
        self.sky.set_shader(self.unlit)
        self.sky.set_shader_input("use_fog", 0.0)
        self.sky.set_shader_input("glow", 1.0)
        self.sky.set_texture(self.bank.get("white"))
        self.sky.set_light_off()
        self.sky.set_two_sided(True)
        self.sky.set_bin("background", 0)
        self.sky.set_depth_write(False)
        self.sky.hide(SHADOW_BIT)

    def set_preset(self, name: str) -> None:
        self.atm.set(name)
        if self.sky is not None:
            self.sky.remove_node()
        self._build_sky()
        self.render.set_shader_input("glow", 1.0)
        if "lamp" in self.layer_nodes:
            self.layer_nodes["lamp"].set_shader_input("glow", 2.4 if name in ("night", "dusk") else 0.55)

    def set_cutaway(self, on: bool, x: float = 0.0, z: float = 0.0, radius: float = 38.0) -> None:
        """While the car is in a tunnel, hide the surface layers around it so the tunnel is visible from above."""
        from panda3d.core import Vec2
        self.render.set_shader_input("cut_on", 1.0 if on else 0.0)
        if on:
            self.render.set_shader_input("cut_c", Vec2(x, z))
            self.render.set_shader_input("cut_r", float(radius))

    def follow_sky(self, x: float, y: float, z: float) -> None:
        if self.sky is not None:
            self.sky.set_pos(x, 0.0, z)

    # ---------------------------------------------------------------------- traffic-light lamps (dynamic colours)
    def _build_signal_lamps(self) -> None:
        m = self.scene.light_mesh
        if m is None or m.empty:
            return
        self.lamps = mesh_to_node(m, "signal-lamps", self.root, dynamic=True)
        self.lamps.set_shader(self.unlit)
        self.lamps.set_texture(self.bank.get("white"))
        self.lamps.set_shader_input("glow", 1.6)
        self.lamps.set_two_sided(True)
        self.lamps.hide(SHADOW_BIT)
        self._lamp_cols = m.c.copy()
        self._lamp_state = None

    def update_signals(self, infra, t: float, force: bool = False) -> None:
        """Set lamp colours from the signal plan at world time ``t``."""
        if self.lamps is None:
            return
        cols = self._lamp_cols
        off = np.array([0.10, 0.10, 0.10, 1.0], dtype=np.float32)
        on = {0: np.array([0.15, 1.0, 0.35, 1.0], dtype=np.float32),      # green
              1: np.array([1.0, 0.82, 0.10, 1.0], dtype=np.float32),      # yellow
              2: np.array([1.0, 0.14, 0.12, 1.0], dtype=np.float32)}      # red
        ctrl = infra.controls
        state = []
        for h in self.scene.lights:
            c = ctrl[h.control]
            state.append(infra.light_state(c.node, c.group, t))
        state_t = tuple(state)
        if state_t == self._lamp_state and not force:
            return
        self._lamp_state = state_t
        cols[:] = off
        for h, s in zip(self.scene.lights, state):
            lamp = {2: 0, 1: 1, 0: 2}[s]                                    # which of the three lamps is lit
            a = h.first_vertex + lamp * h.verts_per_lamp
            cols[a:a + h.verts_per_lamp] = on[s]
        update_vertex_colors(self.lamps, cols)
