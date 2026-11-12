"""Interactive / headless 3-D digital twin of a city with a driving AI (Ursina + Panda3D).

    python -m astana_pilot.viz.app3d --map assets/maps/astana_expo.pkl                 # expert driver, window
    python -m astana_pilot.viz.app3d --map assets/maps/astana_expo.pkl --model runs/night1/final_model.zip
    python -m astana_pilot.viz.app3d --record assets/videos/demo.mp4 --cinematic --duration 90 --headless
    astana view --random-route --seed 7 --model runs/night2/best_model.zip          # navigator: random A -> B
    astana view --start-node "Nur Alem" --goal-node "Kabanbay" --model ...          # navigator: your A -> B

Keys:  C / 1-7 camera (chase, cockpit, bird, top, drone, orbit, overview)   T weather (day, dusk, night, snow)
       L lidar   H HUD   R new episode   SPACE pause   +/- sim speed   G hand over to the keyboard (WASD / arrows)
       F12 screenshot   V video   K cinematic director   ESC quit
       navigator: M open the map (click A / B, choose a route, change the destination while driving)
                  B replay the route overview   O alternatives on/off   Z mini-map zoom   ENTER skip the overview
"""
from __future__ import annotations

import builtins
import math
import sys
import time as pytime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from panda3d.core import (LineSegs, NodePath, TransparencyAttrib, Vec3, Vec4)

from ..env.lidar import ray_angles
from . import quality as Q
from .actors import ActorPool, car_models, panda_heading
from .city_scene import build_scene_cached
from .gfx import mesh_to_node
from .hud import Hud
from .mesh import MeshBuilder
from .renderer import SHADOW_BIT, StaticRenderer
from .session import DriveSession, SimRunner, make_snap
from .canvas import AMBER as C_AMBER, DIM as C_DIM, GREEN as C_GREEN, TEXT as C_TEXT
from .stills import create_app, default_window, gpu_name, save_screenshot

CAMERAS = ("chase", "cockpit", "bird", "top", "drone", "orbit", "overview", "free")
DIRECTOR = (("chase", 8.0), ("drone", 7.0), ("cockpit", 6.0), ("side", 6.0), ("bird", 6.0), ("front", 6.0),
            ("chase", 6.0), ("roadside", 7.0), ("top", 5.0))
WEATHER = ("day", "dusk", "night", "snow")
def color_white():
    from ursina import color
    return color.white


KIND_NAMES = {0: "wall", 1: "parked car", 2: "car", 3: "bus", 4: "pedestrian"}


@dataclass
class ViewerOptions:
    map_path: str = "assets/maps/astana_expo.pkl"
    model: Optional[str] = None
    seed: int = 0
    quality: str = "auto"
    preset: str = "day"
    headless: bool = False
    size: Optional[Tuple[int, int]] = None
    record: Optional[str] = None
    fps: int = 30
    duration: float = 0.0              # seconds of video / headless run (0 = until closed)
    cinematic: bool = False
    manual: bool = False
    sim_speed: float = 1.0
    episodes: int = 0                  # headless: stop after N episodes
    hud: bool = True
    camera: str = "chase"
    shots_dir: str = "assets/shots"
    difficulty: float = 1.0
    fullscreen: bool = False
    landmarks: bool = True
    ego_color: Tuple[float, float, float] = (0.10, 0.46, 0.95)
    hotspots: Optional[str] = None     # audit hotspots.csv: show the risky places as beacons in the 3-D city
    tour: bool = False                 # showcase: long eventful routes + cinematic camera
    observer: bool = False             # fly around the city: free camera, no HUD, no ego car, traffic still lives
    recovery: bool = False             # spawn some episodes facing backwards / sideways (training scenario)
    shot: Optional[str] = None
    shot_at: float = 5.0
    # ---- point-to-point navigation (navigator chooses the route, the network drives it)
    start: Optional[str] = None        # node id | "x,y" | landmark / street name
    goal: Optional[str] = None
    random_route: bool = False         # a new random A -> B every episode (reproducible with --seed)
    route_km: Tuple[float, float] = (0.6, 1.5)
    verified: bool = False             # the reference expert must be able to drive the scenario first
    intro: float = 4.0                 # seconds of "route overview" camera before the car sets off (0 = off)
    pick: bool = False                 # start with the navigator map open: the user clicks A and B


class CameraRig:
    """Smooth camera with several modes; everything is computed in viewer coordinates (x east, y up, z north)."""

    def __init__(self, camera, extent_center: Tuple[float, float], extent: float) -> None:
        self.cam = camera
        self.mode = "chase"
        self.pos = np.array([0.0, 20.0, -20.0])
        self.look = np.zeros(3)
        self.fov = 62.0
        self.blend = 0.0
        self.center, self.extent = extent_center, extent
        self.orbit = [0.6, 0.35, 16.0]             # yaw, pitch, distance
        self.drone_a = 0.0
        self.ortho = False
        self.zoom = 1.0
        self.road_cam: Optional[np.ndarray] = None
        self.fpos = np.array([extent_center[0], 160.0, extent_center[1] - 0.45 * extent])
        self.fyaw, self.fpitch, self.fspeed = 0.0, 22.0, 40.0          # free-fly ("observer") camera
        self.tyaw, self.tpitch = 0.0, 22.0
        self.fvel = np.zeros(3)
        self.fly_target = None

    def free_move(self, keys: dict, dlook: Tuple[float, float], wheel: float, dt: float) -> None:
        """Smooth fly camera: WASD / QE with inertia, Shift = fast, mouse look (dlook in screen fractions), wheel = speed.
        A pending ``fly_to`` glides the camera to a target (hot-spot jump) until the user steers."""
        self.tyaw += dlook[0] * 90.0
        self.tpitch = float(np.clip(self.tpitch + dlook[1] * 90.0, -89.0, 89.0))
        k = 1.0 - math.exp(-16.0 * dt)
        self.fyaw += (self.tyaw - self.fyaw) * k
        self.fpitch += (self.tpitch - self.fpitch) * k
        self.fspeed = float(np.clip(self.fspeed * (1.0 + 0.15 * wheel), 3.0, 600.0))
        yaw = math.radians(self.fyaw)
        f = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
        r = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        d = f * (keys.get("w", 0) - keys.get("s", 0)) + r * (keys.get("d", 0) - keys.get("a", 0)) \
            + np.array([0.0, keys.get("e", 0) - keys.get("q", 0), 0.0])
        steering = float(np.abs(d).sum()) > 0.0
        if steering:
            self.fly_target = None
        target_v = d * self.fspeed * (3.0 if keys.get("shift", 0) else 1.0)
        self.fvel += (target_v - self.fvel) * (1.0 - math.exp(-7.0 * dt))
        self.fpos += self.fvel * dt
        if self.fly_target is not None:
            tp, ty, tpi = self.fly_target
            g = 1.0 - math.exp(-2.6 * dt)
            self.fpos += (tp - self.fpos) * g
            self.tyaw += ((ty - self.tyaw + 180.0) % 360.0 - 180.0) * g
            self.tpitch += (tpi - self.tpitch) * g
            if float(np.linalg.norm(tp - self.fpos)) < 1.5:
                self.fly_target = None
        self.fpos[1] = max(1.5, self.fpos[1])

    def fly_to(self, pos, yaw: float, pitch: float) -> None:
        self.fly_target = (np.asarray(pos, dtype=float), yaw, pitch)

    def set_mode(self, mode: str) -> None:
        if mode != self.mode:
            self.mode = mode
            self.blend = 1.0
            self.road_cam = None
        self._apply_projection()

    def _apply_projection(self) -> None:
        want = self.mode == "top"
        if want != self.ortho:
            self.cam.orthographic = want
            self.ortho = want

    def update(self, ego: Tuple[float, float, float, float, float], ref_xy: Optional[np.ndarray], dt: float,
               yaw_move: float = 0.0) -> None:
        x, z, y, th, v = ego                       # sim x, sim y (->z), height, heading, speed
        f = np.array([math.cos(th), 0.0, math.sin(th)])
        left = np.array([-math.sin(th), 0.0, math.cos(th)])
        c = np.array([x, y, z])
        up = np.array([0.0, 1.0, 0.0])
        m = self.mode
        k = 7.0
        fov = 62.0
        if m == "free":
            self.pos = self.fpos.copy()
            self.cam.position = tuple(self.fpos)
            self.cam.rotation = (self.fpitch, self.fyaw, 0.0)
            self.cam.fov = 62.0
            self.cam.clip_plane_near = 1.5
            yaw = math.radians(self.fyaw)
            self.look = self.fpos + np.array([math.sin(yaw), 0.0, math.cos(yaw)]) * 60.0
            return
        if m == "chase":
            pos = c - f * (11.0 + 0.14 * abs(v)) / self.zoom + up * (4.5 + 0.03 * abs(v))
            look = c + f * 13.0 - up * 0.4
            fov, k = 58.0 + min(abs(v), 25.0) * 0.45, 11.0
        elif m == "cockpit":
            pos = c + f * 0.15 + left * 0.34 + up * 1.22
            look = pos + f * 40.0 + up * 0.5
            fov, k = 78.0, 30.0
        elif m == "bird":
            pos = c - f * 34.0 / self.zoom + up * 62.0 / self.zoom
            look = c + f * 6.0
            fov, k = 58.0, 4.0
        elif m == "top":
            pos = c + up * 420.0
            look = c.copy()
            fov, k = 150.0 / self.zoom, 12.0
        elif m == "drone":
            self.drone_a += dt * 0.22
            r = 30.0 / self.zoom
            pos = c + np.array([math.cos(self.drone_a) * r, 15.0 + 4 * math.sin(self.drone_a * 0.7), math.sin(self.drone_a) * r])
            look = c + up * 1.0
            fov, k = 56.0, 5.0
        elif m == "orbit":
            yw, pt, d = self.orbit
            yw += yaw_move
            pos = c + np.array([math.cos(yw) * math.cos(pt), math.sin(pt), math.sin(yw) * math.cos(pt)]) * d / self.zoom
            look = c + up * 1.0
            fov, k = 55.0, 10.0
        elif m == "side":
            pos = c - left * 6.5 - f * (3.0 - 0.15 * abs(v)) + up * 1.5
            look = c + f * (1.0 + 0.2 * abs(v)) + up * 0.8
            fov, k = 50.0, 6.0
        elif m == "front":
            pos = c + f * (13.0 + 0.4 * abs(v)) + left * 2.2 + up * 1.0
            look = c + up * 0.9
            fov, k = 52.0, 8.0
        elif m == "roadside":
            if ref_xy is not None and (self.road_cam is None or np.dot(self.road_cam[:3] - c, f) < -8.0):
                i = int(np.argmin(np.hypot(ref_xy[:, 0] - x, ref_xy[:, 1] - z)))
                j = min(len(ref_xy) - 1, i + int(45 + abs(v) * 2.5))
                p = ref_xy[j]
                d = ref_xy[min(len(ref_xy) - 1, j + 2)] - ref_xy[j]
                nrm = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
                self.road_cam = np.array([p[0] + nrm[0] * 5.0, y + 1.7, p[1] + nrm[1] * 5.0])
            pos = self.road_cam if self.road_cam is not None else c - f * 8 + up * 3
            look = c + up * 0.9
            fov, k = 38.0, 20.0
        elif m == "plan" and getattr(self, "plan_box", None) is not None:   # navigator: look at the whole route
            cx, cz, ext = self.plan_box
            pos = np.array([cx, max(110.0, ext * 0.85), cz - ext * 0.62])
            look = np.array([cx, 0.0, cz - ext * 0.05])
            fov, k = 55.0, 2.2
        else:                                       # overview
            cx, cz = self.center
            pos = np.array([cx, self.extent * 0.62 / self.zoom, cz - self.extent * 0.52 / self.zoom])
            look = np.array([cx, 0.0, cz])
            fov, k = 55.0, 3.0
        if self.blend > 0.0:
            self.blend = max(0.0, self.blend - dt / 1.2)
            k = min(k, 2.6)
        a = 1.0 - math.exp(-k * dt)
        self.pos += (pos - self.pos) * a
        self.look += (look - self.look) * min(1.0, a * 1.6)
        self.fov += (fov - self.fov) * min(1.0, 1.0 - math.exp(-4.0 * dt))
        cam = self.cam
        cam.position = tuple(self.pos)
        if self.mode == "top":
            cam.rotation = (90.0, 0.0, 0.0)
        else:
            d = self.look - self.pos
            horiz = math.hypot(d[0], d[2])
            yaw = math.degrees(math.atan2(d[0], d[2]))
            pitch = math.degrees(math.atan2(-d[1], max(horiz, 1e-6)))      # positive = looking down
            cam.rotation = (pitch, yaw, 0.0)
        cam.fov = self.fov
        cam.clip_plane_near = 0.15 if self.mode == "cockpit" else (2.0 if self.mode in ("top", "overview", "bird", "free") else 0.8)


class Viewer:
    def __init__(self, opts: ViewerOptions) -> None:
        from ursina import Entity, camera, held_keys, mouse, time as utime, window
        self.o = opts
        self.held_keys, self.mouse, self.utime, self.window = held_keys, mouse, utime, window
        size = opts.size or (default_window() if not opts.headless else (1600, 900))
        if opts.record:
            size = (size[0] // 2 * 2, size[1] // 2 * 2)
        self.app = create_app(opts.headless, size, 4 if opts.quality in ("auto", "high", "ultra") else
                              (0 if opts.quality == "low" else 0), fullscreen=opts.fullscreen)
        self.gpu = gpu_name()
        self.q = Q.pick(opts.quality, self.gpu)
        print(f"[viewer] GPU: {self.gpu} | quality: {self.q.name}")
        self.render = builtins.base.render                      # type: ignore[attr-defined]
        self.win = builtins.base.win                            # type: ignore[attr-defined]
        self.W, self.H = self.win.get_x_size(), self.win.get_y_size()

        # ---- simulation (graphics-free session shared with the 2-D viewer)
        self.hud: Optional[Hud] = None
        self.sess = DriveSession(opts.map_path, opts.model, opts.seed, opts.difficulty, recovery=opts.recovery, tour=opts.tour,
                                 toast=lambda t, k: self.hud.toast(t, k) if self.hud else None,
                                 flash=lambda k: self.hud.flash(k) if self.hud else None,
                                 start=opts.start, goal=opts.goal, random_route=opts.random_route,
                                 route_km=opts.route_km, verified=opts.verified)
        self.sess.manual = opts.manual
        self.cfg, self.env, self.city = self.sess.cfg, self.sess.env, self.sess.city

        # ---- scene
        t0 = pytime.time()
        self.scene = build_scene_cached(self.city, self.q, opts.map_path, landmarks=opts.landmarks)
        print(f"[viewer] scene built in {pytime.time() - t0:.1f} s  {self.scene.stats}")
        self.sr = StaticRenderer(self.scene, self.q, opts.preset)
        self._build_dynamic()
        self.hud = Hud((self.W, self.H), self.q.hud_hz, self.city)
        if not opts.hud:
            self.hud.toggle()
        x0, y0, x1, y1 = self.city.bounds
        self.rig = CameraRig(camera, ((x0 + x1) / 2, (y0 + y1) / 2), max(x1 - x0, y1 - y0))
        self.rig.set_mode(opts.camera)
        self.cam_idx = CAMERAS.index(opts.camera) if opts.camera in CAMERAS else 0

        # ---- state
        self.speed_mult = opts.sim_speed
        self.show_lidar = True
        self.cinematic = opts.cinematic or bool(opts.tour)
        self.dir_i, self.dir_t = 0, 0.0
        self.fixed_dt = (1.0 / opts.fps) if (opts.record or opts.headless) else 0.0
        self.acc = 0.0
        self.frames = 0
        self.t_wall = 0.0
        self.fps_smooth = 60.0
        self.writer = None
        self.rec_frames = 0
        self.weather_i = WEATHER.index(opts.preset) if opts.preset in WEATHER else 0
        self.route_t = 0.0
        self.done_flag = False
        self._shot_done = False
        self._t_start = pytime.time()
        if opts.observer:
            self.rig.set_mode("free")
            self.cam_idx = CAMERAS.index("free")
            self.show_lidar = False
            self.ego.hide()
            self.goal.hide()
            if self.hud.visible:
                self.hud.toggle()
            self._setup_spectator()
        self.pool.prewarm(self.env.world.actors)
        import gc
        gc.collect()
        gc.freeze()                                              # static scene objects leave the GC's working set
        gc.set_threshold(50000, 20, 20)
        if opts.hotspots:
            self._load_hotspots(opts.hotspots)
        self.hot_rows, self.hot_i, self.follow_id = getattr(self, 'hot_rows', []), 0, None
        self.captured, self._mcap = False, False
        self.help_visible, self.help_pinned = True, False
        self._last_snap_id, self._last_reset = -1, -1
        self._sync_snap, self._sync_id, self._sync_reset = None, 0, 0
        self.runner = None
        if not self.fixed_dt:                                    # interactive: simulate on a worker thread
            sys.setswitchinterval(0.001)
            self.runner = SimRunner(self.sess)
            self.runner.speed = self.speed_mult
            self.runner.start()
        # interactive runs stop after each trip and ask what next; videos / tours keep looping
        self.sess.hold_at_end = not (opts.record or opts.tour or opts.cinematic or opts.observer or
                                     (opts.headless and not opts.pick) or opts.episodes)
        self._setup_ui()
        self._on_new_episode()
        if opts.pick:
            self._open_map("setup")
        elif not opts.observer and not opts.headless:
            self.hud.toast("Esc = menu   ·   M = map: choose / change the route", "info")
        if opts.record:
            self._start_recording(opts.record)
        self.entity = Entity()                              # receives Ursina's update()/input()
        self.entity.update = self._update                   # type: ignore[assignment]
        self.entity.input = self._input                     # type: ignore[assignment]
        self.entity.ignore_paused = True

    # ------------------------------------------------------------------ dynamic nodes
    def _build_dynamic(self) -> None:
        sr = self.sr
        self.fx = self.render.attach_new_node("fx")
        self.pool = ActorPool(self.render, sr.lit, sr.unlit, sr.bank.get("white"), self.q.max_actors)
        # ego car
        paint, glass, wheels, lights = car_models(self.cfg.vehicle.length, self.cfg.vehicle.width)
        self.ego = self.render.attach_new_node("ego")
        for name, mesh, spec in (("paint", paint, 0.55), ("glass", glass, 1.0), ("wheels", wheels, 0.2)):
            n = mesh_to_node(mesh, name, self.ego)
            n.set_shader(sr.lit)
            n.set_texture(sr.bank.get("white"))
            n.set_shader_input("spec_amt", spec)
            n.set_shader_input("world_uv", 0.0)
            if name == "paint":
                n.set_color_scale(Vec4(*self.o.ego_color, 1.0))
        li = mesh_to_node(lights, "lights", self.ego)
        li.set_shader(sr.unlit)
        li.set_texture(sr.bank.get("white"))
        li.set_shader_input("glow", 1.2)
        vp = self.cfg.vehicle
        bl = MeshBuilder()
        for sx in (-1, 1):
            bl.box((sx * vp.width * 0.34, 0.82, -vp.length / 2), (vp.width * 0.30, 0.10, 0.06), color=(1.0, 0.1, 0.08, 1.0))
        self.ego_brake = mesh_to_node(bl.build(), "brake", self.ego)
        self.ego_brake.set_shader(sr.unlit)
        self.ego_brake.set_texture(sr.bank.get("white"))
        self.ego_brake.set_shader_input("glow", 2.4)
        self.ego_brake.hide()
        # goal beacon
        gb = MeshBuilder()
        gb.cylinder((0, 0, 0), 1.4, 70.0, 16, (0.2, 1.0, 0.5, 0.20), caps=False)
        gb.cylinder((0, 0.1, 0), self.cfg.goal_radius, 0.12, 36, (0.2, 1.0, 0.5, 0.35), caps=True)
        self.goal = mesh_to_node(gb.build(), "goal", self.fx)
        self.goal.set_shader(sr.unlit)
        self.goal.set_texture(sr.bank.get("white"))
        self.goal.set_shader_input("glow", 1.4)
        self.goal.set_transparency(TransparencyAttrib.M_alpha)
        self.goal.set_two_sided(True)
        self.goal.set_depth_write(False)
        self.goal.hide(SHADOW_BIT)
        self.route_np: Optional[NodePath] = None
        self.lidar_np: Optional[NodePath] = None
        self.full_np: Optional[NodePath] = None        # whole remaining route (navigator blue)
        self.plan_np: Optional[NodePath] = None        # alternatives + A / B pins of the current plan
        self.plan_labels: list = []
        self.show_alts = True
        self.intro_t = 0.0
        self._mode_after_intro = "chase"
        self._full_t = 0.0
        self.map_open = False                          # navigator map (key M)
        self.map_ui = None
        self.map_card = None
        self.map_dim = None
        self._map_dirty = False
        self._map_t = 0.0
        self._drag = None
        self._awaiting_pick = bool(self.o.pick)

    # ------------------------------------------------------------------ navigator map (click A / B)
    def _mouse_px(self) -> Tuple[float, float]:
        try:
            mp = self.mouse.position
        except Exception:                                      # off-screen buffer (headless): no pointer
            return -1.0, -1.0
        return self.W / 2 + float(mp[0]) * self.H, self.H / 2 - float(mp[1]) * self.H

    def _open_map(self, mode: str) -> None:
        from ursina import Entity, camera, color
        from .hud import _Card
        from .map_picker import MapPanel
        if self.o.observer:
            return
        if self.map_ui is None or (self.map_ui.W, self.map_ui.H) != (self.W, self.H):
            self.map_ui = MapPanel(self.city, self.cfg, (self.W, self.H))
            if self.map_card is not None:
                from ursina import destroy
                destroy(self.map_card.ent)
            x, y, w, h = self.map_ui.box
            self.map_card = _Card("navmap", x, y, w, h, camera.ui, (w, h))
            self.map_card.place(self.W, self.H)
            self.map_card.ent.z = -0.5
            self.map_dim = Entity(parent=camera.ui, model="quad", scale=(20, 2), z=-0.4, color=color.Color(0.0, 0.0, 0.02, 0.62))
        m = self.map_ui
        if mode == "drive":
            S = self.sess
            v = S.env.vehicle
            m.set_drive((v.x, v.y, v.theta), S.env.ref.xy[int(np.searchsorted(S.env.ref.s, S.env._s)):][::3],
                        S.env._route_nodes, S.env.ref, S.env._s)
        else:
            m.mode = "setup"
            m.car = None
        self.map_card.ent.enabled = True
        self.map_dim.enabled = True
        self._hud_was = self.hud.visible
        if self.hud.visible:
            self.hud.toggle()
        self._paused_was = self.sess.paused
        self.sess.paused = True
        self.map_open = True
        self._map_dirty = True

    def _close_map(self) -> None:
        self.map_open = False
        self._drag = None
        if self.map_card is not None:
            self.map_card.ent.enabled = False
            self.map_dim.enabled = False
        if getattr(self, "_hud_was", True) and not self.hud.visible:
            self.hud.toggle()
        self.sess.paused = bool(getattr(self, "_paused_was", False)) and self.intro_t <= 0.0

    def _map_go(self) -> None:
        m, S = self.map_ui, self.sess
        plan = m.plan
        if plan is None:
            return
        plan.chosen = m.sel
        if m.mode == "setup":
            S.use_plan(plan)
            self._awaiting_pick = False
            self._paused_was = False
            self._close_map()
            if self.runner is not None:
                self.runner.request_new_episode()
            else:
                S.episode += 1
                S.new_episode()
                self._sync_reset += 1
        else:
            S.reroute(plan, m.sel)
            if self.runner is not None:
                self.runner.republish()
            else:
                self._sync_snap = None
            self._paused_was = False
            self._close_map()
            self._build_plan_overlay(plan)
            self._full_t = 0.0
            self._start_intro()

    def _map_input(self, key: str) -> None:
        m = self.map_ui
        if key == "left mouse down":
            px, py = m.local(*self._mouse_px())
            self._drag = (px, py, m.cx, m.cy, False)
        elif key == "left mouse up" and self._drag is not None:
            px, py, _, _, moved = self._drag
            self._drag = None
            if not moved:
                act = m.click(px, py, "left")
                self._map_dirty = True
                if act == "go":
                    self._map_go()
                elif act == "close":
                    self._close_map()
        elif key == "right mouse down":
            px, py = m.local(*self._mouse_px())
            m.click(px, py, "right")
            self._map_dirty = True
        elif key in ("scroll up", "scroll down"):
            px, py = m.local(*self._mouse_px())
            m.zoom(px, py, 1.25 if key == "scroll up" else 0.8)
            self._map_dirty = True
        elif key == "enter":
            self._map_go()
        elif key == "r":
            m.random()
            self._map_dirty = True
        elif key in ("m", "escape") and (m.mode == "drive" or not self._awaiting_pick):
            self._close_map()
        elif key == "escape":                                  # first trip not chosen yet: the menu (with Exit)
            self._close_map()
            self._open_menu("pause")

    def _map_update(self, dt: float) -> None:
        m = self.map_ui
        if self._drag is not None and self.mouse.left:
            x0, y0, cx0, cy0, moved = self._drag
            px, py = m.local(*self._mouse_px())
            if moved or abs(px - x0) + abs(py - y0) > 6:
                m.cx, m.cy = cx0, cy0
                m.pan(px - x0, py - y0)
                self._drag = (x0, y0, cx0, cy0, True)
                self._map_dirty = True
        self._map_t -= dt
        if self._map_dirty and self._map_t <= 0.0:
            self._map_dirty = False
            self._map_t = 1.0 / 40.0
            self.map_card.set(m.render())

    # ------------------------------------------------------------------ menus (pause / trip finished) + toolbar
    def _ui_card(self, name: str, x: float, y: float, w: int, h: int, z: float):
        from ursina import camera
        from .hud import _Card
        c = _Card(name, x, y, w, h, camera.ui, (w, h))
        c.place(self.W, self.H)
        c.ent.z = z
        return c

    def _setup_ui(self) -> None:
        from ursina import Entity, camera, color
        from .menus import MenuPanel, Toolbar
        self.menu = None
        self.menu_card = None
        self._menu_size = None
        self._menu_t = 0.0
        self.toolbar = None
        self._toolbar_dirty = True
        self._toolbar_t = 0.0
        if self.o.observer or (self.o.headless and not self.o.pick):
            return
        k = max(0.6, min(self.H / 900.0, self.W / 1600.0))
        self.menu_panel = MenuPanel(k)
        self.toolbar = Toolbar(k)
        m = int(round(22 * k))
        self._toolbar_xy = (m, self.H - m - int(round(230 * k)) - int(round(52 * k)))
        self.toolbar_card = self._ui_card("toolbar", *self._toolbar_xy, self.toolbar.w, self.toolbar.h, -0.15)
        self.ui_dim = Entity(parent=camera.ui, model="quad", scale=(20, 2), z=-0.4, color=color.Color(0.0, 0.0, 0.02, 0.55))
        self.ui_dim.enabled = False

    def _menu_items(self, kind: str):
        S = self.sess
        if kind == "end":
            r = S.last_result or {}
            ok = bool(r.get("success"))
            title = "Destination reached" if ok else "Trip ended: " + str(r.get("status", "")).title()
            s = S.session
            lines = [(f"{r.get('km', 0):.2f} km in {r.get('minutes', 0):.1f} min  ·  average {r.get('speed', 0):.0f} km/h", C_TEXT),
                     (f"Rule violations on this trip: {r.get('violations', 0)}", C_GREEN if r.get("violations", 0) == 0 else C_AMBER),
                     (f"This session: {s['success']} of {s['n']} trips completed  ·  driver: {S.model_name}", C_DIM)]
            items = [("Choose the next destination", "M", "next", "primary"),
                     ("Drive the same trip again", "R", "restart", "normal"),
                     ("Random route", "", "random", "normal"),
                     ("More options", "Esc", "pause", "normal"),
                     ("Exit to main menu (Studio)", "", "quit", "danger")]
            return title, lines, items, ((47, 214, 122, 255) if ok else (255, 79, 90, 255))
        nav = S.nav_info() or {}
        lines = [(f"{self.city.raw.name.replace('_', ' ').title()}  ·  driver: {S.model_name}", C_DIM)]
        if S.plan is not None and not self._awaiting_pick:
            lines.append((f"Trip to {S.plan.goal_name}: {nav.get('remain_m', 0) / 1000:.1f} km left", C_TEXT))
        cam = self.rig.mode if self.rig.mode != "plan" else self._mode_after_intro
        items = [("Resume", "Esc", "resume", "primary")] if not self._awaiting_pick else []
        items += [("Change destination" if not self._awaiting_pick else "Choose A and B on the map", "M", "map", "normal"),
                  ("New trip: choose A and B", "", "newtrip", "normal"),
                  ("Restart this trip", "R", "restart", "normal"),
                  ("Random route", "", "random", "normal"),
                  ("View", "", "", "section"),
                  (f"Camera: {cam}", "C", "camera", "normal"),
                  (f"Weather: {WEATHER[self.weather_i]}", "T", "weather", "normal"),
                  (f"Simulation speed: x{self.speed_mult:g}", "+ / -", "speed", "normal"),
                  (f"Lidar rays: {'on' if self.show_lidar else 'off'}  ·  HUD: {'on' if self._hud_on() else 'off'}", "L / H", "lidar", "normal"),
                  (f"Driver: {'keyboard (WASD)' if S.manual else 'autopilot'}", "G", "driver", "normal"),
                  ("", "", "", "section"),
                  ("Exit to main menu (Studio)", "", "quit", "danger")]
        return "Paused" if not self._awaiting_pick else "Menu", lines, items, (37, 120, 255, 255)

    def _hud_on(self) -> bool:
        return bool(getattr(self, "_hud_was", True)) if self.menu is not None else self.hud.visible

    def _open_menu(self, kind: str) -> None:
        if self.toolbar is None:
            return
        if self.intro_t > 0.0:
            self._end_intro()
        if self.menu is None:
            self._menu_paused_was = self.sess.paused
            self._hud_was = self.hud.visible
        self.menu = kind
        self.sess.paused = True
        if self.hud.visible:
            self.hud.toggle()
        self.toolbar_card.ent.enabled = False
        self.ui_dim.enabled = True
        self._render_menu()

    def _render_menu(self) -> None:
        title, lines, items, accent = self._menu_items(self.menu)
        arr = self.menu_panel.render(title, lines, items, accent)
        h, w = arr.shape[:2]
        if self._menu_size != (w, h):
            if self.menu_card is not None:
                from ursina import destroy
                destroy(self.menu_card.ent)
            self._menu_xy = ((self.W - w) // 2, max(10, (self.H - h) // 2))
            self.menu_card = self._ui_card("menu", *self._menu_xy, w, h, -0.5)
            self._menu_size = (w, h)
        self.menu_card.ent.enabled = True
        self.menu_card.set(arr)

    def _close_menu(self, resume: bool = True) -> None:
        self.menu = None
        if self.menu_card is not None:
            self.menu_card.ent.enabled = False
        self.ui_dim.enabled = False
        if getattr(self, "_hud_was", True) and not self.hud.visible:
            self.hud.toggle()
        self.toolbar_card.ent.enabled = True
        self._toolbar_dirty = True
        if resume:
            self.sess.paused = False

    def _menu_input(self, key: str) -> None:
        act = None
        if key == "left mouse up":
            px, py = self._mouse_px()
            act = self.menu_panel.hit(px - self._menu_xy[0], py - self._menu_xy[1])
        elif key == "escape":
            act = "resume" if self.menu == "pause" and not self._awaiting_pick else ("pause" if self.menu == "end" else None)
        elif key == "enter":
            act = "next" if self.menu == "end" else ("resume" if not self._awaiting_pick else "map")
        elif key == "m":
            act = "next" if self.menu == "end" else "map"
        elif key in ("r", "n"):
            act = "restart"
        elif key in ("c", "t", "l", "g", "h", "+", "=", "-"):
            act = {"c": "camera", "t": "weather", "l": "lidar", "g": "driver", "h": "hud"}.get(key, "speed" if key in "+=" else "slower")
        if act:
            self._menu_action(act)

    def _menu_action(self, act: str) -> None:
        S = self.sess
        if act == "resume":
            if S.waiting_next:                                 # nothing to resume after a finished trip
                self._open_menu("end")
                return
            self._close_menu(resume=True)
        elif act == "pause":
            self._open_menu("pause")
        elif act in ("map", "next", "newtrip"):
            setup = act in ("next", "newtrip") or self._awaiting_pick or S.waiting_next
            self._close_menu(resume=False)
            self._open_map("setup" if setup else "drive")
            m = self.map_ui
            if setup:
                m.B, m.plan, m.msg = None, None, ""
                m.A = None
                if act == "next" and S.plan is not None:       # continue from where the last trip ended
                    m.A = S.plan.goal
                self._map_dirty = True
        elif act in ("restart", "random"):
            self._close_menu(resume=False)
            self._new_trip(act)
        elif act == "quit":
            self._quit()
        else:
            key = {"camera": "c", "weather": "t", "lidar": "l", "driver": "g", "hud": None}.get(act)
            if act == "speed":
                steps = (0.5, 1.0, 2.0, 4.0)
                self.speed_mult = steps[(steps.index(self.speed_mult) + 1) % len(steps)] if self.speed_mult in steps else 1.0
            elif act == "slower":
                self.speed_mult = max(0.25, self.speed_mult / 2)
            elif act == "lidar":
                self.show_lidar = not self.show_lidar
            elif act == "hud":
                self._hud_was = not getattr(self, "_hud_was", True)
            elif key:
                self._hotkey(key)
            self._render_menu()

    def _new_trip(self, how: str) -> None:
        """restart = the same A -> B again (new traffic); random = a random verified A -> B."""
        S = self.sess
        if how == "restart" and S.plan is not None:
            S.use_plan(S.plan)
        elif how == "random":
            S.forced_plan = None
            S.start_spec = S.goal_spec = None
            S._fixed_plan = None
            S.nav_mode = True
        self._awaiting_pick = False
        S.paused = False
        if self.menu is not None:
            self._close_menu(resume=True)
        if self.runner is not None:
            self.runner.request_new_episode()
        else:
            S.episode += 1
            S.new_episode()
            self._sync_reset += 1

    def _toolbar_action(self, act: str) -> None:
        if act == "menu":
            self._open_menu("pause")
        elif act == "map":
            self._open_map("setup" if (self._awaiting_pick or self.sess.waiting_next) else "drive")
        elif act == "pause":
            if self.intro_t > 0.0:
                self._end_intro()
            self.sess.paused = not self.sess.paused
            self._toolbar_dirty = True

    def _ui_update(self, dt: float) -> None:
        if self.toolbar is None:
            return
        S = self.sess
        if S.waiting_next and self.menu is None and not self.map_open:
            self._open_menu("end")
        self._toolbar_t -= dt
        if self._toolbar_t <= 0.0:
            self._toolbar_t = 0.08
            px, py = self._mouse_px()
            if self.menu is not None:
                hv = self.menu_panel.hit(px - self._menu_xy[0], py - self._menu_xy[1])
                if hv != self.menu_panel.hover:
                    self.menu_panel.hover = hv
                    self._render_menu()
            elif self.toolbar_card.ent.enabled:
                hv = self.toolbar.hit(px - self._toolbar_xy[0], py - self._toolbar_xy[1])
                if hv != self.toolbar.hover:
                    self.toolbar.hover = hv
                    self._toolbar_dirty = True
        show = self.menu is None and not self.map_open
        if self.toolbar_card.ent.enabled != show:
            self.toolbar_card.ent.enabled = show
        if self._toolbar_dirty and show:
            self._toolbar_dirty = False
            self.toolbar_card.set(self.toolbar.render(S.paused and self.intro_t <= 0.0))

    def _quit(self) -> None:
        self._stop_recording()
        if self.runner is not None:
            self.runner.stop()
        from ursina import application
        application.quit()

    # ------------------------------------------------------------------ navigator overlay
    @staticmethod
    def _ribbon(mb: MeshBuilder, xy: np.ndarray, z: Optional[np.ndarray], half_w: float, rgba, lift: float,
                fade: Optional[np.ndarray] = None) -> None:
        """Flat band along a polyline, ``lift`` metres above the road (like a route line in a navigation app)."""
        if len(xy) < 2:
            return
        zz = z if z is not None else np.zeros(len(xy))
        d = np.gradient(xy, axis=0)
        n = np.column_stack([-d[:, 1], d[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        L = np.column_stack([xy[:, 0] + n[:, 0] * half_w, zz + lift, xy[:, 1] + n[:, 1] * half_w])
        R = np.column_stack([xy[:, 0] - n[:, 0] * half_w, zz + lift, xy[:, 1] - n[:, 1] * half_w])
        P = np.stack([L[:-1], R[:-1], R[1:], L[1:]], axis=1)
        col = np.zeros((len(P), 4), dtype=np.float32)
        col[:] = rgba
        if fade is not None:
            col[:, 3] = fade[:-1]
        mb.quads(P, col, None, np.tile([0, 1, 0], (len(P), 1)))

    def _overlay_node(self, mb: MeshBuilder, name: str, glow: float = 1.3) -> NodePath:
        np_ = mesh_to_node(mb.build(), name, self.fx)
        np_.set_shader(self.sr.unlit)
        np_.set_texture(self.sr.bank.get("white"))
        np_.set_shader_input("glow", glow)
        np_.set_transparency(TransparencyAttrib.M_alpha)
        np_.set_depth_write(False)
        np_.set_two_sided(True)
        np_.hide(SHADOW_BIT)
        return np_

    def _build_plan_overlay(self, plan) -> None:
        """Grey alternative routes, A (start) and B (finish) pins with labels; camera box for the intro shot."""
        from ursina import Text, destroy, scene
        for attr in ("plan_np", "plan_wide_np"):
            node = getattr(self, attr, None)
            if node is not None:
                node.remove_node()
            setattr(self, attr, None)
        for tx in self.plan_labels:
            destroy(tx)
        self.plan_labels = []
        if plan is None or self.o.observer:
            return
        b = plan.best.ref
        a_xy, b_xy = b.xy[0], b.xy[-1]
        a_z, b_z = b.z_at(0.0), b.z_at(b.length)
        ext = float(np.ptp(np.vstack([o.ref.xy for o in plan.options]), axis=0).max()) + 120.0
        # narrow layer: seen from the car (alternatives as thin grey lines, small pins)
        mb = MeshBuilder()
        for i, o in enumerate(plan.options):
            if i != plan.chosen:
                r = o.ref
                self._ribbon(mb, r.xy[::2], r.z[::2] if r.z is not None else None, 0.95, (0.62, 0.68, 0.80, 0.55), 0.07)
        # wide layer: the "navigation app" overview shown from the sky before departure
        mw = MeshBuilder()
        hw = max(3.0, ext / 260.0)
        for i, o in enumerate(plan.options):
            r = o.ref
            chosen = i == plan.chosen
            self._ribbon(mw, r.xy[::3], r.z[::3] if r.z is not None else None, hw * (1.0 if chosen else 0.8),
                         (0.15, 0.47, 1.0, 0.95) if chosen else (0.70, 0.74, 0.84, 0.80), 0.35 if chosen else 0.25)
        pole = max(26.0, ext / 22.0)
        for (px, py), pz, col in ((a_xy, a_z, (0.15, 0.85, 0.45, 0.9)), (b_xy, b_z, (1.0, 0.30, 0.32, 0.9))):
            for m_, s_ in ((mb, 1.0), (mw, ext / 500.0)):
                s_ = max(1.0, s_)
                m_.cylinder((float(px), pz, float(py)), 0.45 * s_, pole, 12, col, caps=True)
                m_.cylinder((float(px), pz + pole, float(py)), 2.2 * s_, 1.6 * s_, 20, col, caps=True)
                m_.cylinder((float(px), pz + 0.05, float(py)), 5.0 * s_, 0.1, 32, (col[0], col[1], col[2], 0.35), caps=True)
        self.plan_np = self._overlay_node(mb, "plan", 1.4)
        self.plan_wide_np = self._overlay_node(mw, "plan_wide", 1.6)
        self.plan_wide_np.hide()
        if not self.show_alts:
            self.plan_np.hide()
        bo = plan.best
        labels = ((a_xy, a_z, "A  start"), (b_xy, b_z, f"B  finish  {bo.length / 1000:.1f} km  {bo.eta_s / 60:.0f} min"))
        for (px, py), pz, txt in labels:
            glyph_m = max(5.0, ext / 85.0)                  # letter height in metres (Ursina text: 0.025 per unit of scale)
            self.plan_labels.append(Text(text=txt, parent=scene, billboard=True, scale=glyph_m / Text.size, color=color_white(),
                                         position=(float(px), pz + pole + 6.0 * max(1.0, ext / 500.0), float(py)), origin=(0, 0)))
        allxy = np.vstack([o.ref.xy for o in plan.options])
        (x0, y0), (x1, y1) = allxy.min(axis=0), allxy.max(axis=0)
        self.rig.plan_box = (float(x0 + x1) / 2, float(y0 + y1) / 2, float(max(x1 - x0, y1 - y0)) + 120.0)

    def _start_intro(self) -> None:
        if self.o.intro <= 0 or not self.sess.nav_mode or self.o.observer or self.map_open or self._awaiting_pick:
            return
        if self.rig.mode != "plan":
            self._mode_after_intro = self.rig.mode
        self.rig.set_mode("plan")
        self.intro_t = self.o.intro
        self.sess.paused = True
        self.sess.planning = True
        if getattr(self, "plan_wide_np", None) is not None:
            self.plan_wide_np.show()
        if self.plan_np is not None:
            self.plan_np.hide()
        if self.hud is not None:
            self.hud.toast("Navigator: route chosen - the network drives from A to B", "info")

    def _end_intro(self) -> None:
        self.intro_t = 0.0
        self.rig.set_mode(self._mode_after_intro)
        self.sess.paused = False
        self.sess.planning = False
        if getattr(self, "plan_wide_np", None) is not None:
            self.plan_wide_np.hide()
        if self.plan_np is not None and self.show_alts:
            self.plan_np.show()

    # ------------------------------------------------------------------ spectator mode
    def _mouse_look(self):
        """Mouse look that never hits the screen edge: while the right button is held (or after Tab) the cursor is hidden,
        re-centred every frame and only its movement is used."""
        if self.o.headless:
            return (0.0, 0.0)
        from panda3d.core import WindowProperties
        want = bool(self.mouse.right) or self.captured
        wp = WindowProperties()
        cx, cy = self.W // 2, self.H // 2
        if want and not self._mcap:
            wp.set_cursor_hidden(True)
            self.win.request_properties(wp)
            self.win.move_pointer(0, cx, cy)
            self._mcap = True
            return (0.0, 0.0)
        if not want and self._mcap:
            wp.set_cursor_hidden(False)
            self.win.request_properties(wp)
            self._mcap = False
            return (0.0, 0.0)
        if not want:
            return (0.0, 0.0)
        md = self.win.get_pointer(0)
        dx, dy = md.get_x() - cx, md.get_y() - cy
        self.win.move_pointer(0, cx, cy)
        return (dx * 0.11 / 90.0, dy * 0.11 / 90.0)

    def _setup_spectator(self) -> None:
        from ursina import camera
        self.spec_root = __import__("ursina").Entity(parent=camera.ui, name="spectator")
        self.spec_cards = {}
        self._spec_layout()
        self.spec_t = 0.0
        self.help_until = pytime.time() + 14.0

    def _spec_layout(self) -> None:
        from .hud import _Card
        for c in getattr(self, "spec_cards", {}).values():
            __import__("ursina").destroy(c.ent)
        s = max(0.6, min(self.H / 900.0, self.W / 1600.0))
        self.spec_s = s
        m = int(22 * s)
        geo = {"help": (m, self.H - m - int(300 * s), int(330 * s), int(300 * s)),
               "info": (m, m, int(360 * s), int(96 * s))}
        self.spec_cards = {}
        for k, (x, y, w, h) in geo.items():
            c = _Card(f"spec_{k}", x, y, w, h, self.spec_root, (w, h))
            c.place(self.W, self.H)
            self.spec_cards[k] = c
        self.spec_geo = geo
        self.spec_dirty = True

    def _draw_spectator(self, sn) -> None:
        from . import canvas as C
        from .canvas import Canvas
        s = self.spec_s
        ss = 2 if s < 1.25 else 1
        show_help = self.help_visible and (pytime.time() < self.help_until or self.help_pinned)
        _, _, w, h = self.spec_geo["help"]
        cv = Canvas(w, h, ss)
        if show_help:
            cv.rect(0, 0, w, h, fill=(8, 10, 16, 222), outline=C.STROKE, r=22 * s)
            cv.text(20 * s, 24 * s, "Spectator controls", int(14 * s), C.TEXT, True, "lm")
            rows = [("W A S D", "fly"), ("Q / E", "down / up"), ("Shift", "faster"), ("Right mouse  or  Tab", "look around"),
                    ("Mouse wheel", "flight speed"), ("[  ]  or  1-9", "jump to hot-spot"), ("F", "follow a car"),
                    ("T", "time of day"), ("H", "hide this help"), ("Esc", "quit")]
            y = 52 * s
            for a, b in rows:
                cv.text(20 * s, y, a, int(12 * s), C.ACCENT, True, "lm")
                cv.text(w - 20 * s, y, b, int(12 * s), C.DIM, False, "rm")
                y += 23 * s
        self.spec_cards["help"].set(cv.finish())
        _, _, w, h = self.spec_geo["info"]
        cv = Canvas(w, h, ss)
        cv.rect(0, 0, w, h, fill=(8, 10, 16, 222), outline=C.STROKE, r=22 * s)
        cv.text(20 * s, 26 * s, self.city.raw.name.replace("_", " ").upper(), int(14 * s), C.TEXT, True, "lm")
        peds = sum(1 for q in sn.actors if q.kind == "ped" and q.alive)
        cars = sum(1 for q in sn.actors if q.kind in ("car", "bus") and q.alive)
        cv.text(w - 20 * s, 26 * s, f"{cars} vehicles  \u00b7  {peds} people", int(11 * s), C.DIM, False, "rm")
        cv.text(20 * s, 52 * s, f"speed {self.rig.fspeed:.0f} m/s  \u00b7  {self.sr.atm.name}" + ("  \u00b7  following" if self.follow_id is not None else ""),
                int(11 * s), C.DIM, False, "lm")
        if self.hot_rows:
            r = self.hot_rows[self.hot_i]
            cv.text(20 * s, 76 * s, f"Hot-spot #{r['rank']}  {r['street'][:26]}", int(12 * s), C.AMBER, True, "lm")
            cv.text(w - 20 * s, 76 * s, f"{r['per_km']:.0f} /km", int(12 * s), C.AMBER, True, "rm")
        else:
            cv.text(20 * s, 76 * s, "Run the audit to see hot-spots here", int(11 * s), C.FAINT, False, "lm")
        self.spec_cards["info"].set(cv.finish())

    def _goto_hotspot(self, i: int) -> None:
        if not self.hot_rows:
            self.hud.toast("No hot-spots: run the audit first", "info")
            return
        self.hot_i = i % len(self.hot_rows)
        r = self.hot_rows[self.hot_i]
        self.follow_id = None
        self.rig.mode = "free"
        # hover 70 m south-west of the place, 55 m up, looking at it
        px, pz = r["x"] - 55.0, r["y"] - 55.0
        self.rig.fly_to((px, 55.0, pz), math.degrees(math.atan2(r["x"] - px, r["y"] - pz)), 33.0)
        self.hud.toast(f"Hot-spot #{r['rank']}: {r['street'][:30]}", "warn")

    def _follow_nearest(self, sn) -> None:
        if self.follow_id is not None:
            self.follow_id = None
            self.rig.mode = "free"
            return
        cx, cz = self.rig.pos[0], self.rig.pos[2]
        cand = [q for q in sn.actors if q.alive and q.kind in ("car", "bus") and q.v > 0.5]
        if cand:
            q = min(cand, key=lambda a: (a.x - cx) ** 2 + (a.y - cz) ** 2)
            self.follow_id = q.id

    def _load_hotspots(self, path: str) -> None:
        """Beacons over the places the audit ranked as most stressful (height grows with the score)."""
        import csv
        from ursina import Text, scene
        rows = list(csv.DictReader(open(path, encoding="utf-8")))[:10]
        if not rows:
            return
        self.hot_rows = [{"rank": r["rank"], "x": float(r["x"]), "y": float(r["y"]), "street": r["street"] or "unnamed street",
                          "per_km": float(r["per_km"])} for r in rows]
        top = max(float(r["per_km"]) for r in rows) or 1.0
        mb = MeshBuilder()
        for r in rows:
            x, z = float(r["x"]), float(r["y"])
            k = float(r["per_km"]) / top
            col = (1.0, 0.25 + 0.5 * (1 - k), 0.15, 0.30)
            mb.cylinder((x, 0.0, z), 4.0, 40.0 + 90.0 * k, 14, col, caps=False)
            mb.cylinder((x, 0.1, z), 16.0, 0.15, 28, (1.0, 0.35, 0.2, 0.35))
        np_ = mesh_to_node(mb.build(), "hotspots", self.fx)
        np_.set_shader(self.sr.unlit)
        np_.set_texture(self.sr.bank.get("white"))
        np_.set_shader_input("glow", 1.6)
        np_.set_transparency(TransparencyAttrib.M_alpha)
        np_.set_depth_write(False)
        np_.set_two_sided(True)
        np_.hide(SHADOW_BIT)
        self._hotspot_labels = []
        for r in rows:
            k = float(r["per_km"]) / top
            tx = Text(text=f"#{r['rank']}  {r['street'][:28]}", parent=scene, billboard=True, scale=6.0 / Text.size, color=color_white(),
                      position=(float(r["x"]), 55.0 + 90.0 * k, float(r["y"])), origin=(0, 0))
            self._hotspot_labels.append(tx)
        self.hud.toast(f"Audit hot-spots: top #{rows[0]['rank']} {rows[0]['street'][:30]}", "warn")

    # ------------------------------------------------------------------ episodes
    def _on_new_episode(self, sn=None) -> None:
        """Visual side of a (re)started episode."""
        self.acc = 0.0
        for st in list(self.pool.nodes.values()):
            st["prev"] = None
            st.pop("next", None)
        ref = sn.ref if sn is not None else self.env.ref
        g = ref.xy[-1]
        gz = ref.z_at(ref.length)
        self.goal.set_pos(float(g[0]), float(gz), float(g[1]))
        self.route_t = 0.0
        self._full_t = 0.0
        plan = sn.plan if sn is not None else self.sess.plan
        self._build_plan_overlay(plan if self.sess.nav_mode and not self._awaiting_pick else None)
        self._start_intro()

    # ------------------------------------------------------------------ frame
    def _snapshot(self, dt: float):
        """Latest simulation snapshot and the interpolation factor between its previous and current state."""
        S = self.sess
        if self.runner is not None:
            sn = self.runner.snap
            alpha = 1.0 if sn.paused else min(1.0, (pytime.perf_counter() - sn.t_pub) * self.speed_mult / self.cfg.dt)
            return sn, alpha
        if S.manual:
            hk = self.held_keys
            S.manual_action = np.array([float(hk["a"] or hk["left arrow"]) - float(hk["d"] or hk["right arrow"]),
                                        float(hk["w"] or hk["up arrow"]) - float(hk["s"] or hk["down arrow"])], dtype=np.float32)
        before = S.cur_pose
        self.acc, finished = S.advance(dt, self.acc, self.speed_mult)
        if finished and S.hold_at_end:
            S.waiting_next, S.paused, finished = True, True, False
            self._sync_snap = None
        if finished:
            if self.o.episodes and S.session["n"] >= self.o.episodes:
                self.done_flag = True
            else:
                S.episode += 1
                S.new_episode()
                self._sync_reset += 1
        if S.cur_pose is not before or self._sync_snap is None or finished:
            self._sync_id += 1
            self._sync_snap = make_snap(S, self._sync_id, self._sync_reset, 0.0)
        return self._sync_snap, (min(1.0, self.acc / self.cfg.dt) if not S.paused else 1.0)

    def _update(self) -> None:
        dt = self.fixed_dt or min(self.utime.dt, 0.1)
        self.frames += 1
        if self.frames % 20 == 0 and not self.o.headless:
            w, h = self.win.get_x_size(), self.win.get_y_size()
            if (w, h) != (self.W, self.H) and w > 0 and h > 0:            # the user resized / maximised the window
                self.W, self.H = w, h
                self.hud.set_window(w, h)
                if self.o.observer and hasattr(self, 'spec_cards'):
                    self._spec_layout()
        self._governor(dt)
        if self.map_open:
            self._map_update(dt)
        self._ui_update(dt)
        self.t_wall += dt
        self.fps_smooth += (min(1.0 / max(self.utime.dt, 1e-4), 400.0) - self.fps_smooth) * 0.05
        S = self.sess
        if S.manual:
            hk = self.held_keys
            S.manual_action = np.array([float(hk["a"] or hk["left arrow"]) - float(hk["d"] or hk["right arrow"]),
                                        float(hk["w"] or hk["up arrow"]) - float(hk["s"] or hk["down arrow"])], dtype=np.float32)
        if self.runner is not None:
            self.runner.speed = self.speed_mult
        sn, alpha = self._snapshot(dt)
        if sn.id != self._last_snap_id:
            self._last_snap_id = sn.id
            self.pool.commit()
        if sn.reset_id != self._last_reset:
            self._last_reset = sn.reset_id
            self._on_new_episode(sn)

        # ---- ego pose (interpolated)
        px, py, pth, pz = sn.prev_pose
        cx, cy, cth, cz = sn.cur_pose
        x, y = px + (cx - px) * alpha, py + (cy - py) * alpha
        d = (cth - pth + math.pi) % (2 * math.pi) - math.pi
        th = pth + d * alpha
        z = pz + (cz - pz) * alpha
        self.ego.set_pos(x, z + 0.05, y)
        self.ego.set_h(panda_heading(th))
        (self.ego_brake.show if sn.braking else self.ego_brake.hide)()
        v_ms = sn.v

        # ---- world
        self.pool.update(sn.actors, alpha, dt)
        if self.frames % 3 == 0:
            self.sr.update_signals(self.city.infra, sn.world_t)
        underground = sn.layer < 0
        self.sr.set_cutaway(underground, x, y)
        night = self.sr.atm.name in ("night", "dusk") or underground
        self.render.set_shader_input("head_on", 1.0 if night else 0.0)
        c, s_ = math.cos(th), math.sin(th)
        self.render.set_shader_input("head_pos", Vec3(x + c * 2.3, z + 0.75, y + s_ * 2.3))
        self.render.set_shader_input("head_dir", Vec3(c, -0.07, s_))

        # ---- camera
        if self.intro_t > 0.0:
            self.intro_t -= dt
            if self.intro_t <= 0.0:
                self._end_intro()
        elif self.cinematic:
            self.dir_t += dt
            if self.dir_t >= DIRECTOR[self.dir_i][1]:
                self.dir_i = (self.dir_i + 1) % len(DIRECTOR)
                self.dir_t = 0.0
                self.rig.set_mode(DIRECTOR[self.dir_i][0])
        if self.rig.mode == "orbit" and self.mouse.right:
            self.rig.orbit[0] += -self.mouse.velocity[0] * 3.0
            self.rig.orbit[1] = float(np.clip(self.rig.orbit[1] + self.mouse.velocity[1] * 2.0, 0.05, 1.45))
        if self.rig.mode == "free":
            hk = self.held_keys
            keys = {k_: float(hk[k_]) for k_ in ("w", "a", "s", "d", "q", "e")}
            keys["shift"] = float(hk["shift"])
            if self.follow_id is not None and any(keys[k_] for k_ in ("w", "a", "s", "d", "q", "e")):
                self.follow_id = None                                 # steering takes the camera back from a followed car
            dl = list(self._mouse_look())
            dl[0] += (float(hk["right arrow"]) - float(hk["left arrow"])) * dt * 1.0      # arrows still work
            dl[1] += (float(hk["down arrow"]) - float(hk["up arrow"])) * dt * 0.7
            self.rig.free_move(keys, dl, getattr(self, "_wheel", 0.0), dt)
            self._wheel = 0.0
        pose = (x, y, z, th, v_ms)
        if self.follow_id is not None:                                # chase a chosen car / bus instead of the ego
            fa = next((q for q in sn.actors if q.id == self.follow_id and q.alive), None)
            if fa is None:
                self.follow_id = None
            else:
                pose = (fa.x, fa.y, fa.z, fa.theta, fa.v)
                self.rig.mode = "chase"
        elif self.o.observer and self.rig.mode == "chase":
            self.rig.mode = "free"
        self.rig.update(pose, sn.ref.xy, dt)
        cam_pos = self.rig.pos
        if self.rig.mode == "free":
            mx, mz = 0.5 * (cam_pos[0] + self.rig.look[0]), 0.5 * (cam_pos[2] + self.rig.look[2])
            self.sr.atm.follow(mx, 0.0, mz)
        else:
            self.sr.atm.follow(x, z, y)
        self.sr.follow_sky(cam_pos[0], 0.0, cam_pos[2])

        # ---- effects
        if self.show_lidar:
            self._update_lidar(sn, x, y, z)
        elif self.lidar_np is not None:
            self.lidar_np.remove_node()
            self.lidar_np = None
        self.route_t -= dt
        if self.route_t <= 0.0:
            self.route_t = 0.15
            self._update_route(sn)
        self._full_t -= dt
        if self._full_t <= 0.0 and self.sess.nav_mode and not self.o.observer:
            self._full_t = 0.6
            self._update_full_route(sn)

        if self.o.observer and hasattr(self, 'spec_cards'):
            self.spec_t -= dt
            if self.spec_t <= 0.0 or self.spec_dirty:
                self.spec_t, self.spec_dirty = 0.25, False
                self._draw_spectator(sn)
        # ---- HUD
        if self.hud.visible:
            tele = sn.tele
            tele.fps = self.fps_smooth
            tele.ego = (x, y, th)
            if getattr(self.sess, "planning", False):
                tele.status, tele.status_kind = "ROUTE OVERVIEW: A → B", "goal"
            self.hud.update(tele)
        # ---- recording / headless end conditions
        if self.writer is not None:
            self._write_frame()
        if self.o.shot and not self._shot_done and self.t_wall >= self.o.shot_at:
            save_screenshot(self.o.shot)
            self._shot_done = True
            print("[viewer] screenshot", self.o.shot)
            if self.o.headless and self.o.duration <= 0:
                self.done_flag = True
        if self.o.duration > 0 and self.t_wall >= self.o.duration:
            self.done_flag = True
        if self.done_flag and not self.o.headless:                       # windowed run with --duration / --episodes
            self._stop_recording()
            wall = pytime.time() - self._t_start
            print(f"[viewer] {self.frames} frames in {wall:.1f} s -> {self.frames / max(wall, 1e-6):.0f} fps on screen")
            from ursina import application
            application.quit()

    def _governor(self, dt: float) -> None:
        """Keep slow machines usable: if the frame rate stays below ~38 fps, shrink the shadow map, then switch shadows off."""
        if self.fixed_dt or self.q.shadows <= 0 or self.frames < 120:
            return
        self._slow = getattr(self, "_slow", 0.0) + (dt if self.fps_smooth < 38 else -dt)
        self._slow = max(0.0, self._slow)
        if self._slow > 4.0:
            self._slow = 0.0
            self._shadow_level = getattr(self, "_shadow_level", self.q.shadows)
            nxt = self._shadow_level // 2 if self._shadow_level > 1024 else 0
            self._shadow_level = nxt
            self.sr.atm.set_shadow_res(nxt)
            self.hud.toast("Low frame rate: shadow quality reduced" if nxt else "Low frame rate: shadows off", "info")

    # ------------------------------------------------------------------ effects
    def _update_lidar(self, sn, x: float, y: float, z: float) -> None:
        perc = sn.perc
        if perc is None:
            return
        n = self.cfg.n_rays
        rng = self.cfg.lidar_range
        d = perc.lidar_d * rng
        ang = ray_angles(sn.theta, n)
        ls = LineSegs("lidar")
        ls.set_thickness(1.6)
        pts = LineSegs("hits")
        pts.set_thickness(6.0)
        h = z + 0.55
        for i in range(n):
            hit = d[i] < rng - 0.5
            ex, ey = x + math.cos(ang[i]) * d[i], y + math.sin(ang[i]) * d[i]
            if perc.lidar_ped[i] > 0.5 and hit:
                col = (1.0, 0.32, 0.36)
            elif perc.lidar_veh[i] > 0.5 and hit:
                col = (1.0, 0.66, 0.25)
            else:
                col = (0.30, 0.88, 1.0)
            ls.set_color(*col, 0.55 if hit else 0.16)
            ls.move_to(x, h, y)
            ls.draw_to(ex, h, ey)
            if hit:
                pts.set_color(*col, 0.95)
                pts.move_to(ex, h, ey)
        if self.lidar_np is not None:
            self.lidar_np.remove_node()
        root = self.fx.attach_new_node("lidar")
        root.attach_new_node(ls.create())
        root.attach_new_node(pts.create())
        root.set_shader_off()
        root.set_light_off()
        root.set_transparency(TransparencyAttrib.M_alpha)
        root.set_depth_write(False)
        root.hide(SHADOW_BIT)
        self.lidar_np = root

    def _update_full_route(self, sn) -> None:
        """The whole remaining route as a blue band (the part already driven disappears, as in a navigation app)."""
        ref = sn.ref
        i0 = int(np.searchsorted(ref.s, max(0.0, sn.s - 2.0)))
        xy = ref.xy[i0::2]
        if len(xy) < 2:
            if self.full_np is not None:
                self.full_np.hide()
            return
        zz = ref.z[i0::2] if ref.z is not None else None
        mb = MeshBuilder()
        self._ribbon(mb, xy, zz, 1.25, (0.15, 0.47, 1.0, 0.62), 0.10)
        if self.full_np is None:
            self.full_np = mesh_to_node(mb.build(), "full_route", self.fx, dynamic=True)
            self.full_np.set_shader(self.sr.unlit)
            self.full_np.set_texture(self.sr.bank.get("white"))
            self.full_np.set_shader_input("glow", 1.2)
            self.full_np.set_transparency(TransparencyAttrib.M_alpha)
            self.full_np.set_depth_write(False)
            self.full_np.hide(SHADOW_BIT)
        else:
            from .gfx import mesh_to_geom
            gn = self.full_np.node()
            gn.remove_all_geoms()
            gn.add_geom(mesh_to_geom(mb.build(), "full_route", True))
            self.full_np.show()

    def _update_route(self, sn) -> None:
        if self.o.observer:
            return
        ref = sn.ref
        s = sn.s
        i0 = int(np.searchsorted(ref.s, max(0.0, s - 4.0)))
        i1 = int(np.searchsorted(ref.s, s + 140.0))
        if i1 - i0 < 2:
            if self.route_np is not None:
                self.route_np.hide()
            return
        xy = ref.xy[i0:i1:2]
        zz = (ref.z[i0:i1:2] if ref.z is not None else np.zeros(len(xy)))
        d = np.gradient(xy, axis=0)
        n = np.column_stack([-d[:, 1], d[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        w = 0.55
        L = np.column_stack([xy[:, 0] + n[:, 0] * w, zz + 0.13, xy[:, 1] + n[:, 1] * w])
        R = np.column_stack([xy[:, 0] - n[:, 0] * w, zz + 0.13, xy[:, 1] - n[:, 1] * w])
        fade = np.clip(1.0 - np.arange(len(xy)) / len(xy), 0.1, 1.0)
        mb = MeshBuilder()
        P = np.stack([L[:-1], R[:-1], R[1:], L[1:]], axis=1)
        col = np.zeros((len(P), 4), dtype=np.float32)
        col[:] = (0.15, 0.85, 1.0, 0.0)
        col[:, 3] = 0.30 + 0.35 * fade[:-1]
        mb.quads(P, col, None, np.tile([0, 1, 0], (len(P), 1)))
        if self.route_np is None:
            self.route_np = mesh_to_node(mb.build(), "route", self.fx, dynamic=True)
            self.route_np.set_shader(self.sr.unlit)
            self.route_np.set_texture(self.sr.bank.get("white"))
            self.route_np.set_shader_input("glow", 1.5)
            self.route_np.set_transparency(TransparencyAttrib.M_alpha)
            self.route_np.set_depth_write(False)
            self.route_np.hide(SHADOW_BIT)
        else:                                          # swap the geometry in place (no node churn -> no GC hiccup)
            from .gfx import mesh_to_geom
            gn = self.route_np.node()
            gn.remove_all_geoms()
            gn.add_geom(mesh_to_geom(mb.build(), "route", True))
            self.route_np.show()

    # ------------------------------------------------------------------ input
    def _input(self, key: str) -> None:
        if self.map_open:                                      # the navigator map takes all input while open
            self._map_input(key)
            return
        if self.menu is not None:                              # pause / trip-finished menu is modal
            self._menu_input(key)
            return
        if key == "left mouse down" and self.toolbar is not None and self.toolbar_card.ent.enabled:
            px, py = self._mouse_px()
            x, y = self._toolbar_xy
            act = self.toolbar.hit(px - x, py - y)
            if act is not None:
                self._toolbar_action(act)
            return
        if key == "escape":
            self._open_menu("pause")
            return
        if key == "space" and self.intro_t <= 0.0:
            self.sess.paused = not self.sess.paused
            self._toolbar_dirty = True
            return
        if key in ("r", "n"):
            self._new_trip("restart")
            return
        self._hotkey(key)

    def _hotkey(self, key: str) -> None:
        if self.intro_t > 0.0 and (key in ("space", "enter", "c", "k", "m") or (len(key) == 1 and key in "12345678")):
            self._end_intro()                                  # any camera / go key ends the route overview
            if key in ("space", "enter"):
                return
        if key == "m" and not self.o.observer:
            self._open_map("setup" if self._awaiting_pick else "drive")
            return
        if key == "z":
            self.hud.toast(self.hud.cycle_map_zoom(), "info")
            return
        if key == "o":
            self.show_alts = not self.show_alts
            if self.plan_np is not None:
                (self.plan_np.show if self.show_alts else self.plan_np.hide)()
            for tx in self.plan_labels:
                tx.enabled = self.show_alts
            self.hud.toast("Navigator overlay " + ("on" if self.show_alts else "off"), "info")
            return
        if key == "b" and self.sess.nav_mode:                 # replay the route overview
            self._start_intro()
            return
        if key in ("c",):
            self.cam_idx = (self.cam_idx + 1) % len(CAMERAS)
            self.cinematic = False
            self.rig.set_mode(CAMERAS[self.cam_idx])
            self.hud.toast(f"Camera: {CAMERAS[self.cam_idx]}", "info")
        elif self.o.observer and key in ("[", "]", "left bracket", "right bracket", "tab", "f", "h") or (self.o.observer and key in "123456789" and len(key) == 1):
            if key in ("[", "left bracket"):
                self._goto_hotspot(self.hot_i - 1)
            elif key in ("]", "right bracket"):
                self._goto_hotspot(self.hot_i + 1)
            elif key == "tab":
                self.captured = not self.captured
                self.hud.toast("Mouse look on (Tab to release)" if self.captured else "Mouse look off", "info")
            elif key == "f":
                self._follow_nearest(self.runner.snap if self.runner is not None else self._sync_snap)
            elif key == "h":
                self.help_visible = not self.help_visible
                self.help_pinned = self.help_visible
                self.spec_dirty = True
            else:
                self._goto_hotspot(int(key) - 1)
        elif key in "12345678" and len(key) == 1:
            self.cam_idx = int(key) - 1
            self.cinematic = False
            self.rig.set_mode(CAMERAS[self.cam_idx])
        elif key == "k":
            self.cinematic = not self.cinematic
            self.hud.toast("Cinematic director " + ("on" if self.cinematic else "off"), "info")
        elif key == "t":
            self.weather_i = (self.weather_i + 1) % len(WEATHER)
            self.sr.set_preset(WEATHER[self.weather_i])
            self.hud.toast(f"Scene: {WEATHER[self.weather_i]}", "info")
        elif key == "l":
            self.show_lidar = not self.show_lidar
        elif key == "h":
            self.hud.toggle()
        elif key == "space":
            self.sess.paused = not self.sess.paused
        elif key in ("r", "n"):
            if self.runner is not None:
                self.runner.request_new_episode()
            else:
                self.sess.episode += 1
                self.sess.new_episode()
                self._sync_reset += 1
        elif key in ("+", "=", "numpad +"):
            self.speed_mult = min(8.0, self.speed_mult * 1.5)
            self.hud.toast(f"Sim speed x{self.speed_mult:.1f}", "info")
        elif key in ("-", "numpad -"):
            self.speed_mult = max(0.2, self.speed_mult / 1.5)
            self.hud.toast(f"Sim speed x{self.speed_mult:.1f}", "info")
        elif key == "g":
            self.sess.manual = not self.sess.manual
            self.hud.toast("Manual control (WASD / arrows)" if self.sess.manual else "Autopilot engaged", "info")
        elif key in ("f12", "p"):
            p = f"{self.o.shots_dir}/shot_{int(pytime.time())}.png"
            save_screenshot(p)
            self.hud.toast(f"Saved {p}", "ok")
        elif key == "v":
            if self.writer is None:
                self._start_recording(f"assets/videos/rec_{int(pytime.time())}.mp4")
            else:
                self._stop_recording()
        elif key == "scroll up":
            self.rig.zoom = min(4.0, self.rig.zoom * 1.12)
            self._wheel = 1.0
        elif key == "scroll down":
            self.rig.zoom = max(0.35, self.rig.zoom / 1.12)
            self._wheel = -1.0
        elif key == "escape":
            self._stop_recording()
            if self.runner is not None:
                self.runner.stop()
            from ursina import application
            application.quit()

    # ------------------------------------------------------------------ recording
    def _start_recording(self, path: str) -> None:
        import imageio
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(path, fps=self.o.fps, codec="libx264", quality=6.5, macro_block_size=2,
                                         pixelformat="yuv420p")
        self.rec_path = path
        self.fixed_dt = 1.0 / self.o.fps
        print(f"[record] writing {path} @ {self.o.fps} fps")

    def _write_frame(self) -> None:
        tex = self.win.get_screenshot()
        if tex is None:
            return
        buf = tex.get_ram_image_as("RGB")
        w, h = tex.get_x_size(), tex.get_y_size()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)[::-1]
        self.writer.append_data(np.ascontiguousarray(arr))
        self.rec_frames += 1

    def _stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            print(f"[record] saved {self.rec_path} ({self.rec_frames} frames)")

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        if self.o.headless or (self.o.record and self.o.duration > 0):
            n = 0
            t0 = pytime.time()
            while not self.done_flag:
                self.app.step()
                n += 1
                if n % 120 == 0:
                    print(f"[viewer] t={self.t_wall:5.1f} s  episodes={self.sess.session['n']}  success={self.sess.session['success']}")
            wall = pytime.time() - t0
            print(f"[viewer] {n} frames in {wall:.1f} s -> {n / max(wall, 1e-6):.0f} fps (render + simulation + HUD)")
            self._stop_recording()
        else:
            self.app.run()
            self._stop_recording()


def run_viewer(opts: ViewerOptions) -> None:
    Viewer(opts).run()


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--model", default=None, help="trained .zip (default: rule-based expert)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality", default="auto", choices=["auto", "low", "medium", "high", "ultra"])
    ap.add_argument("--preset", default="day", choices=list(WEATHER))
    ap.add_argument("--camera", default="chase", choices=list(CAMERAS))
    ap.add_argument("--size", default=None, help="WxH, e.g. 1920x1080")
    ap.add_argument("--headless", action="store_true", help="no window (offscreen); use with --record")
    ap.add_argument("--record", default=None, help="write an MP4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (video / headless run)")
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--cinematic", action="store_true")
    ap.add_argument("--manual", action="store_true")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--hotspots", default=None, help="hotspots.csv of an audit: draw them as beacons")
    ap.add_argument("--tour", action="store_true", help="showcase: long, eventful routes with the cinematic camera")
    ap.add_argument("--observer", action="store_true", help="free-fly camera over the city (no car, no HUD)")
    ap.add_argument("--recovery", action="store_true", help="include sideways / backwards starts (hard recovery scenarios)")
    ap.add_argument("--shot", default=None, help="save a screenshot to this PNG (see --shot-at)")
    ap.add_argument("--shot-at", type=float, default=5.0, help="simulated seconds before the screenshot")
    g = ap.add_argument_group("point-to-point navigation (the navigator plans A -> B, the network drives it)")
    g.add_argument("--start-node", "--from", dest="start", default=None,
                   help="start A: node id, 'x,y' metres, or a landmark / street name ('Nur Alem', 'Kabanbay')")
    g.add_argument("--goal-node", "--to", dest="goal", default=None, help="goal B: same formats as --start-node")
    g.add_argument("--random-route", action="store_true", help="a random A -> B every episode (same --seed = same routes)")
    g.add_argument("--route-km", default="0.6-1.5", help="length range of random routes, km (default 0.6-1.5)")
    g.add_argument("--verified", action="store_true", help="random routes only if the rule-based expert can drive them")
    g.add_argument("--intro", type=float, default=4.0, help="seconds of route overview before departure (0 = off)")
    g.add_argument("--pick", action="store_true", help="open the map first: click A and B yourself (M opens it any time)")
    a = ap.parse_args(argv)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    km = tuple(float(v) for v in a.route_km.replace(",", "-").split("-"))
    run_viewer(ViewerOptions(a.map, a.model, a.seed, a.quality, a.preset, a.headless, size, a.record, a.fps, a.duration,
                             a.cinematic, a.manual, 1.0, a.episodes, not a.no_hud, a.camera, difficulty=a.difficulty,
                             fullscreen=a.fullscreen, landmarks=not a.no_landmarks, shot=a.shot, shot_at=a.shot_at, recovery=a.recovery, observer=a.observer, hotspots=a.hotspots, tour=a.tour or None,
                             start=a.start, goal=a.goal, random_route=a.random_route, route_km=(km[0], km[-1]),
                             verified=a.verified, intro=a.intro, pick=a.pick))


if __name__ == "__main__":
    main()

