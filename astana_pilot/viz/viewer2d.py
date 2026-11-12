"""Light-weight 2-D viewer (pygame) - the same driving session and HUD as the 3-D viewer, any laptop.

    python -m astana_pilot.viz.viewer2d --map assets/maps/astana_expo.pkl
    python -m astana_pilot.viz.viewer2d --map synthetic --model runs/night1/final_model.zip

Keys: SPACE pause | R new episode | +/- or mouse wheel zoom | L lidar | H HUD | G manual drive (WASD / arrows)
      F12 screenshot | ESC quit
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pygame

from ..env.city_map import LAYER_TUNNEL, CityMap, polyline_length
from ..env.infrastructure import GREEN, RED, YELLOW
from .hud import HudPanels, Telemetry
from .session import DriveSession

BG = (14, 20, 32)
PARK, WATER, BUILD, BUILD_EDGE = (24, 50, 44), (24, 52, 88), (46, 60, 92), (76, 98, 140)
ROAD, CASING = (104, 114, 134), (36, 44, 60)
FLY, FLY_CASE, TUN = (255, 176, 80), (120, 70, 20), (96, 170, 255)
ZEBRA = (230, 232, 238)
CAR_COL = {"car": (236, 238, 244), "bus": (70, 200, 120), "parked": (150, 156, 170), "ped": (255, 96, 106)}
LIGHT_COL = {GREEN: (70, 235, 130), YELLOW: (255, 200, 70), RED: (255, 84, 94)}


class BaseMap:
    """The whole city pre-rendered once at ``px_per_m`` pixels per metre (vector -> one big surface)."""

    def __init__(self, city: CityMap, px_per_m: float = 1.5) -> None:
        self.city, self.S = city, px_per_m
        x0, y0, x1, y1 = city.bounds
        self.x0, self.y0 = x0 - 40.0, y0 - 40.0
        self.W, self.H = int((x1 - x0 + 80) * px_per_m), int((y1 - y0 + 80) * px_per_m)
        s = pygame.Surface((self.W, self.H))
        s.fill(BG)
        raw = city.raw
        for ring in getattr(raw, "green", []) or []:
            pygame.draw.polygon(s, PARK, [self.px(p) for p in ring])
        for ring in getattr(raw, "water", []) or []:
            pygame.draw.polygon(s, WATER, [self.px(p) for p in ring])
        for ring, _ in raw.buildings:
            pts = [self.px(p) for p in ring]
            pygame.draw.polygon(s, BUILD, pts)
            pygame.draw.polygon(s, BUILD_EDGE, pts, 1)
        seen = set()
        order = []
        for i, e in enumerate(raw.edges):
            key = (min(e.u, e.v), max(e.u, e.v), len(e.xy), round(polyline_length(e.xy), 1))
            if key in seen:
                continue
            seen.add(key)
            lay = city.edge_layers[i]
            order.append((0 if lay == {0} else (1 if lay == {LAYER_TUNNEL} else 2), i, e))
        order.sort(key=lambda t: t[0])
        for pass_, casing in ((0, True), (1, False)):                 # casings first, then fills
            for lv, i, e in order:
                w = max(2, int(2 * city.edge_hw[i] * px_per_m))
                pts = [self.px(p) for p in e.xy]
                if len(pts) < 2:
                    continue
                if lv == 0:
                    col = CASING if casing else ROAD
                    ww = w + (3 if casing else 0)
                elif lv == 1:
                    col = (CASING if casing else (40, 70, 120))
                    ww = w + (3 if casing else 0)
                else:
                    col = FLY_CASE if casing else FLY
                    ww = w + (4 if casing else 0)
                pygame.draw.lines(s, col, False, pts, ww)
                for p in (pts[0], pts[-1]):
                    pygame.draw.circle(s, col, p, ww // 2)
        for cw in city.infra.crosswalks:                              # zebras
            h = cw.heading
            fx, fy = math.cos(h), math.sin(h)
            lx, ly = -fy, fx
            for t in np.arange(-cw.half_len + 0.6, cw.half_len - 0.4, 1.1):
                c = (cw.xy[0] + lx * t, cw.xy[1] + ly * t)
                a = self.px((c[0] - fx * 2.0, c[1] - fy * 2.0))
                b = self.px((c[0] + fx * 2.0, c[1] + fy * 2.0))
                pygame.draw.line(s, ZEBRA, a, b, max(1, int(0.5 * px_per_m)))
        self.surf = s

    def px(self, p) -> Tuple[int, int]:
        return (int((p[0] - self.x0) * self.S), int(self.H - (p[1] - self.y0) * self.S))


class Viewer2D:
    def __init__(self, map_path: str, model: Optional[str], seed: int, size: Tuple[int, int], difficulty: float,
                 shots_dir: str = "assets/shots") -> None:
        pygame.init()
        pygame.display.set_caption("Astana Pilot - 2D")
        self.screen = pygame.display.set_mode(size)
        self.W, self.H = size
        self.shots_dir = shots_dir
        self.hud: Optional[HudPanels] = None
        self.sess = DriveSession(map_path, model, seed, difficulty, toast=lambda t, k: self.hud.toast(t, k) if self.hud else None,
                                 flash=lambda k: self.hud.flash(k) if self.hud else None)
        self.city = self.sess.city
        t0 = time.time()
        self.base = BaseMap(self.city)
        self.hud = HudPanels(size, 12.0, self.city)
        print(f"[viewer2d] base map {self.base.W}x{self.base.H} px in {time.time() - t0:.1f} s")
        self.scale = 4.0                       # screen px per metre
        self.show_lidar, self.show_hud = True, True
        self.speed_mult = 1.0
        self.acc = 0.0
        self.surfs: Dict[str, pygame.Surface] = {}
        self.clock = pygame.time.Clock()
        self.fps = 60.0

    # ------------------------------------------------------------------ drawing
    def _tx(self, cam: np.ndarray):
        s = self.scale
        return lambda x, y: (self.W / 2 + (x - cam[0]) * s, self.H / 2 - (y - cam[1]) * s)

    def _draw_world(self, cam: np.ndarray, pose: Tuple[float, float, float]) -> None:
        scr = self.screen
        scr.fill(BG)
        b, s = self.base, self.scale
        # visible window in base pixels
        hw, hh = self.W / (2 * s), self.H / (2 * s)
        bx0, by1 = b.px((cam[0] - hw, cam[1] - hh))
        bx1, by0 = b.px((cam[0] + hw, cam[1] + hh))
        rect = pygame.Rect(bx0, by0, bx1 - bx0, by1 - by0)
        clip = rect.clip(pygame.Rect(0, 0, b.W, b.H))
        if clip.w > 1 and clip.h > 1:
            sub = b.surf.subsurface(clip)
            dw, dh = max(1, int(clip.w * (s / b.S))), max(1, int(clip.h * (s / b.S)))
            img = pygame.transform.smoothscale(sub, (dw, dh)) if s / b.S < 1.0 else pygame.transform.scale(sub, (dw, dh))
            scr.blit(img, ((clip.x - rect.x) * (s / b.S), (clip.y - rect.y) * (s / b.S)))
        T = self._tx(cam)
        env = self.sess.env
        # traffic lights
        t = env.world.t
        infra = self.city.infra
        for c in infra.controls:
            x, y = T(*c.xy)
            if -20 < x < self.W + 20 and -20 < y < self.H + 20:
                if c.kind == "light":
                    st = infra.light_state(c.node, c.group, t)
                    pygame.draw.circle(scr, LIGHT_COL[st], (int(x), int(y)), max(3, int(0.9 * s)))
                else:
                    pygame.draw.circle(scr, (255, 90, 90) if c.kind == "stop" else (255, 255, 255), (int(x), int(y)), max(2, int(0.6 * s)))
        # route
        ref = env.ref
        i0 = int(np.searchsorted(ref.s, max(0.0, env._s - 5.0)))
        pts = [T(*p) for p in ref.xy[i0:i0 + 400:3]]
        if len(pts) > 1:
            pygame.draw.lines(scr, (56, 220, 255), False, pts, max(2, int(0.6 * s)))
        g = T(*ref.xy[-1])
        pygame.draw.circle(scr, (80, 255, 150), (int(g[0]), int(g[1])), int(self.sess.cfg.goal_radius * s), 2)
        # actors
        for a in env.world.actors:
            if not a.alive:
                continue
            x, y = T(a.x, a.y)
            if not (-30 < x < self.W + 30 and -30 < y < self.H + 30):
                continue
            col = CAR_COL.get(a.kind, (200, 200, 200))
            if a.kind == "ped":
                pygame.draw.circle(scr, col, (int(x), int(y)), max(2, int(0.45 * s)))
                continue
            self._box(x, y, a.theta, a.length, a.width, col, edge=(20, 24, 34))
            if getattr(a, "braking", False):
                bx, by = T(a.x - math.cos(a.theta) * a.length / 2, a.y - math.sin(a.theta) * a.length / 2)
                pygame.draw.circle(scr, (255, 60, 60), (int(bx), int(by)), max(2, int(0.5 * s)))
        # lidar + ego
        perc = env._perc
        v = env.vehicle
        ex, ey = T(pose[0], pose[1])
        if self.show_lidar and perc is not None:
            ang = np.linspace(0, 2 * math.pi, self.sess.cfg.n_rays, endpoint=False) + v.theta
            d = perc.lidar_d * self.sess.cfg.lidar_range
            for i in range(len(d)):
                hit = d[i] < self.sess.cfg.lidar_range - 0.5
                col = (255, 96, 106) if (hit and perc.lidar_ped[i] > 0.5) else (255, 176, 80) if (hit and perc.lidar_veh[i] > 0.5) else (64, 200, 255)
                px, py = T(pose[0] + math.cos(ang[i]) * d[i], pose[1] + math.sin(ang[i]) * d[i])
                pygame.draw.line(scr, tuple(int(c * (0.9 if hit else 0.35)) for c in col), (ex, ey), (px, py), 1)
                if hit:
                    pygame.draw.circle(scr, col, (int(px), int(py)), 2)
        vp = self.sess.cfg.vehicle
        glow = pygame.Surface((int(vp.length * s * 2.2), int(vp.length * s * 2.2)), pygame.SRCALPHA)
        pygame.draw.circle(glow, (56, 160, 255, 60), (glow.get_width() // 2, glow.get_height() // 2), glow.get_width() // 2)
        scr.blit(glow, (ex - glow.get_width() / 2, ey - glow.get_height() / 2))
        self._box(ex, ey, pose[2], vp.length, vp.width, (40, 130, 255), edge=(255, 255, 255))

    def _box(self, x: float, y: float, th: float, L: float, W: float, col, edge) -> None:
        s = self.scale
        c, sn = math.cos(th), math.sin(th)
        pts = []
        for lx, ly in ((L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)):
            pts.append((x + (lx * c - ly * sn) * s, y - (lx * sn + ly * c) * s))
        pygame.draw.polygon(self.screen, col, pts)
        pygame.draw.polygon(self.screen, edge, pts, 1)

    def _draw_hud(self, t: Telemetry) -> None:
        h = self.hud
        panels = h.render_all(t) if not self.surfs else h.due_panels(t)
        for k, arr in panels.items():
            hh, ww = arr.shape[:2]
            self.surfs[k] = pygame.image.frombuffer(arr.tobytes(), (ww, hh), "RGBA").convert_alpha()
        for k, surf in self.surfs.items():
            x, y, _, _ = h._geom[k]
            self.screen.blit(surf, (x, y))

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        S = self.sess
        running = True
        while running:
            dt = min(self.clock.tick(60) / 1000.0, 0.1)
            self.fps += (1.0 / max(dt, 1e-3) - self.fps) * 0.05
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.MOUSEWHEEL:
                    self.scale = float(np.clip(self.scale * (1.12 ** e.y), 0.6, 18.0))
                elif e.type == pygame.KEYDOWN:
                    k = e.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_SPACE:
                        S.paused = not S.paused
                    elif k == pygame.K_r:
                        S.episode += 1
                        S.new_episode()
                        self.acc = 0.0
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.scale = min(18.0, self.scale * 1.2)
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.scale = max(0.6, self.scale / 1.2)
                    elif k == pygame.K_l:
                        self.show_lidar = not self.show_lidar
                    elif k == pygame.K_h:
                        self.show_hud = not self.show_hud
                    elif k == pygame.K_g:
                        S.manual = not S.manual
                    elif k == pygame.K_PERIOD:
                        self.speed_mult = min(8.0, self.speed_mult * 1.5)
                    elif k == pygame.K_COMMA:
                        self.speed_mult = max(0.25, self.speed_mult / 1.5)
                    elif k == pygame.K_F12:
                        Path(self.shots_dir).mkdir(parents=True, exist_ok=True)
                        p = f"{self.shots_dir}/shot2d_{int(time.time())}.png"
                        pygame.image.save(self.screen, p)
                        self.hud.toast(f"Saved {p}", "ok")
            if S.manual:
                kk = pygame.key.get_pressed()
                S.manual_action = np.array([float(kk[pygame.K_a] or kk[pygame.K_LEFT]) - float(kk[pygame.K_d] or kk[pygame.K_RIGHT]),
                                            float(kk[pygame.K_w] or kk[pygame.K_UP]) - float(kk[pygame.K_s] or kk[pygame.K_DOWN])],
                                           dtype=np.float32)
            self.acc, finished = S.advance(dt, self.acc, self.speed_mult)
            if finished:
                S.episode += 1
                S.new_episode()
                self.acc = 0.0
            alpha = min(1.0, self.acc / S.cfg.dt) if not S.paused else 1.0
            px, py, pth, _ = S.prev_pose
            cx, cy, cth, _ = S.cur_pose
            d = (cth - pth + math.pi) % (2 * math.pi) - math.pi
            pose = (px + (cx - px) * alpha, py + (cy - py) * alpha, pth + d * alpha)
            self._draw_world(np.array(pose[:2]), pose)
            if self.show_hud:
                self._draw_hud(S.telemetry(self.fps, pose))
            pygame.display.flip()
        pygame.quit()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", default="assets/maps/astana_expo.pkl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--difficulty", type=float, default=1.0)
    a = ap.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    Viewer2D(a.map, a.model, a.seed, (w, h), a.difficulty).run()


if __name__ == "__main__":
    main()
