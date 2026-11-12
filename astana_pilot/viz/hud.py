"""Heads-up display: speedometer, rule-compliance board, "what the AI sees" radar, decision explanation, mini-map,
session statistics, event toasts.

Panels are drawn with Pillow (see :mod:`canvas`) into small RGBA arrays. Static parts (frames, labels) are cached and
only the dynamic content is redrawn; at most one panel is refreshed per call, so the HUD costs ~1 ms per frame.
:class:`HudPanels` needs no graphics engine (the pygame viewer uses it too); :class:`Hud` shows the panels as cards in
Ursina's ``camera.ui``.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from ..env.infrastructure import GREEN as L_GREEN
from ..env.infrastructure import RED as L_RED
from ..env.infrastructure import YELLOW as L_YELLOW
from . import canvas as C
from .canvas import Canvas

RULES = (("red_light", "Red light"), ("yellow_light", "Yellow light"), ("stop_sign", "Stop sign"),
         ("yield", "Give way"), ("zebra", "Pedestrian crossing"), ("speeding", "Speed limit"), ("wrong_side", "Opposite lane"))


@dataclass
class Telemetry:
    """Everything the HUD needs, assembled once per frame by the viewer."""
    speed_kmh: float = 0.0
    gear: str = "N"
    limit_kmh: float = 50.0
    target_kmh: float = 0.0
    steer_deg: float = 0.0
    steer_cmd: float = 0.0
    pedal: float = 0.0
    e_y: float = 0.0
    progress: float = 0.0
    status: str = "DRIVING"
    status_kind: str = "ok"                 # ok | goal | bad
    reason: str = "Free road"
    reason_kind: str = "ok"                 # ok | info | warn
    constraints: List[Tuple[str, str, str]] = field(default_factory=list)   # (label, value, kind)
    aeb: bool = False
    ttc: float = math.inf
    gap: float = math.inf
    light: Optional[Tuple[int, float, float]] = None        # (state, distance, seconds to change)
    ep_viol: Dict[str, float] = field(default_factory=dict)
    ep_speeding_frac: float = 0.0
    mu: float = 1.0
    episode: int = 0
    t_sim: float = 0.0
    fps: float = 0.0
    n_actors: int = 0
    model_name: str = "expert"
    map_name: str = "Astana"
    session: Dict[str, float] = field(default_factory=dict)
    perception: object = None
    ego: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lidar: Optional[np.ndarray] = None
    lidar_angles: Optional[np.ndarray] = None
    circles: Optional[np.ndarray] = None
    circle_kind: Optional[np.ndarray] = None
    route_ahead: Optional[np.ndarray] = None
    weather: str = "day"
    nav: Optional[Dict] = None                      # navigator card (see DriveSession.nav_info)
    route_xy: Optional[np.ndarray] = None           # whole chosen route (decimated), for the mini-map
    alts_xy: Optional[List[np.ndarray]] = None      # alternative routes the navigator rejected


def explain(perc, v_ms: float, limit_ms: float, v_target_ms: float, lead_kind_names=None) -> Tuple[str, str]:
    """Human-readable reason for the current driving behaviour (what the policy "is thinking")."""
    from ..env.infrastructure import RED, YELLOW
    names = lead_kind_names or {0: "wall", 1: "parked car", 2: "car", 3: "bus", 4: "pedestrian"}
    if perc is None:
        return "Initialising", "info"
    if perc.ctrl_d < 60 and perc.ctrl_event is not None and perc.ctrl_event.kind == "light" and perc.ctrl_state == RED:
        return (f"Red light ahead, {perc.ctrl_d:.0f} m" if v_ms > 0.5 else "Waiting for the green light"), "warn"
    if perc.ctrl_d < 45 and perc.ctrl_event is not None and perc.ctrl_event.kind == "light" and perc.ctrl_state == YELLOW:
        return "Yellow light: stopping if safe", "warn"
    if perc.ctrl_d < 45 and perc.ctrl_event is not None and perc.ctrl_event.kind == "stop":
        return "Stop sign ahead", "warn"
    if perc.ctrl_d < 40 and perc.ctrl_event is not None and perc.ctrl_event.kind == "yield":
        return "Giving way at the junction", "info"
    if perc.zebra_busy and perc.zebra_d < 45:
        return "Yielding to pedestrians on the crossing", "warn"
    if perc.blocked:
        return "Lane blocked: waiting / passing when the opposite lane is free", "warn"
    if perc.gap < 30.0 and perc.lead_kind >= 0:
        kind = names.get(int(perc.lead_kind), "vehicle")
        if perc.v_lead < 0.5 and v_ms < 1.0:
            return f"Waiting behind a stopped {kind}", "info"
        return f"Following a {kind} at {perc.gap:.0f} m", "info"
    if perc.zebra_d < 30:
        return "Approaching a pedestrian crossing", "info"
    if v_ms < 0.5:
        return "Standing", "info"
    if v_target_ms < 0.7 * limit_ms:
        return "Slowing for the road geometry", "info"
    return "Free road: cruising at the speed limit", "ok"


class MiniMapData:
    """Vector data of the city for the circular mini-map (drawn crisp at any resolution, culled per frame)."""

    def __init__(self, city) -> None:
        from ..env.city_map import polyline_length
        seen, pts, wid, lv = set(), [], [], []
        for i, e in enumerate(city.raw.edges):
            key = (min(e.u, e.v), max(e.u, e.v), len(e.xy), round(polyline_length(e.xy), 1))
            if key in seen:
                continue
            seen.add(key)
            layers = city.edge_layers[i]
            pts.append(np.asarray(e.xy, dtype=np.float32))
            wid.append(2.0 * city.edge_hw[i])
            lv.append(0 if layers == {0} else (1 if max(layers) <= 0 else 2))
        self.pts, self.wid, self.lv = pts, np.array(wid), np.array(lv)
        self.bbox = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in pts])
        self.b_rings = [np.asarray(r, dtype=np.float32) for r, _ in city.raw.buildings]
        self.b_bbox = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in self.b_rings]) \
            if self.b_rings else np.zeros((0, 4))
        self.g_rings = [np.asarray(r, dtype=np.float32) for r in (getattr(city.raw, "green", []) or [])]
        self.g_bbox = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in self.g_rings]) \
            if self.g_rings else np.zeros((0, 4))

    @staticmethod
    def _vis(bbox: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
        if len(bbox) == 0:
            return np.zeros(0, dtype=int)
        m = (bbox[:, 2] >= cx - r) & (bbox[:, 0] <= cx + r) & (bbox[:, 3] >= cy - r) & (bbox[:, 1] <= cy + r)
        return np.nonzero(m)[0]


# ====================================================================================== panels (pure drawing)
class HudPanels:
    """Minimal, glanceable HUD drawn with Pillow (no graphics engine needed; the pygame viewer uses it too).

    Design rules (automotive HMI guidance): driving information first and bottom-centre, everything else small and
    quiet, dynamic content only when relevant (rule board stays a green chip until something is violated), one accent
    colour, no title bars. Layout is anchored to the screen corners and scaled by ``min(H/900, W/1600)`` so any aspect
    ratio (16:9, 16:10 QHD, ultrawide) and any resolution stays crisp; panels are tiny textures, drawn 2x
    super-sampled, with cached static parts - about 1 ms per frame.
    """

    PANEL_HZ = {"speed": 12.0, "radar": 8.0, "map": 8.0, "toast": 6.0, "rules": 4.0, "nav": 4.0, "top": 3.0, "stats": 1.0}
    MAP_RANGES = (150.0, 450.0, 1200.0)          # mini-map radius [m]: street / district / whole route (key M)

    def __init__(self, win_size: Tuple[int, int], rate_hz: float = 15.0, city=None) -> None:
        self.W, self.H = win_size
        self.rate = rate_hz
        self.visible = True
        self.toasts: Deque[Tuple[float, str, str]] = deque(maxlen=3)
        self._flash: Dict[str, float] = {}
        self._bgc: Dict[tuple, Image.Image] = {}
        self._mask: Dict[tuple, Image.Image] = {}
        self._t_panel: Dict[str, float] = {k: 0.0 for k in self.PANEL_HZ}
        self.mm = MiniMapData(city) if city is not None else None
        self.map_zoom = 0
        self._layout()

    def cycle_map_zoom(self) -> str:
        self.map_zoom = (self.map_zoom + 1) % len(self.MAP_RANGES)
        return f"Mini-map radius {self.MAP_RANGES[self.map_zoom]:.0f} m"

    def _layout(self) -> None:
        s = max(0.6, min(self.H / 900.0, self.W / 1600.0))
        self.s = s
        self.ss = 2 if s < 1.25 else 1          # high-density screens are already crisp: skip the super-sampling
        sz = lambda v: int(round(v * s))
        m = sz(22)
        W, H = self.W, self.H
        self._geom = {
            "top": (m, m, sz(430), sz(78)),
            "nav": (m, m + sz(84), sz(390), sz(206)),
            "rules": (W - m - sz(300), m, sz(300), sz(200)),
            "radar": (W - m - sz(290), H - m - sz(345), sz(290), sz(345)),
            "speed": (W // 2 - sz(210), H - m - sz(168), sz(420), sz(168)),
            "map": (m, H - m - sz(230), sz(230), sz(230)),
            "stats": (W - m - sz(290), H - m - sz(345) - sz(34), sz(290), sz(30)),
            "toast": (W // 2 - sz(260), m, sz(520), sz(120)),
        }
        self._bgc.clear()
        self._mask.clear()

    def set_window(self, w: int, h: int) -> None:
        self.W, self.H = w, h
        self._layout()

    def toast(self, text: str, kind: str = "warn") -> None:
        now = time.time()
        if any(t == text and now - ts < 2.5 for ts, t, _ in self.toasts):
            return
        self.toasts.append((now, text, kind))

    def flash(self, key: str) -> None:
        self._flash[key] = time.time()

    # ------------------------------------------------------------------ scheduling
    def render_all(self, t: Telemetry) -> Dict[str, np.ndarray]:
        return {k: self._draw(k, t) for k in self.PANEL_HZ}

    def _draw(self, key: str, t: Telemetry) -> np.ndarray:
        if key == "toast":
            return self._toasts()
        return {"top": self._top, "rules": self._rules, "radar": self._radar, "speed": self._speed,
                "map": self._minimap, "stats": self._stats, "nav": self._navcard}[key](t)

    def due_panels(self, t: Telemetry, now: Optional[float] = None) -> Dict[str, np.ndarray]:
        now = time.time() if now is None else now
        scale = self.rate / 15.0
        best, over = None, 1.0
        for k, hz in self.PANEL_HZ.items():
            ratio = (now - self._t_panel[k]) * hz * scale
            if ratio >= over:
                best, over = k, ratio
        if best is None:
            return {}
        self._t_panel[best] = now
        return {best: self._draw(best, t)}

    # ------------------------------------------------------------------ helpers
    def _canvas(self, key: tuple, w: int, h: int, static: Callable[[Canvas], None]) -> Canvas:
        img = self._bgc.get(key)
        if img is None:
            cv = Canvas(w, h, self.ss)
            static(cv)
            img = self._bgc[key] = cv.img
        return Canvas.from_image(img.copy(), w, h, self.ss)

    def _circle_mask(self, d: int) -> Image.Image:
        key = ("m", d)
        if key not in self._mask:
            m = Image.new("L", (d * 2, d * 2), 0)
            ImageDraw.Draw(m).ellipse([0, 0, d * 2 - 1, d * 2 - 1], fill=255)
            self._mask[key] = m
        return self._mask[key]

    @staticmethod
    def _glass(cv: Canvas, x, y, w, h, r) -> None:
        cv.rect(x, y, w, h, fill=C.GLASS, outline=C.STROKE, r=r)

    # ------------------------------------------------------------------ top-left: status + route
    def _top(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["top"]
        k = self.s
        col = {"ok": C.GREEN, "goal": C.ACCENT, "bad": C.RED}.get(t.status_kind, C.GREEN)

        def static(cv: Canvas) -> None:
            self._glass(cv, 0, 0, w, 46 * k, 23 * k)
            cv.text(18 * k, 56 * k, f"{t.map_name}  ·  {t.model_name}", int(11 * k), C.DIM, False)

        cv = self._canvas(("top", w, h, t.map_name, t.model_name), w, h, static)
        cv.circle(24 * k, 23 * k, 5 * k, fill=col)
        cv.text(40 * k, 23 * k, t.status, int(14 * k), C.TEXT, True, "lm")
        cv.text(w - 18 * k, 23 * k, f"{t.progress * 100:.0f}%", int(14 * k), C.DIM, True, "rm")
        cv.bar(18 * k, 42 * k, w - 36 * k, 2 * k, t.progress, col, back=(255, 255, 255, 30), r=k)
        return cv.finish()

    # ------------------------------------------------------------------ top-left, below: navigator card
    @staticmethod
    def _arrow(cv: Canvas, cx: float, cy: float, r: float, kind: str, col) -> None:
        """Manoeuvre pictogram (white on the blue tile, like a navigation app)."""
        wd = 0.30 * r
        if kind == "arrive":
            cv.line([(cx - 0.45 * r, cy + 0.9 * r), (cx - 0.45 * r, cy - 0.9 * r)], col, 0.16 * r)
            cv.poly([(cx - 0.45 * r, cy - 0.9 * r), (cx + 0.75 * r, cy - 0.55 * r), (cx - 0.45 * r, cy - 0.2 * r)], fill=col)
            return
        if kind in ("ramp_up", "ramp_down"):
            up = kind == "ramp_up"
            cv.line([(cx - 0.8 * r, cy + 0.6 * r), (cx - 0.1 * r, cy + 0.6 * r), (cx + 0.5 * r, cy - 0.2 * r if up else cy + 0.9 * r)], col, wd)
            cv.text(cx + 0.35 * r, cy - 0.65 * r if up else cy - 0.2 * r, "↑" if up else "↓", int(0.9 * r), col, True, "mm")
            return
        sgn = -1.0 if "left" in kind else 1.0
        if kind == "uturn":
            pts = [(cx + 0.45 * r, cy + 0.9 * r), (cx + 0.45 * r, cy - 0.35 * r), (cx + 0.25 * r, cy - 0.7 * r),
                   (cx - 0.25 * r, cy - 0.7 * r), (cx - 0.45 * r, cy - 0.35 * r), (cx - 0.45 * r, cy + 0.2 * r)]
            tip, d = (cx - 0.45 * r, cy + 0.65 * r), (0.0, 1.0)
        elif kind.startswith("slight"):
            pts = [(cx - 0.15 * sgn * r, cy + 0.9 * r), (cx - 0.15 * sgn * r, cy + 0.1 * r), (cx + 0.35 * sgn * r, cy - 0.45 * r)]
            tip, d = (cx + 0.6 * sgn * r, cy - 0.72 * r), (0.707 * sgn, -0.707)
        elif kind in ("left", "right"):
            pts = [(cx - 0.3 * sgn * r, cy + 0.9 * r), (cx - 0.3 * sgn * r, cy - 0.2 * r), (cx + 0.35 * sgn * r, cy - 0.2 * r)]
            tip, d = (cx + 0.85 * sgn * r, cy - 0.2 * r), (sgn, 0.0)
        else:
            pts = [(cx, cy + 0.9 * r), (cx, cy - 0.35 * r)]
            tip, d = (cx, cy - 0.9 * r), (0.0, -1.0)
        cv.line(pts, col, wd)
        nx_, ny_ = -d[1], d[0]
        bx, by = tip[0] - d[0] * 0.6 * r, tip[1] - d[1] * 0.6 * r
        cv.poly([tip, (bx + nx_ * 0.42 * r, by + ny_ * 0.42 * r), (bx - nx_ * 0.42 * r, by - ny_ * 0.42 * r)], fill=col)

    @staticmethod
    def _fit(cv: Canvas, s: str, size: int, width: float, bold: bool = False) -> str:
        if cv.text_w(s, size, bold) <= width:
            return s
        while s and cv.text_w(s + "…", size, bold) > width:
            s = s[:-1]
        return s + "…"

    def _navcard(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["nav"]
        k = self.s
        cv = Canvas(w, h, self.ss)
        n = t.nav
        if not n:
            return cv.finish()
        opts = n["options"] if n.get("nav_mode") else []
        hh = (112 + (22 + 22 * len(opts) if opts else 0)) * k
        self._glass(cv, 0, 0, w, min(h, hh), 20 * k)
        tile = 66 * k
        blue = (37, 120, 255, 240) if n["kind"] != "arrive" else (40, 190, 110, 240)
        cv.rect(12 * k, 12 * k, tile, tile, fill=blue, r=15 * k)
        self._arrow(cv, 12 * k + tile / 2, 12 * k + tile / 2, 23 * k, n["kind"], C.WHITE)
        d = n["dist"]
        dist = f"{d:.0f} m" if d < 950 else f"{d / 1000:.1f} km"
        cv.text(92 * k, 32 * k, dist, int(25 * k), C.TEXT, True, "lm")
        cv.text(92 * k, 62 * k, self._fit(cv, n["text"], int(12 * k), w - 104 * k), int(12 * k), C.DIM, False, "lm")
        y = 96 * k
        cv.text(16 * k, y, f"{n['remain_m'] / 1000:.1f} km  ·  {max(1, round(n['remain_s'] / 60))} min  ·  arrive {n['eta_clock']}",
                int(13 * k), C.ACCENT, True, "lm")
        if opts:
            y += 24 * k
            cv.text(16 * k, y, f"NAVIGATOR  ·  {len(opts)} ROUTE{'S' if len(opts) > 1 else ''} COMPARED", int(9 * k), C.DIM, True, "lm")
            for i, (km, mins, chosen, tag, via) in enumerate(opts):
                y += 22 * k
                col = (90, 160, 255, 255) if chosen else C.FAINT
                if chosen:
                    cv.rect(8 * k, y - 10 * k, w - 16 * k, 20 * k, fill=(37, 120, 255, 70), outline=(90, 160, 255, 140), r=10 * k)
                cv.circle(22 * k, y, 5 * k, fill=col)
                cv.text(34 * k, y, f"{km} · {mins}", int(12 * k), C.TEXT if chosen else C.DIM, chosen, "lm")
                tag_txt = tag.upper() if chosen else tag
                tag_w = cv.text_w(tag_txt, int(11 * k), True)
                cv.text(148 * k, y, self._fit(cv, "via " + via, int(10 * k), w - 172 * k - tag_w), int(10 * k),
                        (200, 212, 230, 255) if chosen else C.DIM, False, "lm")
                cv.text(w - 16 * k, y, tag_txt, int(11 * k), (120, 255, 170, 255) if chosen else C.DIM, True, "rm")
        return cv.finish()

    # ------------------------------------------------------------------ top-right: rules chip (expands on violations)
    def _rules(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["rules"]
        k = self.s
        cv = Canvas(w, h, self.ss)
        bad: List[Tuple[str, str]] = []
        total = 0
        for key, label in RULES:
            if key == "speeding":
                fr = t.ep_speeding_frac
                if fr > 0.05:
                    bad.append((label, f"{fr * 100:.0f}% of time"))
            elif key == "wrong_side":
                fr = t.ep_viol.get("wrong_side_frac", 0.0)
                if fr > 0.05:
                    bad.append((label, f"{fr * 100:.0f}% of time"))
            else:
                n = int(t.ep_viol.get(key, 0))
                if n:
                    bad.append((label, f"×{n}"))
                    total += n
        now = time.time()
        flashing = any(now - ts < 1.2 for ts in self._flash.values())
        col = C.GREEN if not bad else (C.RED if flashing else C.AMBER)
        rows = len(bad)
        hh = (40 + 24 * rows + (8 if rows else 0)) * k
        cv.rect(w - 230 * k, 0, 230 * k, hh, fill=(10, 12, 18, 150 if not flashing else 190), outline=(col[0], col[1], col[2], 90), r=20 * k)
        cv.circle(w - 208 * k, 20 * k, 5 * k, fill=col)
        cv.text(w - 194 * k, 20 * k, "Traffic rules", int(13 * k), C.TEXT, True, "lm")
        cv.text(w - 18 * k, 20 * k, "all clear" if not bad else f"{max(total, 1)} issue" + ("s" if total > 1 else ""),
                int(13 * k), col, True, "rm")
        for i, (label, val) in enumerate(bad):
            y = (46 + 24 * i) * k
            cv.text(w - 208 * k, y, label, int(12 * k), C.TEXT, False, "lm")
            cv.text(w - 18 * k, y, val, int(12 * k), C.AMBER, True, "rm")
        return cv.finish()

    # ------------------------------------------------------------------ bottom-right: perception + decision
    def _radar(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["radar"]
        k = self.s
        D = 196 * k
        cx, cy = w / 2, D / 2 + 6 * k
        R = D / 2
        sc = R / 45.0

        def static(cv: Canvas) -> None:
            self._glass(cv, 0, 0, w, h, 24 * k)
            cv.circle(cx, cy + 6 * k, R + 4 * k, fill=(0, 0, 0, 70))
            for rr in (15, 30, 45):
                cv.circle(cx, cy + 6 * k, rr * sc, outline=(255, 255, 255, 22))
            cv.text(18 * k, h - 18 * k, "", 1)

        cv = self._canvas(("radar", w, h), w, h, static)
        cy += 6 * k
        ex, ey, eth = t.ego
        fx, fy = math.cos(eth), math.sin(eth)
        rx, ry = math.sin(eth), -math.cos(eth)

        def to_px(px, py):
            dx, dy = px - ex, py - ey
            return cx + (dx * rx + dy * ry) * sc, cy - (dx * fx + dy * fy) * sc

        if t.route_ahead is not None and len(t.route_ahead) > 1:
            pts = [to_px(p[0], p[1]) for p in t.route_ahead]
            cv.line(pts, (110, 231, 255, 70), 5 * k)
            cv.line(pts, C.ACCENT, 1.6 * k)
        if t.lidar is not None:
            rng = float(np.max(t.lidar)) if len(t.lidar) else 50.0
            ped = getattr(t.perception, "lidar_ped", None)
            veh = getattr(t.perception, "lidar_veh", None)
            for i, (d, a) in enumerate(zip(t.lidar, t.lidar_angles)):
                if d >= rng - 0.5:
                    continue
                px, py = to_px(ex + math.cos(a) * d, ey + math.sin(a) * d)
                col = C.RED if (ped is not None and ped[i] > 0.5) else (C.AMBER if (veh is not None and veh[i] > 0.5) else (150, 170, 200, 230))
                cv.circle(px, py, 1.8 * k, fill=col)
        if t.circles is not None:
            for (x, y, r), kind in zip(t.circles, t.circle_kind):
                px, py = to_px(x, y)
                if (px - cx) ** 2 + (py - cy) ** 2 > (R - 2) ** 2:
                    continue
                col = {4: C.RED, 3: C.AMBER, 2: C.AMBER, 1: (255, 214, 90, 255)}.get(int(kind), C.DIM)
                cv.circle(px, py, max(2.2 * k, r * sc), outline=col, width=1.3)
        cv.poly([(cx, cy - 7 * k), (cx + 5 * k, cy + 6 * k), (cx - 5 * k, cy + 6 * k)], fill=C.WHITE)
        y = cy + R + 20 * k
        col = {"ok": C.GREEN, "info": C.ACCENT, "warn": C.AMBER}.get(t.reason_kind, C.ACCENT)
        cv.text(w / 2, y, t.reason, int(13 * k), col, True, "mm")
        y += 20 * k
        for label, val, kind in t.constraints[:3]:
            cc = {"ok": C.TEXT, "warn": C.AMBER, "bad": C.RED}.get(kind, C.TEXT)
            cv.text(20 * k, y, label, int(11 * k), C.DIM, False, "lm")
            cv.text(w - 20 * k, y, val, int(11 * k), cc, True, "rm")
            y += 16 * k
        return cv.finish()

    # ------------------------------------------------------------------ bottom-centre: the driving cluster
    def _speed(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["speed"]
        k = self.s
        cx, cy, R = w / 2, 84 * k, 62 * k
        a0, a1 = 135, 405

        def static(cv: Canvas) -> None:
            cv.circle(cx, cy, R + 16 * k, fill=(8, 10, 16, 150))
            cv.arc(cx, cy, R, a0, a1, (255, 255, 255, 34), 5 * k)
            cv.text(cx, cy + 26 * k, "km/h", int(11 * k), C.DIM, False, "mm")

        cv = self._canvas(("speed", w, h), w, h, static)
        vmax = 100.0
        frac = min(1.0, t.speed_kmh / vmax)
        col = C.speed_color(t.speed_kmh / max(t.limit_kmh, 1.0))
        if frac > 0.004:
            cv.arc(cx, cy, R, a0, a0 + (a1 - a0) * frac, col, 5 * k)
        for val, c2, ln in ((t.limit_kmh, C.WHITE, 10), (t.target_kmh, C.ACCENT, 6)):
            if 0 < val <= vmax:
                ang = math.radians(a0 + (a1 - a0) * val / vmax)
                cv.line([(cx + (R - ln * k) * math.cos(ang), cy + (R - ln * k) * math.sin(ang)),
                         (cx + (R + 5 * k) * math.cos(ang), cy + (R + 5 * k) * math.sin(ang))], c2, 2 * k)
        cv.text(cx, cy - 6 * k, f"{t.speed_kmh:.0f}", int(50 * k), C.TEXT, True, "mm")
        gc = {"D": C.DIM, "R": C.AMBER, "N": C.DIM}.get(t.gear, C.DIM)
        cv.text(cx, cy + 42 * k, {"D": "", "R": "REVERSE", "N": "P"}.get(t.gear, ""), int(11 * k), gc, True, "mm")
        C.roundel(cv, cx + 112 * k, cy - 4 * k, 24 * k, f"{t.limit_kmh:.0f}")
        # pedals: two thin vertical bars on the left
        gas, brk = max(t.pedal, 0.0), max(-t.pedal, 0.0)
        bh = 64 * k
        for x, v, c in ((cx - 128 * k, gas, C.GREEN), (cx - 112 * k, brk, C.RED)):
            cv.rect(x, cy - bh / 2, 5 * k, bh, fill=(255, 255, 255, 28), r=2.5 * k)
            if v > 0.02:
                cv.rect(x, cy + bh / 2 - bh * v, 5 * k, max(bh * v, 5 * k), fill=c, r=2.5 * k)
        cv.center_bar(cx - 40 * k, h - 8 * k, 80 * k, 3 * k, t.steer_cmd, (255, 255, 255, 200))
        # only when relevant
        if t.aeb:
            cv.rect(cx - 62 * k, 0, 124 * k, 26 * k, fill=(255, 82, 92, 235), r=13 * k)
            cv.text(cx, 13 * k, "EMERGENCY BRAKE", int(11 * k), C.WHITE, True, "mm")
        elif (np.isfinite(t.ttc) and t.ttc < 8) or (np.isfinite(t.gap) and t.gap < 25):
            s = f"gap {t.gap:.0f} m" + (f"  ·  TTC {t.ttc:.1f} s" if np.isfinite(t.ttc) and t.ttc < 8 else "")
            cv.text(cx, 10 * k, s, int(12 * k), C.AMBER, True, "mm")
        if t.light is not None and t.light[1] < 90:
            st, d, _ = t.light
            lc = {L_GREEN: C.GREEN, L_YELLOW: C.AMBER, L_RED: C.RED}[st]
            cv.circle(cx + 112 * k, cy + 38 * k, 6 * k, fill=lc)
            cv.text(cx + 112 * k, cy + 56 * k, f"{d:.0f} m", int(10 * k), C.DIM, True, "mm")
        return cv.finish()

    # ------------------------------------------------------------------ session line
    def _stats(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["stats"]
        k = self.s
        cv = Canvas(w, h, self.ss)
        s = t.session
        txt = f"run {t.episode}  ·  {s.get('success', 0) * 100:.0f}% of {int(s.get('n', 0))}  ·  " \
              f"{int(s.get('collisions', 0))} collisions  ·  {s.get('km', 0):.1f} km"
        tw = cv.text_w(txt, int(11 * k), False) + 24 * k
        cv.rect(w - tw, 0, tw, h, fill=(10, 12, 18, 130), r=h / 2)
        cv.text(w - 12 * k, h / 2, txt, int(11 * k), (200, 210, 224, 255), False, "rm")
        return cv.finish()

    # ------------------------------------------------------------------ bottom-left: circular vector mini-map (heading up)
    def _minimap(self, t: Telemetry) -> np.ndarray:
        _, _, w, h = self._geom["map"]
        k = self.s
        cv = Canvas(w, h, self.ss)
        D = w - 8 * k
        cx = cy = w / 2
        R = D / 2
        view_r = self.MAP_RANGES[self.map_zoom]          # metres from the centre to the rim
        sc = R / view_r
        cv.circle(cx, cy, R, fill=(11, 14, 22, 235))
        if self.mm is not None:
            m = self.mm
            ex, ey, eth = t.ego
            fx, fy = math.cos(eth), math.sin(eth)
            rx, ry = math.sin(eth), -math.cos(eth)
            rr = view_r * 1.45

            def tf(p: np.ndarray) -> list:
                d = p - np.array([ex, ey], dtype=np.float32)
                sx = cx + (d[:, 0] * rx + d[:, 1] * ry) * sc
                sy = cy - (d[:, 0] * fx + d[:, 1] * fy) * sc
                return list(zip(sx.tolist(), sy.tolist()))

            for i in m._vis(m.g_bbox, ex, ey, rr):
                cv.poly(tf(m.g_rings[i]), fill=(30, 62, 56, 255))
            for i in m._vis(m.b_bbox, ex, ey, rr):
                cv.poly(tf(m.b_rings[i]), fill=(34, 42, 60, 255))
            vis = m._vis(m.bbox, ex, ey, rr)
            for lvl, col, extra in ((0, (86, 98, 122, 255), 0.0), (1, (80, 140, 220, 255), 0.0), (2, (255, 176, 80, 255), 1.5)):
                for i in vis:
                    if m.lv[i] == lvl:
                        wpx = max(1.2 * k, m.wid[i] * sc + extra * k)
                        cv.line(tf(m.pts[i]), col, wpx)
            for alt in t.alts_xy or []:                  # routes the navigator compared but did not take
                if len(alt) > 1:
                    cv.line(tf(alt), (150, 160, 182, 150), 2.4 * k)
            i0 = int(t.nav["route_i0"]) if t.nav else 0
            if t.route_xy is not None and len(t.route_xy) - max(0, i0 - 1) > 1:
                rest = t.route_xy[max(0, i0 - 1):]
                cv.line(tf(rest), (37, 120, 255, 255), 4.2 * k)          # whole remaining route, navigator blue
                gx, gy = tf(rest[-1:])[0]
                if (gx - cx) ** 2 + (gy - cy) ** 2 < (R - 6 * k) ** 2:
                    cv.circle(gx, gy, 6 * k, fill=(255, 82, 92, 255), outline=C.WHITE, width=1.5 * k)
            if t.route_ahead is not None and len(t.route_ahead) > 1:
                cv.line(tf(np.asarray(t.route_ahead, dtype=np.float32)), C.ACCENT, 2.6 * k)
            if t.circles is not None and len(t.circles):
                pts = tf(np.asarray(t.circles[:, :2], dtype=np.float32))
                for (px, py), kind in zip(pts, t.circle_kind):
                    if (px - cx) ** 2 + (py - cy) ** 2 < (R - 3) ** 2:
                        cv.circle(px, py, 2.2 * k, fill={4: C.RED, 3: C.AMBER}.get(int(kind), (255, 255, 255, 190)))
        cv.poly([(cx, cy - 8 * k), (cx + 5.5 * k, cy + 7 * k), (cx, cy + 4 * k), (cx - 5.5 * k, cy + 7 * k)], fill=C.WHITE)
        # round the corners: multiply alpha with a circular mask, then draw the rim
        mask = self._circle_mask(int(w))
        cv.img.putalpha(ImageChops.multiply(cv.img.getchannel("A"), mask.resize(cv.img.size)))
        cv.circle(cx, cy, R, outline=(255, 255, 255, 40), width=1.2 * k)
        # north marker on the rim (the map is heading-up, so it rotates)
        eth = t.ego[2]
        ang = math.pi / 2 - eth                          # screen angle of north relative to "up"
        nx, ny = cx + math.sin(-ang) * (R - 9 * k), cy - math.cos(-ang) * (R - 9 * k)
        cv.text(nx, ny, "N", int(10 * k), C.ACCENT, True, "mm")
        return cv.finish()

    # ------------------------------------------------------------------ toasts (top centre)
    def _toasts(self) -> np.ndarray:
        _, _, w, h = self._geom["toast"]
        cv = Canvas(w, h, self.ss)
        k = self.s
        now = time.time()
        y = 0.0
        for ts, text, kind in list(self.toasts)[::-1]:
            age = now - ts
            if age > 3.5:
                continue
            a = int(255 * min(1.0, (3.5 - age) / 0.7))
            col = {"bad": C.RED, "warn": C.AMBER, "ok": C.GREEN, "info": C.ACCENT}.get(kind, C.AMBER)
            tw = cv.text_w(text, int(13 * k)) + 44 * k
            x0 = w / 2 - tw / 2
            cv.rect(x0, y, tw, 32 * k, fill=(10, 12, 18, int(a * 0.62)), outline=(col[0], col[1], col[2], int(a * 0.55)), r=16 * k)
            cv.circle(x0 + 17 * k, y + 16 * k, 4 * k, fill=(col[0], col[1], col[2], a))
            cv.text(x0 + 30 * k, y + 16 * k, text, int(13 * k), (232, 238, 246, a), True, "lm")
            y += 38 * k
        return cv.finish()


# ====================================================================================== Ursina cards
class _Card:
    """A screen-space textured rectangle (pixel coordinates, origin top-left) in Ursina's ``camera.ui``
    (UI units: the window height is 1.0)."""

    def __init__(self, name: str, x: float, y: float, w: int, h: int, parent, tex_size: Tuple[int, int]) -> None:
        from panda3d.core import Texture, TransparencyAttrib
        from ursina import Entity
        from ursina import Texture as UTexture
        self.x, self.y, self.w, self.h = x, y, w, h
        self.utex = UTexture(Image.new("RGBA", tex_size, (0, 0, 0, 0)))
        self.tex = self.utex._texture
        self.tex.set_minfilter(Texture.FT_linear)
        self.tex.set_magfilter(Texture.FT_linear)
        self.ent = Entity(parent=parent, model="quad", texture=self.utex, origin=(-0.5, 0.5), z=-0.1)
        self.ent.set_transparency(TransparencyAttrib.M_alpha)

    def place(self, win_w: int, win_h: int) -> None:
        k = 1.0 / win_h
        self.ent.position = ((self.x - win_w / 2) * k, (win_h / 2 - self.y) * k, -0.1)
        self.ent.scale = (self.w * k, self.h * k)

    def set(self, arr: np.ndarray) -> None:
        from .gfx import update_texture
        update_texture(self.tex, arr)


class Hud(HudPanels):
    """HUD of the 3-D viewer: the panels above shown as cards in Ursina's ``camera.ui``."""

    def __init__(self, win_size: Tuple[int, int], rate_hz: float = 15.0, city=None) -> None:
        super().__init__(win_size, rate_hz, city)
        from ursina import Entity, camera
        self.root = Entity(parent=camera.ui, name="hud")
        self.cards: Dict[str, _Card] = {}
        self._build_cards()

    def _build_cards(self) -> None:
        from ursina import destroy
        for c in self.cards.values():
            destroy(c.ent)
        self.cards = {}
        for k, (x, y, w, h) in self._geom.items():
            c = _Card(f"hud_{k}", x, y, w, h, self.root, (w, h))
            c.place(self.W, self.H)
            self.cards[k] = c

    def set_window(self, w: int, h: int) -> None:
        """New window size: the panels change size too, so their textures are recreated (and redrawn on next update)."""
        super().set_window(w, h)
        self._build_cards()
        self._t_panel = {k: 0.0 for k in self.PANEL_HZ}

    def toggle(self) -> None:
        self.visible = not self.visible
        self.root.enabled = self.visible

    def update(self, t: Telemetry, force: bool = False) -> None:
        panels = self.render_all(t) if force else self.due_panels(t)
        for k, arr in panels.items():
            self.cards[k].set(arr)
