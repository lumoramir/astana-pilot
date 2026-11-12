"""In-game navigator map (key M): click the start A and the destination B on the real city map, compare the routes the
navigator proposes, pick one, press "Drive". While driving, open it again and click a new destination - the route is
re-planned from where the car is now.

Engine-free (Pillow only): :class:`MapPanel` renders one RGBA image and answers mouse events in panel pixels, so the
3-D viewer only has to show the image and forward clicks. Interaction:

    left click            set A, then B (later clicks move B)          right click   move A (before departure)
    left drag             move the map                                 wheel         zoom at the cursor
    click a route card    choose that route                            ENTER         drive / go this way
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..env import navigator as nav
from . import canvas as C
from .canvas import Canvas
from .minimap import GROUND, render_minimap

BLUE, ALT, A_COL, B_COL = (37, 120, 255, 255), (150, 160, 182, 255), (47, 214, 122, 255), (255, 79, 90, 255)


class MapPanel:
    def __init__(self, city, cfg, win: Tuple[int, int], alternatives: int = 3) -> None:
        self.city, self.cfg, self.alternatives = city, cfg, alternatives
        x0, y0, x1, y1 = city.bounds
        mpp = max(0.6, max(x1 - x0, y1 - y0) / 2600.0)
        img, info = render_minimap(city, mpp)
        self.base = Image.fromarray(img).convert("RGB")
        self.bx0, self.by0, self.mpp = info["x0"], info["y0"], info["res"]
        self.bh = self.base.size[1]
        self.extent = (x0, y0, x1, y1)
        self.mode = "setup"                   # "setup": choose A and B | "drive": the car is A, choose a new B
        self.A: Optional[int] = None
        self.B: Optional[int] = None
        self.plan = None
        self.sel = 0
        self.msg = ""
        self.car: Optional[Tuple[float, float, float]] = None
        self.cur_route: Optional[np.ndarray] = None
        self.drive_ctx = None                 # (route_nodes, ref, s) for planning from the car
        self.hits: List[Tuple[Tuple[float, float, float, float], str]] = []
        self.layout(win)
        self.fit()

    # ------------------------------------------------------------------ geometry
    def layout(self, win: Tuple[int, int]) -> None:
        W, H = win
        self.W, self.H = W, H
        s = max(0.6, min(H / 900.0, W / 1600.0))
        self.k = s
        m = int(H * 0.045)
        side = int(370 * s)
        mh = H - 2 * m
        mw = int(min(W - 2 * m - side, mh * 1.35))
        self.box = ((W - mw - side) // 2, m, mw + side, mh)          # whole card in window pixels
        self.mw, self.mh, self.side = mw, mh, side

    def fit(self) -> None:
        x0, y0, x1, y1 = self.extent
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.ppm = 0.96 * min(self.mw / (x1 - x0), self.mh / (y1 - y0))
        self.ppm_min, self.ppm_max = self.ppm * 0.7, self.ppm * 9.0

    def to_px(self, X, Y):
        return (self.mw / 2 + (np.asarray(X) - self.cx) * self.ppm, self.mh / 2 - (np.asarray(Y) - self.cy) * self.ppm)

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        return self.cx + (px - self.mw / 2) / self.ppm, self.cy - (py - self.mh / 2) / self.ppm

    def local(self, wx: float, wy: float) -> Tuple[float, float]:
        """Window pixels -> panel pixels."""
        return wx - self.box[0], wy - self.box[1]

    def in_map(self, px: float, py: float) -> bool:
        return 0 <= px < self.mw and 0 <= py < self.mh

    # ------------------------------------------------------------------ interaction
    def zoom(self, px: float, py: float, factor: float) -> None:
        if not self.in_map(px, py):
            return
        wx, wy = self.to_world(px, py)
        self.ppm = float(np.clip(self.ppm * factor, self.ppm_min, self.ppm_max))
        nx_, ny_ = self.to_world(px, py)
        self.cx += wx - nx_
        self.cy += wy - ny_

    def pan(self, dx: float, dy: float) -> None:
        self.cx -= dx / self.ppm
        self.cy += dy / self.ppm

    def hit(self, px: float, py: float) -> Optional[str]:
        for (x, y, w, h), act in self.hits:
            if x <= px <= x + w and y <= py <= y + h:
                return act
        return None

    def click(self, px: float, py: float, button: str = "left") -> Optional[str]:
        """Returns an action for the viewer ("go", "close") or None. Map clicks set A / B and re-plan."""
        act = self.hit(px, py)
        if act is not None:
            if act.startswith("route:"):
                self.sel = int(act.split(":")[1])
                return None
            if act == "random":
                self.random()
                return None
            if act == "swap" and self.mode == "setup":
                self.A, self.B = self.B, self.A
                self.replan()
                return None
            if act == "clear":
                self.B = None if self.mode == "drive" else self.B
                if self.mode == "setup":
                    self.A = self.B = None
                self.plan, self.msg = None, ""
                return None
            if act == "go" and self.plan is None:
                return None
            return act
        if not self.in_map(px, py):
            return None
        X, Y = self.to_world(px, py)
        if self.mode == "drive":
            self.B = nav.nearest_node(self.city, X, Y, "goal")
        elif button == "right" or self.A is None:
            self.A = nav.nearest_node(self.city, X, Y, "start")
        else:
            self.B = nav.nearest_node(self.city, X, Y, "goal")
        self.replan()
        return None

    def random(self) -> None:
        rng = np.random.default_rng()
        if self.mode == "drive":
            cand = nav._candidates(self.city, "goal")
            for _ in range(20):
                self.B = int(cand[int(rng.integers(len(cand)))])
                if self.replan():
                    return
            return
        p = nav.random_plan(self.city, self.cfg, rng, 600, 1800, self.alternatives)
        if p is not None:
            self.A, self.B, self.plan, self.sel, self.msg = p.start, p.goal, p, 0, ""

    def replan(self) -> bool:
        self.plan, self.sel, self.msg = None, 0, ""
        if self.mode == "drive" and self.B is not None and self.drive_ctx is not None:
            nodes, ref, s = self.drive_ctx
            self.plan = nav.plan_from_car(self.city, self.cfg, nodes, ref, s, self.B, self.alternatives)
        elif self.A is not None and self.B is not None:
            self.plan = nav.plan_route(self.city, self.cfg, self.A, self.B, self.alternatives)
        else:
            return False
        if self.plan is None:
            self.msg = "No drivable route to this point - try a spot on a bigger street."
            return False
        return True

    def set_drive(self, car, route_xy, nodes, ref, s) -> None:
        self.mode = "drive"
        self.car, self.cur_route, self.drive_ctx = car, route_xy, (list(nodes), ref, s)
        self.A, self.B, self.plan, self.msg = None, None, None, ""

    # ------------------------------------------------------------------ drawing
    def render(self) -> np.ndarray:
        mw, mh, side, k = self.mw, self.mh, self.side, self.k
        X0, Y1 = self.to_world(0, 0)
        X1, Y0 = self.to_world(mw, mh)
        box = ((X0 - self.bx0) / self.mpp, self.bh - (Y1 - self.by0) / self.mpp,
               (X1 - self.bx0) / self.mpp, self.bh - (Y0 - self.by0) / self.mpp)
        mp = self.base.transform((mw, mh), Image.EXTENT, box, Image.BILINEAR, fillcolor=GROUND[:3]).convert("RGBA")
        d = ImageDraw.Draw(mp, "RGBA")
        lw = max(3, int(5 * k * min(2.0, max(0.8, self.ppm / (self.ppm_min / 0.7)) ** 0.35)))

        def line(xy, col, w):
            if xy is None or len(xy) < 2:
                return
            sx, sy = self.to_px(xy[:, 0], xy[:, 1])
            d.line(list(zip(sx.tolist(), sy.tolist())), fill=col, width=int(w), joint="curve")

        for lm in getattr(self.city.raw, "landmarks", []) or []:
            sx, sy = self.to_px(*lm["xy"])
            d.text((float(sx) + 8, float(sy) - 8), str(lm.get("name", "")), fill=(251, 191, 36, 255), font=C.get_font(int(13 * k), True))
            d.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(251, 191, 36, 255))
        if self.plan is None and self.cur_route is not None:
            line(self.cur_route, (255, 255, 255, 200), lw + 4)
            line(self.cur_route, BLUE, lw)
        if self.plan is not None:
            for i, o in enumerate(self.plan.options):
                if i != self.sel:
                    line(o.ref.xy[::2], (40, 46, 60, 255), lw + 3)
                    line(o.ref.xy[::2], ALT, lw)
            o = self.plan.options[self.sel]
            line(o.ref.xy[::2], (255, 255, 255, 230), lw + 5)
            line(o.ref.xy[::2], BLUE, lw + 1)

        def pin(xy, col, label):
            sx, sy = (float(v) for v in self.to_px(*xy))
            r = 13 * k
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col, outline=(255, 255, 255, 255), width=max(2, int(2.5 * k)))
            f = C.get_font(int(14 * k), True)
            d.text((sx, sy), label, fill=(255, 255, 255, 255), font=f, anchor="mm")

        if self.mode == "setup":
            if self.A is not None:
                pin(nav.node_xy(self.city, self.A) if self.plan is None else self.plan.options[self.sel].ref.xy[0], A_COL, "A")
        if self.B is not None:
            pin(nav.node_xy(self.city, self.B) if self.plan is None else self.plan.options[self.sel].ref.xy[-1], B_COL, "B")
        if self.car is not None and self.mode == "drive":
            x, y, th = self.car
            sx, sy = (float(v) for v in self.to_px(x, y))
            r = 15 * k
            c, s_ = math.cos(th), -math.sin(th)                          # screen y points down
            pts = [(sx + c * r, sy + s_ * r), (sx - c * r * 0.7 - s_ * r * 0.65, sy - s_ * r * 0.7 + c * r * 0.65),
                   (sx - c * r * 0.35, sy - s_ * r * 0.35), (sx - c * r * 0.7 + s_ * r * 0.65, sy - s_ * r * 0.7 - c * r * 0.65)]
            d.ellipse([sx - r * 1.3, sy - r * 1.3, sx + r * 1.3, sy + r * 1.3], fill=(37, 120, 255, 70))
            d.polygon(pts, fill=(255, 255, 255, 255), outline=(37, 120, 255, 255))
        # scale bar
        m = 100.0 if self.ppm * 100 > 60 else 500.0
        bx, by = 18 * k, mh - 22 * k
        d.line([(bx, by), (bx + m * self.ppm, by)], fill=(232, 238, 246, 220), width=max(2, int(3 * k)))
        d.text((bx, by - 8 * k), f"{m:.0f} m", fill=(232, 238, 246, 220), font=C.get_font(int(11 * k), True), anchor="lb")
        if self.msg:
            f = C.get_font(int(14 * k), True)
            tw = d.textlength(self.msg, font=f)
            d.rounded_rectangle([mw / 2 - tw / 2 - 16 * k, 16 * k, mw / 2 + tw / 2 + 16 * k, 50 * k], 16 * k, fill=(120, 20, 30, 220))
            d.text((mw / 2, 33 * k), self.msg, fill=(255, 255, 255, 255), font=f, anchor="mm")

        out = Image.new("RGBA", (mw + side, mh), (0, 0, 0, 0))
        mask = Image.new("L", (mw, mh), 0)
        ImageDraw.Draw(mask).rounded_rectangle([0, 0, mw - 1, mh - 1], int(20 * k), fill=255)
        mp.putalpha(mask)
        out.paste(mp, (0, 0), mp)
        sp = Image.fromarray(self._side())
        out.paste(sp, (mw, 0), sp)
        return np.asarray(out)

    def _btn(self, cv: Canvas, x, y, w, h, text, act, primary=False, enabled=True) -> None:
        k = self.k
        fill = (37, 120, 255, 255) if primary and enabled else ((30, 34, 44, 255) if enabled else (18, 21, 28, 255))
        cv.rect(x, y, w, h, fill=fill, outline=(255, 255, 255, 40) if not primary else None, r=h / 2)
        cv.text(x + w / 2, y + h / 2, text, int((15 if primary else 12) * k), C.WHITE if enabled else C.FAINT, True, "mm")
        if enabled:
            self.hits.append(((self.mw + x, y, w, h), act))

    def _side(self) -> np.ndarray:
        w, h, k = self.side, self.mh, self.k
        cv = Canvas(w, h, 2 if k < 1.25 else 1)
        cv.rect(10 * k, 0, w - 10 * k, h, fill=(10, 13, 20, 255), outline=(255, 255, 255, 26), r=20 * k)
        self.hits = []
        x0, x1 = 30 * k, w - 20 * k
        y = 34 * k
        cv.text(x0, y, "Navigator", int(22 * k), C.TEXT, True, "lm")
        y += 26 * k
        sub = "Your car is point A. Click the map to choose a new destination." if self.mode == "drive" else \
            "Choose where to start and where to go."
        for line in _wrap(cv, sub, int(12 * k), x1 - x0):
            cv.text(x0, y, line, int(12 * k), C.DIM, False, "lm")
            y += 17 * k
        y += 10 * k
        if self.mode == "setup":
            steps = (("A", "Start: click the map", self.A is not None, A_COL), ("B", "Destination: click again", self.B is not None, B_COL))
        else:
            steps = (("B", "Destination: click the map", self.B is not None, B_COL),)
        for lab, txt, done, col in steps:
            cv.circle(x0 + 12 * k, y + 14 * k, 12 * k, fill=col if done else (40, 44, 54, 255))
            cv.text(x0 + 12 * k, y + 14 * k, lab, int(12 * k), C.WHITE, True, "mm")
            name = ""
            if done and self.plan is not None:
                name = self.plan.start_name if lab == "A" else self.plan.goal_name
            cv.text(x0 + 34 * k, y + 7 * k, txt if not done else ("Start" if lab == "A" else "Destination"), int(12 * k),
                    C.TEXT if done else C.DIM, done, "lm")
            if name:
                cv.text(x0 + 34 * k, y + 24 * k, _fit(cv, name, int(11 * k), x1 - x0 - 34 * k), int(11 * k), C.DIM, False, "lm")
            y += 40 * k
        y += 6 * k
        if self.plan is not None:
            cv.text(x0, y, f"{len(self.plan.options)} ROUTE{'S' if len(self.plan.options) > 1 else ''} - CLICK ONE TO CHOOSE",
                    int(10 * k), C.DIM, True, "lm")
            y += 16 * k
            best = min(o.eta_s for o in self.plan.options)
            others = [set(o.streets) for o in self.plan.options]
            for i, o in enumerate(self.plan.options):
                sel = i == self.sel
                hh = 66 * k
                cv.rect(x0 - 8 * k, y, x1 - x0 + 8 * k, hh, fill=(18, 44, 88, 255) if sel else (20, 23, 31, 255),
                        outline=(90, 160, 255, 200) if sel else (255, 255, 255, 22), r=14 * k)
                self.hits.append(((self.mw + x0 - 8 * k, y, x1 - x0 + 8 * k, hh), f"route:{i}"))
                cv.text(x0 + 4 * k, y + 18 * k, f"{max(1, round(o.eta_s / 60))} min", int(19 * k), C.TEXT, True, "lm")
                cv.text(x0 + 80 * k, y + 18 * k, f"{o.length / 1000:.1f} km", int(13 * k), C.DIM, True, "lm")
                tag = "fastest" if abs(o.eta_s - best) < 1 else f"+{max(1, round((o.eta_s - best) / 60))} min"
                cv.text(x1 - 6 * k, y + 18 * k, tag, int(11 * k), (120, 255, 170, 255) if tag == "fastest" else C.DIM, True, "rm")
                rest = set().union(*[s for j, s in enumerate(others) if j != i]) if len(others) > 1 else set()
                via = next((st for st in o.streets if st not in rest), o.via(1))
                cv.text(x0 + 4 * k, y + 39 * k, _fit(cv, "via " + via, int(11 * k), x1 - x0 - 10 * k), int(11 * k),
                        (210, 220, 236, 255) if sel else C.DIM, False, "lm")
                cv.text(x0 + 4 * k, y + 55 * k, f"{o.lights} traffic lights  ·  {o.crossings} crossings  ·  {len(o.maneuvers) - 1} turns",
                        int(10 * k), C.DIM, False, "lm")
                y += hh + 8 * k
        # buttons
        by = h - 150 * k
        bw = x1 - x0 + 8 * k
        go = "Go this way" if self.mode == "drive" else "Drive"
        self._btn(cv, x0 - 8 * k, by, bw, 46 * k, go, "go", primary=True, enabled=self.plan is not None)
        half = (bw - 8 * k) / 2
        self._btn(cv, x0 - 8 * k, by + 56 * k, half, 34 * k, "Random", "random")
        if self.mode == "setup":
            self._btn(cv, x0 + half, by + 56 * k, half, 34 * k, "Swap A / B", "swap", enabled=self.A is not None and self.B is not None)
        else:
            self._btn(cv, x0 + half, by + 56 * k, half, 34 * k, "Close  (M)", "close")
        cv.text(x0, h - 30 * k, "drag = move  ·  wheel = zoom  ·  right click = move A", int(10 * k), C.FAINT, False, "lm")
        return cv.finish()


def _fit(cv: Canvas, s: str, size: int, width: float) -> str:
    if cv.text_w(s, size, False) <= width:
        return s
    while s and cv.text_w(s + "…", size, False) > width:
        s = s[:-1]
    return s + "…"


def _wrap(cv: Canvas, s: str, size: int, width: float) -> List[str]:
    out, cur = [], ""
    for wd in s.split():
        t = (cur + " " + wd).strip()
        if cv.text_w(t, size, False) <= width:
            cur = t
        else:
            out.append(cur)
            cur = wd
    return out + ([cur] if cur else [])
