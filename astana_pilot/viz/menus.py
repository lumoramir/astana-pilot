"""Clickable menus of the 3-D viewer (Pillow-drawn, engine-free): the pause menu, the trip-finished screen and the
small on-screen toolbar (Menu / Map / Pause). Each ``render`` returns an RGBA array and fills ``hits`` with
``((x, y, w, h), action)`` rectangles in panel pixels; the viewer forwards mouse clicks to :meth:`hit`."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import canvas as C
from .canvas import Canvas

Item = Tuple[str, str, str, str]          # (label, right-hand value / key hint, action, style: primary|normal|danger|section)

BG = (12, 15, 22, 255)
BTN = (30, 34, 44, 255)
BTN_HOVER = (44, 50, 64, 255)
BLUE = (37, 120, 255, 255)
RED_BG = (90, 26, 34, 255)


class Panel:
    def __init__(self) -> None:
        self.hits: List[Tuple[Tuple[float, float, float, float], str]] = []
        self.hover: Optional[str] = None

    def hit(self, px: float, py: float) -> Optional[str]:
        for (x, y, w, h), act in self.hits:
            if x <= px <= x + w and y <= py <= y + h:
                return act
        return None


class MenuPanel(Panel):
    """Centered modal card: title, subtitle lines, a column of buttons."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k
        self.w = int(440 * k)

    def height(self, items: Sequence[Item], lines: Sequence[Tuple[str, tuple]]) -> int:
        k = self.k
        h = 92 * k + 22 * k * len(lines) + 14 * k
        for it in items:
            h += (26 if it[3] == "section" else (58 if it[3] == "primary" else 46)) * k
        return int(h + 26 * k)

    def render(self, title: str, lines: Sequence[Tuple[str, tuple]], items: Sequence[Item], accent=BLUE) -> np.ndarray:
        k, w = self.k, self.w
        h = self.height(items, lines)
        cv = Canvas(w, h, 2 if k < 1.25 else 1)
        cv.rect(0, 0, w, h, fill=BG, outline=(255, 255, 255, 34), r=24 * k)
        cv.rect(0, 0, w, 6 * k, fill=accent, r=3 * k)
        self.hits = []
        x0, x1 = 28 * k, w - 28 * k
        y = 46 * k
        cv.text(x0, y, title, int(24 * k), C.TEXT, True, "lm")
        y += 34 * k
        for txt, col in lines:
            cv.text(x0, y, txt, int(13 * k), col, False, "lm")
            y += 22 * k
        y += 14 * k
        for label, hint, act, style in items:
            if style == "section":
                cv.text(x0, y + 12 * k, label.upper(), int(10 * k), C.DIM, True, "lm")
                y += 26 * k
                continue
            bh = (48 if style == "primary" else 38) * k
            fill = BLUE if style == "primary" else (RED_BG if style == "danger" else (BTN_HOVER if self.hover == act else BTN))
            cv.rect(x0 - 6 * k, y, x1 - x0 + 12 * k, bh, fill=fill, r=bh / 2)
            size = int((15 if style == "primary" else 13) * k)
            cv.text(x0 + 14 * k, y + bh / 2, label, size, C.WHITE, True, "lm")
            if hint:
                cv.text(x1 - 8 * k, y + bh / 2, hint, int(11 * k), (200, 214, 236, 255) if style == "primary" else C.DIM, True, "rm")
            self.hits.append(((x0 - 6 * k, y, x1 - x0 + 12 * k, bh), act))
            y += bh + 8 * k
        return cv.finish()


class Toolbar(Panel):
    """Always-visible row of buttons while driving (the user should never need to know a hotkey)."""

    ITEMS = (("Menu", "Esc", "menu"), ("Map / route", "M", "map"), ("Pause", "Space", "pause"))

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = k
        self.bw = [int(v * k) for v in (104, 142, 112)]
        self.w = sum(self.bw) + int(8 * k) * (len(self.bw) - 1)
        self.h = int(40 * k)

    def render(self, paused: bool) -> np.ndarray:
        k = self.k
        cv = Canvas(self.w, self.h, 2 if k < 1.25 else 1)
        self.hits = []
        x = 0.0
        for (label, key, act), bw in zip(self.ITEMS, self.bw):
            if act == "pause" and paused:
                label = "Resume"
            primary = act == "map"
            fill = (37, 120, 255, 235) if primary else ((44, 50, 64, 235) if self.hover == act else (14, 17, 25, 215))
            cv.rect(x, 0, bw, self.h, fill=fill, outline=(255, 255, 255, 36), r=self.h / 2)
            cv.text(x + 16 * k, self.h / 2, label, int(13 * k), C.WHITE, True, "lm")
            cv.text(x + bw - 14 * k, self.h / 2, key, int(10 * k), (210, 222, 240, 200) if primary else C.DIM, True, "rm")
            self.hits.append(((x, 0, bw, self.h), act))
            x += bw + 8 * k
        return cv.finish()
