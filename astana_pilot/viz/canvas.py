"""Tiny anti-aliased 2-D canvas on top of Pillow (draws at 2x and down-samples) - used for the HUD panels and the minimap."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .textures import get_font

Col = Tuple[int, int, int, int]

# ---------------------------------------------------------------------- palette (dark glass UI)
BG = (10, 14, 24, 205)
BG_SOLID = (12, 17, 30, 255)
BORDER = (70, 200, 255, 70)
TEXT = (236, 241, 248, 255)
DIM = (130, 150, 178, 255)
FAINT = (70, 86, 110, 255)
ACCENT = (110, 224, 255, 255)
GREEN = (62, 232, 140, 255)
AMBER = (255, 196, 64, 255)
RED = (255, 82, 92, 255)
VIOLET = (170, 130, 255, 255)
WHITE = (255, 255, 255, 255)
GLASS = (10, 12, 18, 150)
STROKE = (255, 255, 255, 26)


def rgba(c: Col, a: Optional[int] = None) -> Col:
    return (c[0], c[1], c[2], c[3] if a is None else a)


class Canvas:
    """Draw in logical pixels; everything is rendered at ``ss`` x and filtered down on :meth:`finish`."""

    def __init__(self, w: int, h: int, ss: int = 2, bg: Col = (0, 0, 0, 0)) -> None:
        self.w, self.h, self.ss = w, h, ss
        self.img = Image.new("RGBA", (w * ss, h * ss), bg)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    @classmethod
    def from_image(cls, img: Image.Image, w: int, h: int, ss: int = 2) -> "Canvas":
        """Continue drawing on top of an existing (cached) background image."""
        self = cls.__new__(cls)
        self.w, self.h, self.ss = w, h, ss
        self.img = img
        self.d = ImageDraw.Draw(self.img, "RGBA")
        return self

    # -- helpers
    def _p(self, v) -> float:
        return v * self.ss

    def _pts(self, pts) -> list:
        return [(x * self.ss, y * self.ss) for x, y in pts]

    # -- shapes
    def rect(self, x, y, w, h, fill: Optional[Col] = None, outline: Optional[Col] = None, r: float = 0, width: float = 1) -> None:
        if w * self.ss < 2 or h * self.ss < 2:
            return                                                        # too small to draw (PIL would raise)
        box = [self._p(x), self._p(y), self._p(x + w) - 1, self._p(y + h) - 1]
        if r > 0:
            r = min(r, w / 2.0, h / 2.0)
            self.d.rounded_rectangle(box, radius=self._p(r), fill=fill, outline=outline, width=int(max(1, width * self.ss)))
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=int(max(1, width * self.ss)))

    def circle(self, cx, cy, r, fill: Optional[Col] = None, outline: Optional[Col] = None, width: float = 1) -> None:
        self.d.ellipse([self._p(cx - r), self._p(cy - r), self._p(cx + r), self._p(cy + r)], fill=fill, outline=outline,
                       width=int(max(1, width * self.ss)))

    def line(self, pts: Sequence[Tuple[float, float]], color: Col, width: float = 1.0) -> None:
        if len(pts) >= 2:
            self.d.line(self._pts(pts), fill=color, width=int(max(1, width * self.ss)), joint="curve")

    def poly(self, pts, fill: Optional[Col] = None, outline: Optional[Col] = None) -> None:
        self.d.polygon(self._pts(pts), fill=fill, outline=outline)

    def arc(self, cx, cy, r, a0: float, a1: float, color: Col, width: float = 3.0) -> None:
        """Arc from angle ``a0`` to ``a1`` degrees (0 = east, clockwise on screen)."""
        self.d.arc([self._p(cx - r), self._p(cy - r), self._p(cx + r), self._p(cy + r)], a0, a1, fill=color,
                   width=int(max(1, width * self.ss)))

    def text(self, x, y, s: str, size: int = 14, color: Col = TEXT, bold: bool = True, anchor: str = "la", mono: bool = False) -> None:
        f = get_font(int(size * self.ss), bold, mono)
        self.d.text((self._p(x), self._p(y)), s, font=f, fill=color, anchor=anchor)

    def text_w(self, s: str, size: int = 14, bold: bool = True, mono: bool = False) -> float:
        f = get_font(int(size * self.ss), bold, mono)
        return f.getlength(s) / self.ss

    def bar(self, x, y, w, h, frac: float, color: Col, back: Col = (255, 255, 255, 28), r: float = 3) -> None:
        self.rect(x, y, w, h, fill=back, r=r)
        f = max(0.0, min(1.0, frac))
        if f > 0.002:
            self.rect(x, y, max(h, w * f), h, fill=color, r=r)

    def center_bar(self, x, y, w, h, v: float, color: Col, back: Col = (255, 255, 255, 28)) -> None:
        self.rect(x, y, w, h, fill=back, r=h / 2)
        v = max(-1.0, min(1.0, v))
        mid = x + w / 2
        if abs(v) > 0.01:
            self.rect(min(mid, mid + v * w / 2), y, abs(v) * w / 2, h, fill=color, r=h / 2)
        self.rect(mid - 1, y - 2, 2, h + 4, fill=(255, 255, 255, 140))

    def panel(self, x, y, w, h, title: str = "", accent: Col = ACCENT, r: float = 12, fill: Col = BG) -> None:
        self.rect(x, y, w, h, fill=fill, outline=BORDER, r=r)
        if title:
            self.rect(x + 14, y + 12, 3, 13, fill=accent, r=1.5)
            self.text(x + 23, y + 9, title.upper(), 11, DIM)

    # -- output
    def finish(self) -> np.ndarray:
        img = self.img.reduce(self.ss) if self.ss != 1 else self.img       # box filter: ~10x faster than LANCZOS
        return np.asarray(img)

    def paste(self, arr: np.ndarray, x: int, y: int) -> None:
        im = Image.fromarray(arr)
        self.img.alpha_composite(im.resize((im.width * self.ss, im.height * self.ss)), (int(x * self.ss), int(y * self.ss)))


def speed_color(frac: float) -> Col:
    """Green below the limit -> amber near it -> red above."""
    if frac < 0.9:
        return GREEN
    if frac < 1.05:
        return AMBER
    return RED


def roundel(cv: Canvas, cx: float, cy: float, r: float, text: str, ring: Col = (214, 36, 46, 255)) -> None:
    cv.circle(cx, cy, r, fill=ring)
    cv.circle(cx, cy, r * 0.74, fill=(250, 250, 250, 255))
    cv.text(cx, cy + 1, text, int(r * (0.95 if len(text) <= 2 else 0.78)), (20, 20, 20, 255), True, "mm")
