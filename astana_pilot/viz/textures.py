"""Procedural textures (numpy + Pillow) - nothing to download, everything is generated at start-up.

All images are ``(H, W, 4)`` uint8 with row 0 at the top. Facade tiles are one window bay x one floor; the
alpha channel is the *window mask* used by the shader (sky reflection by day, lit windows at night).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

_FONT_CANDIDATES = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "DejaVuSans.ttf")
_FONT_REGULAR = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
_MONO = ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf")


@lru_cache(maxsize=64)
def get_font(size: int, bold: bool = True, mono: bool = False) -> ImageFont.ImageFont:
    names = _MONO if mono else (_FONT_CANDIDATES if bold else _FONT_REGULAR)
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)          # Pillow >= 10.1: scalable fallback
    except Exception:                                # pragma: no cover
        return ImageFont.load_default()


def _tileable_noise(size: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = gaussian_filter(rng.standard_normal((size, size)), sigma, mode="wrap")
    n = (n - n.mean()) / (n.std() + 1e-9)
    return n


def _rgba(rgb: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if a is None:
        a = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    return np.dstack([rgb, a.astype(np.uint8)])


def asphalt(size: int = 256, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 0.55 * _tileable_noise(size, 1.2, rng) + 0.35 * _tileable_noise(size, 5.0, rng) + 0.25 * rng.standard_normal((size, size))
    base = 118 + 11 * n
    return _rgba(np.dstack([base, base, base * 1.03]))


def grass(size: int = 256, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 0.5 * _tileable_noise(size, 2.0, rng) + 0.7 * _tileable_noise(size, 12.0, rng) + 0.2 * rng.standard_normal((size, size))
    r, g, b = 135 + 14 * n, 150 + 16 * n, 98 + 10 * n
    return _rgba(np.dstack([r, g, b]))


def steppe(size: int = 256, seed: int = 3) -> np.ndarray:
    """Dry steppe / bare ground (Astana outside the parks)."""
    rng = np.random.default_rng(seed)
    n = 0.6 * _tileable_noise(size, 3.0, rng) + 0.5 * _tileable_noise(size, 14.0, rng) + 0.2 * rng.standard_normal((size, size))
    return _rgba(np.dstack([160 + 12 * n, 150 + 11 * n, 118 + 9 * n]))


def paving(size: int = 128, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 170.0)
    step = size // 4
    for i in range(0, size, step):
        for j in range(0, size, step):
            img[i:i + step, j:j + step] += rng.uniform(-9, 9)
    img[::step, :] -= 22
    img[:, ::step] -= 22
    img += 5 * rng.standard_normal((size, size))
    return _rgba(np.dstack([img, img, img * 0.98]))


def concrete(size: int = 128, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 0.6 * _tileable_noise(size, 1.5, rng) + 0.4 * _tileable_noise(size, 8.0, rng)
    v = 205 + 9 * n
    return _rgba(np.dstack([v, v, v]))


def facade(style: str, size: int = 128) -> np.ndarray:
    """One bay x one floor. ``style``: ``panel`` | ``glass`` | ``stone`` | ``shop``."""
    S = size
    img = Image.new("RGBA", (S, S), (236, 236, 232, 0))
    d = ImageDraw.Draw(img)
    glass_top, glass_bot = (60, 86, 118), (26, 38, 58)

    def window(x0, y0, x1, y1, mull_v=0, mull_h=0, frame=(246, 246, 244)):
        x0, y0, x1, y1 = [int(v * S) for v in (x0, y0, x1, y1)]
        d.rectangle([x0 - 3, y0 - 3, x1 + 3, y1 + 3], fill=frame + (0,))
        for y in range(y0, y1):
            t = (y - y0) / max(1, y1 - y0 - 1)
            col = tuple(int(glass_top[k] * (1 - t) + glass_bot[k] * t) for k in range(3))
            d.line([(x0, y), (x1, y)], fill=col + (255,))
        for k in range(1, mull_v + 1):
            x = x0 + (x1 - x0) * k // (mull_v + 1)
            d.line([(x, y0), (x, y1)], fill=frame + (255,), width=3)
        for k in range(1, mull_h + 1):
            y = y0 + (y1 - y0) * k // (mull_h + 1)
            d.line([(x0, y), (x1, y)], fill=frame + (255,), width=3)

    if style == "glass":
        window(0.03, 0.10, 0.97, 0.90, mull_v=1, frame=(190, 200, 210))
        d.rectangle([0, int(0.90 * S), S, S], fill=(200, 205, 210, 0))
    elif style == "stone":
        window(0.28, 0.20, 0.72, 0.82, mull_v=1, mull_h=1)
        d.rectangle([0, 0, int(0.07 * S), S], fill=(205, 203, 196, 0))
        d.rectangle([int(0.93 * S), 0, S, S], fill=(205, 203, 196, 0))
        d.rectangle([int(0.24 * S), int(0.84 * S), int(0.76 * S), int(0.88 * S)], fill=(222, 220, 214, 0))
    elif style == "shop":
        window(0.05, 0.12, 0.95, 0.88, mull_v=2)
    else:                                              # "panel": Soviet / modern residential
        window(0.20, 0.26, 0.80, 0.78, mull_v=1)
        d.rectangle([int(0.14 * S), int(0.80 * S), int(0.86 * S), int(0.86 * S)], fill=(212, 210, 205, 0))
        d.line([(0, S - 2), (S, S - 2)], fill=(205, 203, 198, 0), width=2)
    arr = np.asarray(img).copy()
    # soften the window mask a little for nicer mip-mapping
    arr[..., 3] = np.asarray(Image.fromarray(arr[..., 3]).filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(0.8)))
    return arr


# ------------------------------------------------------------------------------------------ signs
def _circle_sign(text: str, ring=(210, 28, 40), bg=(250, 250, 250), size: int = 256) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([2, 2, size - 3, size - 3], fill=ring + (255,))
    m = int(size * 0.16)
    d.ellipse([m, m, size - m, size - m], fill=bg + (255,))
    f = get_font(int(size * (0.44 if len(text) <= 2 else 0.34)), True)
    bb = d.textbbox((0, 0), text, font=f)
    d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (size - (bb[3] - bb[1])) / 2 - bb[1]), text, font=f, fill=(20, 20, 20, 255))
    return np.asarray(img).copy()


def speed_sign(kmh: int) -> np.ndarray:
    return _circle_sign(str(kmh))


def stop_sign(size: int = 256) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    import math
    pts = [(size / 2 + (size / 2 - 2) * math.cos(math.radians(22.5 + 45 * k)),
            size / 2 + (size / 2 - 2) * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    d.polygon(pts, fill=(255, 255, 255, 255))
    pts2 = [(size / 2 + (size / 2 - 14) * math.cos(math.radians(22.5 + 45 * k)),
             size / 2 + (size / 2 - 14) * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    d.polygon(pts2, fill=(200, 25, 35, 255))
    f = get_font(int(size * 0.27), True)
    bb = d.textbbox((0, 0), "STOP", font=f)
    d.text(((size - (bb[2] - bb[0])) / 2 - bb[0], (size - (bb[3] - bb[1])) / 2 - bb[1]), "STOP", font=f, fill=(255, 255, 255, 255))
    return np.asarray(img).copy()


def yield_sign(size: int = 256) -> np.ndarray:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(4, 14), (size - 4, 14), (size / 2, size - 8)], fill=(210, 28, 40, 255))
    d.polygon([(40, 38), (size - 40, 38), (size / 2, size - 56)], fill=(255, 255, 255, 255))
    return np.asarray(img).copy()


def white_pixel() -> np.ndarray:
    return np.full((4, 4, 4), 255, dtype=np.uint8)
