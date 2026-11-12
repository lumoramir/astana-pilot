"""Graphics quality presets for the 3-D viewer (laptop RTX 4060 -> ``high``; integrated graphics -> ``low``)."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict


@dataclass(frozen=True)
class Quality:
    name: str
    shadows: int            # shadow-map resolution (0 = off)
    shadow_size: float      # half extent of the shadowed area around the camera focus [m]
    msaa: int               # multisampling samples (0 = off)
    trees: float            # density multiplier of street / park trees
    props: float            # street lamps, signs density multiplier
    far: float              # far clip plane [m]
    textures: bool          # facade / asphalt textures
    max_buildings: int
    hud_hz: float           # HUD refresh rate
    window: tuple           # default window size
    max_actors: int = 140   # animated road users that are drawn


PRESETS: Dict[str, Quality] = {
    "low": Quality("low", 0, 60.0, 0, 0.25, 0.4, 700.0, False, 700, 8.0, (1280, 720)),
    "medium": Quality("medium", 1024, 90.0, 0, 0.6, 0.8, 1400.0, True, 2500, 12.0, (1600, 900)),
    "high": Quality("high", 2048, 110.0, 4, 1.0, 1.0, 2600.0, True, 6000, 15.0, (1600, 900)),
    "ultra": Quality("ultra", 4096, 130.0, 4, 1.4, 1.3, 4000.0, True, 12000, 20.0, (1920, 1080)),
}

INTEGRATED_HINTS = ("intel", "uhd", "iris", "radeon graphics", "radeon(tm) graphics", "vega", "microsoft basic")


def pick(name: str = "auto", gpu: str = "") -> Quality:
    if name in PRESETS:
        return PRESETS[name]
    low = gpu.lower()
    if low and any(h in low for h in INTEGRATED_HINTS) and "nvidia" not in low:
        return PRESETS["low"]
    return PRESETS["high"]


def with_window(q: Quality, size: tuple) -> Quality:
    return replace(q, window=size)
