"""Render still images of a city from arbitrary camera poses (headless, no window) - for docs, posters and debugging.

    python -m astana_pilot.viz.stills --map synthetic --out assets/shots/test.png --pos 0,120,-200 --look 0,0,0
    python -m astana_pilot.viz.stills --map assets/maps/astana_expo.pkl --shots overview,street --quality high
"""
from __future__ import annotations

import argparse
import builtins
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from panda3d.core import Filename, Vec3, loadPrcFileData

from . import quality as Q


def monitor_size() -> Tuple[int, int]:
    """True pixel size of the primary monitor (and opt in to DPI awareness so Windows does not bitmap-stretch the window)."""
    try:
        import ctypes
        u = ctypes.windll.user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            u.SetProcessDPIAware()
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def default_window(frac: float = 0.86) -> Tuple[int, int]:
    """A window that keeps the monitor's aspect ratio (16:9, 16:10, ultrawide ...) and leaves room for the taskbar."""
    w, h = monitor_size()
    return int(w * frac) // 2 * 2, int(h * frac) // 2 * 2


def create_app(headless: bool, size: Tuple[int, int], msaa: int = 0, title: str = "Astana AV", vsync: bool = True,
               fullscreen: bool = False):
    loadPrcFileData("", f"framebuffer-multisample {1 if msaa else 0}\nmultisamples {msaa}\n"
                        f"sync-video {1 if (vsync and not headless) else 0}\nnotify-level-glgsg error\n"
                        "audio-library-name null\n")
    from ursina import Ursina, camera, window
    app = Ursina(title=title, window_type="offscreen" if headless else "onscreen", size=size, borderless=headless,
                 development_mode=False, vsync=vsync and not headless, fullscreen=fullscreen, forced_aspect_ratio=None)
    if headless:       # Ursina only does this for real windows
        camera.ui_lens.set_film_size(camera.ui_size * 0.5 * window.aspect_ratio, camera.ui_size * 0.5)
        camera.perspective_lens.set_aspect_ratio(window.aspect_ratio)
    return app


def gpu_name() -> str:
    try:
        return str(builtins.base.win.get_gsg().get_driver_renderer())  # type: ignore[attr-defined]
    except Exception:
        return "unknown"


def save_screenshot(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    builtins.base.win.save_screenshot(Filename.from_os_specific(str(Path(path).resolve())))  # type: ignore[attr-defined]


def look_at_pose(camera, pos: Sequence[float], target: Sequence[float], fov: float = 60.0, far: float = 5000.0) -> None:
    camera.position = tuple(pos)
    camera.look_at(Vec3(*target))
    camera.fov = fov
    camera.clip_plane_near = 0.5
    camera.clip_plane_far = far


def render_stills(map_path: str, shots: Sequence[Tuple[str, Sequence[float], Sequence[float], float]], out_dir: str,
                  quality: str = "high", preset: str = "day", size: Tuple[int, int] = (1600, 900),
                  landmarks: bool = True) -> None:
    from ursina import camera

    from ..env.city_map import load_city_map
    from .city_scene import build_scene
    from .renderer import StaticRenderer

    q = Q.pick(quality, "nvidia")
    app = create_app(True, size, q.msaa)
    t0 = time.time()
    city = load_city_map(map_path)
    scene = build_scene(city, q, landmarks=landmarks)
    print(f"[stills] scene built in {time.time() - t0:.1f} s: {scene.stats}")
    sr = StaticRenderer(scene, q, preset)
    sr.update_signals(city.infra, 0.0, force=True)
    for name, pos, target, fov in shots:
        look_at_pose(camera, pos, target, fov, q.far)
        sr.atm.follow(*target)
        sr.follow_sky(*pos)
        for _ in range(4):
            app.step()
        out = str(Path(out_dir) / f"{name}.png")
        save_screenshot(out)
        print("[stills] wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--map", default="synthetic")
    ap.add_argument("--out", default="assets/shots")
    ap.add_argument("--quality", default="high")
    ap.add_argument("--preset", default="day", choices=["day", "dusk", "night", "snow"])
    ap.add_argument("--pos", default=None, help="x,y,z camera (viewer coords: y is up)")
    ap.add_argument("--look", default="0,0,0")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--name", default="still")
    ap.add_argument("--ego-seed", type=int, default=None, help="camera behind the car of episode <seed> (street view)")
    ap.add_argument("--back", type=float, default=14.0)
    ap.add_argument("--height", type=float, default=5.0)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    pos = tuple(float(v) for v in a.pos.split(",")) if a.pos else (0.0, 300.0, -300.0)
    look = tuple(float(v) for v in a.look.split(","))
    if a.ego_seed is not None:
        from ..env import DrivingEnv, EnvConfig
        env = DrivingEnv(EnvConfig(map_path=a.map))
        env.reset(seed=a.ego_seed)
        v = env.vehicle
        c, s = np.cos(v.theta), np.sin(v.theta)
        pos = (v.x - c * a.back, a.height, v.y - s * a.back)
        look = (v.x + c * 25.0, 1.5, v.y + s * 25.0)
    render_stills(a.map, [(a.name, pos, look, a.fov)], a.out, a.quality, a.preset, (w, h))


if __name__ == "__main__":
    main()
