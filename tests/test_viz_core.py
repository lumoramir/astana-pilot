"""Visualisation core: mesh toolkit, scene builder, HUD canvas, session, 2-D base map (no GPU / window needed)."""
import math

import numpy as np
import pytest

from astana_pilot.viz import canvas as C
from astana_pilot.viz.mesh import MeshBuilder, ear_clip, instances, split_into_tiles


def test_up_quad_is_front_facing_and_normal_up():
    b = MeshBuilder()
    b.quad((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1), (1, 1, 1, 1))
    m = b.build()
    assert np.allclose(m.n, [0, 1, 0])
    # front face = counter-clockwise on a north-up map (x east, z north) -> positive signed area
    tri = m.v[m.t[0]][:, [0, 2]]
    a, b = tri[1] - tri[0], tri[2] - tri[0]
    assert 0.5 * (a[0] * b[1] - a[1] * b[0]) > 0


def test_box_normals_point_outwards():
    b = MeshBuilder()
    b.box((0, 1, 0), (2, 2, 2), yaw=0.3)
    m = b.build()
    centre = np.array([0, 1, 0])
    for tri, n in zip(m.t, m.n[m.t[:, 0]]):
        face_c = m.v[tri].mean(axis=0)
        assert float(np.dot(face_c - centre, n)) > 0


def test_ear_clip_area_is_preserved():
    ring = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [4, 3], [4, 4], [0, 4]], dtype=float)   # a "C" shape
    tris = ear_clip(ring)
    def tri_area(a, b, c):
        u, v = ring[b] - ring[a], ring[c] - ring[a]
        return abs(0.5 * (u[0] * v[1] - u[1] * v[0]))
    area = sum(tri_area(a, b, c) for a, b, c in tris)
    assert abs(area - 10.0) < 1e-6
    assert len(tris) == len(ring) - 2


def test_extrude_has_facade_uvs_and_roof():
    b = MeshBuilder()
    ring = np.array([[0, 0], [10, 0], [10, 7], [0, 7]], dtype=float)
    b.extrude(ring, 0, 13.0, (0.9, 0.9, 0.9, 1), (0.5, 0.5, 0.5, 1), ear_clip(ring), tile_w=3.3, tile_h=3.3)
    m = b.build()
    assert m.uv.max() >= 3.0                                           # 10 m wall / 3.3 m bays, 4 floors
    assert np.isclose(m.v[:, 1].max(), 13.0)


def test_instances_and_tiles():
    b = MeshBuilder()
    b.cylinder((0, 0, 0), 0.2, 2.0, 6)
    t = b.build()
    m = instances(t, np.array([[150, 0, 150], [650, 0, 150], [150, 0, 950]], dtype=float), np.array([0, 1, 2.0]), 1.0)
    assert len(m.v) == 3 * len(t.v)
    tiles = split_into_tiles(m, 300.0)
    assert len(tiles) == 3 and sum(len(tm.t) for _, tm in tiles) == len(m.t)


def test_loft_car_is_closed_and_sane():
    from astana_pilot.viz.actors import bus_models, car_models, pedestrian_models
    paint, glass, wheels, lights = car_models()
    assert len(paint.t) > 100 and len(wheels.t) > 50
    lo, hi = paint.bounds()
    assert 4.3 < hi[2] - lo[2] < 4.7 and 1.7 < hi[0] - lo[0] < 2.1     # 4.5 x 1.9 m
    bp, *_ = bus_models()
    lo, hi = bp.bounds()
    assert 11.5 < hi[2] - lo[2] < 12.5
    assert set(pedestrian_models()) == {"torso", "head", "leg", "arm"}


def test_scene_builds_on_the_synthetic_city():
    from astana_pilot.env.city_map import load_city_map
    from astana_pilot.viz.city_scene import build_scene
    from astana_pilot.viz.quality import PRESETS
    city = load_city_map("synthetic")
    sd = build_scene(city, PRESETS["low"])
    assert sd.stats["buildings"] > 10
    for name in ("road", "walk", "mark", "ground"):
        assert sd.layers[name].meshes, name
    for layer in sd.layers.values():
        for _, m in layer.meshes:
            assert np.all(np.isfinite(m.v)) and np.all(np.isfinite(m.c))
            assert m.t.max() < len(m.v)
    assert sd.lights and sd.light_mesh is not None


def test_scene_has_a_flyover_structure(mini_city):
    from astana_pilot.viz.city_scene import build_scene
    from astana_pilot.viz.quality import PRESETS
    sd = build_scene(mini_city, PRESETS["low"], landmarks=False)
    con = [m for _, m in sd.layers["concrete"].meshes]
    assert con and max(m.v[:, 1].max() for m in con) > 6.0            # girders / parapets / piers reach the deck


def test_textures():
    from astana_pilot.viz import textures as T
    for style in ("panel", "glass", "stone", "shop"):
        f = T.facade(style)
        assert f.shape == (128, 128, 4) and f[..., 3].max() > 200 and f[..., 3].min() < 50     # window mask present
    assert T.asphalt().shape == (256, 256, 4)
    assert T.speed_sign(60).shape[2] == 4


def test_canvas_and_hud_panels():
    from astana_pilot.viz.hud import HudPanels, Telemetry
    cv = C.Canvas(100, 60)
    cv.rect(0, 0, 100, 60, fill=(255, 0, 0, 255))
    arr = cv.finish()
    assert arr.shape == (60, 100, 4) and arr[30, 50, 0] == 255
    h = HudPanels((1600, 900))
    t = Telemetry(speed_kmh=42, limit_kmh=50, lidar=np.full(48, 40.0), lidar_angles=np.linspace(0, 6.28, 48, endpoint=False))
    panels = h.render_all(t)
    assert set(panels) == set(h.PANEL_HZ)
    for k, a in panels.items():
        assert a.shape[2] == 4 and a.shape[:2] == (h._geom[k][3], h._geom[k][2])
    assert h.due_panels(t, now=1e9)                                    # something is due


def test_session_runs_and_reports():
    from astana_pilot.viz.session import DriveSession
    s = DriveSession("synthetic", seed=2)
    acc = 0.0
    for _ in range(60):
        acc, _ = s.advance(0.1, acc)
    t = s.telemetry()
    assert t.progress >= 0.0 and t.limit_kmh > 0 and len(t.route_ahead) == 22
    assert len(t.lidar) == s.cfg.n_rays
    assert isinstance(t.reason, str) and t.reason


def test_2d_base_map_renders(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    pygame.display.set_mode((64, 64))
    from astana_pilot.env.city_map import load_city_map
    from astana_pilot.viz.viewer2d import BaseMap
    bm = BaseMap(load_city_map("synthetic"), 0.5)
    assert bm.surf.get_width() > 100


def test_heading_convention():
    """A model whose front is +z, turned with ``panda_heading(theta)``, must point along (cos theta, sin theta) in the
    viewer's (x east, z north) frame. (Regression: all cars used to drive sideways.) Runs in a subprocess because the
    coordinate system is fixed per process by Ursina."""
    pytest.importorskip("ursina")
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import math, builtins
        from astana_pilot.viz.stills import create_app
        app = create_app(True, (200, 150), 0)
        from panda3d.core import Point3
        from astana_pilot.viz.actors import panda_heading
        n = builtins.base.render.attach_new_node("car")
        for th in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.7, 2.4):
            n.set_h(panda_heading(th))
            p = builtins.base.render.get_relative_point(n, Point3(0, 0, 1))
            assert abs(p[0] - math.cos(th)) < 1e-3 and abs(p[2] - math.sin(th)) < 1e-3, (th, tuple(p))
        print("HEADING_OK")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "HEADING_OK" in r.stdout, r.stdout[-500:] + r.stderr[-800:]
