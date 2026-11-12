"""Point-to-point navigation: the navigator plans A -> B with alternatives, the environment drives exactly that route."""
import numpy as np
import pytest

from astana_pilot.env import DrivingEnv, EnvConfig
from astana_pilot.env import navigator as nav
from astana_pilot.env.city_map import load_city_map


@pytest.fixture(scope="module")
def city():
    return load_city_map("synthetic")


@pytest.fixture(scope="module")
def cfg():
    return EnvConfig(map_path="synthetic")


def test_random_plan_is_drivable_and_fastest_first(city, cfg):
    plan = nav.random_plan(city, cfg, np.random.default_rng(3), 500, 1400)
    assert plan is not None and plan.options
    etas = [o.eta_s for o in plan.options]
    assert plan.chosen == 0 and etas == sorted(etas)              # the chosen route is the fastest
    b = plan.best
    assert b.nodes[0] == plan.start and b.nodes[-1] == plan.goal
    assert b.maneuvers[-1].kind == "arrive"
    assert b.eta_left(0.0) == pytest.approx(b.eta_s) and b.eta_left(b.length) == pytest.approx(0.0)


def test_alternatives_differ(city, cfg):
    plan = nav.random_plan(city, cfg, np.random.default_rng(3), 500, 1400, alternatives=3)
    for i, o in enumerate(plan.options[1:], 1):
        assert o.nodes != plan.options[0].nodes


def test_resolve_point_formats(city):
    n = next(iter(city.G.nodes))
    assert nav.resolve_point(city, str(n))[0] in city.G
    x, y = nav.node_xy(city, n)
    assert nav.resolve_point(city, f"{x:.1f},{y:.1f}")[0] in city.G
    with pytest.raises(ValueError):
        nav.resolve_point(city, "no such place 12345")


def test_env_drives_the_given_route(city, cfg):
    plan = nav.random_plan(city, cfg, np.random.default_rng(5), 400, 1000)
    env = DrivingEnv(cfg)
    env.reset(seed=1, options={"route_nodes": plan.best.nodes, "ref": plan.best.ref})
    assert env._route_nodes == plan.best.nodes
    assert env.ref is plan.best.ref
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == env.observation_space.shape                # same observation: the network sees no map


def test_reroute_from_moving_car(city, cfg):
    env = DrivingEnv(cfg)
    env.reset(seed=4)
    for _ in range(40):
        env.step(np.array([0.0, 0.4], dtype=np.float32))
    goals = nav._candidates(city, "goal")
    plan = None
    for g in goals[::7]:
        plan = nav.plan_from_car(city, cfg, env._route_nodes, env.ref, env._s, g)
        if plan is not None:
            break
    assert plan is not None
    obs = env.reroute(plan.best.nodes, plan.best.ref)
    assert obs.shape == env.observation_space.shape and env.ref is plan.best.ref
    env.step(np.zeros(2, dtype=np.float32))


def test_map_panel_click_flow(city, cfg):
    from astana_pilot.viz.map_picker import MapPanel
    m = MapPanel(city, cfg, (1280, 720))
    assert m.click(m.mw * 0.3, m.mh * 0.3) is None and m.A is not None
    m.click(m.mw * 0.7, m.mh * 0.7)
    img = m.render()
    assert img.shape[:2] == (m.mh, m.mw + m.side)
    if m.plan is not None:
        go = next(r for r, a in m.hits if a == "go")
        assert m.click(go[0] + 5, go[1] + 5) == "go"


def test_session_navigator_mode(cfg):
    from astana_pilot.viz.session import DriveSession
    s = DriveSession("synthetic", seed=2, random_route=True, route_km=(0.4, 1.0))
    assert s.nav_mode and s.plan is not None
    info = s.nav_info()
    assert info["options"] and info["remain_m"] > 0
    first = s.plan.best.nodes
    s.episode += 1
    s.new_episode()
    s2 = DriveSession("synthetic", seed=2, random_route=True, route_km=(0.4, 1.0))
    assert s2.plan.best.nodes == first                            # same seed -> same route (reproducible demo)
