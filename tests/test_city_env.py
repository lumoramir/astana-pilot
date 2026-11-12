"""Tests of the city map, route generation and the Gymnasium environment (synthetic city)."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

from astana_pilot.env import DrivingEnv, EnvConfig                       # noqa: E402
from astana_pilot.env.route import build_reference_path                  # noqa: E402
from astana_pilot.env.synthetic_map import build_synthetic_map           # noqa: E402
from astana_pilot.viz.policies import HeuristicPolicy          # noqa: E402

CFG = EnvConfig(map_path="synthetic", recovery_prob=0.0)


@pytest.fixture(scope="module")
def city():
    return build_synthetic_map()


def test_sdf_sign(city):
    node = np.array([city.raw.nodes[200]])
    assert city.sdf(node)[0] > 2.0                           # junction is on the road
    assert city.sdf(np.array([[-900.0, -900.0]]))[0] < 0.0   # far away: not drivable


def test_sdf_is_roughly_distance_to_wall(city):
    # walk perpendicular to a straight road; sdf must reach ~0 near the road edge
    e = city.raw.edges[0]
    a, b = e.xy[0], e.xy[-1]
    mid = (a + b) / 2
    n = np.array([-(b - a)[1], (b - a)[0]]); n /= np.linalg.norm(n)
    vals = city.sdf(np.stack([mid + n * t for t in np.linspace(-40, 40, 161)]))
    assert (vals > 0).any() and (vals < 0).any()


def test_reference_paths_stay_on_road(city):
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(40):
        ref = build_reference_path(city, city.sample_route(rng, 150, 400), CFG)
        if ref is None:
            continue
        n_ok += 1
        assert city.sdf(ref.xy).min() >= CFG.vehicle.body_radius + 0.2
        assert np.abs(ref.kappa).max() <= 0.23
        assert ref.v_prof.max() <= CFG.max_target_speed + 1e-6
    assert n_ok >= 30


def test_env_api_and_determinism():
    e1, e2 = DrivingEnv(CFG), DrivingEnv(CFG)
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    assert o1.shape == e1.observation_space.shape and np.allclose(o1, o2)
    a = np.array([0.1, 0.5], dtype=np.float32)
    for _ in range(20):
        o1, r1, *_ = e1.step(a)
        o2, r2, *_ = e2.step(a)
    assert np.allclose(o1, o2) and r1 == r2
    assert o1.min() >= -1.0 and o1.max() <= 1.0 and np.isfinite(o1).all()


def test_baseline_driver_completes_easy_routes():
    env = DrivingEnv(CFG)
    env.set_difficulty(0.0)
    pol = HeuristicPolicy(CFG)
    wins = 0
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, _, te, tr, info = env.step(pol.act(obs))
            done = te or tr
        wins += info["is_success"]
        assert info["ep_lane_dev"] < 0.6                     # follows the lane, no wobble
    assert wins >= 8


def test_leaving_the_road_is_a_collision():
    env = DrivingEnv(CFG)
    env.reset(seed=0)
    for _ in range(200):
        _, _, te, tr, info = env.step(np.array([1.0, 1.0], dtype=np.float32))   # full lock + throttle
        if te or tr:
            break
    assert info["collision"] or info["lost"]


def test_wrong_way_spawn_and_reverse_progress():
    env = DrivingEnv(EnvConfig(map_path="synthetic", recovery_prob=1.0))
    env.set_difficulty(1.0)
    for seed in range(60):
        env.reset(seed=seed)
        if env.spawn_kind == "wrong_way":
            break
    assert env.spawn_kind == "wrong_way"
    s0 = env._s
    for _ in range(25):
        env.step(np.array([0.0, -1.0], dtype=np.float32))
    assert env.vehicle.v < -1.0 and env._s > s0              # reversing moves along the route
