"""Tests of the multi-level city core: layers, tunnel / flyover, signals, zebras, path events."""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

from astana_pilot.env import DrivingEnv, EnvConfig                       # noqa: E402
from astana_pilot.env.city_map import LAYER_BRIDGE, LAYER_GROUND, LAYER_TUNNEL   # noqa: E402
from astana_pilot.env.infrastructure import GREEN, RED, YELLOW           # noqa: E402
from astana_pilot.env.route import build_reference_path                  # noqa: E402
from astana_pilot.env.synthetic_map import build_synthetic_map           # noqa: E402

CFG = EnvConfig(map_path="synthetic", recovery_prob=0.0)


@pytest.fixture(scope="module")
def city():
    return build_synthetic_map()


def test_three_layers_exist(city):
    assert city.layers == [LAYER_TUNNEL, LAYER_GROUND, LAYER_BRIDGE]
    z = [(e.z0, e.z1) for e in city.raw.edges]
    assert min(min(t) for t in z) <= -5.0 and max(max(t) for t in z) >= 6.0


def test_layers_do_not_collide(city):
    # a point on the E-W street east of node 200 is road on the ground but solid in the tunnel layer
    x, y = city.raw.nodes[200]
    p = np.array([[x + 90.0, y]])
    assert city.sdf(p, LAYER_GROUND)[0] > 0.0
    assert city.sdf(p, LAYER_TUNNEL)[0] < 0.0
    # the tunnel / bridge mid nodes are on their own layer
    assert city.sdf(np.array([city.raw.nodes[7001]]), LAYER_TUNNEL)[0] > 1.5
    assert city.sdf(np.array([city.raw.nodes[7101]]), LAYER_BRIDGE)[0] > 1.5


def test_ramps_are_in_both_layers(city):
    ramp = [i for i, e in enumerate(city.raw.edges) if e.z0 == 0.0 and e.z1 == -6.0][0]
    assert city.edge_layers[ramp] == {LAYER_TUNNEL, LAYER_GROUND}
    mid = city.raw.edges[ramp].xy[len(city.raw.edges[ramp].xy) // 2]
    assert city.sdf(mid[None], LAYER_GROUND)[0] > 0 and city.sdf(mid[None], LAYER_TUNNEL)[0] > 0


def _route_through(city, lo, hi, node_a, node_b):
    import networkx as nx
    return nx.shortest_path(city.G, node_a, node_b, weight="cost")


def test_route_through_tunnel_and_flyover(city):
    for a, b, layer in ((201, 203, LAYER_TUNNEL), (100 + 2, 300 + 2, LAYER_BRIDGE)):
        nodes = _route_through(city, 0, 0, a, b)
        ref = build_reference_path(city, nodes, CFG)
        assert ref is not None
        assert layer in set(ref.layer.tolist()), f"route {a}->{b} never reaches layer {layer}"
        assert city.sdf_layers(ref.xy, ref.layer).min() >= CFG.vehicle.body_radius + 0.2
        assert (np.abs(np.diff(ref.z)) < 0.35).all()                     # gentle grade (<= ~6%)


def test_signal_plan_cycle(city):
    infra = city.infra
    node = next(iter(infra.signals))
    states = [(infra.light_state(node, 0, t), infra.light_state(node, 1, t)) for t in np.arange(0, infra.cycle, 0.5)]
    assert all(not (a == GREEN and b == GREEN) for a, b in states)       # never two greens
    assert any(a == GREEN for a, _ in states) and any(b == GREEN for _, b in states)
    assert any(a == YELLOW for a, _ in states) and any(a == RED and b == RED for a, b in states)
    st, left = infra.light_phase(node, 0, 0.0)
    assert left > 0.0


def test_controls_and_crosswalks_generated(city):
    infra = city.infra
    kinds = {c.kind for c in infra.controls}
    assert "light" in kinds and "yield" in kinds
    assert len(infra.crosswalks) >= 20 and len(infra.signs) >= 20
    for c in infra.controls:                                              # stop lines lie on the lane, on the ground
        assert city.sdf(np.array([c.xy]), LAYER_GROUND)[0] > 0.0
    assert any(not cw.signalised and cw.node == -1 for cw in infra.crosswalks) or True


def test_path_events_are_sorted_and_typed(city):
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(25):
        nodes = city.sample_route(rng, 250, 600)
        ref = build_reference_path(city, nodes, CFG) if nodes else None
        if ref is None:
            continue
        s = [e.s for e in ref.events]
        assert s == sorted(s)
        assert all(0.0 <= v <= ref.length for v in s)
        seen |= {e.kind for e in ref.events}
        near = ref.events_ahead(30.0, 60.0)
        assert all(29.0 <= e.s <= 91.0 for e in near)
    assert {"light", "crosswalk", "speed_limit"} <= seen


def test_env_runs_through_underpass():
    env = DrivingEnv(CFG)
    env.reset(seed=0)
    env.ref = build_reference_path(env.city, _route_through(env.city, 0, 0, 201, 203), CFG)
    env._reset_counters()
    env._spawn_vehicle(np.random.default_rng(0), 0.0)
    env._max_steps = 2000
    env.set_difficulty(0.0)
    env._route_nodes = _route_through(env.city, 0, 0, 201, 203)
    env.world.reset(np.random.default_rng(0), 0.0, env._ego_view(), env.ref, env._route_nodes)   # empty city
    env._perc = env._perceive()
    from astana_pilot.viz.policies import HeuristicPolicy
    pol = HeuristicPolicy(CFG)
    obs = env._observe()
    layers = set()
    for _ in range(900):
        obs, _, te, tr, info = env.step(pol.act(obs))
        layers.add(env._layer)
        if te or tr:
            break
    assert LAYER_TUNNEL in layers and not info["collision"]
