"""OpenStreetMap -> RawMap -> CityMap -> environment (offline, using the synthetic Overpass extract in conftest)."""
import numpy as np

from astana_pilot.env.city_map import LAYER_BRIDGE, edge_layer_set, polyline_length
from astana_pilot.tools.osm_builder import Projector, building_height, describe, rdp


def test_projection_roundtrip():
    p = Projector(51.0895, 71.4150)
    xy = p.fwd(51.0905, 71.4170)
    lat, lon = p.inv(xy[0], xy[1])
    assert abs(lat - 51.0905) < 1e-9 and abs(lon - 71.4170) < 1e-9
    assert 100 < xy[1] < 120 and 130 < xy[0] < 160                  # ~111 m north, ~140 m east


def test_graph_basics(mini_raw):
    raw = mini_raw
    names = {e.name for e in raw.edges}
    assert "East-West Avenue" in names and "North Street" in names
    assert all(e.highway != "service" for e in raw.edges)             # the parking aisle was dropped
    # the main avenue is split at the junction and carries the explicit lane count
    ew = [e for e in raw.edges if e.name == "East-West Avenue"]
    assert len(ew) >= 4 and all(e.lanes == 4 for e in ew)
    # two-way streets have both directions
    pairs = {(e.u, e.v) for e in raw.edges}
    assert all((v, u) in pairs for (u, v) in pairs if any(e.name == "North Street" and (e.u, e.v) == (u, v) for e in raw.edges))


def test_oneway_minus_one_is_reversed(mini_raw):
    one = [e for e in mini_raw.edges if e.highway == "residential"]
    assert len(one) == 1 and one[0].oneway
    # digitised from x=-500 to x=-100 but tagged oneway=-1: traffic flows towards x=-500
    assert one[0].xy[0, 0] > one[0].xy[-1, 0]


def test_bridge_gets_ramps_and_a_flyover_layer(mini_raw, mini_city):
    z = [(e.z0, e.z1) for e in mini_raw.edges if e.name == "Flyover"]
    assert z, "bridge edges missing"
    assert max(max(a, b) for a, b in z) > 5.5                          # deck at +6.5 m
    ramps = [e for e in mini_raw.edges if len(edge_layer_set(e)) > 1]
    assert len(ramps) >= 2                                             # up and down (both directions -> >= 2 edges)
    assert LAYER_BRIDGE in mini_city.layers
    # the slope never exceeds 8 %
    for e in mini_raw.edges:
        L = polyline_length(e.xy)
        if L > 5:
            assert abs(e.z1 - e.z0) / L <= 0.085


def test_signal_snapped_to_the_junction(mini_raw):
    assert mini_raw.signal_nodes, "traffic signal not snapped"
    n = next(iter(mini_raw.signal_nodes))
    x, y = mini_raw.nodes[n]
    assert np.hypot(x, y) < 5.0                                        # the central junction


def test_buildings_parks_and_heights(mini_raw):
    assert len(mini_raw.buildings) == 1
    ring, h = mini_raw.buildings[0]
    assert abs(h - (9 * 3.3 + 1.0)) < 0.5                              # building:levels = 9
    assert mini_raw.building_meta[0]["name"] == "Test House"
    assert len(mini_raw.green) == 1
    assert building_height({"height": "42 m"}, 500)[0] == 42.0


def test_rdp_keeps_corners():
    pts = np.array([[0, 0], [5, 0.01], [10, 0], [10, 10]], dtype=float)
    out = rdp(pts, 0.5)
    assert len(out) == 3


def test_describe_numbers(mini_raw):
    d = describe(mini_raw)
    assert d["flyover_km"] > 0.1 and d["signalised_junctions"] == 1 and d["buildings"] == 1


def test_environment_runs_on_an_osm_map(mini_city, tmp_path):
    """The whole stack (routes, infrastructure, world, perception, rewards) works on a map built from OSM data."""
    p = tmp_path / "mini.pkl"
    mini_city.save(p)
    from astana_pilot.env import DrivingEnv, EnvConfig
    env = DrivingEnv(EnvConfig(map_path=str(p)))
    obs, _ = env.reset(seed=1)
    assert obs.shape == (351,) and np.all(np.isfinite(obs))
    for _ in range(60):
        obs, r, te, tr, info = env.step(env.action_space.sample() * 0.2)
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if te or tr:
            break
    rs = env.get_render_state()
    assert rs.city.raw.name == "mini"


def test_old_pickles_still_load(mini_city, tmp_path):
    """Maps baked before the package layout / the visual extras existed must keep loading."""
    import pickle
    p = tmp_path / "old.pkl"
    raw = mini_city.raw
    for f in ("building_meta", "green", "water", "trees", "landmarks", "half_size", "drive_side"):
        raw.__dict__.pop(f, None)
    with open(p, "wb") as f:
        pickle.dump({"raw": raw, "sdfs": mini_city.grids, "x0": mini_city.x0, "y0": mini_city.y0, "res": mini_city.res}, f)
    from astana_pilot.env.city_map import CityMap
    c = CityMap.load(p)
    assert c.raw.green == [] and c.raw.drive_side == "right"
