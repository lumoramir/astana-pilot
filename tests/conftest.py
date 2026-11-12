"""Shared fixtures: a tiny, fully synthetic OpenStreetMap extract (no network) shaped like the Overpass answers."""
from __future__ import annotations

import pytest

from astana_pilot.tools.osm_builder import Projector

LAT0, LON0 = 51.0895, 71.4150


def _ll(proj: Projector, x: float, y: float):
    lat, lon = proj.inv(x, y)
    return float(lat), float(lon)


def mini_osm(half: float = 700.0) -> dict:
    """A cross of two avenues with a traffic light, a one-way street, a 320 m bridge, a building, a park, one crossing."""
    proj = Projector(LAT0, LON0)
    nid = [1000]
    nodes = {}

    def node(x, y):
        nid[0] += 1
        nodes[nid[0]] = _ll(proj, x, y)
        return nid[0]

    def way(wid, pts, tags):
        ids = [node(*p) if not isinstance(p, int) else p for p in pts]
        return {"type": "way", "id": wid, "nodes": ids,
                "geometry": [{"lat": nodes[i][0], "lon": nodes[i][1]} for i in ids], "tags": tags}

    c = node(0, 0)                                                     # the junction
    ew = [node(-600, 0), node(-300, 0), c, node(300, 0), node(600, 0)]
    ns = [node(0, -600), node(0, -300), c, node(0, 300), node(0, 600)]
    bl, bm, br = node(-160, -500), node(0, -500), node(160, -500)           # the 320 m bridge
    a1, a2, b1, b2 = node(-300, -500), node(-300, -300), node(300, -500), node(300, -300)
    ways = [
        way(1, ew, {"highway": "secondary", "name": "East-West Avenue", "lanes": "4", "maxspeed": "60"}),
        way(2, ns, {"highway": "tertiary", "name": "North Street"}),
        # one-way street tagged "-1" (digitised against the direction of travel), joined to North Street
        way(3, [node(-500, 300), ns[3]], {"highway": "residential", "oneway": "-1"}),
        # a short parking aisle that must be ignored
        way(4, [node(100, 100), node(120, 100)], {"highway": "service", "service": "parking_aisle"}),
        # long bridge -> flyover layer with ramps; ground roads at both ends so the ramps have something to join
        way(5, [bl, bm, br], {"highway": "primary", "bridge": "yes", "name": "Flyover"}),
        way(6, [bl, a1, a2], {"highway": "secondary"}),
        way(7, [br, b1, b2], {"highway": "secondary"}),
        way(8, [a2, ew[1]], {"highway": "secondary"}),
        way(9, [b2, ew[3]], {"highway": "secondary"}),
    ]
    sig = {"type": "node", "id": 9001, "lat": nodes[c][0] + 0.00003, "lon": nodes[c][1], "tags": {"highway": "traffic_signals"}}
    cross = {"type": "node", "id": 9002, "lat": nodes[c][0] + 0.0004, "lon": nodes[c][1], "tags": {"highway": "crossing"}}
    sq = [(-80, 80), (-40, 80), (-40, 120), (-80, 120), (-80, 80)]
    bld = {"type": "way", "id": 7001, "tags": {"building": "apartments", "building:levels": "9", "name": "Test House"},
           "geometry": [{"lat": _ll(proj, *p)[0], "lon": _ll(proj, *p)[1]} for p in sq]}
    park = {"type": "way", "id": 7002, "tags": {"leisure": "park"},
            "geometry": [{"lat": _ll(proj, *p)[0], "lon": _ll(proj, *p)[1]} for p in [(100, 100), (300, 100), (300, 250), (100, 250), (100, 100)]]}
    return {"roads": ways, "signals": [sig], "crossings": [cross], "buildings": [bld], "green": [park], "water": [], "trees": [],
            "landmarks": []}


@pytest.fixture(scope="session")
def osm():
    return mini_osm()


@pytest.fixture(scope="session")
def mini_raw(osm):
    from astana_pilot.tools.osm_builder import build_rawmap
    return build_rawmap(osm, (LAT0, LON0), 700.0, "mini", verbose=False)


@pytest.fixture(scope="session")
def mini_city(mini_raw):
    from astana_pilot.env.city_map import CityMap
    return CityMap.build(mini_raw, res_build=1.0, res_store=1.0, curb_radius=2.0)
