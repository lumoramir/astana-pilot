"""OpenStreetMap (Overpass JSON) -> :class:`RawMap` for the simulator.

Why not osmnx?  We need things osmnx throws away (bridge / tunnel / layer profiles, ramps, signals snapped to
junctions, crossings, building heights, parks, trees) and we want ``pip install`` to stay light on Windows.
Everything here is plain numpy + networkx + requests.

Pipeline::

    Overpass JSON ──► way filter (drivable, no parking aisles) ──► cut to the square ──► split at junctions
        ──► heights: bridge / tunnel ways get ramps (grade <= 8 %), deck at +6.5 m / -6 m
        ──► RawEdge list (two directed edges for two-way streets) ──► largest connected component
        ──► signals snapped to junction nodes, crossings, buildings (+height), parks, water, trees, landmarks
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from ..env.city_map import RawEdge, RawMap, polyline_length
from .overpass import bbox_around, run_query

R_EARTH = 6_371_008.8

DRIVABLE = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "service", "motorway_link", "trunk_link", "primary_link", "secondary_link",
            "tertiary_link")
SKIP_SERVICE = {"parking_aisle", "driveway", "drive-through", "emergency_access", "alley"}
LINK_BASE = {"motorway_link": "motorway", "trunk_link": "trunk", "primary_link": "primary",
             "secondary_link": "secondary", "tertiary_link": "tertiary"}

BRIDGE_H, TUNNEL_H = 6.5, -6.0          # deck heights [m]
MAX_GRADE = 0.08                        # ramp slope
MIN_LAYER_H = 3.2                       # lower decks are treated as ground (same collision layer)


# ------------------------------------------------------------------------------------ projection
class Projector:
    """Equirectangular projection around a centre (error < 0.3 % over a few km - irrelevant for a simulator)."""

    def __init__(self, lat0: float, lon0: float) -> None:
        self.lat0, self.lon0 = lat0, lon0
        self.kx = R_EARTH * math.cos(math.radians(lat0)) * math.pi / 180.0
        self.ky = R_EARTH * math.pi / 180.0

    def fwd(self, lat, lon) -> np.ndarray:
        lat, lon = np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)
        return np.stack([(lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky], axis=-1)

    def inv(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return self.lat0 + y / self.ky, self.lon0 + x / self.kx


# ------------------------------------------------------------------------------------ tag helpers
def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(v))
    return float(m.group(0).replace(",", ".")) if m else None


def _speed_ms(v: Any) -> float:
    n = _num(v)
    if n is None or str(v).strip().lower() in ("none", "signals", "walk"):
        return 0.0
    return n * (0.44704 if "mph" in str(v).lower() else 1.0 / 3.6)


def _name(tags: Dict[str, str]) -> str:
    return tags.get("name:en") or tags.get("int_name") or tags.get("name") or ""


# ------------------------------------------------------------------------------------ fetching
def fetch_osm(lat: float, lon: float, half: float, verbose: bool = True, cache: bool = True) -> Dict[str, Any]:
    """Download everything we need around ``(lat, lon)`` (square of half-size ``half`` metres)."""
    pad = half + 120.0                      # roads slightly beyond the square are cut afterwards
    bb = bbox_around(lat, lon, pad)
    hw = "|".join(DRIVABLE)
    out: Dict[str, Any] = {}

    def q(name: str, stmts: str, clause: str, optional: bool = False) -> None:
        if verbose:
            print(f"      query: {name}")
        try:
            out[name] = run_query(f"[out:json][timeout:180];({stmts});{clause};", cache=cache, verbose=verbose,
                                  retries=2 if optional else 6)["elements"]
        except RuntimeError as exc:                         # decoration only: the simulation never needs these
            if not optional:
                raise
            print(f"      (skipped optional layer '{name}': {exc})")
            out[name] = []

    q("roads", f'way["highway"~"^({hw})$"]({bb});', "out body geom")
    q("signals", f'node["highway"="traffic_signals"]({bb});node["crossing"="traffic_signals"]({bb});', "out body")
    q("crossings", f'node["highway"="crossing"]({bb});', "out body")
    q("buildings", f'way["building"]({bb});', "out tags geom")
    q("green", f'way["leisure"~"^(park|garden|playground|pitch|golf_course)$"]({bb});'
               f'way["landuse"~"^(grass|forest|meadow|recreation_ground|village_green|orchard)$"]({bb});'
               f'way["natural"~"^(wood|scrub|grassland)$"]({bb});', "out tags geom", optional=True)
    q("water", f'way["natural"="water"]({bb});way["waterway"="riverbank"]({bb});', "out tags geom", optional=True)
    q("trees", f'node["natural"="tree"]({bb});', "out body", optional=True)
    q("landmarks", f'way["name"]["tourism"~"^(attraction|museum)$"]({bb});way["man_made"~"^(tower|obelisk)$"]["name"]({bb});'
                   f'node["tourism"="attraction"]["name"]({bb});', "out tags geom", optional=True)
    return out


# ------------------------------------------------------------------------------------ roads -> graph
@dataclass
class _Way:
    wid: int
    tags: Dict[str, str]
    nodes: List[int]
    xy: np.ndarray                # (n, 2)
    oneway: bool = False
    elev: int = 0                 # +1 bridge, -1 tunnel, 0 ground


def _elevation_kind(tags: Dict[str, str]) -> int:
    br, tu = tags.get("bridge", "no"), tags.get("tunnel", "no")
    if tu not in ("no", "building_passage", ""):
        return -1
    if br not in ("no", ""):
        return 1
    return 0


def _geom_length(g) -> float:
    if not g or len(g) < 2:
        return 0.0
    lat = np.array([p["lat"] for p in g])
    lon = np.array([p["lon"] for p in g])
    dy = np.diff(lat) * 111_320.0
    dx = np.diff(lon) * 111_320.0 * math.cos(math.radians(float(lat.mean())))
    return float(np.hypot(dx, dy).sum())


def _parse_ways(elements: Iterable[dict], proj: Projector, half: float, service_min_len: float = 70.0) -> List[_Way]:
    ways: List[_Way] = []
    lim = half + 4.0
    for e in elements:
        t = e.get("tags", {})
        hw = t.get("highway")
        if hw not in DRIVABLE or e.get("type") != "way":
            continue
        if hw == "service" and t.get("service") in SKIP_SERVICE:
            continue
        if t.get("access") in ("no", "private") or t.get("motor_vehicle") in ("no", "private") or t.get("area") == "yes":
            continue
        if t.get("tunnel") == "building_passage":
            continue
        if hw == "service" and not t.get("name") and not t.get("bridge") and not t.get("tunnel") and \
                _geom_length(e.get("geometry")) < service_min_len:
            continue                                  # short courtyard / parking links: tight, dead-endy, useless
        g = e.get("geometry")
        ids = e.get("nodes")
        if not g or not ids or len(g) != len(ids):
            continue
        xy = proj.fwd([p["lat"] for p in g], [p["lon"] for p in g])
        oneway = (t.get("oneway") in ("yes", "true", "1", "-1", "reverse")
                  or t.get("junction") in ("roundabout", "circular") or hw in ("motorway", "motorway_link"))
        if t.get("oneway") in ("-1", "reverse"):
            xy, ids = xy[::-1].copy(), list(ids)[::-1]
        w = _Way(int(e["id"]), t, [int(i) for i in ids], xy, oneway, _elevation_kind(t))
        # cut to the square: keep contiguous runs of vertices inside
        inside = (np.abs(xy[:, 0]) <= lim) & (np.abs(xy[:, 1]) <= lim)
        if inside.all():
            ways.append(w)
            continue
        i, n = 0, len(xy)
        while i < n:
            if not inside[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            if j > i:
                ways.append(_Way(w.wid, t, w.nodes[i:j + 1], xy[i:j + 1].copy(), oneway, w.elev))
            i = j + 1
    return ways


def _guess_lanes(highway: str, oneway: bool) -> int:
    """Many Astana avenues carry no ``lanes`` tag; real carriageways are wider than the generic defaults.
    Lanes in the direction of travel if ``oneway`` else the total of both directions."""
    base = LINK_BASE.get(highway, highway)
    if oneway:
        return {"motorway": 3, "trunk": 3, "primary": 3, "secondary": 3, "tertiary": 2}.get(base, 2 if base != "service" else 1)
    return {"motorway": 6, "trunk": 6, "primary": 4, "secondary": 4, "tertiary": 2}.get(base, 2)


def _split_points(xy: np.ndarray, targets: Sequence[float]) -> List[Tuple[int, float]]:
    """Locate arc-length ``targets`` on a polyline -> list of (segment index, fraction)."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    out = []
    for s in targets:
        k = int(min(max(np.searchsorted(cum, s, side="right") - 1, 0), len(seg) - 1))
        out.append((k, float((s - cum[k]) / max(seg[k], 1e-9))))
    return out


def build_graph(ways: List[_Way], verbose: bool = True) -> Tuple[Dict[int, Tuple[float, float]], List[RawEdge]]:
    """Split ways at junctions, give every piece a height profile, emit directed ``RawEdge`` s."""
    use: Dict[int, int] = defaultdict(int)
    ground_nodes: Set[int] = set()
    elev_nodes: Dict[Tuple[int, int], int] = defaultdict(int)    # (node, kind) -> number of elevated ways
    for w in ways:
        for n in set(w.nodes):
            use[n] += 1
        if w.elev == 0:
            ground_nodes.update((w.nodes[0], w.nodes[-1]))
            ground_nodes.update(w.nodes)
        else:
            for n in (w.nodes[0], w.nodes[-1]):
                elev_nodes[(n, w.elev)] += 1

    node_xy: Dict[int, Tuple[float, float]] = {}
    edges: List[RawEdge] = []
    virt = -1                                                    # synthetic ids for ramp split points

    for w in ways:
        xy, ids = w.xy.copy(), list(w.nodes)
        L = polyline_length(xy)
        if L < 1.0:
            continue
        hmax = BRIDGE_H if w.elev > 0 else TUNNEL_H
        H = hmax if w.elev else 0.0
        if w.elev:
            H = math.copysign(min(abs(hmax), MAX_GRADE * L / 2.0), hmax)
            if abs(H) < MIN_LAYER_H:
                H = 0.0
        # ramps only where the elevated way meets ground (or ends in the void at the cut)
        def ramp_at(node: int) -> float:
            if H == 0.0:
                return 0.0
            shared = elev_nodes.get((node, w.elev), 0) >= 2 and node not in ground_nodes
            return 0.0 if shared else abs(H) / MAX_GRADE
        r0, r1 = ramp_at(ids[0]), ramp_at(ids[-1])

        def height(s: float) -> float:
            if H == 0.0:
                return 0.0
            a = 1.0 if r0 <= 0.0 else min(1.0, s / r0)
            b = 1.0 if r1 <= 0.0 else min(1.0, (L - s) / r1)
            return H * min(a, b)

        # insert virtual vertices at the ramp ends
        extra = [s for s in ((r0, L - r1) if H != 0.0 else ()) if 1.0 < s < L - 1.0]
        if extra:
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            for s in sorted(extra, reverse=True):
                k, f = _split_points(xy, [s])[0]
                p = xy[k] + f * (xy[k + 1] - xy[k])
                xy = np.insert(xy, k + 1, p, axis=0)
                ids.insert(k + 1, virt)
                virt -= 1
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]

        key = [0] + [i for i in range(1, len(ids) - 1) if use.get(ids[i], 0) >= 2 or ids[i] < 0] + [len(ids) - 1]
        hw = w.tags["highway"]
        lanes = int(_num(w.tags.get("lanes")) or 0)
        width = float(_num(w.tags.get("width")) or 0.0)
        if lanes == 0 and width == 0.0:
            lanes = _guess_lanes(hw, w.oneway)
        vmax = _speed_ms(w.tags.get("maxspeed"))
        name = _name(w.tags)
        for a, b in zip(key[:-1], key[1:]):
            piece = xy[a:b + 1]
            if polyline_length(piece) < 0.5 or ids[a] == ids[b]:
                continue
            u, v = ids[a], ids[b]
            node_xy[u], node_xy[v] = tuple(piece[0]), tuple(piece[-1])
            z0, z1 = height(float(cum[a])), height(float(cum[b]))
            edges.append(RawEdge(u, v, piece.copy(), hw, lanes, w.oneway, width, vmax, z0, z1, name))
            if not w.oneway:
                edges.append(RawEdge(v, u, piece[::-1].copy(), hw, lanes, False, width, vmax, z1, z0, name))
    if verbose:
        print(f"      {len(edges)} directed edges from {len(ways)} ways")
    return node_xy, edges


def largest_component(node_xy: Dict[int, Tuple[float, float]], edges: List[RawEdge]):
    G = nx.Graph()
    for e in edges:
        G.add_edge(e.u, e.v)
    if G.number_of_nodes() == 0:
        return node_xy, edges
    comp = max(nx.connected_components(G), key=len)
    edges = [e for e in edges if e.u in comp and e.v in comp]
    return {n: p for n, p in node_xy.items() if n in comp}, edges


# ------------------------------------------------------------------------------------ signals / crossings
def snap_signals(node_xy: Dict[int, Tuple[float, float]], edges: List[RawEdge], osm_nodes: List[dict],
                 proj: Projector, radius: float = 40.0) -> Optional[Set[int]]:
    """Map OSM traffic-signal nodes to graph junction nodes (>= 3 neighbours) within ``radius`` metres."""
    if not osm_nodes:
        return None
    nbrs: Dict[int, Set[int]] = defaultdict(set)
    for e in edges:
        if abs(e.z0) < 1.0 and abs(e.z1) < 1.0:
            nbrs[e.u].add(e.v)
            nbrs[e.v].add(e.u)
    junc = [n for n, s in nbrs.items() if len(s) >= 3 and n in node_xy]
    if not junc:
        return None
    jxy = np.array([node_xy[n] for n in junc])
    out: Set[int] = set()
    for e in osm_nodes:
        p = proj.fwd(e["lat"], e["lon"])
        d = np.hypot(jxy[:, 0] - p[0], jxy[:, 1] - p[1])
        k = int(d.argmin())
        if d[k] <= radius:
            out.add(junc[k])
    return out or None


# ------------------------------------------------------------------------------------ polygons
def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Douglas-Peucker simplification of an open polyline."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        ab = pts[b] - pts[a]
        L = float(np.hypot(*ab))
        seg = pts[a + 1:b] - pts[a]
        d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L if L > 1e-9 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(d.argmax())
        if d[k] > eps:
            keep[a + 1 + k] = True
            stack += [(a, a + 1 + k), (a + 1 + k, b)]
    return pts[keep]


def _ring(el: dict, proj: Projector, eps: float = 0.4) -> Optional[np.ndarray]:
    g = el.get("geometry")
    if not g or len(g) < 4:
        return None
    xy = proj.fwd([p["lat"] for p in g], [p["lon"] for p in g])
    if np.hypot(*(xy[0] - xy[-1])) > 0.5:
        return None
    xy = rdp(xy[:-1], eps)
    return xy if len(xy) >= 3 else None


def _area(r: np.ndarray) -> float:
    return 0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))


_LEVELS_BY_TYPE = {"apartments": 9, "residential": 7, "house": 2, "detached": 2, "terrace": 3, "dormitory": 6,
                   "commercial": 4, "office": 8, "retail": 3, "supermarket": 2, "hotel": 8, "public": 4,
                   "school": 3, "kindergarten": 2, "university": 5, "hospital": 5, "industrial": 2, "warehouse": 2,
                   "garage": 1, "garages": 1, "parking": 3, "shed": 1, "roof": 1, "hut": 1, "service": 1,
                   "church": 2, "mosque": 2, "civic": 4, "government": 5, "stadium": 4, "construction": 4,
                   "yes": 5}


def building_height(tags: Dict[str, str], area: float) -> Tuple[float, int]:
    """``(height m, levels)`` from tags, falling back to a type-based guess."""
    h = _num(tags.get("height"))
    lv = _num(tags.get("building:levels"))
    kind = tags.get("building", "yes")
    if h is None:
        if lv is None:
            lv = float(_LEVELS_BY_TYPE.get(kind, 5))
            if kind == "yes" and area > 1500:
                lv = 8.0
            if area < 60:
                lv = min(lv, 1.0)
        h = lv * 3.3 + 1.0
    elif lv is None:
        lv = max(1.0, round(h / 3.3))
    return float(min(max(h, 3.0), 160.0)), int(lv)


# ------------------------------------------------------------------------------------ landmarks
LANDMARK_PATTERNS = [
    ("nur_alem", re.compile(r"nur\s*alem|нұр\s*әлем|нур\s*алем|kazakhstan pavilion", re.I)),
    ("khan_shatyr", re.compile(r"khan\s*shatyr|хан\s*шатыр", re.I)),
    ("bayterek", re.compile(r"bayterek|baiterek|байтерек", re.I)),
    ("akorda", re.compile(r"akorda|ақорда|ак\s*орда", re.I)),
    ("mega_silk_way", re.compile(r"mega\s*silk\s*way", re.I)),
    ("peace_palace", re.compile(r"palace of peace|дворец мира", re.I)),
]


def find_landmarks(osm: Dict[str, Any], proj: Projector, buildings_meta: List[dict],
                   buildings: List[Tuple[np.ndarray, float]]) -> List[dict]:
    out: List[dict] = []
    seen: Set[str] = set()
    for el in list(osm.get("landmarks", [])):
        t = el.get("tags", {})
        nm = _name(t) + " " + t.get("name", "")
        for key, pat in LANDMARK_PATTERNS:
            if key in seen or not pat.search(nm):
                continue
            if el["type"] == "node":
                c = proj.fwd(el["lat"], el["lon"])
            else:
                g = el.get("geometry") or []
                if not g:
                    continue
                c = proj.fwd([p["lat"] for p in g], [p["lon"] for p in g]).mean(axis=0)
            out.append({"kind": key, "name": _name(t) or key, "xy": (float(c[0]), float(c[1]))})
            seen.add(key)
    for (ring, h), meta in zip(buildings, buildings_meta):
        nm = meta.get("name", "")
        for key, pat in LANDMARK_PATTERNS:
            if key in seen or not nm or not pat.search(nm):
                continue
            c = ring.mean(axis=0)
            out.append({"kind": key, "name": nm, "xy": (float(c[0]), float(c[1])),
                        "radius": float(np.sqrt(abs(_area(ring)) / math.pi)), "height": float(h)})
            seen.add(key)
    return out


# ------------------------------------------------------------------------------------ top level
def build_rawmap(osm: Dict[str, Any], center: Tuple[float, float], half: float, name: str,
                 verbose: bool = True) -> RawMap:
    lat, lon = center
    proj = Projector(lat, lon)
    ways = _parse_ways(osm["roads"], proj, half)
    if not ways:
        raise RuntimeError("No drivable roads in the downloaded area - check --center / --half-size.")
    node_xy, edges = build_graph(ways, verbose)
    node_xy, edges = largest_component(node_xy, edges)
    edges = [e for e in edges if e.u != e.v or polyline_length(e.xy) > 8.0]      # drop degenerate loops
    used = {e.u for e in edges} | {e.v for e in edges}
    node_xy = {n: p for n, p in node_xy.items() if n in used}

    signal_nodes = snap_signals(node_xy, edges, osm.get("signals", []), proj)
    crossing_pts = [tuple(float(c) for c in proj.fwd(e["lat"], e["lon"])) for e in osm.get("crossings", [])]
    crossing_pts = [p for p in crossing_pts if abs(p[0]) <= half and abs(p[1]) <= half]

    buildings, meta = [], []
    lim = half + 60.0
    for el in osm.get("buildings", []):
        ring = _ring(el, proj)
        if ring is None or np.abs(ring).max() > lim:
            continue
        a = _area(ring)
        if abs(a) < 18.0:
            continue
        if a < 0:
            ring = ring[::-1].copy()
        t = el.get("tags", {})
        h, lv = building_height(t, abs(a))
        buildings.append((ring, h))
        meta.append({"type": t.get("building", "yes"), "name": _name(t), "levels": lv,
                     "colour": t.get("building:colour") or t.get("colour") or "",
                     "roof": t.get("roof:shape", "")})

    def polys(key: str) -> List[np.ndarray]:
        out = []
        for el in osm.get(key, []):
            r = _ring(el, proj, 1.0)
            if r is not None and abs(_area(r)) > 80.0 and np.abs(r).max() < lim + 100:
                out.append(r)
        return out

    trees = np.array([proj.fwd(e["lat"], e["lon"]) for e in osm.get("trees", [])], dtype=np.float64).reshape(-1, 2)
    trees = trees[(np.abs(trees[:, 0]) < lim) & (np.abs(trees[:, 1]) < lim)] if len(trees) else trees
    raw = RawMap(node_xy, edges, buildings, (lat, lon), name, signal_nodes, crossing_pts)
    raw.building_meta = meta
    raw.green = polys("green")
    raw.water = polys("water")
    raw.trees = trees
    raw.landmarks = find_landmarks(osm, proj, meta, buildings)
    raw.half_size = float(half)
    return raw


def describe(raw: RawMap) -> Dict[str, Any]:
    """Headline numbers of a map (used by the README / pitch)."""
    from ..env.city_map import edge_layer_set
    km = sum(polyline_length(e.xy) for e in raw.edges) / 1000.0
    bridge = sum(polyline_length(e.xy) for e in raw.edges if e.z0 > 3.0 and e.z1 > 3.0) / 1000.0
    tunnel = sum(polyline_length(e.xy) for e in raw.edges if e.z0 < -3.0 and e.z1 < -3.0) / 1000.0
    ramps = sum(1 for e in raw.edges if len(edge_layer_set(e)) > 1)
    classes: Dict[str, int] = defaultdict(int)
    for e in raw.edges:
        classes[e.highway] += 1
    return {"name": raw.name, "nodes": len(raw.nodes), "directed_edges": len(raw.edges), "directed_km": round(km, 1),
            "flyover_km": round(bridge, 2), "tunnel_km": round(tunnel, 2), "ramp_edges": ramps,
            "signalised_junctions": None if raw.signal_nodes is None else len(raw.signal_nodes),
            "crossings_mapped": len(raw.crossing_pts or []), "buildings": len(raw.buildings),
            "landmarks": [m["kind"] for m in getattr(raw, "landmarks", [])], "road_classes": dict(classes)}
