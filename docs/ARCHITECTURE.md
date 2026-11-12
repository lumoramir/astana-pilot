# Architecture

## Coordinate frames
* **Simulation:** `x` east, `y` north, metres, heading θ counter-clockwise from east; height `z` per road edge.
* **Viewer (Ursina/Panda3D `y-up-left`):** `x` east, `y` up, `z` north → sim `(x, y, h)` ⇒ viewer `(x, h, y)`; a car's local `+z` is its front, yaw = π/2 − θ.
* **Winding:** a face is front-facing when it is counter-clockwise *on screen*; the builder (`viz/mesh.py`) documents this and tests check it.

## Map pipeline (`tools/`)
`overpass.py` (mirrors, back-off, cache) → `osm_builder.py`:
1. keep drivable ways; drop parking aisles / short courtyard service roads; cut to the square;
2. split ways at junctions; oneway handling (`-1`, roundabouts, motorways);
3. **heights:** a bridge/tunnel way gets a deck at +6.5 / −6 m with linear ramps (≤ 8 % grade; short structures stay at ground level); every
   split piece becomes a directed `RawEdge(u, v, xy, z0, z1, highway, lanes, width, maxspeed)`;
4. largest connected component; traffic signals snapped to junction nodes (≥ 3 neighbours, ≤ 40 m); mapped crossings; buildings with heights;
   parks, water, trees, landmark names → `RawMap`.

`env/city_map.py` rasterises every road level into its own **signed-distance field** (positive on the road, negative outside; kerb fillets by
morphological closing) → bilinear lookups answer "did I hit the kerb?" and "how far is the wall?" in O(1). Ramps are painted into both layers.

## Simulator (`env/`)
| module | responsibility |
|---|---|
| `route.py` | shortest path → smooth lane path in Frenet coordinates, speed profile (curvature, limits), events ahead (lights, stops, zebras, limits) |
| `infrastructure.py` | junctions, two-phase signal plans, stop lines, zebras (junction legs + mid-block), speed signs |
| `world.py` | IDM cars, buses with stops, pedestrians (zebra/jay-walk/walkers), parked cars, streaming within 170 m of the agent |
| `perception.py` | the **351-D observation**: 48 rays × 4 channels + ego, lane, look-ahead, path object, signal, zebra, limit, level, 8 objects |
| `rules.py` | monitor: red/yellow/stop/yield/zebra violations, speeding, opposite lane — reports only |
| `rewards.py` | reward v4 (progress, comfort, per-violation, collision by kind, TTC, *justified-stop-aware* slowness) |
| `driving_env.py` | Gymnasium env, AEB layer, termination, `RenderState` |

## Learning (`agents/`, `training/`)
`CityExtractor` (circular Conv1d over the rays + MLP over the rest → 384 features) + [256, 256] heads, SB3 PPO with gSDE. `train.py`: demos → BC →
critic warm-up → DAgger → PPO with adaptive curriculum, autosave, resume, TensorBoard. `evaluate.py`: paired seeds, seen/unseen, Wilson intervals.

## Visualisation (`viz/`)
* `mesh.py` — NumPy mesh toolkit (quads, ribbons, boxes, cylinders, spheres, lofts, extrusion with facade UVs, instancing, tile splitting).
* `city_scene.py` — static city → named layers (asphalt, sidewalks, markings, concrete, facades, trees, signs…) split into 260 m tiles for
  frustum culling; flyover girders/piers, tunnel trenches + terrain holes; fully unit-testable without a GPU.
* `renderer.py`, `shaders.py`, `gfx.py` — Panda3D bridge: custom GLSL (sun + hemisphere ambient, 3×3 PCF shadows with slope bias, fog, night
  windows, headlights), atmosphere presets, sky dome, dynamic signal lamps.
* `actors.py` — procedural cars/buses/pedestrians and the pool that interpolates 10 Hz simulation states for smooth rendering.
* `session.py` — graphics-free episode loop shared by both viewers; `hud.py` + `canvas.py` — Pillow panels (cached static parts, one panel per
  frame); `app3d.py` — camera rig, cinematic director, recording; `viewer2d.py` — pygame twin.

## Audit (`audit/`)
`run_audit.py`: workers run episodes (with optional what-if mutation of the city), log events with positions and the exposure per 25 m cell;
aggregation → ranked hot-spots (per km), heat-map, CSV/GeoJSON, paired bootstrap for variants.

## Design decisions worth knowing
* **No absolute coordinates in the observation** → the policy cannot memorise a map; unseen districts are a fair test.
* **Per-level SDFs** instead of a 3-D collision world: cheap, exact enough for road vehicles, and it makes tunnels/flyovers trivial.
* **Rule monitor separate from the reward** → the same numbers are metrics, reward terms, HUD counters and audit events.
* **Our own Overpass client** instead of osmnx → we keep tags osmnx discards, and the install stays small.
