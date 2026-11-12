# Part 1 / 5 — Core: road levels, signals, zebras, signs

Files: `env/city_map.py`, `env/route.py`, `env/infrastructure.py` (new), `env/synthetic_map.py`,
`env/config.py` (+`InfraParams`), `env/driving_env.py` (layer-aware), `tests/test_levels_infra.py`.

* Every OSM edge has a height profile `z0 -> z1`. Layers: -1 tunnel (z <= -3), 0 ground, +1 flyover (z >= 3).
  Each layer has its own SDF, so a flyover never "collides" with the road under it. Ramps are in both layers.
* `CityMap.sdf(pts, layer=0)`, `CityMap.sdf_layers(pts, layers)`, `ReferencePath.layer_at(s)`, `.z_at(s)`.
* `city.infra`: signalised junctions (2-group plan green/yellow/all-red), stop lines (`light`/`yield`/`stop`),
  zebras (junction legs + mid-block), speed signs. `infra.light_phase(node, group, t)`.
* `ref.events` / `ref.events_ahead(s, horizon)`: what lies ahead on the route (consumed by Part 2 rules + obs).
* Old maps (`.pkl` with one SDF) still load; OSM `signal_nodes` override the heuristic when present (Part 4).
