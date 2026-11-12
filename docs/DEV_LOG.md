# Development log — decisions and problems (edit and extend in your own words)

A decision log is the best evidence that you understand and own the project. Below are the real decisions of the "Part 4" build; add dates, what
*you* decided, and what you tested yourself.

| # | Problem | Decision | Why / evidence |
|---|---|---|---|
| 1 | The project was spread over four overlay archives with top-level packages `env`, `training`, … | One installable package `astana_pilot` with a single CLI (`astana …`), `pyproject.toml`, tests, CI | `pip install -e .`, no import clashes, one entry point for map / view / train / eval / audit |
| 2 | osmnx throws away tunnel/bridge/layer tags and pulls heavy dependencies | Own Overpass client (mirrors, back-off, cache) + own OSM → graph builder | Needed ramps, signals snapped to junctions, building heights; installs in seconds on Windows |
| 3 | Bridges and tunnels are a single OSM way, but the simulator wants `z0 → z1` per edge | Split structures into ramps + deck (≤ 8 % grade; short structures stay at ground level) | Test `test_bridge_gets_ramps_and_a_flyover_layer`; flyover renders with piers in the 3-D twin |
| 4 | The first real map had *"service" roads everywhere* (864 ways in 1.5 km) | Drop parking aisles and short unnamed courtyard links; estimate lanes per road class when untagged | Fewer dead-end traps, wider carriageways, same connectivity (largest component) |
| 5 | Expert driver hit walls on the real EXPO map (21 of 56 episodes) | Diagnosed with plots + step traces: swerve to pass a parked car ignored the kerb on one-way streets → pass offset limited by the free width | Success 46 % → 68 %; failure categories now match the documented weak spots |
| 6 | 3-D viewer must look good *and* run on a laptop | Custom GLSL (PCF shadows, fog, night windows), tile-culled static meshes, procedural textures, instanced trees, cached HUD panels | ≈ 48 fps in a 1600×900 window with 4×MSAA while a training job runs |
| 7 | HUD cost 130 ms per refresh (PIL at 2× SSAA) | Cache static panel parts, box-filter downscale, refresh one panel per frame | ≈ 1–2 ms per frame average |
| 8 | Windows: DAgger / evaluation workers crashed with `WinError 1114` (every worker imported CUDA torch) | Cap learner workers at 5 and hide the GPU from them | `astana train` and `astana eval` now run end-to-end on the real maps |
| 9 | `pip install stable-baselines3` silently replaced CUDA torch by a CPU build | Install torch **last** in `setup_windows.bat`; `astana check` prints CUDA status | Found while testing; documented in the runbook |
| 10 | "Is the city safe?" is not answerable with accuracy numbers | Road-safety audit with surrogate safety measures, per-km normalisation, paired bootstrap for what-if variants | `astana audit`; honest limits in `docs/IMPACT.md` |

## Things I (the author) personally checked or changed — fill in
- [ ] changed … in `env/config.py` and observed …
- [ ] ran the ablation `--no-bc` and found …
- [ ] opened the 3-D viewer on … and noticed …
- [ ] broke one test on purpose to see …
