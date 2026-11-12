# v3 upgrade: real Astana map, bidirectional control, smooth driving

## What changed
| Area | Before (v2) | Now (v3) |
|---|---|---|
| World | 80x80 m arena, circles | OSM road network of Astana (EXPO / MEGA Silk Way), kerbs + buildings = solid walls |
| Geometry | - | lat/lon -> UTM -> local metres; roads rasterised to a signed-distance field (SDF) |
| Sensing | 16-ray lidar vs circles/arena | 24-ray lidar vs SDF + obstacles; look-ahead lane points; lane offset; heading error |
| Task | reach a point | follow a route through junctions (shortest path on the road graph) |
| Action | steer, accel (no reverse) | steer, pedal in [-1, 1]: gas / brake / reverse gear |
| Dynamics | bicycle model | + actuator lag (inertia), rate-limited steering, rolling resistance, real braking distance |
| Speed | max 36 km/h | curvature-aware target profile up to 60 km/h (OSM `maxspeed` / road class) |
| Smoothness | 0.05*|d steer| | quadratic steering-rate penalty + previous action in obs + gSDE exploration |
| Traffic | random balls | parked cars at the kerb + oncoming cars in the opposite lane |
| Recovery | none | wrong-way / sideways spawns -> agent must reverse or U-turn |

## Pipeline
```
osmnx -> RawMap(edges, buildings) -> CityMap.build -> SDF grid (.pkl)
                                            |
        route sampler (networkx) -> ReferencePath (smoothed lane centre-line, speed profile)
                                            |
                         DrivingEnv: obs / reward / termination  -> PPO (SB3) -> viewers
```

## Reward (per step)
`+0.6*ds` progress, `+0.12*exp(-((v-v*)/3)^2)` speed tracking, `-0.2*(e_y/2)^2` lane, `-0.1*(1-cos(psi_err))`
heading, **`-1.5*(u_steer,t - u_steer,t-1)^2`** anti-wobble, `-0.6*|d delta|`, pedal-rate, lateral-g comfort,
reverse/time costs, proximity; terminal: +100 goal, -50 collision, -25 stall/lost, -10 timeout.

## Metrics logged to TensorBoard (`custom/*`)
success / collision / stall rate, `ep_mean_speed_kmh`, `ep_steer_jerk` (wobble), `ep_lane_dev`, `ep_progress`.
Compare with `python -m astana_pilot.training.evaluate --baseline` (classical pure-pursuit) vs `--model ...`.
