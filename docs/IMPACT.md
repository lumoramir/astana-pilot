# Why this matters — practical use, novelty, honest limits

## 1. The problem

Cities cannot audit autonomous (or human) driving, and they cannot test street changes before building them.

* Road traffic injury is one of the leading causes of death worldwide, and about half of the victims are pedestrians, cyclists and
  motorcyclists (WHO, *Global status report on road safety 2023* — look up the current figure and add the Kazakhstan / Hong Kong numbers
  yourself, with a citation, before you present).
* A self-driving system fails in cities in two opposite ways: it **breaks rules** (jumps a light, does not yield at a zebra) or it is
  **over-cautious** ("phantom braking", stopping for harmless objects beside the road), which causes rear-end crashes and congestion.
  Both are rarely measured separately in open projects.
* Planners decide on speed limits, signal timing and crossings with sparse crash data: crashes are rare, and waiting for them means waiting for victims.

Context for Hong Kong: in 2024 pedestrians were 42 of 88 traffic fatalities ([Police Traffic Report 2024](https://www.police.gov.hk/info/doc/statistics/traffic_report_2024_en.pdf)), and pilot licences for autonomous vehicles are being issued ([Transport Department](https://www.info.gov.hk/gia/general/202508/14/P2025081400789p.htm)). The surrogate-safety approach is established but its predictive validity is still debated ([QUT review](https://eprints.qut.edu.au/209080)); see `docs/DEFENSE_EN.md`.

## 2. What this project offers

An **open, laptop-sized digital twin of any OpenStreetMap district with a rule-compliant driving agent in it**, used in three ways.

### A. Road-safety audit (implemented: `astana audit`)
Traffic engineering uses *surrogate safety measures* — near-misses observable long before crashes (time-to-collision, hard braking,
conflicts at crossings). The agent drives thousands of random trips; each near-miss is logged with its position, normalised by the metres
driven through that place, and aggregated into a heat-map, a ranked list of hot-spots (nearest junction, street name, GPS, GeoJSON for QGIS) and
a *what-if* comparison of infrastructure variants on **identical trips** with bootstrap confidence intervals (speed limits ×k, signal green time).

*Who could use it:* a municipality's road-safety unit, a university transport lab, a student NGO. *What it replaces:* nothing — it is a cheap
screening step that tells experts where to look first and quantifies the trade-off (safety events vs. mean trip time).

### B. Research sandbox with the right metrics (implemented)
Per-rule violation counters, *justified vs. unjustified* stopping (phantom-braking rate), AEB reliance, collision by kind (pedestrian is
weighted 3×), seen/unseen district evaluation with Wilson intervals. Multi-level roads (tunnels, flyovers, ramps) with a separate collision
field per level, which hobby simulators usually skip.

### C. Education / outreach (implemented as manual mode; workshops are a suggestion)
`G` hands the car to the keyboard: the same rule monitor scores a human driver (red light, zebra, speed), the AI drives the same route for comparison.
A school workshop "can you out-drive the AI in your own district?" teaches both traffic rules and what machine learning is.

## 3. What is new — and what is not

| System | What it is | What this project adds (honestly) |
|---|---|---|
| Tesla FSD / Autopilot | closed, camera end-to-end, fleet data | Nothing to compete with. We provide an *open, auditable* alternative where every rule violation is counted and explained on screen. |
| Waymo, Apollo, Autoware | lidar+HD-map stacks, huge engineering | No HD map: only OpenStreetMap, and the observation contains no absolute coordinates, so generalisation to an unseen district is *tested*, not assumed. |
| CARLA, LGSVL | photorealistic, Unreal-based, fixed towns, heavy | Runs on a laptop GPU; builds any district in a minute; native multi-level roads; rule-first metrics. It is less photoreal by design. |
| SUMO / MATSim | traffic-flow simulators | We simulate one ego agent with perception, learning and rule monitoring inside a 3-D world, not network-level flows. |
| Surrogate-safety tools (e.g. FHWA SSAM) | post-process trajectories from a traffic simulator | The probe is a *learned, rule-bound driver* in the actual geometry of a real district, with what-if infrastructure changes. |

The specific contributions that can be defended:

1. A map pipeline that turns OSM into **multi-layer signed-distance fields** with ramps (≤ 8 % grade), signals snapped to junctions and mapped crossings.
2. An evaluation protocol that separates **rule violations**, **collisions by kind**, **needless slowness** and **safety-layer reliance**.
3. **Zero-shot district transfer** as a first-class experiment (train on district A, evaluate on B and on synthetic cities).
4. The **audit loop** (agent → near-miss events → per-km hot-spots → paired what-if with intervals).
5. A fully reproducible **overnight training recipe** (BC → DAgger → PPO) for a single 8 GB GPU.

Not new: PPO, behaviour cloning, IDM, pure pursuit, SDF collision checks, surrogate safety measures. The value is in combining them into a
rule-focused, reproducible, map-agnostic and *auditable* whole, and in making the result tangible (3-D twin, explanation overlay).

## 4. What can and cannot be claimed

| Claim | Status |
|---|---|
| The pipeline builds a multi-level 3-D district from OSM and drives in it | demonstrated (EXPO, Nurzhol; tests) |
| The rule-based expert drives ≈ 70–85 % of random trips without incident, 0 rule violations | measured (`docs/RESULTS.md`) |
| The trained network beats the expert / reaches ≈ 0 % collisions | **to be measured by your training run; no result is promised** |
| Audit hot-spots coincide with real crash locations | **not tested** — requires local crash/near-miss data; a natural collaboration with a traffic department |
| The agent is safe for real vehicles | **no** — it is a simulation study |

## 5. Roadmap (what a lab would do next)

1. Left-hand traffic and Hong Kong districts (flyover- and tunnel-rich), same tool.
2. Calibrate the world to real data: pedestrian gap acceptance, speed distributions, a handful of real near-miss reports for the audit validation.
3. Cyclists / scooters, buses with real timetables, winter friction (snow, ice) — friction is already randomised in the environment.
4. A camera perception front-end (BEV segmentation) replacing the lidar-style input; domain randomisation of the 3-D twin.
5. Publish the audit method and the benchmark as a short paper / workshop poster.
