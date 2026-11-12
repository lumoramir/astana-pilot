# Results — what has been measured, and where your numbers go

Everything below was produced by the code in this repository on an RTX 4060 Laptop GPU / 16-thread CPU / Windows 11 / Python 3.11 /
PyTorch 2.6 (CUDA 12.4). Commands are given so that you can reproduce each line. Trained-network results are in [section 7](#7-trained-networks-measured-2026-11-12-30-paired-trips-per-group-difficulty-10).

## 1. Maps built from OpenStreetMap (`astana map --preset …`)

| | EXPO / MEGA Silk Way | Nurzhol (government quarter) |
|---|---|---|
| area | 2.8 × 2.8 km | 2.8 × 2.8 km |
| graph | 1 160 nodes, 2 655 directed edges, 223 km | 1 250 nodes, 2 790 edges, 192 km |
| signalised junctions (from OSM) | 72 | 59 |
| mapped crossings | 457 | 239 |
| flyovers / ramps | – (ring roads, wide avenues) | 3.16 km, 26 ramp edges, 2 road levels |
| buildings (with heights) | 732 | 1 127 |
| landmarks modelled | Nur Alem sphere | Bayterek, Akorda |
| build time (SDF, 1 m grid) | 31 s | 62 s |
| pickle size | 17 MB | 34 MB |

## 2. The rule-based expert (baseline teacher), paired seeds, difficulty 1.0

| map | episodes | success | main failures |
|---|---|---|---|
| synthetic training cities (part 3 measurement) | 80 | 85 % (94 % without backward starts) | backward starts, pedestrians |
| **real EXPO, before the fix** | 56 | 46 % (26/56) | 21 wall hits: the expert swerved to pass parked cars on narrow one-way streets, into the kerb |
| **real EXPO, after** the fix (pass offset limited by road width) | 56 | **68 % (38/56)** | 7 backward / sideways starts into walls, 4 pedestrians, 5 time-outs, 1 car |
| real EXPO, 96 trips of the audit | 96 | 75 % (72/96) | same categories |
| rule violations (all runs) | – | **0** red / yellow / stop / yield events | zebra: 1 in 96 trips |

Reproduce: `astana eval --baseline-only --episodes 60 --seen-maps assets\maps\astana_expo.pkl --unseen-maps assets\maps\astana_nurzhol.pkl`.
The fix is in `viz/policies.py` (room to the left kerb is read from the `level` block of the observation).
Lesson for the report: *moving from a synthetic to a real city exposed a failure mode (swerving into a kerb) that synthetic tests never showed.*

## 3. Training pipeline on the real maps (smoke test, **not** a result)

`astana train --run-name smoke_real --maps expo.pkl,nurzhol.pkl --timesteps 30000 --demos-episodes 18 --bc-epochs 3 --dagger-rounds 1 --n-envs 6`:
demos (expert success 78 %) → BC on CUDA (val MSE 0.087) → critic warm-up (MSE 1.65 → 0.60) → DAgger → PPO 30 720 steps at ≈ 180 steps/s
(while other programs were running). `astana eval` on EXPO (seen) / Nurzhol (unseen) then ran correctly with 5 worker processes.
After 30 k steps the network drives ≈ 0–10 % of routes — as expected; the real run is 5–8 M steps (`scripts\run_night.bat 8 night1`).

## 4. Road-safety audit (illustration, expert driver, 96 trips ≈ 36 km on EXPO)

Events per 100 km: collisions 47, AEB 308, TTC < 1.5 s 255, hard braking 1 154, needless crawling 324, pedestrian < 1.2 m 5.5, zebra violations 2.7,
other rule violations 0. Weighted rate 3 311 [95 % CI 2 842 … 3 847] per 100 km. The ranked hot-spots (`reports/audit_expo/`) are mostly priority
junctions of the secondary roads in the north of the district. **96 trips are an illustration; use ≥ 500 for rankings you want to defend.**
In a 28-trip pilot, a speed-limit reduction ×0.8 did not change the weighted rate significantly (the tool reports the interval and says so).

## 5. Performance
* Scene build: 7–8 s (EXPO), cached geometry is not needed. 9 577 trees, 730 buildings, 470 lamps, tile-culled.
* 3-D viewer, 1600×900, quality *high* (4×MSAA, 2048 shadow map), real window: **≈ 48 fps** with a training job running in the background;
  headless 1080p ≈ 69 fps without HUD. `--quality low` ≈ 98 fps. HUD cost ≈ 1–2 ms per frame (cached panels, one panel per frame).
* 2-D viewer: > 140 fps.
* Simulator: ≈ 400 steps/s per core at full traffic.

## 6. Tests
`pytest` — 87 tests (61 from parts 1–3 + 26 new: OSM builder with a synthetic Overpass extract, mesh/scene/HUD/session, audit, CLI, cache). No GPU or network needed.

## 7. Trained networks (measured 2026-11-12, 30 paired trips per group, difficulty 1.0)

Seen = EXPO, Nurzhol, synthetic:0, synthetic:1 (training cities); unseen = synthetic:100–103. Same seeds for every row.

| driver | training | success seen | success unseen | collisions s / u | violation trips s / u | needless slow s / u |
|---|---|---|---|---|---|---|
| rule-based expert | – | 60.0 % [42–75] | 56.7 % [39–73] | 30.0 / 30.0 % | 0.0 / 3.3 % | 15.4 / 9.7 % |
| `night2` best | BC + DAgger + PPO 0.16 M steps | 43.3 % [27–61] | 70.0 % [52–83] | 56.7 / 26.7 % | 36.7 / 36.7 % | 19.6 / 14.9 % |
| **`night3` best = `demo_weights/astana_pilot.zip`** | fine-tune of night2, violation penalties × 2.5, selected at 0.88 M | **56.7 %** [39–73] | **53.3 %** [36–70] | 36.7 / 46.7 % | 36.7 / 23.3 % | 16.8 / 12.9 % |
| `night3` checkpoint 5.0 M (for comparison) | same run, last numbered checkpoint | 53.3 % [36–70] | 63.3 % [46–78] | 46.7 / 36.7 % | 26.7 / 36.7 % | 13.7 / 13.5 % |

The published network is the one the stability guard selected *during training*; the 5.0 M checkpoint was evaluated afterwards and is
equivalent within the intervals. Picking the published model by these test trips would tune it to the test set, so it was not done.
Learning curve of every checkpoint: `docs/img/progress_night3.png` (`astana progress night3`).

Reproduce: `astana eval --model demo_weights\astana_pilot.zip --episodes 30 --seen-maps assets\maps\astana_expo.pkl,assets\maps\astana_nurzhol.pkl,synthetic:0,synthetic:1 --unseen-maps synthetic:100,synthetic:101,synthetic:102,synthetic:103`

What happened in `night3` (6.4 h, 5.1 M steps): during the first half hour the share of training trips with a violation fell from 23 % to 8 %;
when the adaptive curriculum reached full traffic (difficulty 1.0) the training success fell to ~16 % and the stability guard restored the best
weights six times. Conclusion: a one-off violation penalty is weak next to a trip-ending collision; next: red-light crossing ends the trip, slower curriculum.

| still to do | command |
|---|---|
| ablation: no behaviour cloning | `astana train --no-bc …` |
| leave-one-district-out: train without Nurzhol, test on it | `astana train --maps assets\maps\astana_expo.pkl,synthetic:0,synthetic:1 …` then `astana eval --unseen-maps assets\maps\astana_nurzhol.pkl` |
| audit with the trained model, variants | `astana audit --model demo_weights\astana_pilot.zip --variants "baseline;limit0.8;green30"` |
