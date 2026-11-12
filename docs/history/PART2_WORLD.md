# Part 2 / 5 — World: traffic, buses, pedestrians, rules, perception

Install: unzip `part2-world.zip` over the project (replace files). Part 1 must already be installed.
Check: `python -m pytest tests -q` (40 tests).  Picture of the world: `python -m astana_pilot.tools.preview_world --map synthetic`.

## New / replaced files
`env/world.py` (new), `env/perception.py` (new), `env/rules.py` (new), `env/lidar.py`, `env/config.py`,
`env/driving_env.py`, `visualization/policies.py`, `tools/preview_world.py` (new), tests.

## What the world does (every episode is different)
* Random start and goal, random friction, random light phase, random traffic. Nothing absolute is observed.
* NPC cars and 12 m buses drive their own routes with IDM car following. They stop for red / yellow lights,
  serve stop signs, give way at yield signs and let pedestrians cross. Buses stop at bus stops 8-14 s and block the lane;
  passengers walk to the door / from the door to the kerb.
* Pedestrians wait at zebras, cross on the walk signal (signalised) or when the gap is big enough (unsignalised);
  ~15 % are assertive (small gap), some jaywalk mid-block. Decorative walkers on pavements are not seen by sensors.
* Traffic streams in/out around the agent (radius 170 m), so cost stays flat on the whole of Astana.
* Roads on other levels: NPCs use tunnels and flyovers too; sensors only see actors within 2.5 m of the agent's height.

## What the agent perceives (observation 350 numbers for 48 rays)
* radar lidar: distance + closing speed + pedestrian / vehicle flag per ray (48 rays, 7.5 deg)
* `path`: object in MY corridor (gap, speed, closing speed, TTC, class: car / parked / bus / pedestrian),
  free space left/right, predicted conflict along the route and along my current trajectory, "lane blocked"
* `ctrl`: next light / stop / yield: distance, colour, time to change, required deceleration, "already stopped here", cross traffic
* `zebra`: signalised?, walk signal, people on it / waiting, where they are and which way they move
* `limit`, `level` (tunnel / flyover / grade / lane side), `objs` (8 nearest road users, relative position and velocity)
Objects beside the road or in the opposite lane do not enter `path`, so there is no reason to brake for them.

## Rules monitor (`info["violations"]`, `info["ep_*"]`)
red_light, yellow_light, stop_sign, yield, zebra, speeding, wrong_side — reported only; the reward uses them in part 3.
Standing still is "justified" (red light, pedestrian, blocked lane...) and never counts as stalling; `ep_unneeded_slow_frac`
measures crawling without reason. Safety layer: `cfg.aeb` brakes when contact is predicted within 1.3 s (`info["aeb"]`).

## Honest status
* The reward is still the v3 one. Do NOT start a long training yet: it does not punish rule violations. Part 3 does that.
* `HeuristicPolicy` is a reference driver, not the final expert. On the synthetic city it finishes ~100 % of empty-city
  routes and ~65 % at full traffic with 0 rule violations; the rest are parked-car passes and dashing pedestrians.
* NPC drivers ignore parked cars (they would otherwise queue forever behind them); parked cars only matter for the agent.
* Speed: ~400 env steps/s on one core at full traffic (v3 was faster, traffic costs ~2.5 ms/step).
* Everything was verified on the synthetic city with a gymnasium stand-in; real Astana map and PyTorch were not run here.
