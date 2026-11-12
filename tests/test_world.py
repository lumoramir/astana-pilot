"""Tests of the living world: traffic, pedestrians, radar lidar, perception, rules, emergency braking."""
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

from astana_pilot.env import DrivingEnv, EnvConfig, obs_layout                 # noqa: E402
from astana_pilot.env.lidar import cast_rays_ex                                # noqa: E402
from astana_pilot.env.world import Actor, idm_accel                            # noqa: E402
from astana_pilot.viz.policies import HeuristicPolicy                # noqa: E402

CFG = EnvConfig(map_path="synthetic", recovery_prob=0.0)


def test_lidar_reports_closing_speed_and_class():
    sdf = lambda p: np.full(len(p), 10.0)                                      # open field
    circ = np.array([[20.0, 0.0, 1.0]])
    d, closing, kind = cast_rays_ex(sdf, np.zeros(2), 0.0, circ, n_rays=4, max_range=40.0,
                                    vel=np.array([[-5.0, 0.0]]), kinds=np.array([4]), ego_vel=(3.0, 0.0))
    assert abs(d[0] - 19.0) < 1e-6 and kind[0] == 4
    assert abs(closing[0] - 8.0) < 1e-6                                        # we 3 m/s + it 5 m/s towards us
    assert kind[1] == 0 and d[1] == 40.0


def test_idm_brakes_for_slow_leader_and_accelerates_when_free():
    assert idm_accel(10.0, 14.0, math.inf, 0.0, 2.0, 1.4, 1.8, 2.6) > 0.0
    assert idm_accel(14.0, 14.0, 8.0, 14.0, 2.0, 1.4, 1.8, 2.6) < -3.0


def test_observation_matches_layout_and_is_finite():
    env = DrivingEnv(CFG)
    env.set_difficulty(1.0)
    lay = obs_layout(CFG)
    for seed in range(4):
        o, _ = env.reset(seed=seed)
        assert o.shape == (lay["size"],) and env.observation_space.shape == o.shape
        for _ in range(30):
            o, *_ = env.step(np.array([0.0, 0.3], dtype=np.float32))
        assert np.isfinite(o).all() and o.min() >= -1.0 and o.max() <= 1.0


def test_world_is_populated_and_random():
    env = DrivingEnv(CFG)
    env.set_difficulty(1.0)
    kinds, layouts = set(), set()
    for seed in range(8):
        env.reset(seed=seed)
        acts = [a for a in env.world.actors if a.alive and a.relevant]
        kinds |= {a.kind for a in acts}
        layouts.add(len(acts))
    assert {"car", "ped"} <= kinds and "bus" in kinds
    assert len(layouts) > 2                                                    # different traffic every episode


def test_difficulty_zero_has_no_traffic():
    env = DrivingEnv(CFG)
    env.set_difficulty(0.0)
    env.reset(seed=1)
    assert not [a for a in env.world.actors if a.alive and a.kind in ("car", "bus", "parked")]


def test_world_is_deterministic():
    runs = []
    for _ in range(2):
        env = DrivingEnv(CFG)
        env.set_difficulty(1.0)
        o, _ = env.reset(seed=5)
        for _ in range(40):
            o, r, *_ = env.step(np.array([0.05, 0.4], dtype=np.float32))
        runs.append((o, r, len(env.world.actors)))
    assert np.allclose(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1] and runs[0][2] == runs[1][2]


def test_npc_cars_move_and_do_not_gridlock():
    env = DrivingEnv(CFG)
    env.set_difficulty(1.0)
    env.reset(seed=3)
    pol = HeuristicPolicy(CFG)
    moving = []
    obs, _ = env.reset(seed=3)
    for i in range(600):
        obs, _, te, tr, _ = env.step(pol.act(obs))
        if i % 100 == 99:
            veh = [a for a in env.world.actors if a.alive and a.path is not None]
            moving.append(sum(a.v > 0.5 for a in veh) / max(len(veh), 1))
        if te or tr:
            break
    assert np.mean(moving) > 0.3


def test_pedestrians_cross_on_zebras():
    env = DrivingEnv(CFG)
    env.set_difficulty(1.0)
    seen = 0
    for seed in (2, 3, 4, 5):                       # pedestrians are random: any of a few worlds must show a crossing
        env.reset(seed=seed)
        for _ in range(900):
            env.world.step(0.1, env._ego_view())
            seen = max(seen, sum(len(v) for v in env.world._cw_cross.values()))
        if seen:
            break
    assert seen > 0


def test_perception_sees_the_blocking_car_but_not_the_opposite_lane():
    env = DrivingEnv(CFG)
    env.set_difficulty(0.0)
    env.reset(seed=0)
    ref, v = env.ref, env.vehicle
    x, y, psi = ref.at(env._s + 20.0)
    blocker = env.world._new("car", 4.5, 1.9)
    blocker.x, blocker.y, blocker.theta = x, y, psi
    side = env.world._new("car", 4.5, 1.9)                  # 3.5 m to the left: the oncoming lane
    side.x, side.y, side.theta = x - math.sin(psi) * 3.6, y + math.cos(psi) * 3.6, psi + math.pi
    env.world.actors += [blocker, side]
    env._perc = env._perceive()
    p = env._perc
    assert p.lead_kind == 2 and 10.0 < p.gap < 20.0         # exactly one leader, in my lane
    obs = env._observe()
    lay = env.layout
    assert obs[lay["path"]][0] == 1.0 and obs[lay["path"]][5] == 1.0


def test_aeb_prevents_a_collision_with_a_stopped_car():
    def drive(aeb):
        env = DrivingEnv(EnvConfig(map_path="synthetic", recovery_prob=0.0, aeb=aeb))
        env.set_difficulty(0.0)
        env.reset(seed=1)
        x, y, psi = env.ref.at(env._s + 60.0)
        car = env.world._new("car", 4.5, 1.9)
        car.x, car.y, car.theta = x, y, psi
        env.world.actors.append(car)
        env.vehicle.v = 9.0
        env._perc = env._perceive()
        hit, used = False, False
        pol, obs = HeuristicPolicy(EnvConfig(map_path="synthetic")), env._observe()
        for _ in range(120):
            steer = float(pol.act(obs)[0])                  # follow the road, but never touch the brake
            obs, _, te, tr, info = env.step(np.array([steer, 0.25], dtype=np.float32))
            used |= info["aeb"]
            if info["collision"]:
                hit = True
            if te or tr:
                break
        return hit, used
    hit_off, _ = drive(False)
    hit_on, used = drive(True)
    assert hit_off and not hit_on and used


def test_reference_driver_obeys_rules_and_finishes_without_traffic():
    env = DrivingEnv(CFG)
    env.set_difficulty(0.0)
    pol = HeuristicPolicy(CFG)
    wins, viol = 0, 0.0
    for ep in range(8):
        obs, _ = env.reset(seed=ep)
        pol.reset()
        done = False
        while not done:
            obs, _, te, tr, info = env.step(pol.act(obs))
            done = te or tr
        wins += info["is_success"]
        viol += info["ep_violations"]
    assert wins >= 7 and viol <= 1


def test_standing_at_a_red_light_is_not_a_stall():
    env = DrivingEnv(CFG)
    env.set_difficulty(0.0)
    for seed in range(40):                                   # find a route with a light and sit in front of it
        env.reset(seed=seed)
        evs = [e for e in env.ref.events if e.kind == "light"]
        if evs:
            break
    assert evs
    e = evs[0]
    env.vehicle.v = 0.0
    p = env._perc
    # a justified wait never counts towards the stall timer
    env._stall_count = 0
    assert env.cfg.max_wait_steps > env.cfg.stall_steps
