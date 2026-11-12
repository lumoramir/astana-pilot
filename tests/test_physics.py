"""Unit tests for the pure-NumPy core: vehicle dynamics, lidar, reward."""
import math

import numpy as np

from astana_pilot.env.config import RewardWeights, VehicleParams
from astana_pilot.env.lidar import cast_rays
from astana_pilot.env.rewards import compute_reward
from astana_pilot.env.vehicle import VehicleState, step_vehicle


def _run(state, steer, pedal, seconds, p=VehicleParams(), mu=1.0, drag=0.0, dt=0.1):
    for _ in range(int(seconds / dt)):
        step_vehicle(state, steer, pedal, dt, p, mu, drag)
    return state


def test_straight_line_motion():
    s = VehicleState(0, 0, 0.0, 5.0, 0.0)
    _run(s, 0.0, 0.0, 1.0)
    assert abs(s.y) < 1e-9 and s.x > 4.0


def test_yaw_rate_matches_bicycle_model():
    """omega = (v / l_r) * sin(beta), beta = atan(l_r / L * tan(delta))."""
    p = VehicleParams()
    s = VehicleState(0, 0, 0.0, 3.0, p.max_steer)
    step_vehicle(s, 1.0, 0.0, 0.01, p, mu=1.0, drag=0.0)
    beta = math.atan(p.rear_length / p.wheelbase * math.tan(s.delta))
    assert abs(s.yaw_rate - s.v / p.rear_length * math.sin(beta)) < 1e-9


def test_friction_limits_steering():
    p = VehicleParams()
    s = VehicleState(0, 0, 0.0, 18.0, 0.0)
    _run(s, 1.0, 0.0, 3.0, p, mu=0.5)
    assert s.v**2 * math.tan(s.delta) / p.wheelbase <= 0.5 * p.gravity + 1e-6


def test_steering_is_rate_limited():
    p = VehicleParams()
    s = VehicleState()
    step_vehicle(s, 1.0, 0.0, 0.1, p, 1.0, 0.0)
    assert abs(s.delta) <= p.max_steer_rate * 0.1 + 1e-12      # cannot snap to full lock


def test_throttle_has_inertia():
    p = VehicleParams()
    s = VehicleState()
    step_vehicle(s, 0.0, 1.0, 0.1, p, 1.0, 0.0)
    assert 0.0 < s.a < p.max_accel * 0.5                       # actuator lag: not instantaneous


def test_speed_limits_forward_and_reverse():
    p, s = VehicleParams(), VehicleState()
    _run(s, 0.0, 1.0, 60.0, p, drag=0.05)
    assert 15.0 < s.v <= p.max_speed                           # can reach urban speeds (> 54 km/h)
    _run(s, 0.0, -1.0, 15.0, p, drag=0.05)                     # brake ... then reverse gear
    assert -p.max_reverse_speed - 1e-9 <= s.v < -1.0


def test_brake_stops_without_reversing_and_distance_is_realistic():
    p = VehicleParams()
    s = VehicleState(0, 0, 0.0, 14.0, 0.0)
    x0, vmin = s.x, 14.0
    for _ in range(80):
        step_vehicle(s, 0.0, -1.0, 0.1, p, 1.0, 0.0)
        vmin = min(vmin, s.v)
        if s.v <= 0.05:
            break
    # first time it reaches ~0 it must not have gone negative yet
    assert vmin > -0.5
    assert 8.0 < s.x - x0 < 25.0                               # ~ v^2 / 2a = 12 m + actuator lag


def test_reverse_moves_car_backwards_along_heading():
    s = VehicleState(0, 0, 0.0, 0.0, 0.0)
    _run(s, 0.0, -1.0, 3.0)
    assert s.v < -1.0 and s.x < -1.0


def test_lidar_corridor_distances():
    sdf = lambda p: 5.0 - np.abs(p[:, 1])                      # corridor |y| < 5
    d = cast_rays(sdf, np.array([0.0, 1.0]), 0.0, np.zeros((0, 3)), n_rays=4, max_range=40.0)
    assert np.allclose(d, [40.0, 4.0, 40.0, 6.0], atol=0.2)    # forward, left, back, right


def test_lidar_circle_obstacle():
    sdf = lambda p: np.full(len(p), 100.0)
    circ = np.array([[20.0, 0.0, 2.0]])
    d = cast_rays(sdf, np.array([0.0, 0.0]), 0.0, circ, n_rays=4, max_range=40.0)
    assert abs(d[0] - 18.0) < 1e-6 and d[1] == 40.0


def _r(**kw):
    base = dict(progress=1.0, v=10.0, v_along=10.0, v_target=10.0, e_y=0.0, heading_err=0.0,
                steer_cmd=0.0, prev_steer_cmd=0.0, thr_cmd=0.0, prev_thr_cmd=0.0, delta_change=0.0,
                a_lat=0.0, obstacle_clearance=50.0, v_max=22.0)
    base.update(kw)
    return compute_reward(RewardWeights(), **base)


def test_reward_penalises_steering_flips_hard():
    r_smooth, _ = _r()
    r_wobble, parts = _r(steer_cmd=1.0, prev_steer_cmd=-1.0, delta_change=0.08)
    assert r_wobble < r_smooth - 5.0 and parts["steer_rate"] < -5.0


def test_reward_prefers_target_speed_and_lane_centre():
    good, _ = _r()
    slow, _ = _r(v_along=2.0, progress=0.2)
    off, _ = _r(e_y=2.0)
    assert good > slow and good > off


def test_reward_terminal_terms():
    assert _r(collided=True)[1]["collision"] == -RewardWeights().collision_penalty
    assert _r(reached_goal=True)[1]["goal"] == RewardWeights().goal_bonus
    assert _r(v=-3.0, v_along=-3.0)[1]["reverse"] < 0
