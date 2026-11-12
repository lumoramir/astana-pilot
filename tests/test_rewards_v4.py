"""Pure-NumPy/python tests of the v4 reward: what is punished, and (as important) what is NOT."""
import math

from astana_pilot.env.config import RewardWeights
from astana_pilot.env.rewards import compute_reward

W = RewardWeights()


def _r(**kw):
    base = dict(progress=1.0, v=10.0, v_along=10.0, v_target=10.0, e_y=0.0, heading_err=0.0,
                steer_cmd=0.0, prev_steer_cmd=0.0, thr_cmd=0.0, prev_thr_cmd=0.0, delta_change=0.0,
                a_lat=0.0, obstacle_clearance=50.0, v_max=22.0)
    base.update(kw)
    return compute_reward(W, **base)


def test_violations_are_punished_and_ordered():
    ok, _ = _r()
    for key, weight in (("red_light", W.red_light), ("stop_sign", W.stop_sign), ("yield", W.yield_sign),
                        ("zebra", W.zebra), ("yellow_light", W.yellow_light)):
        r, parts = _r(violations={key: 1.0})
        assert math.isclose(r, ok - weight, abs_tol=1e-9), key
    assert W.red_light > W.stop_sign > W.yield_sign > W.yellow_light
    assert _r(violations={"speed_excess": 4.0})[0] < ok - 0.5
    assert _r(violations={"speed_excess": 40.0})[0] >= ok - W.speeding * 25 - 1e-9      # capped


def test_collision_severity_by_kind():
    pen = {k: -_r(collided=True, collision_kind=k, v=8.0)[1]["collision"] for k in
           ("wall", "parked", "car", "bus", "pedestrian")}
    assert pen["pedestrian"] > pen["bus"] >= pen["car"] > pen["parked"] > pen["wall"] > 0
    assert pen["pedestrian"] > 2.5 * pen["wall"]
    # slow touch hurts less than a crash, but never below 70 %
    slow = -_r(collided=True, collision_kind="car", v=0.5)[1]["collision"]
    assert 0.7 * W.collision_penalty * W.collision_by_kind["car"] <= slow < pen["car"]
    # unknown kind (old callers): plain penalty
    assert -_r(collided=True)[1]["collision"] == W.collision_penalty


def test_aeb_is_a_near_accident():
    ok, _ = _r()
    onset, p1 = _r(aeb=True, aeb_onset=True)
    cont, p2 = _r(aeb=True)
    assert onset < cont < ok and p1["aeb"] == -(W.aeb_step + W.aeb_onset)


def test_standing_still_with_a_reason_is_free():
    # red light / pedestrian / blocked lane: stopped, nothing extra is charged
    stop = dict(v=0.0, v_along=0.0, progress=0.0, v_target=10.0, justified_stop=True, thr_cmd=-1.0, prev_thr_cmd=-1.0)
    _, parts = _r(**stop)
    assert parts["unneeded_slow"] == 0.0 and parts["unneeded_brake"] == 0.0
    # behind a stopped bus: leader close, no penalty either (and no tail-gating at v = 0)
    _, parts = _r(v=0.0, v_along=0.0, progress=0.0, lead_gap=4.0, lead_v=0.0, lead_kind=3, thr_cmd=-0.5, prev_thr_cmd=-0.5)
    assert parts["unneeded_slow"] == 0.0 and parts["tailgate"] == 0.0 and parts["ttc"] == 0.0


def test_crawling_without_reason_is_punished_braking_too():
    _, parts = _r(v=1.0, v_along=1.0, progress=0.1)
    assert parts["unneeded_slow"] < -0.05
    _, parts = _r(thr_cmd=-0.8, prev_thr_cmd=-0.8)
    assert parts["unneeded_brake"] < 0.0
    # but slowing for a red light that is close is not
    _, parts = _r(v=3.0, v_along=3.0, ctrl_d=20.0, thr_cmd=-0.8, prev_thr_cmd=-0.8)
    assert parts["unneeded_slow"] == 0.0


def test_following_gap_in_seconds_not_pixels():
    # same 12 m gap: tail-gating at 14 m/s (0.86 s) is punished, at 5 m/s (2.4 s) it is rewarded
    _, fast = _r(v=14.0, v_along=14.0, lead_gap=12.0, lead_v=14.0, lead_kind=2)
    _, slow = _r(v=5.0, v_along=5.0, v_target=14.0, lead_gap=12.0, lead_v=5.0, lead_kind=2, ctrl_d=100.0)
    assert fast["tailgate"] < -0.1 and slow["tailgate"] == 0.0 and slow["follow"] > 0.0
    # a parked car ahead is not a "leader to follow"
    _, parked = _r(v=6.0, v_along=6.0, lead_gap=5.0, lead_v=0.0, lead_kind=1)
    assert parked["tailgate"] == 0.0 and parked["follow"] == 0.0


def test_closing_on_a_leader_and_trajectory_conflict():
    _, far = _r(lead_gap=60.0, lead_v=5.0, lead_kind=2)
    _, near = _r(lead_gap=8.0, lead_v=2.0, lead_kind=2)
    assert far["ttc"] == 0.0 and near["ttc"] < -0.1
    _, c = _r(t_conflict=0.5)
    assert c["conflict"] < -0.1 and _r(t_conflict=math.inf)[1]["conflict"] == 0.0
    assert _r(t_conflict=0.5, v=0.0, v_along=0.0)[1]["conflict"] == 0.0     # standing still: no conflict


def test_passing_a_parked_car_is_cheap():
    """Partial excursion into the opposite lane: only the tiny wrong-side term; passing at 1.5 m is free."""
    ok, _ = _r()
    r, parts = _r(violations={"wrong_side": 1.0}, obstacle_clearance=1.5)
    assert ok - r <= W.wrong_side + 1e-9 and parts["proximity"] == 0.0
    assert _r(obstacle_clearance=0.2)[1]["proximity"] < 0.0                  # grazing is not free


def test_old_anti_wobble_terms_untouched():
    r_smooth, _ = _r()
    r_wobble, parts = _r(steer_cmd=1.0, prev_steer_cmd=-1.0, delta_change=0.08)
    assert r_wobble < r_smooth - 5.0 and parts["steer_rate"] == -W.steer_rate * 4.0
    assert math.isclose(parts["delta_rate"], -W.delta_rate * 0.08)


def test_hard_braking_not_charged_when_the_aeb_brakes():
    assert _r(a_long=-6.0)[1]["hard_brake"] < 0.0
    assert _r(a_long=-6.0, aeb=True)[1]["hard_brake"] == 0.0
    assert _r(a_long=-2.0)[1]["hard_brake"] == 0.0
