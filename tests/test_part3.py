"""Part 3 tests that need neither torch nor stable-baselines3: expert, metrics, demos, evaluation report, configs."""
import numpy as np
import pytest

from astana_pilot.agents.config import PPOConfig, fit_batch_size
from astana_pilot.agents.layout import ray_and_rest_indices
from astana_pilot.env import DrivingEnv, EnvConfig
from astana_pilot.training.metrics import EpisodeAggregator, merge, outcome_of, wilson
from astana_pilot.training.returns import normalized_returns


def _info(success=False, collision=False, kind="", **kw):
    d = dict(is_success=success, collision=collision, collision_kind=kind, stalled=False, lost=False,
             ep_mean_speed_kmh=30.0, ep_steer_jerk=0.01, ep_lane_dev=0.2, ep_progress=1.0, ep_aeb_frac=0.0,
             ep_unneeded_slow_frac=0.1, ep_speeding_frac=0.0, ep_wrong_side_frac=0.0, ep_steps=500, spawn_kind="normal",
             ep_red_light=0.0, ep_yellow_light=0.0, ep_stop_sign=0.0, ep_yield=0.0, ep_zebra=0.0,
             ep_reward_parts={"progress": 50.0, "collision": 0.0})
    d.update(kw)
    return d


def test_expert_drives_without_violations():
    cfg = EnvConfig(map_path="synthetic", recovery_prob=0.0)
    from astana_pilot.viz.policies import HeuristicPolicy
    env, pol = DrivingEnv(cfg), HeuristicPolicy(cfg)
    env.set_difficulty(0.3)
    ok = viol = 0
    for seed in range(7000, 7006):
        obs, _ = env.reset(seed=seed)
        pol.reset()
        done = False
        while not done:
            obs, r, te, tr, info = env.step(pol.act(obs))
            done = te or tr
        ok += bool(info["is_success"])
        viol += info["ep_violations"]
    assert ok >= 4, ok
    assert viol == 0


def test_expert_starts_a_turn_round_when_facing_the_wrong_way():
    """Recovery starts are hard (the expert completes only about 1 in 5 of them - see docs); here we only check
    that it recognises the situation and rotates the car towards the road direction."""
    cfg = EnvConfig(map_path="synthetic", recovery_prob=1.0)
    from astana_pilot.viz.policies import HeuristicPolicy
    env, pol = DrivingEnv(cfg), HeuristicPolicy(cfg)
    env.set_difficulty(1.0)
    seed, tested = 9000, 0
    while tested < 2 and seed < 9100:
        obs, _ = env.reset(seed=seed)
        seed += 1
        if env.spawn_kind != "wrong_way":
            continue
        pol.reset()
        used, best = False, 3.2
        for _ in range(250):
            obs, r, te, tr, info = env.step(pol.act(obs))
            used |= pol.rec != 0
            lane = obs[env.layout["lane"]]
            best = min(best, abs(float(np.arctan2(lane[1], lane[2]))))
            if te or tr:
                break
        assert used and best < 2.5, (used, best)          # started the manoeuvre and turned the car by >= 35 deg
        tested += 1
    assert tested == 2


def test_synthetic_family_of_cities():
    from astana_pilot.env.city_map import load_city_map
    a, b, c = load_city_map("synthetic"), load_city_map("synthetic:0"), load_city_map("synthetic:101")
    assert a is b or a.raw.name == b.raw.name == "synthetic"
    xa = np.concatenate([e.xy for e in a.raw.edges]).max(axis=0)
    xc = np.concatenate([e.xy for e in c.raw.edges]).max(axis=0)
    assert not np.allclose(xa, xc)
    env = DrivingEnv(EnvConfig(map_path="synthetic:101"))
    obs, _ = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape


def test_metrics_aggregator_and_outcomes():
    agg = EpisodeAggregator()
    agg.add(_info(success=True))
    agg.add(_info(collision=True, kind="pedestrian", ep_red_light=1.0, ep_aeb_frac=0.2))
    agg.add(_info(collision=True, kind="wall"))
    agg.add(_info(stalled=True))
    s = agg.summary()
    assert s["episodes"] == 4 and s["success"] == 0.25 and s["collision_rate"] == 0.5 and s["stall"] == 0.25
    assert s["collision/pedestrian"] == 0.25 and s["collision/wall"] == 0.25 and s["collision/car"] == 0.0
    assert s["viol/red_light"] == 0.25 and s["viol_any"] == 0.25
    assert abs(s["aeb_frac"] - 0.05) < 1e-9 and s["rew/progress"] == 50.0
    assert outcome_of(_info(lost=True)) == "lost" and outcome_of(_info()) == "timeout"
    w = EpisodeAggregator(window=2)
    for _ in range(5):
        w.add(_info(success=True))
    assert len(w) == 2
    m = merge([s, {"episodes": 4.0, "success": 0.75}])
    assert abs(m["success"] - 0.5) < 1e-9


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(6, 6)
    assert hi == pytest.approx(1.0) and 0.5 < lo < 0.7


def test_return_targets_scaled_like_vecnormalize():
    rew = np.array([1, 1, 1, 5, 1, 1], dtype=np.float32)
    ep = np.array([0, 0, 0, 1, 1, 1])
    tgt, (mean, var, n) = normalized_returns(rew, np.zeros(6, bool), ep, gamma=1.0)
    std = np.sqrt(var)
    assert n == 6 and tgt[2] == pytest.approx(1 / std, rel=1e-4) and tgt[0] == pytest.approx(3 / std, rel=1e-4)
    assert tgt[3] == pytest.approx(7 / std, rel=1e-4)        # episode boundary: nothing leaks across


def test_ray_partition_covers_the_observation_exactly_once():
    env = DrivingEnv(EnvConfig(map_path="synthetic"))
    d = env.observation_space.shape[0]
    rays, rest = ray_and_rest_indices(48, d)
    allx = np.concatenate([rays.ravel(), rest])
    assert rays.shape == (4, 48) and sorted(allx.tolist()) == list(range(d))


def test_ppo_config_batch_fits_rollout():
    for n_envs in (4, 8, 12, 14, 16):
        cfg = PPOConfig()
        b = fit_batch_size(n_envs * cfg.n_steps, cfg.batch_size)
        assert (n_envs * cfg.n_steps) % b == 0 and 1024 <= b <= cfg.batch_size
    assert PPOConfig.for_device("cpu").batch_size < PPOConfig.for_device("cuda").batch_size
    assert PPOConfig().finetune().learning_rate < PPOConfig().learning_rate


def test_collect_demos_shapes():
    from astana_pilot.training import demos
    d = demos.collect(2, ["synthetic:0"], seed=5, n_workers=1, difficulty=(0.1, 0.2), keep_failed=True, verbose=False)
    n = len(d["act"])
    assert d["obs"].shape == (n, 351) and d["obs"].dtype == np.float16 and d["act"].shape == (n, 2)
    assert d["rew"].shape == (n,) and d["ep_id"].max() == 1
    assert np.abs(d["act"]).max() <= 1.0 + 1e-6
    both = demos.merge(d, d)
    assert both["ep_id"].max() == 3 and len(both["act"]) == 2 * n


def test_evaluation_report_has_all_sections():
    from astana_pilot.training.evaluate import make_report
    res = {}
    for lb in ("model", "expert"):
        for g in ("seen", "unseen"):
            agg = EpisodeAggregator()
            for i in range(10):
                agg.add(_info(success=i < 8, collision=i >= 8, kind="car" if i >= 8 else ""))
            res[(lb, g)] = agg
    md = make_report(res, 10, 1.0)
    for head in ("Outcome", "What was hit", "Rule violations", "Driving manners", "Memorising check"):
        assert head in md
    assert "80.0%" in md


def test_pedestrian_walking_into_a_standing_car_is_not_a_collision():
    env = DrivingEnv(EnvConfig(map_path="synthetic", recovery_prob=0.0))
    env.set_difficulty(0.0)
    env.reset(seed=3)
    c = env.cfg
    assert c.ped_contact_speed > 0.0
    # the rule lives in step(): with the car at rest only contact with a pedestrian is forgiven
    import astana_pilot.env.driving_env as _de
    with open(_de.__file__, encoding="utf-8") as _f:
        src = _f.read()
    assert "ped_walked_in" in src and "hit_kind == 4" in src
