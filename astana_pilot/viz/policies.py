"""Controllers that map observations to actions (used by the viewers and for sanity checks)."""
from __future__ import annotations

import math
from typing import Optional, Protocol

import numpy as np

from astana_pilot.env.config import EnvConfig
from astana_pilot.env.driving_env import obs_layout


class Policy(Protocol):
    """Anything with ``act(obs) -> action``."""

    def act(self, obs: np.ndarray) -> np.ndarray: ...


class ModelPolicy:
    """Trained PPO model."""

    def __init__(self, path: str) -> None:
        from astana_pilot.agents import PPOAgent
        self._agent = PPOAgent.load(path)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._agent.predict(obs)


class HeuristicPolicy:
    """Rule-based expert driver (no learning) - v2 for part 3.

    It reads exactly the observation the RL agent gets, so it is a fair reference and the teacher for the
    behaviour-cloning warm-up. What it does:

    * lateral: pure pursuit on an *interpolated* look-ahead point (short look-ahead -> no corner cutting),
      plus a lateral offset around parked cars that is planned from the **gap to the car** (not from time),
      so the lane change is finished before the bumper arrives; it waits behind the car while an oncoming
      vehicle is in the way and drives round only into a free opposite lane;
    * longitudinal: the smallest speed allowed by every perceived constraint: limit, leader in the corridor,
      red / yellow light, stop / yield sign, zebra, predicted conflict and **pedestrians that will be in my
      corridor when I get there** (dashing / jay-walking people), nothing else - objects beside the road
      or in the opposite lane never make it brake;
    * recovery: three-point turn (reverse / forward with full lock) when the car starts facing the wrong way.
    """

    B = 3.0                 # comfortable braking used for planning [m/s^2]
    PASS_OFFSET_MARGIN = 0.55
    LD_A, LD_B, LD_MIN, LD_MAX = 0.3, 3.0, 3.5, 9.0     # look-ahead = LD_A * v + LD_B (clipped)
    BEND_GAIN, BEND_K0, BEND_MAX = 10.0, 0.03, 0.9       # outward swing in tight bends [m per 1/m]
    ZEBRA_STANDOFF = 5.0                                # [m] stop this far before the middle of a crossing
    HEADWAY = 1.1                                       # [s] time gap kept to a moving leader
    K_EY = 0.3                                          # extra steering per metre of lane error

    def __init__(self, cfg: Optional[EnvConfig] = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.lay = obs_layout(self.cfg)
        self.reset()

    def reset(self) -> None:
        self.nudge = 0.0            # current lateral offset from the route [m] (+ = left)
        self.hold_m = 0.0           # distance still to drive with the offset after the obstacle is passed
        self.rec = 0                # recovery phase: 0 off, +1 forward arc, -1 reverse arc
        self.rec_dir = 0.0          # rotation direction wanted (+1 = counter-clockwise)
        self._goal_prev = 0.0
        self._passing = False
        self._pass_off = 0.0
        self.backing = False
        self._bias = 0.0

    @staticmethod
    def _stop_speed(d: float, b: float) -> float:
        """Speed from which the car can still stop within ``d`` metres."""
        return math.sqrt(2.0 * b * max(d, 0.0))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _smooth(x: float) -> float:
        x = min(1.0, max(0.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _pursuit(self, look: np.ndarray, v: float, nudge: float) -> float:
        """Pure-pursuit steering command in [-1, 1] towards a point ``ld`` metres along the (offset) route."""
        c, vp = self.cfg, self.cfg.vehicle
        pts = np.vstack([[0.0, 0.0], look])
        # tight bend ahead: swing towards the OUTSIDE of the bend (the lane centre line of a junction turn passes
        # only a few decimetres from the inner kerb corner)
        k8 = 2.0 * look[1, 1] / (look[1, 0] ** 2 + look[1, 1] ** 2 + 1e-6)
        self._bias = -math.copysign(min(self.BEND_MAX, self.BEND_GAIN * max(abs(k8) - self.BEND_K0, 0.0)), k8)
        nudge = nudge + self._bias
        if abs(nudge) > 1e-3:
            seg = np.diff(pts, axis=0)
            ang = np.arctan2(seg[:, 1], np.maximum(seg[:, 0], 1e-3))
            ang = np.concatenate([[ang[0]], ang])
            pts = pts + np.stack([-np.sin(ang), np.cos(ang)], axis=1) * nudge * np.r_[0.0, np.ones(len(look))][:, None]
        seglen = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        ld = float(np.clip(self.LD_A * abs(v) + self.LD_B, self.LD_MIN, self.LD_MAX))
        ld = min(ld, cum[-1] - 1e-3)
        j = int(np.searchsorted(cum, ld))
        j = min(max(j, 1), len(cum) - 1)
        f = (ld - cum[j - 1]) / max(cum[j] - cum[j - 1], 1e-6)
        x, y = pts[j - 1] + f * (pts[j] - pts[j - 1])
        alpha = math.atan2(y, max(x, 0.5))
        return float(np.clip(math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(ld, 1.0)) / vp.max_steer, -1.0, 1.0))

    def _recovery(self, obs: np.ndarray, he: float, v: float) -> Optional[np.ndarray]:
        """Three-point turn. Returns an action while the car must turn round, else ``None``."""
        lay, c = self.lay, self.cfg
        lid = obs[lay["lidar"]] * c.lidar_range
        n = len(lid)
        ang = np.arange(n) * (2.0 * math.pi / n)                        # ray 0 = straight ahead, counter-clockwise
        off = abs(c.vehicle.body_offsets[0])                                # centre of the end circles [m]
        cf, cr = np.cos(ang), -np.cos(ang)
        d_front = np.sqrt(np.maximum(lid ** 2 - 2.0 * off * lid * cf + off * off, 0.0))
        d_rear = np.sqrt(np.maximum(lid ** 2 - 2.0 * off * lid * cr + off * off, 0.0))
        front = float(d_front[cf > -0.17].min())        # clearance of the front / rear end circle to what the radar sees
        rear = float(d_rear[cr > -0.17].min())
        if self.rec == 0:
            if abs(he) > 1.25 and abs(v) < 3.0:
                self.rec = 1 if front > 4.0 else -1
                left = float(lid[1:n // 4 + 1].mean())              # free room on the left / right: turn that way
                right = float(lid[-n // 4:].mean())
                self.rec_dir = -1.0 if he > 0.0 else 1.0
                if abs(abs(he) - math.pi) < 0.5:
                    self.rec_dir = 1.0 if left > right else -1.0
            else:
                return None
        elif abs(he) < 0.35:
            self.rec = 0
            return None
        # switch direction when the way is blocked; when BOTH ways are tight (narrow street) keep going until
        # really close, otherwise the car would dither between forward and reverse
        f_lim, r_lim = 1.8, 1.8
        if front < f_lim and rear < r_lim:
            f_lim = r_lim = 1.45
        if self.rec == 1 and front < f_lim:
            self.rec = -1
        elif self.rec == -1 and rear < r_lim:
            self.rec = 1
        taper = float(np.clip(abs(he) / 0.8, 0.35, 1.0))               # ease the lock off as the car lines up
        if self.rec == 1:        # forward: steer in the rotation direction
            steer, pedal = self.rec_dir * taper, (0.5 if v < 2.0 else 0.0)
            if front < f_lim:
                pedal = -1.0
        else:                    # reverse: steer against the rotation direction
            steer, pedal = -self.rec_dir * taper, (-0.5 if v > -1.8 else 0.0)
            if rear < r_lim:
                pedal = 1.0
        return np.array([steer, pedal], dtype=np.float32)

    # ------------------------------------------------------------------ main
    def act(self, obs: np.ndarray) -> np.ndarray:
        c, vp, lay, B = self.cfg, self.cfg.vehicle, self.lay, self.B
        goal = float(obs[lay["goal"]][0])
        if goal > self._goal_prev + 0.3:             # new episode
            self.reset()
        self._goal_prev = goal
        look = obs[lay["look"]].reshape(-1, 2) * (c.lookahead[-1] + 8.0)       # car frame [m]
        v = float(obs[lay["ego"]][0]) * vp.max_speed
        lane = obs[lay["lane"]]
        he = math.atan2(float(lane[1]), float(lane[2]))
        rec = self._recovery(obs, he, v)
        if rec is not None:
            return rec
        v_t = float(obs[lay["speed"]][0]) * c.max_target_speed
        path, ctrl, zeb = obs[lay["path"]], obs[lay["ctrl"]], obs[lay["zebra"]]
        lim = float(obs[lay["limit"]][0]) * 22.2
        H = c.leader_horizon
        dt = c.dt
        objs = obs[lay["objs"]].reshape(-1, 11)
        allowed = min(v_t, lim)

        has_lead = path[0] > 0.5
        gap = path[1] * H
        v_lead = path[2] * 20.0
        lead_parked = bool(has_lead and path[14] > 0.5)

        # ---------------- parked car in my lane: wait for a free opposite lane, then pass with a planned offset
        lat_p = 0.0
        if lead_parked:
            cand = [o for o in objs if o[10] > 0.5 and o[8] > 0.5 and o[0] > 0.0]
            if cand:
                lat_p = float(min(cand, key=lambda o: o[0])[4]) * 6.0
        want = 0.0
        # room to the LEFT edge of the road at the lane path: never plan a swerve into a kerb / wall (one-way streets)
        lvl = obs[lay["level"]]
        hw_r = float(lvl[5]) * 8.0
        room_path = hw_r + float(lvl[6]) * 2.0 * hw_r + float(lane[0]) * 4.0 - 1.05 - 0.30
        can_pass = True
        if lead_parked and gap < 38.0:
            need = lat_p + 0.95 + 0.95 + self.PASS_OFFSET_MARGIN
            can_pass = need <= room_path + 0.25
            if can_pass:
                self._passing = True
                self._pass_off = float(np.clip(need, 1.4, min(3.6, max(room_path, 1.4))))
            else:
                self._passing = False                    # not enough road: wait behind it rather than hit the kerb
        ahead_free = True
        if self._passing:
            # oncoming (or crossing) moving vehicle that will be beside the parked car while I pass
            t_pass = (max(gap, 0.0) + 8.0) / max(v, 3.0)
            for o in objs:
                if o[6] > 0.5 and o[8] < 0.5 and o[7] < 2.0 and -6.0 < o[0] * 50.0 < 60.0:
                    lat_o = o[4] * 6.0
                    if lat_o > self._pass_off - 2.6 and lat_o < self._pass_off + 2.4:
                        vclose = -o[2] * 15.0                          # closing speed along my axis
                        x_o = o[0] * 50.0
                        if vclose > 1.0 and x_o / vclose < t_pass + 1.5:
                            ahead_free = False
                        elif -3.0 < x_o < 14.0 and abs(o[2] * 15.0) < 1.0:
                            ahead_free = False                         # something standing in the pass lane
            if lead_parked:
                ramp = self._smooth((24.0 - gap) / 17.0)
                if ahead_free or gap < 6.0:
                    want = self._pass_off * ramp if ahead_free else self.nudge
                else:
                    want = 0.0
                self.hold_m = 8.0
            else:
                want = self.nudge
                self.hold_m -= max(v, 0.3) * dt
                if self.hold_m <= 0.0:
                    self._passing = False
                elif self.hold_m < 6.0:
                    want = self._pass_off * self._smooth(self.hold_m / 6.0)
        # too close to start the manoeuvre: back up a little (a human would), then go round
        if self._passing and lead_parked and ahead_free and gap < 5.2 and v < 1.2 and self.nudge < 0.7 * self._pass_off:
            self.backing = True
        if self.backing:
            lid = obs[lay["lidar"]] * c.lidar_range
            n_r = len(lid)
            rear = float(lid[n_r // 2 - 3:n_r // 2 + 4].min())
            if gap >= 8.0 or rear < 3.4 or not lead_parked:
                self.backing = False
            else:
                self.nudge *= 0.9
                # reversing: steer so that the heading error shrinks (in reverse gear the sign flips)
                return np.array([float(np.clip(he / 0.6, -1.0, 1.0)), -0.5 if v > -1.8 else 0.0], dtype=np.float32)
        self.nudge += float(np.clip(want - self.nudge, -0.10, 0.10))
        steer = self._pursuit(look, v, self.nudge)
        steer = float(np.clip(steer - float(np.clip(self.K_EY * (float(lane[0]) * 4.0 - self.nudge - self._bias), -0.2, 0.2)),
                               -1.0, 1.0))

        # ---------------- longitudinal
        if has_lead:
            lead_v = max(v_lead, 0.0)
            if lead_parked:
                if ahead_free and gap < 24.0 and self.nudge > 0.6 * self._pass_off:
                    allowed = min(allowed, 6.0 if gap > 3.0 else max(v, 2.0))
                else:
                    stop_gap = 6.5 if self._passing else 3.0
                    allowed = min(allowed, self._stop_speed(gap - stop_gap, B))
            else:
                d_star = 3.0 + self.HEADWAY * v                      # desired gap: a time gap, not metres
                allowed = min(allowed, math.sqrt(lead_v ** 2 + 2.0 * B * max(gap - 3.0, 0.0)),
                              max(0.0, lead_v + 0.6 * (gap - d_star)))
        if ctrl[0] > 0.5:
            d = ctrl[1] * 80.0
            if ctrl[2] > 0.5:                                           # traffic light
                red, yellow = ctrl[7] > 0.5, ctrl[6] > 0.5
                can_pass = d < v * v / (2.0 * 3.0) + 1.0
                if red or (yellow and not can_pass):
                    allowed = min(allowed, self._stop_speed(d - 1.0, B) if d > 1.0 else 0.0)
            elif ctrl[3] > 0.5 and ctrl[10] < 0.5:                      # stop sign, not yet served
                allowed = min(allowed, self._stop_speed(d - 0.6, B) if d > 0.8 else 0.0)
            elif ctrl[4] > 0.5:                                         # yield
                if d < 30.0:
                    allowed = min(allowed, 5.0)
                if d < 14.0 and ctrl[11] > 0.0:
                    allowed = min(allowed, self._stop_speed(d - 1.0, B) if d > 1.0 else 0.0)
        if zeb[0] > 0.5:
            zd = zeb[1] * 80.0                                          # front bumper -> crossing centre
            dz = zd - self.ZEBRA_STANDOFF
            if zeb[4] > 0.0:                                            # pedestrians on the crossing
                allowed = min(allowed, self._stop_speed(dz, B) if dz > 0.5 else (0.0 if dz > -2.0 else allowed))
            elif zeb[5] > 0.0 and dz < 25.0:                            # people waiting at the kerb
                allowed = min(allowed, 5.0)
            # never stop ON a crossing: if the leader is stopped just beyond it, wait before it
            if has_lead and not lead_parked and v_lead < 4.5 and gap < zd + 8.5 and zd > 0.5:
                allowed = min(allowed, self._stop_speed(dz, B) if dz > 0.5 else 0.0)
        # pedestrians that will be in my corridor when I arrive (dashing out, jay-walking)
        for o in objs:
            if o[5] < 0.5:
                continue
            x, y = o[0] * 50.0, o[1] * 50.0
            if not (-2.0 < x < 38.0) or abs(y) > 9.0:           # includes people level with the car's side, not those already behind it
                continue
            vx_p, vy_p = o[2] * 15.0 + v, o[3] * 15.0
            half = 2.15                                                 # my half-width + safety
            if abs(vy_p) < 0.25 and x < 1.0 and abs(y) > 1.35:
                continue                    # level with / behind the car, not touching and not moving sideways: no reason to stop
            if abs(vy_p) < 0.25:
                t_in, t_out = (0.0, 99.0) if abs(y) < half else (99.0, 99.0)
            elif y * vy_p < 0.0:                                        # walking towards my line
                t_in, t_out = max(0.0, (abs(y) - half) / abs(vy_p)), (abs(y) + half) / abs(vy_p)
            else:                                                       # walking away
                t_in, t_out = (0.0, (half - abs(y)) / abs(vy_p)) if abs(y) < half else (99.0, 99.0)
            t_me = max(x - 2.25, 0.0) / max(v, 1.0)             # front bumper reaches the person's line
            t_rear = max(x + 2.25, 0.0) / max(v, 1.0)           # the WHOLE body has passed it (a crossing person hits the side)
            if t_in < t_rear + 1.0 and t_out > t_me - 1.2:
                allowed = min(allowed, self._stop_speed(x - 2.25 - 2.0, B) if x > 4.5 else 0.0)
        if path[13] > 0.0:                                              # contact predicted along my trajectory
            t_conf = 3.0 / max(path[13], 1e-3)
            if t_conf < 2.6 and not (lead_parked and ahead_free and self.nudge > 0.6 * self._pass_off):
                allowed = min(allowed, max(0.0, v - 4.0 * (2.6 - t_conf)))
        # a person walks towards the (nearly) stopped car: give room (pedestrians do not step round a standing car)
        if abs(v) < 1.0:
            for o in objs:
                if o[5] > 0.5 and 0.0 < o[0] * 50.0 < 5.2 and abs(o[1] * 50.0) < 2.3 and o[2] * 15.0 + v < -0.2:
                    lid = obs[lay["lidar"]] * c.lidar_range
                    n_r = len(lid)
                    if float(lid[n_r // 2 - 3:n_r // 2 + 4].min()) > 4.0:
                        return np.array([0.0, -0.5 if v > -1.5 else 0.0], dtype=np.float32)
        # not yet lined up with the road (after a turn-round or a swerve): stay slow
        allowed = min(allowed, 2.5 + 10.0 * max(0.0, math.cos(he)) ** 4, 5.0 if abs(float(lane[0])) * 4.0 > 1.6 else 99.0)
        pedal = float(np.clip(0.7 * (allowed - v), -1.0, 1.0)) if allowed >= v else \
            float(np.clip(0.9 * (allowed - v), -1.0, 0.0))
        return np.array([steer, pedal], dtype=np.float32)


class RandomPolicy:
    """Uniform random actions."""

    def __init__(self, action_space) -> None:
        self._space = action_space

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self._space.sample()


def build_policy(model: Optional[str], cfg: Optional[EnvConfig] = None) -> Policy:
    """Pick a policy: trained model if given, else the pure-pursuit baseline."""
    return ModelPolicy(model) if model else HeuristicPolicy(cfg)
