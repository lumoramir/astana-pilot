"""Graphics-independent driving session shared by the 3-D and 2-D viewers: environment, policy, episode bookkeeping,
rule-violation events, session statistics and the telemetry record the HUD draws."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..env import DrivingEnv, EnvConfig
from ..env.infrastructure import RED, YELLOW
from ..env.lidar import ray_angles
from .hud import Telemetry, explain
from .policies import HeuristicPolicy, ModelPolicy, Policy

KIND_NAMES = {0: "wall", 1: "parked car", 2: "car", 3: "bus", 4: "pedestrian"}
VIOL_TEXT = {"red_light": "RED LIGHT VIOLATION", "yellow_light": "YELLOW LIGHT VIOLATION",
             "stop_sign": "STOP SIGN VIOLATION", "yield": "GIVE-WAY VIOLATION", "zebra": "CROSSING VIOLATION"}


class DriveSession:
    """Owns the :class:`DrivingEnv` and a policy; the viewers only draw what happens here."""

    def __init__(self, map_path: str, model: Optional[str] = None, seed: int = 0, difficulty: float = 1.0,
                 toast: Optional[Callable[[str, str], None]] = None, flash: Optional[Callable[[str], None]] = None,
                 end_pause: float = 2.4, recovery: bool = False, tour: bool = False,
                 start: Optional[str] = None, goal: Optional[str] = None, random_route: bool = False,
                 route_km: Tuple[float, float] = (0.6, 1.5), verified: bool = False, alternatives: int = 3) -> None:
        # demos start facing forward; recovery (backwards / sideways) starts are a training scenario, opt-in
        kw = dict(map_path=map_path) if recovery else dict(map_path=map_path, recovery_prob=0.0)
        if tour:                                                    # showcase: long, eventful routes through the whole district
            kw.update(route_len_easy=(1400.0, 2600.0), route_len_hard=(1800.0, 3600.0), route_pick=8)
        self.cfg = EnvConfig(**kw)
        self.env = DrivingEnv(self.cfg)
        self.env.set_difficulty(difficulty)
        self.city = self.env.city
        self.policy: Policy = ModelPolicy(model) if model else HeuristicPolicy(self.cfg)
        self.model_name = (Path(model).parent.name + "/" + Path(model).stem) if model else "rule-based expert"
        self.toast = toast or (lambda text, kind: None)
        self.flash = flash or (lambda key: None)
        self.end_pause = end_pause
        self.episode = seed
        self.manual = False
        self.manual_action = np.zeros(2, dtype=np.float32)
        self.status, self.status_kind, self.end_timer = "DRIVING", "ok", 0.0
        self.session = dict(n=0, success=0, collisions=0, violations=0, km=0.0, steps=0, speed_sum=0.0)
        self.last_info: Dict = {}
        self.prev_pose = self.cur_pose = (0.0, 0.0, 0.0, 0.0)       # x, y, theta, z (for render interpolation)
        self.prev_aeb = False
        self.paused = False
        self.hold_at_end = False          # True: stop after a trip instead of starting the next one (interactive viewer)
        self.waiting_next = False         # a trip ended and the user has not chosen what to do next
        self.last_result: Dict = {}
        # ---- navigator (point-to-point): A / B given by the user, or a random verified pair every episode
        from ..env import navigator as nav
        self._nav = nav
        self.nav_mode = bool(start is not None or goal is not None or random_route)
        self.route_km = (float(route_km[0]), float(route_km[1]))
        self.verified, self.alternatives = verified, max(1, int(alternatives))
        self.start_spec, self.goal_spec = None, None
        if start is not None:
            self.start_spec = nav.resolve_point(self.city, start, "start")
        if goal is not None:
            self.goal_spec = nav.resolve_point(self.city, goal, "goal")
        self.plan = None                                             # navigator.RoutePlan of the current episode
        self._fixed_plan = None
        self.new_episode(first=True)

    # ------------------------------------------------------------------ navigator
    def _make_plan(self):
        """RoutePlan for this episode (``None`` outside navigator mode)."""
        nav = self._nav
        if getattr(self, "forced_plan", None) is not None:           # chosen by the user on the map
            return self.forced_plan
        if self.start_spec and self.goal_spec:                       # fixed A and B: plan once, replay with new traffic
            if self._fixed_plan is None:
                self._fixed_plan = nav.plan_route(self.city, self.cfg, self.start_spec[0], self.goal_spec[0], self.alternatives,
                                                  start_name=self.start_spec[1], goal_name=self.goal_spec[1])
                if self._fixed_plan is None:
                    raise SystemExit(f"No drivable route from {self.start_spec[1]} to {self.goal_spec[1]} "
                                     "(one-way streets / disconnected graph). Pick other points.")
            return self._fixed_plan
        rng = np.random.default_rng(1_000_003 * (self.episode + 1))
        lo, hi = self.route_km[0] * 1000.0, self.route_km[1] * 1000.0
        for attempt in range(12 if self.verified else 1):
            plan = nav.random_plan(self.city, self.cfg, rng, lo, hi, self.alternatives,
                                   start=self.start_spec[0] if self.start_spec else None,
                                   goal=self.goal_spec[0] if self.goal_spec else None)
            if plan is None:
                raise SystemExit("The navigator could not find a drivable route of that length on this map. "
                                 "Try --route-km 0.3-1.0")
            if self.start_spec:
                plan.start_name = self.start_spec[1]
            if self.goal_spec:
                plan.goal_name = self.goal_spec[1]
            if not self.verified or self._expert_can_drive(plan):
                return plan
            print(f"[navigator] scenario {attempt + 1} rejected: the reference expert could not complete it", flush=True)
        return plan

    def _expert_can_drive(self, plan) -> bool:
        """'Verified' route: the rule-based expert drives the same route with the same traffic seed first."""
        if not hasattr(self, "_shadow"):
            self._shadow = DrivingEnv(self.cfg)
            self._shadow_pol = HeuristicPolicy(self.cfg)
        env, pol = self._shadow, self._shadow_pol
        env.set_difficulty(self.env.difficulty)
        obs, _ = env.reset(seed=self.episode, options={"route_nodes": plan.best.nodes, "ref": plan.best.ref})
        pol.reset()
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(pol.act(obs))
            done = term or trunc
        return bool(info.get("is_success"))

    # ------------------------------------------------------------------ episodes
    def new_episode(self, first: bool = False) -> None:
        self.waiting_next = False
        if self.nav_mode:
            self.plan = self._make_plan()
            b = self.plan.best
            self.obs, _ = self.env.reset(seed=self.episode, options={"route_nodes": b.nodes, "ref": b.ref})
            print("[navigator] " + self.plan.summary().replace("\n", "\n[navigator] "), flush=True)
        else:
            self.obs, _ = self.env.reset(seed=self.episode)
            self.plan = self._nav.plan_from_nodes(self.city, self.cfg, list(self.env._route_nodes), self.env.ref)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.status, self.status_kind, self.end_timer = "DRIVING", "ok", 0.0
        v = self.env.vehicle
        z = self.env.ref.z_at(self.env._s)
        self.prev_pose = self.cur_pose = (v.x, v.y, v.theta, z)
        self.prev_aeb = False
        self._route_vis()
        if self.nav_mode:
            b = self.plan.best
            n = len(self.plan.options)
            self.toast(f"Navigator: {n} route{'s' if n > 1 else ''} found, fastest chosen - "
                       f"{b.length / 1000:.1f} km, {b.eta_s / 60:.0f} min", "info")
        elif not first:
            self.toast(f"Episode {self.episode}: new route {self.env.ref.length:.0f} m", "info")

    def _route_vis(self) -> None:
        ref = self.env.ref
        k = max(1, len(ref.xy) // 400)                               # whole route, decimated, for the mini-map
        self.route_xy = np.asarray(ref.xy[::k], dtype=np.float32)
        self.route_s = np.asarray(ref.s[::k])
        self.alts_xy = [np.asarray(o.ref.xy[::max(1, len(o.ref.xy) // 300)], dtype=np.float32)
                        for i, o in enumerate(self.plan.options) if i != self.plan.chosen]

    def use_plan(self, plan) -> None:
        """The user chose A, B and one of the routes on the map: drive exactly that from the next episode on."""
        self.nav_mode = True
        self.forced_plan = plan

    def reroute(self, plan, choice: int = 0) -> None:
        """New destination while driving: switch the route under the moving car (traffic and time keep running)."""
        plan.chosen = int(choice)
        b = plan.best
        self.obs = self.env.reroute(b.nodes, b.ref)
        self.plan = plan
        self.nav_mode = True
        self._via_plan = None
        self.status, self.status_kind = "DRIVING", "ok"
        self._route_vis()
        v = self.env.vehicle
        self.prev_pose = self.cur_pose = (v.x, v.y, v.theta, b.ref.z_at(self.env._s))
        self.toast(f"New route to {plan.goal_name}: {max(0.0, b.length - self.env._s) / 1000:.1f} km", "info")

    def nav_info(self) -> Optional[Dict]:
        """What the navigation card shows: next manoeuvre, distance / time left, the alternatives considered."""
        if self.plan is None:
            return None
        s = self.env._s
        b = self.plan.best
        nm = b.next_maneuver(s)
        remain = max(0.0, b.length - s)
        left = b.eta_left(s)
        if getattr(self, "_via_plan", None) is not self.plan:            # the street that makes each option different
            self._via_plan, self._via = self.plan, []
            for i, o in enumerate(self.plan.options):
                others = {st for j, p in enumerate(self.plan.options) if j != i for st in p.streets}
                uniq = [st for st in o.streets if st not in others]
                self._via.append(uniq[0] if uniq else o.via(1))
        opts = [(f"{o.length / 1000:.1f} km", f"{max(1, round(o.eta_s / 60))} min", i == self.plan.chosen,
                 "fastest" if i == self.plan.chosen else f"+{max(1, round((o.eta_s - b.eta_s) / 60))} min", self._via[i])
                for i, o in enumerate(self.plan.options)]
        return dict(kind=nm[0].kind if nm else "arrive", dist=nm[1] if nm else remain,
                    text=nm[0].text() if nm else "Arrive at destination", remain_m=remain, remain_s=left,
                    eta_clock=time.strftime("%H:%M", time.localtime(time.time() + left)), options=opts,
                    start=self.plan.start_name, goal=self.plan.goal_name, nav_mode=self.nav_mode,
                    route_i0=int(np.searchsorted(self.route_s, s)))

    def _end_episode(self, info: Dict) -> None:
        s = self.session
        s["n"] += 1
        if info["is_success"]:
            s["success"] += 1
            self.status, self.status_kind = "DESTINATION REACHED", "goal"
            self.toast("Destination reached", "ok")
        else:
            reason = ("COLLISION: " + str(info.get("collision_kind", "")).upper()) if info["collision"] else \
                "STALLED" if info["stalled"] else "OFF ROUTE" if info["lost"] else "TIME OUT"
            self.status, self.status_kind = reason, "bad"
            self.toast(reason, "bad")
        s["collisions"] += int(info["collision"])
        s["violations"] += int(info.get("ep_violations", 0))
        s["km"] += info.get("ep_progress", 0.0) * self.env.ref.length / 1000.0
        s["speed_sum"] += info.get("ep_mean_speed_kmh", 0.0) * info.get("ep_steps", 0)
        s["steps"] += info.get("ep_steps", 0)
        self.end_timer = self.end_pause
        self.last_result = dict(success=bool(info["is_success"]), status=self.status,
                                km=info.get("ep_progress", 0.0) * self.env.ref.length / 1000.0,
                                minutes=info.get("ep_steps", 0) * self.cfg.dt / 60.0,
                                speed=info.get("ep_mean_speed_kmh", 0.0), violations=int(info.get("ep_violations", 0)),
                                progress=float(info.get("ep_progress", 0.0)),
                                goal=self.plan.goal_name if self.plan is not None else "")

    # ------------------------------------------------------------------ stepping
    def action(self) -> np.ndarray:
        return self.manual_action if self.manual else self.policy.act(self.obs)

    def step(self) -> None:
        env = self.env
        self.prev_pose = self.cur_pose
        self.obs, _, term, trunc, info = env.step(self.action())
        self.last_info = info
        v = env.vehicle
        self.cur_pose = (v.x, v.y, v.theta, env.ref.z_at(env._s))
        for k, val in info["violations"].items():
            if k in VIOL_TEXT and val > 0:
                self.toast(VIOL_TEXT[k], "bad")
                self.flash(k)
        if info.get("aeb") and not self.prev_aeb:
            self.toast("Emergency braking engaged", "warn")
        self.prev_aeb = bool(info.get("aeb"))
        if term or trunc:
            self._end_episode(info)

    def advance(self, dt: float, acc: float, speed_mult: float = 1.0, max_steps: int = 8) -> Tuple[float, bool]:
        """Advance the simulation by ``dt`` wall seconds. Returns ``(new accumulator, episode_finished_and_pause_over)``.
        The caller starts the next episode when the second value is True."""
        finished = False
        if self.end_timer > 0.0:
            self.end_timer -= dt
            if self.end_timer <= 0.0:
                finished = True
            return acc, finished
        if self.paused:
            return acc, False
        acc += dt * speed_mult
        n = 0
        while acc >= self.cfg.dt and self.end_timer <= 0.0 and n < max_steps:
            acc -= self.cfg.dt
            self.step()
            n += 1
        if n >= max_steps:
            acc = 0.0
        return acc, False

    # ------------------------------------------------------------------ telemetry
    def telemetry(self, fps: float = 60.0, ego_override: Optional[Tuple[float, float, float]] = None) -> Telemetry:
        env = self.env
        perc = env._perc
        ref = env.ref
        v = env.vehicle
        v_ms = v.v
        lim = perc.limit_now if perc is not None else 13.9
        vt = ref.speed_at(env._s)
        reason, rkind = explain(perc, v_ms, lim, vt)
        cons: List[Tuple[str, str, str]] = [("Speed limit", f"{lim * 3.6:.0f} km/h", "ok")]
        if perc is not None:
            if perc.ctrl_d < 90 and perc.ctrl_event is not None:
                name = {"light": "Signal", "stop": "Stop sign", "yield": "Give way"}.get(perc.ctrl_event.kind, "Control")
                st = {RED: "RED", YELLOW: "YELLOW"}.get(perc.ctrl_state, "GREEN") if perc.ctrl_event.kind == "light" else "ahead"
                cons.append((name, f"{st} {perc.ctrl_d:.0f} m", "bad" if st == "RED" else "warn" if st == "YELLOW" else "ok"))
            if perc.gap < 60 and perc.lead_kind >= 0:
                cons.append(("Leader", f"{KIND_NAMES.get(int(perc.lead_kind), 'obj')} {perc.gap:.0f} m", "warn" if perc.gap < 12 else "ok"))
            if perc.zebra_d < 60:
                cons.append(("Crossing", f"{'BUSY' if perc.zebra_busy else 'free'} {perc.zebra_d:.0f} m", "warn" if perc.zebra_busy else "ok"))
        light = None
        if perc is not None and perc.ctrl_event is not None and perc.ctrl_event.kind == "light":
            light = (int(perc.ctrl_state), float(perc.ctrl_d), 0.0)
        mon = env.monitor
        n = max(mon.steps, 1)
        viol = dict(mon.counts)
        viol["wrong_side_frac"] = mon.wrong_side_steps / n
        s = self.session
        sess = dict(n=s["n"], success=(s["success"] / s["n"]) if s["n"] else 0.0, collisions=s["collisions"],
                    violations=s["violations"] + int(sum(mon.counts.values())), km=s["km"],
                    avg_kmh=(s["speed_sum"] / s["steps"]) if s["steps"] else 0.0)
        look = ref.lookahead(env._s, np.linspace(0, 60, 22))
        stat = "ROUTE OVERVIEW: A → B" if getattr(self, "planning", False) else "PAUSED" if self.paused else ("MANUAL DRIVING" if (self.manual and self.status_kind == "ok") else self.status)
        ego = ego_override or (v.x, v.y, v.theta)
        return Telemetry(
            speed_kmh=abs(v_ms) * 3.6, gear="R" if v_ms < -0.3 else ("D" if v_ms > 0.3 else "N"), limit_kmh=lim * 3.6,
            target_kmh=vt * 3.6, steer_deg=math.degrees(v.delta), steer_cmd=float(env._last_action[0]),
            pedal=float(env._last_action[1]), e_y=env._e_y, progress=min(1.0, env._s / ref.length),
            status=stat, status_kind=self.status_kind, reason=reason, reason_kind=rkind, constraints=cons,
            aeb=bool(self.last_info.get("aeb", False)), ttc=float(perc.t_conflict) if perc else math.inf,
            gap=float(perc.gap) if perc else math.inf, light=light, ep_viol=viol, ep_speeding_frac=mon.speeding_steps / n,
            mu=env.mu, episode=self.episode, fps=fps, model_name=self.model_name,
            map_name=self.city.raw.name.replace("_", " ").upper(), session=sess, perception=perc, ego=ego,
            lidar=(perc.lidar_d * self.cfg.lidar_range) if perc is not None else None,
            lidar_angles=ray_angles(v.theta, self.cfg.n_rays), circles=perc.circles if perc is not None else None,
            circle_kind=perc.circle_kind if perc is not None else None, route_ahead=look,
            nav=self.nav_info(), route_xy=getattr(self, "route_xy", None), alts_xy=getattr(self, "alts_xy", None))


class Snap:
    """Immutable-by-convention picture of the simulation after one step: everything the renderer needs, so the
    render thread never touches the live environment."""
    __slots__ = ("id", "reset_id", "t_pub", "prev_pose", "cur_pose", "actors", "ref", "s", "perc", "theta", "v",
                 "braking", "aeb", "layer", "world_t", "tele", "paused", "plan")


def make_snap(sess: "DriveSession", step_id: int, reset_id: int, t_pub: float) -> Snap:
    env = sess.env
    v = env.vehicle
    sn = Snap()
    sn.id, sn.reset_id, sn.t_pub = step_id, reset_id, t_pub
    sn.prev_pose, sn.cur_pose = sess.prev_pose, sess.cur_pose
    sn.actors = list(env.world.actors)
    sn.ref, sn.s, sn.perc = env.ref, env._s, env._perc
    sn.theta, sn.v = v.theta, v.v
    sn.braking = bool(env._last_action[1] < -0.08 or env._aeb)
    sn.aeb = bool(env._aeb)
    sn.layer = env.ref.layer_at(env._s)
    sn.world_t = env.world.t
    sn.tele = sess.telemetry(60.0, (v.x, v.y, v.theta))
    sn.paused = sess.paused
    sn.plan = sess.plan
    return sn


class SimRunner:
    """Runs the simulation on its own thread at the requested speed and publishes a :class:`Snap` after every step.
    The render loop only reads ``runner.snap`` - a slow ``env.step`` can no longer cause a dropped frame."""

    def __init__(self, sess: "DriveSession") -> None:
        import threading
        self.sess = sess
        self.speed = 1.0
        self.reset_id = 0
        self.step_id = 0
        self._want_reset = False
        self._stop = False
        self._lock = threading.Lock()
        self.snap = make_snap(sess, 0, 0, time.perf_counter())
        self.error = None
        self.thread = threading.Thread(target=self._loop, name="sim", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop = True

    def request_new_episode(self) -> None:
        self._want_reset = True

    def republish(self) -> None:
        """Publish a fresh snapshot now (after a change made from the UI thread while the simulation is paused)."""
        self.step_id += 1
        self.snap = make_snap(self.sess, self.step_id, self.reset_id, time.perf_counter())

    def _loop(self) -> None:
        S = self.sess
        last = time.perf_counter()
        acc = 0.0
        try:
            while not self._stop:
                now = time.perf_counter()
                dt = min(now - last, 0.25)
                last = now
                if self._want_reset:
                    self._want_reset = False
                    S.waiting_next = False
                    S.episode += 1
                    S.new_episode()
                    acc = 0.0
                    self.reset_id += 1
                    self.step_id += 1
                    self.snap = make_snap(S, self.step_id, self.reset_id, time.perf_counter())
                    continue
                before = S.cur_pose
                acc, finished = S.advance(dt, acc, self.speed)
                if finished and S.hold_at_end:                # stop after the trip: the viewer shows the end menu
                    S.waiting_next = True
                    S.paused = True
                    acc = 0.0
                    self.step_id += 1
                    self.snap = make_snap(S, self.step_id, self.reset_id, time.perf_counter())
                elif finished:
                    S.episode += 1
                    S.new_episode()
                    acc = 0.0
                    self.reset_id += 1
                    self.step_id += 1
                    self.snap = make_snap(S, self.step_id, self.reset_id, time.perf_counter())
                elif S.cur_pose is not before:
                    self.step_id += 1
                    self.snap = make_snap(S, self.step_id, self.reset_id, time.perf_counter())
                # sleep until the next step is due: a polling thread would fight the render thread for the GIL
                wait = 0.02 if (S.paused or S.end_timer > 0.0) else max(0.001, min((S.cfg.dt - acc) / max(self.speed, 0.1), 0.05))
                time.sleep(wait)
        except Exception as exc:                      # never take the viewer down with the simulation
            import traceback
            traceback.print_exc()
            self.error = exc
