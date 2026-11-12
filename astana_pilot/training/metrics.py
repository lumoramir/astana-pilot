"""Episode statistics shared by the TensorBoard callback and ``training.evaluate`` (pure Python / NumPy, no torch).

Every finished episode of the environment reports an ``info`` dict (see ``DrivingEnv.step``). ``EpisodeAggregator``
turns a stream of those dicts into the numbers we care about:

* ``success``                 reached the goal
* ``collision_rate`` and ``collision/<kind>``  by what was hit: wall / parked / car / bus / pedestrian
* ``stall`` / ``lost`` / ``timeout``  other ways to fail
* ``viol/<kind>``             rule violations per episode (red_light, yellow_light, stop_sign, yield, zebra)
* ``viol_any``                share of episodes with at least one violation
* ``speeding_frac``, ``wrong_side_frac``  share of time
* ``aeb_frac``                share of time the emergency brake had to take over (a near accident)
* ``unneeded_slow_frac``      share of time crawling although nothing required it
* ``speed_kmh``, ``steer_jerk``, ``lane_dev_m``, ``progress``, ``ep_len``
* ``rew/<part>``              mean per-episode sum of every reward component
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

COLLISION_KINDS = ("wall", "parked", "car", "bus", "pedestrian")
VIOLATION_KINDS = ("red_light", "yellow_light", "stop_sign", "yield", "zebra")


def outcome_of(info: Mapping[str, Any]) -> str:
    """One word for how an episode ended: success / collision / stall / lost / timeout."""
    if info.get("is_success"):
        return "success"
    if info.get("collision"):
        return "collision"
    if info.get("stalled"):
        return "stall"
    if info.get("lost"):
        return "lost"
    return "timeout"


def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % Wilson score interval of a proportion k / n (honest error bars for small evaluation sets)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class EpisodeAggregator:
    """Collects finished-episode infos (optionally only the last ``window``) and summarises them."""

    def __init__(self, window: Optional[int] = None) -> None:
        self.window = window
        self.rows: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "outcome": outcome_of(info),
            "kind": str(info.get("collision_kind", "")) if info.get("collision") else "",
            "spawn": str(info.get("spawn_kind", "")),
            "speed_kmh": float(info.get("ep_mean_speed_kmh", 0.0)),
            "steer_jerk": float(info.get("ep_steer_jerk", 0.0)),
            "lane_dev_m": float(info.get("ep_lane_dev", 0.0)),
            "progress": float(info.get("ep_progress", 0.0)),
            "aeb_frac": float(info.get("ep_aeb_frac", 0.0)),
            "unneeded_slow_frac": float(info.get("ep_unneeded_slow_frac", 0.0)),
            "speeding_frac": float(info.get("ep_speeding_frac", 0.0)),
            "wrong_side_frac": float(info.get("ep_wrong_side_frac", 0.0)),
            "ep_len": float(info.get("ep_steps", 0.0)),
            "violations": {k: float(info.get(f"ep_{k}", 0.0)) for k in VIOLATION_KINDS},
            "parts": {k: float(v) for k, v in (info.get("ep_reward_parts") or {}).items()},
        }
        self.rows.append(row)
        if self.window and len(self.rows) > self.window:
            del self.rows[0]
        return row

    def summary(self) -> Dict[str, float]:
        n = len(self.rows)
        if n == 0:
            return {}
        out: Dict[str, float] = {"episodes": float(n)}
        cnt = defaultdict(int)
        for r in self.rows:
            cnt[r["outcome"]] += 1
        out["success"] = cnt["success"] / n
        out["collision_rate"] = cnt["collision"] / n
        out["stall"], out["lost"], out["timeout"] = cnt["stall"] / n, cnt["lost"] / n, cnt["timeout"] / n
        for k in COLLISION_KINDS:
            out[f"collision/{k}"] = sum(1 for r in self.rows if r["outcome"] == "collision" and r["kind"] == k) / n
        for k in VIOLATION_KINDS:
            out[f"viol/{k}"] = sum(r["violations"][k] for r in self.rows) / n
        out["viol_any"] = sum(1 for r in self.rows if sum(r["violations"].values()) > 0) / n
        for k in ("aeb_frac", "unneeded_slow_frac", "speeding_frac", "wrong_side_frac", "speed_kmh", "steer_jerk",
                  "lane_dev_m", "progress", "ep_len"):
            out[k] = sum(r[k] for r in self.rows) / n
        parts: Dict[str, float] = defaultdict(float)
        for r in self.rows:
            for k, v in r["parts"].items():
                parts[k] += v
        for k, v in parts.items():
            out[f"rew/{k}"] = v / n
        return out

    def success_ci(self) -> Tuple[float, float]:
        n = len(self.rows)
        return wilson(sum(1 for r in self.rows if r["outcome"] == "success"), n)


def merge(summaries: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    """Episode-weighted mean of several ``summary()`` dicts."""
    tot: Dict[str, float] = defaultdict(float)
    n = 0.0
    for s in summaries:
        w = s.get("episodes", 0.0)
        n += w
        for k, v in s.items():
            if k != "episodes":
                tot[k] += v * w
    out = {k: v / n for k, v in tot.items()} if n else {}
    out["episodes"] = n
    return out
