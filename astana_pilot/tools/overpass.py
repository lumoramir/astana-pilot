"""Tiny, dependency-light Overpass API client with mirror fail-over, back-off and an on-disk cache.

Overpass is a free community service and is often busy, so every request is cached
(``assets/osm_cache/<sha1>.json``) and retried over several mirrors with exponential back-off.
Please keep the area small and do not hammer the servers.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Sequence

MIRRORS = (
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "astana-av-student-project/1.0 (open-source research; github astana-av)"
CACHE_DIR = Path("assets/osm_cache")


def run_query(query: str, cache: bool = True, retries: int = 6, timeout: float = 100.0,
              verbose: bool = True, mirrors: Sequence[str] = MIRRORS) -> Dict[str, Any]:
    """POST an Overpass QL query, return the decoded JSON. Cached by the hash of the query text."""
    import requests

    key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.json"
    if cache and path.exists():
        if verbose:
            print(f"      (cache hit {path.name})")
        return json.loads(path.read_text(encoding="utf-8"))

    last = "no attempt"
    delay = 4.0
    for attempt in range(retries):
        for url in mirrors:
            try:
                r = requests.post(url, data={"data": query}, timeout=timeout,
                                  headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                if r.status_code == 200 and r.text.lstrip().startswith("{"):
                    data = r.json()
                    if "remark" in data and "runtime error" in str(data.get("remark", "")) and not data.get("elements"):
                        last = f"{url}: {data['remark']}"
                        continue
                    if cache:
                        CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(data), encoding="utf-8")
                    return data
                last = f"{url}: HTTP {r.status_code}"
            except Exception as exc:                      # network hiccups are expected here
                last = f"{url}: {type(exc).__name__}"
            if verbose:
                print(f"      overpass: {last}")
        if verbose:
            print(f"      all mirrors busy, retry {attempt + 1}/{retries} in {delay:.0f} s ...")
        time.sleep(delay)
        delay = min(delay * 1.7, 60.0)
    raise RuntimeError(f"Overpass is unavailable ({last}). Try again in a few minutes, or use --osm-json <file>.")


def bbox_around(lat: float, lon: float, half_m: float) -> str:
    """Overpass bbox ``south,west,north,east`` of a square of half-size ``half_m`` metres."""
    import math
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * math.cos(math.radians(lat)))
    return f"{lat - dlat:.6f},{lon - dlon:.6f},{lat + dlat:.6f},{lon + dlon:.6f}"
