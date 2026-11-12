"""Road-safety audit + CLI wiring."""
import importlib
import importlib.util
import json

import numpy as np
import pytest


def test_every_cli_command_points_to_a_module_with_main():
    from astana_pilot.cli import COMMANDS
    for cmd, mod in COMMANDS.items():
        assert importlib.util.find_spec(mod) is not None, f"{cmd} -> {mod}"
    for cmd in ("map", "audit", "eval", "view", "view2d", "train", "studio"):
        assert hasattr(importlib.import_module(COMMANDS[cmd]), "main")


def test_audit_variants_change_the_city(mini_city):
    import copy
    from astana_pilot.audit.run_audit import apply_variant
    c = copy.deepcopy(mini_city)
    before = list(c.edge_vlim)
    assert apply_variant(c, "limit0.5").startswith("speed limits")
    assert np.allclose(c.edge_vlim, np.array(before) * 0.5)
    assert apply_variant(c, "green31").startswith("signal green")
    assert c.infra.params.light_green == 31.0
    assert apply_variant(c, "baseline") == "baseline"
    with pytest.raises(ValueError):
        apply_variant(c, "teleport")


def test_audit_end_to_end_on_a_tiny_run(tmp_path):
    from astana_pilot.audit.run_audit import run
    summ = run("synthetic", "expert", ["baseline", "limit0.9"], episodes=2, difficulty=0.6, workers=1, out=str(tmp_path / "a"))
    for f in ("report.md", "risk_heatmap.png", "hotspots.csv", "hotspots.geojson", "summary.json"):
        assert (tmp_path / "a" / f).exists(), f
    assert set(summ) == {"baseline", "limit0.9"}
    s = summ["baseline"]
    assert s["trips"] == 2 and s["km"] > 0 and len(s["weighted_ci95"]) == 2
    assert "change_vs_baseline" in summ["limit0.9"]
    gj = json.loads((tmp_path / "a" / "hotspots.geojson").read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection"


def test_overpass_cache_roundtrip(tmp_path, monkeypatch):
    """A cached answer is returned without touching the network."""
    from astana_pilot.tools import overpass
    monkeypatch.setattr(overpass, "CACHE_DIR", tmp_path)
    q = "[out:json];node(1);out;"
    import hashlib
    key = hashlib.sha1(q.encode("utf-8")).hexdigest()[:16]
    (tmp_path / f"{key}.json").write_text(json.dumps({"elements": [{"id": 1}]}), encoding="utf-8")
    assert overpass.run_query(q, verbose=False)["elements"][0]["id"] == 1

