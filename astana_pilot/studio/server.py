"""Astana Pilot Studio - the single front door of the project.

    python -m astana_pilot            (or: astana)            opens the Studio

A tiny local web app (standard library only, bound to 127.0.0.1) shown in an app-style window of Edge/Chrome, so the interface
can be modern, responsive and crisp on any screen - and everything it starts is an ordinary `astana ...` command
running as a child process. Results of an audit (heat-map, ranking) are shown inside the Studio itself.
"""
from __future__ import annotations

import glob
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(os.getcwd())
HERE = Path(__file__).parent
PRESETS = {"expo": ("Astana EXPO", "Nur Alem sphere, ring roads, 72 signals"),
           "nurzhol": ("Nurzhol", "Bayterek, Akorda, Ishim flyovers")}
ALLOWED_QUALITY = {"auto", "low", "medium", "high", "ultra"}
ALLOWED_LIGHT = {"day", "dusk", "night", "snow"}


class Job:
    """One child process at a time, with a rolling log."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.title = ""
        self.log: List[str] = []
        self.code: Optional[int] = None
        self.started = 0.0
        self.kind = ""

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, kind: str, title: str, args: List[str]) -> None:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        self.kind, self.title, self.log, self.code, self.started = kind, title, [], None, time.time()
        self.proc = subprocess.Popen([sys.executable, "-m", "astana_pilot"] + args, cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        noisy = ("prc(", "display:", "pnmimage", "aux display", "Known pipe", "wglGraphicsPipe", "package_folder", "asset_folder",
                 "set window", "Using primary monitor", "changed aspect")
        for line in self.proc.stdout:
            if not any(n in line for n in noisy) and line.strip():
                self.log.append(line.rstrip())
                del self.log[:-400]
        self.code = self.proc.wait()

    def stop(self) -> None:
        if self.running:
            self.proc.terminate()                                   # type: ignore[union-attr]


JOB = Job()


# ------------------------------------------------------------------------------------------------ state helpers
def _maps() -> List[Dict[str, Any]]:
    out = []
    for key, (title, sub) in PRESETS.items():
        p = f"assets/maps/astana_{key}.pkl"
        out.append({"id": key, "title": title, "sub": sub, "path": p, "ready": (ROOT / p).exists()})
    known = {m["path"] for m in out}
    for p in sorted(glob.glob(str(ROOT / "assets/maps/*.pkl"))):
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        if rel not in known:
            out.append({"id": Path(rel).stem, "title": Path(rel).stem.replace("_", " ").title(), "sub": "custom district",
                        "path": rel, "ready": True})
    out.append({"id": "synthetic", "title": "Test city", "sub": "synthetic, works offline", "path": "synthetic", "ready": True})
    return out


def _models() -> List[Dict[str, str]]:
    shipped = sorted(glob.glob(str(ROOT / "demo_weights/*.zip")))          # the published network (comes with the repo)
    found = sorted(glob.glob(str(ROOT / "runs/*/best_model.zip")) + glob.glob(str(ROOT / "runs/*/final_model.zip"))
                   + glob.glob(str(ROOT / "runs/imported/*/final_model.zip")) + glob.glob(str(ROOT / "runs/*/checkpoints/latest.zip")),
                   key=os.path.getmtime, reverse=True)

    def label(p: str) -> str:
        P = Path(p)
        if P.parent.name == "demo_weights":
            return f"{P.stem} (published)"
        if P.name == "latest.zip":
            return P.parent.parent.name + " (autosave)"
        return P.parent.name + (" (best)" if P.name == "best_model.zip" else "")
    return [{"path": os.path.relpath(p, ROOT).replace("\\", "/"), "label": label(p)} for p in shipped + found]


def _audit(map_path: str) -> Optional[Dict[str, Any]]:
    d = ROOT / f"reports/audit_{Path(map_path).stem}"
    if not (d / "summary.json").exists():
        return None
    out: Dict[str, Any] = {"summary": json.loads((d / "summary.json").read_text(encoding="utf-8")), "hotspots": []}
    csv_p = d / "hotspots.csv"
    if csv_p.exists():
        import csv
        rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))[:8]
        out["hotspots"] = [{"rank": r["rank"], "street": r["street"] or "unnamed street", "where": r["where"] or "mid-block",
                            "per_km": round(float(r["per_km"]), 1), "lat": r["lat"][:8], "lon": r["lon"][:8]} for r in rows]
    out["image"] = f"/files/{d.relative_to(ROOT).as_posix()}/risk_heatmap.png?v={int((d / 'summary.json').stat().st_mtime)}"
    out["dir"] = d.relative_to(ROOT).as_posix()
    return out


def _eval_results() -> Dict[str, Any]:
    """Latest evaluation of every network that has one: model vs. rule-based expert, on seen and unseen districts."""
    out: Dict[str, Any] = {}
    for f in glob.glob(str(ROOT / "runs/*/eval/results.json")) + glob.glob(str(ROOT / "runs/imported/*/eval/results.json")):
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        run = Path(f).parent.parent.relative_to(ROOT / "runs").as_posix()
        rows = []
        for key, v in d.items():
            if not v:
                continue
            pol, grp = key.split("/")
            rows.append({"policy": "network" if pol == "model" else "expert", "cities": grp, "n": int(v.get("episodes", 0)),
                         "success": round(100 * v.get("success", 0), 1), "collision": round(100 * v.get("collision_rate", 0), 1),
                         "violations": round(100 * v.get("viol_any", 0), 1), "kmh": round(v.get("speed_kmh", 0), 1)})
        out[run] = {"rows": sorted(rows, key=lambda r: (r["cities"] != "seen", r["policy"])), "when": int(Path(f).stat().st_mtime)}
    return out


def _import_model() -> Dict[str, Any]:
    """Native 'open file' dialog (the Studio runs on the user's own machine) -> copy the chosen network into runs/imported/."""
    import shutil
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    f = filedialog.askopenfilename(title="Choose a trained network (.zip)", filetypes=[("Network", "*.zip")])
    root.destroy()
    if not f:
        return {"ok": False, "error": "No file chosen."}
    src = Path(f)
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in src.parent.name if True) or "network"
    if src.stem not in ("final_model", "latest", "model"):
        name = f"{name}_{src.stem}"
    dst = ROOT / "runs" / "imported" / name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst / "final_model.zip")
    return {"ok": True, "model": f"runs/imported/{name}/final_model.zip"}


def state(map_path: str = "") -> Dict[str, Any]:
    maps = _maps()
    mp = map_path or next((m["path"] for m in maps if m["ready"]), "synthetic")
    return {"maps": maps, "models": _models(), "job": {"running": JOB.running, "title": JOB.title, "kind": JOB.kind, "code": JOB.code,
                                                       "log": JOB.log[-60:], "elapsed": int(time.time() - JOB.started) if JOB.started else 0},
            "audit": _audit(mp), "evals": _eval_results(), "selected": mp}


# ------------------------------------------------------------------------------------------------ actions
def _safe_map(p: str) -> str:
    if p == "synthetic" or p.startswith("synthetic:"):
        return p
    q = (ROOT / p).resolve()
    if q.parent != (ROOT / "assets/maps").resolve() or q.suffix != ".pkl":
        raise ValueError("bad map path")
    return p


def _safe_model(p: str) -> List[str]:
    if not p:
        return []
    q = (ROOT / p).resolve()
    if (ROOT / "runs").resolve() not in q.parents or q.suffix != ".zip" or not q.exists():
        raise ValueError("bad model path")
    return ["--model", p]


def run_action(req: Dict[str, Any]) -> Dict[str, Any]:
    if JOB.running and req.get("action") != "stop":
        return {"ok": False, "error": "Something is already running. Stop it first."}
    a = req.get("action")
    if a == "stop":
        JOB.stop()
        return {"ok": True}
    if a == "import_model":
        try:
            return _import_model()
        except Exception as exc:
            return {"ok": False, "error": f"Cannot open the file dialog: {exc}"}
    q = req.get("quality", "auto")
    light = req.get("light", "day")
    if q not in ALLOWED_QUALITY or light not in ALLOWED_LIGHT:
        return {"ok": False, "error": "bad option"}
    try:
        if a == "download":
            preset = req["preset"]
            if preset not in PRESETS:
                raise ValueError("unknown preset")
            JOB.start("map", f"Downloading {PRESETS[preset][0]}", ["map", "--preset", preset, "--no-preview"])
            return {"ok": True}
        if a == "custom_map":
            lat, lon = (float(v) for v in str(req["center"]).split(","))
            JOB.start("map", "Building your district", ["map", "--center", f"{lat},{lon}", "--half-size", "1200", "--name",
                                                        f"custom_{abs(int(lat * 1000))}_{abs(int(lon * 1000))}", "--no-preview"])
            return {"ok": True}
        m = _safe_map(req.get("map", "synthetic"))
        model = _safe_model(req.get("model", ""))
        if a == "view":
            JOB.start("view", "3D simulation", ["view", "--map", m, "--quality", q, "--preset", light] + model)
        elif a == "tour":
            JOB.start("view", "Grand tour", ["view", "--map", m, "--quality", q, "--preset", light, "--tour"] + model)
        elif a == "navigate":                                       # random verified A -> B, network drives
            import random
            JOB.start("view", "Navigator A → B", ["view", "--map", m, "--quality", q, "--preset", light, "--random-route",
                                                      "--verified", "--seed", str(random.randint(0, 99999))] + model)
        elif a == "pick_route":                                     # 3-D with the navigator map open: click A and B
            JOB.start("view", "Navigator A → B", ["view", "--map", m, "--quality", q, "--preset", light, "--pick"] + model)
        elif a == "observer":
            JOB.start("view", "Fly around", ["view", "--map", m, "--quality", q, "--preset", light, "--observer"])
        elif a == "view2d":
            JOB.start("view", "2D map", ["view2d", "--map", m] + model)
        elif a == "hotspots":
            csv_p = f"reports/audit_{Path(m).stem}/hotspots.csv"
            if not (ROOT / csv_p).exists():
                return {"ok": False, "error": "Run the audit first."}
            JOB.start("view", "Hot-spots in 3D", ["view", "--map", m, "--quality", q, "--observer", "--hotspots", csv_p])
        elif a == "audit":
            n = max(10, min(int(req.get("episodes", 120)), 3000))
            variants = str(req.get("variants", "baseline;limit0.8"))
            if not all(c.isalnum() or c in ";._" for c in variants):
                raise ValueError("bad variants")
            JOB.start("audit", "Road-safety audit", ["audit", "--map", m, "--episodes", str(n), "--variants", variants,
                                                      "--out", f"reports/audit_{Path(m).stem}"] + model)
        elif a == "train":
            hours = 0.25 if req.get("mode") == "demo" else 8.0
            maps = [x["path"] for x in _maps() if x["ready"] and x["path"] != "synthetic"][:2] + ["synthetic:0", "synthetic:1"]
            extra = ["--demos-episodes", "60", "--bc-epochs", "6", "--dagger-rounds", "1", "--n-envs", "8"] if hours < 1 else []
            JOB.start("train", "Training (15 min)" if hours < 1 else "Training (overnight)",
                      ["train", "--run-name", "demo15" if hours < 1 else "night1", "--hours", str(hours), "--maps", ",".join(maps)] + extra)
        elif a == "eval":
            if not model:
                return {"ok": False, "error": "Choose a trained network first."}
            others = [x["path"] for x in _maps() if x["ready"] and x["path"] not in (m, "synthetic")] + ["synthetic:100"]
            n_ep = max(10, min(int(req.get("episodes", 30)), 300))
            JOB.start("eval", "Testing the network", ["eval", "--model", model[1], "--episodes", str(n_ep), "--seen-maps", m,
                                                      "--unseen-maps", ",".join(others)])
        else:
            return {"ok": False, "error": "unknown action"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


# ------------------------------------------------------------------------------------------------ http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:                              # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            q = self.path.split("map=")[-1] if "map=" in self.path else ""
            try:
                q = _safe_map(q.replace("%2F", "/")) if q else ""
            except ValueError:
                q = ""
            self._send(json.dumps(state(q)).encode(), "application/json")
        elif path.startswith("/files/reports/"):
            f = (ROOT / path[len("/files/"):]).resolve()
            if (ROOT / "reports").resolve() in f.parents and f.exists():
                self._send(f.read_bytes(), mimetypes.guess_type(str(f))[0] or "application/octet-stream")
            else:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        if self.path == "/api/run":
            self._send(json.dumps(run_action(req)).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)


def _open_window(url: str) -> None:
    """Prefer an app-style window (no tabs / address bar) of Edge or Chrome; fall back to the default browser."""
    cands = [os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
             os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
             os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
             os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe")]
    for c in cands:
        if os.path.exists(c):
            subprocess.Popen([c, f"--app={url}", "--window-size=1180,820"])
            return
    webbrowser.open(url)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args(argv)
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        sys.exit("no free port")
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Astana Pilot Studio: {url}   (Ctrl+C to quit)")
    if not a.no_window:
        threading.Timer(0.4, _open_window, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop()


if __name__ == "__main__":
    main()
