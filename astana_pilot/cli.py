"""One entry point for everything:  ``astana <command> [options]``  (or ``python -m astana_pilot <command>``).

    astana app                          desktop Studio: pick district + AI, then watch / fly / audit / train
    astana map --preset expo            download + bake a real district from OpenStreetMap
    astana view --map assets/maps/astana_expo.pkl [--model runs/x/final_model.zip]    3-D digital twin
    astana view --random-route --seed 7 --model ...    navigator picks a random A -> B, the network drives it
    astana view --start-node "Nur Alem" --goal-node "Kabanbay" --model ...      your own A -> B
    astana route --map ... --model ...  2-D navigator window: click A and B, ENTER = drive it in 3-D
    astana progress night3              how the network improved: every checkpoint evaluated on fixed routes
    astana view2d ...                   light-weight 2-D viewer (pygame)
    astana record --out demo.mp4 ...    render a video headless (cinematic camera)
    astana train --run-name night1 --hours 8     BC + DAgger + PPO on the GPU
    astana eval --model runs/night1/final_model.zip     seen / unseen-city evaluation with confidence intervals
    astana audit --map assets/maps/astana_expo.pkl      road-safety audit: risk heat-map from simulated near-misses
    astana plot night1                  training dashboard (success, collisions, violations ...) from the TensorBoard logs
    astana check                        verify torch / CUDA / the whole training stack in ~1 minute
    astana still ...                    render a still image of the city from any camera pose
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "map": "astana_pilot.tools.fetch_astana_map",
    "view": "astana_pilot.viz.app3d",
    "view2d": "astana_pilot.viz.viewer2d",
    "still": "astana_pilot.viz.stills",
    "train": "astana_pilot.training.train",
    "night": "astana_pilot.training.night",
    "eval": "astana_pilot.training.evaluate",
    "demos": "astana_pilot.training.demos",
    "audit": "astana_pilot.audit.run_audit",
    "app": "astana_pilot.studio.server",
    "studio": "astana_pilot.studio.server",
    "plot": "astana_pilot.tools.plot_training",
    "check": "astana_pilot.tools.check_install",
    "bench": "astana_pilot.tools.benchmark_env",
    "preview": "astana_pilot.tools.preview_map",
    "gpu": "astana_pilot.tools.set_gpu_preference",
    "route": "astana_pilot.tools.route_planner",
    "progress": "astana_pilot.tools.progress_chart",
}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:                                                  # double-click / plain `astana`: open the Studio
        argv = ["studio"]
    if argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("commands:", ", ".join(COMMANDS), "+ record")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "record":                                           # sugar: headless video with the cinematic camera
        out = "assets/videos/demo.mp4"
        if "--out" in rest:
            i = rest.index("--out")
            out = rest[i + 1]
            rest = rest[:i] + rest[i + 2:]
        rest = ["--headless", "--record", out, "--cinematic"] + rest
        if "--duration" not in rest:
            rest += ["--duration", "60"]
        cmd = "view"
    if cmd not in COMMANDS:
        sys.exit(f"unknown command '{cmd}'. Try: astana help")
    sys.argv = [f"astana {cmd}"] + rest
    mod = importlib.import_module(COMMANDS[cmd])      # (not runpy: worker pools must be able to re-import the module)
    code = mod.main()
    if isinstance(code, int) and code:
        sys.exit(code)


if __name__ == "__main__":
    main()
