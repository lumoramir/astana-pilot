"""One command for the whole night: self-test -> demos + behaviour cloning -> PPO (resumable) -> evaluation -> report.

    python -m astana_pilot.training.night --hours 8 --run-name night1          # the real thing (use run_night.bat)
    python -m astana_pilot.training.night --smoke                              # ~10 minutes, run_smoke.bat
    python -m astana_pilot.training.night --run-name night1 --hours 8          # after an interruption: continues where it stopped
    python -m astana_pilot.training.night --run-name night3 --hours 8 --init-from runs/night2/best_model.zip   # fine-tune

In the morning read  runs/<run-name>/MORNING.md  (what the network learned, versus the expert, on known and
unknown cities) and look at curves with  tensorboard --logdir runs.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str], log: Path, tag: str) -> int:
    print(f"\n=== {tag}: {' '.join(cmd)}", flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {tag}: {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        return p.wait()


def morning_report(run_dir: Path, name: str, elapsed_h: float) -> str:
    log = (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore") if (run_dir / "train.log").exists() else ""
    bc_lines = [l for l in log.splitlines() if "[bc] epoch" in l][-2:]
    train_lines = [l for l in log.splitlines() if l.startswith("[train]")]
    ev = run_dir / "eval" / "report.md"
    parts = [f"# Morning report: {name}\n", f"Wall clock of this launch: {elapsed_h:.2f} h\n"]
    if (run_dir / "curriculum.json").exists():
        parts.append(f"Curriculum difficulty reached: {(run_dir / 'curriculum.json').read_text()}\n")
    if bc_lines:
        parts += ["## Behaviour cloning (last epochs)\n", "```", *bc_lines, "```\n"]
    if train_lines:
        parts += ["## Training progress (first / middle / last status line)\n", "```",
                  train_lines[0], train_lines[len(train_lines) // 2], train_lines[-1], "```\n"]
    parts.append("## Evaluation: model vs expert, known vs unknown cities\n")
    parts.append(ev.read_text(encoding="utf-8") if ev.exists() else "_evaluation did not run (see train.log)_\n")
    if (run_dir / "progress.md").exists():
        parts += ["\n## Progress of every checkpoint (same routes)\n", (run_dir / "progress.md").read_text(encoding="utf-8")]
    parts.append("\nHonest reading: success / collision / violations above are measured on the simulator only. "
                 "`seen` = the cities of the training run (here possibly real Astana districts), `unseen` = synthetic "
                 "cities with another layout that the network never drove in.")
    return "\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run-name", default="night1")
    p.add_argument("--hours", type=float, default=8.0, help="training wall-clock budget")
    p.add_argument("--timesteps", type=int, default=150_000_000, help="target total steps (the time budget usually ends first)")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (minutes) to see that everything works")
    p.add_argument("--eval-episodes", type=int, default=60)
    p.add_argument("--maps", default=None)
    p.add_argument("--skip-check", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-restarts", type=int, default=3)
    p.add_argument("--init-from", default=None, help="fine-tune this trained network (e.g. runs/night2/best_model.zip)")
    p.add_argument("--start-difficulty", type=float, default=None)
    p.add_argument("--rule-penalty-scale", type=float, default=1.0, help="see training.train --help")
    p.add_argument("--progress-episodes", type=int, default=24, help="routes per checkpoint for the progress chart (0 = skip)")
    a = p.parse_args(argv)

    name = "smoke" if a.smoke and a.run_name == "night1" else a.run_name
    run_dir = Path("runs") / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "train.log"
    t0 = time.time()

    if not a.skip_check:
        if _run([sys.executable, "-m", "astana_pilot.tools.check_install"], log, "self-test") != 0:
            print("\nSelf-test failed - fix that first (see the FAIL block above).")
            return 1

    train = [sys.executable, "-m", "astana_pilot.training.train", "--run-name", name, "--device", a.device]
    if a.maps:
        train += ["--maps", a.maps]
    if a.init_from:
        train += ["--init-from", a.init_from]
    if a.start_difficulty is not None:
        train += ["--start-difficulty", str(a.start_difficulty)]
    if a.rule_penalty_scale != 1.0:
        train += ["--rule-penalty-scale", str(a.rule_penalty_scale)]
    if a.smoke:
        train += ["--maps", "synthetic:0,synthetic:1", "--n-envs", "4", "--timesteps", "40000", "--hours", "0.25",
                  "--demos-episodes", "24", "--bc-epochs", "3", "--critic-epochs", "2", "--dagger-rounds", "1",
                  "--checkpoint-minutes", "1"]
        eval_eps = 8
    else:
        train += ["--timesteps", str(a.timesteps), "--hours", str(a.hours)]
        eval_eps = a.eval_episodes
    deadline = t0 + (a.hours if not a.smoke else 0.25) * 3600.0 + 600.0
    for attempt in range(a.max_restarts + 1):
        rc = _run(train, log, f"training (attempt {attempt + 1})")
        done = (run_dir / "final_model.zip").exists() and rc == 0
        if done or time.time() > deadline:
            break
        print("\n[night] training stopped unexpectedly - resuming from the last autosave", flush=True)
        time.sleep(10)

    model = run_dir / "best_model.zip"                     # the weights the stability guard judged best
    for alt in (run_dir / "final_model.zip", run_dir / "checkpoints" / "latest.zip"):
        if not model.exists() and alt.exists():
            model = alt
    import json
    st = json.loads((run_dir / "state.json").read_text()) if (run_dir / "state.json").exists() else {}
    seen = ",".join(st.get("maps", [])) or "synthetic:0,synthetic:1"
    if model.exists():
        ev = [sys.executable, "-m", "astana_pilot.training.evaluate", "--model", str(model), "--episodes", str(eval_eps),
              "--seen-maps", seen, "--out", str(run_dir / "eval")]
        _run(ev, log, "evaluation")
        _run([sys.executable, "-m", "astana_pilot.tools.plot_training", name], log, "training curves")
        if a.progress_episodes > 0 and not a.smoke:
            _run([sys.executable, "-m", "astana_pilot.tools.progress_chart", name, "--episodes", str(a.progress_episodes)],
                 log, "progress chart")
    (run_dir / "MORNING.md").write_text(morning_report(run_dir, name, (time.time() - t0) / 3600), encoding="utf-8")
    print(f"\nDONE. Read {run_dir / 'MORNING.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
