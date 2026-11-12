"""Windows only: make Python render on the discrete NVIDIA GPU instead of the iGPU.

Equivalent to Settings -> System -> Display -> Graphics -> add python.exe ->
"High performance". Run once, then restart the viewer.

    python -m astana_pilot.tools.set_gpu_preference.py          # prefer the NVIDIA GPU
    python -m astana_pilot.tools.set_gpu_preference.py --reset  # undo
"""
from __future__ import annotations

import argparse
import sys

KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if sys.platform != "win32":
        print("Windows only. On Linux run the viewer with:\n"
              "  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia "
              "python -m astana_pilot.viz.run_demo --mode 3d")
        return
    import winreg

    # A venv's python.exe is a launcher; the process that renders may be the base interpreter.
    exes = {sys.executable, getattr(sys, "_base_executable", sys.executable)}
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, KEY) as key:
        for exe in sorted(exes):
            if args.reset:
                try:
                    winreg.DeleteValue(key, exe)
                    print(f"removed preference for {exe}")
                except FileNotFoundError:
                    pass
            else:
                winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
                print(f"high-performance GPU set for {exe}")
    print("Restart the viewer for the change to take effect.")


if __name__ == "__main__":
    main()
