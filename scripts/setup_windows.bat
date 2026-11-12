@echo off
REM One-time setup for Windows + NVIDIA: virtual environment, project, CUDA build of PyTorch, self-check.
REM Needs Python 3.10-3.12 on PATH. Takes 5-10 minutes (PyTorch CUDA is ~2.5 GB).
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [1/4] creating .venv
  python -m venv .venv || goto :fail
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip || goto :fail
echo [2/4] installing the project and its dependencies
pip install -e ".[train,view,dev]" || goto :fail
echo [3/4] installing PyTorch with CUDA (done last: pip may otherwise replace it by a CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cu124 --upgrade --force-reinstall --no-deps || goto :fail
echo [4/4] self-check
astana check
echo.
echo Setup finished. Next:  .venv\Scripts\astana map --preset expo   and   .venv\Scripts\astana view --map assets\maps\astana_expo.pkl
pause
exit /b 0
:fail
echo SETUP FAILED - see the message above.
pause
exit /b 1
