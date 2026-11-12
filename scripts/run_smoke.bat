@echo off
REM Short end-to-end test (about 10 minutes): shows that the whole night pipeline works on this machine.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m astana_pilot night --smoke
echo.
echo Report: runs\smoke\MORNING.md
pause
