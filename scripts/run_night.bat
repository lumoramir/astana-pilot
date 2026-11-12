@echo off
REM One command for the whole night (Windows). Usage:  scripts\run_night.bat [hours] [run-name]
REM Example: scripts\run_night.bat 8 night1  -> self-test, expert demos, behaviour cloning, PPO for 8 h, evaluation, report.
REM Interrupted (power cut, closed window)? Run the SAME command again - it resumes from the last autosave.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set HOURS=%1
if "%HOURS%"=="" set HOURS=8
set RUN=%2
if "%RUN%"=="" set RUN=night1
python -m astana_pilot night --hours %HOURS% --run-name %RUN%
echo.
echo Morning report: runs\%RUN%\MORNING.md      Curves: tensorboard --logdir runs
pause
