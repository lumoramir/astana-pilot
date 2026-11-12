@echo off
REM Overnight FINE-TUNING of the existing network (no new behaviour cloning).
REM Usage:  scripts\run_finetune.bat [hours] [run-name] [start-model]
REM Default: 8 h, run "night3", starting from runs\night2\best_model.zip
REM Interrupted? Run the SAME command again - it resumes from the last autosave (every 15 min).
REM Morning: runs\<run>\MORNING.md, runs\<run>\progress.png (every checkpoint on the same routes),
REM          runs\<run>\training_curves.png. Live curves meanwhile:  astana plot <run> --watch
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set HOURS=%1
if "%HOURS%"=="" set HOURS=8
set RUN=%2
if "%RUN%"=="" set RUN=night3
set FROM=%3
if "%FROM%"=="" set FROM=runs\night2\best_model.zip
REM rule penalties x2.5: night2 crossed red lights in ~1/3 of the trips - this run is aimed at that
python -m astana_pilot night --hours %HOURS% --run-name %RUN% --init-from %FROM% --rule-penalty-scale 2.5
echo.
echo Morning report: runs\%RUN%\MORNING.md     Progress chart: runs\%RUN%\progress.png
pause
