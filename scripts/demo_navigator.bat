@echo off
REM Committee demo: the navigator map opens first - click A (start) and B (destination), pick one of the
REM proposed routes, press Drive. The neural network then drives it. Esc = menu, M = change the destination.
REM Usage:  scripts\demo_navigator.bat [expo|nurzhol|synthetic] [model.zip] [random]
REM   "random" as the 3rd argument = no map, a random verified route every episode.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set MAP=%1
if "%MAP%"=="" set MAP=expo
if /i "%MAP%"=="expo" (
  if not exist "assets\maps\astana_expo.pkl" python -m astana_pilot map --preset expo --no-preview
  set MAP=assets\maps\astana_expo.pkl
)
if /i "%MAP%"=="nurzhol" (
  if not exist "assets\maps\astana_nurzhol.pkl" python -m astana_pilot map --preset nurzhol --no-preview
  set MAP=assets\maps\astana_nurzhol.pkl
)
set MODEL=%2
if "%MODEL%"=="" if exist "demo_weights\astana_pilot.zip" set MODEL=demo_weights\astana_pilot.zip
if "%MODEL%"=="" if exist "runs\night3\best_model.zip" set MODEL=runs\night3\best_model.zip
if "%MODEL%"=="" set MODEL=runs\night2\best_model.zip
if /i "%3"=="random" (
  python -m astana_pilot view --map %MAP% --model %MODEL% --random-route --verified --seed %RANDOM%
) else (
  python -m astana_pilot view --map %MAP% --model %MODEL% --pick
)
