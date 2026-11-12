@echo off
REM Double-click demo: bakes the EXPO map on first use (cached OSM data) and opens the 3-D viewer.
setlocal
cd /d "%~dp0.."
call ".venv\Scripts\activate.bat"
if not exist "assets\maps\astana_expo.pkl" astana map --preset expo --no-preview
if exist "runs\night1\final_model.zip" (
  astana view --map assets\maps\astana_expo.pkl --model runs\night1\final_model.zip
) else (
  astana view --map assets\maps\astana_expo.pkl
)
