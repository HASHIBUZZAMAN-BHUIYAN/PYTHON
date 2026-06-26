@echo off
:: Activates CNN\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python CNN\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [CNN venv active]  Run scripts as:  python CNN\road_scene_understanding.py
cmd /k
