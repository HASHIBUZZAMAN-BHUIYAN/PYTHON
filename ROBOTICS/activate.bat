@echo off
:: Activates ROBOTICS\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python ROBOTICS\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [ROBOTICS venv active]  Run scripts as:  python ROBOTICS\pathfinding_astar.py
cmd /k
