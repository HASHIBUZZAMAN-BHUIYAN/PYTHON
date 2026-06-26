@echo off
:: Activates ML\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python ML\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [ML venv active]  Run scripts as:  python ML\linear_regression_template.py
cmd /k
