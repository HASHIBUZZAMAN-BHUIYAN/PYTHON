@echo off
:: Activates DL\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python DL\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [DL venv active]  Run scripts as:  python DL\pytorch_training_loop_template.py
cmd /k
