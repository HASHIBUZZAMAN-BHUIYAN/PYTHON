@echo off
:: Activates AGENTIC_AI\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python AGENTIC_AI\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [AGENTIC_AI venv active]  Run scripts as:  python AGENTIC_AI\simple_agent_template.py
cmd /k
