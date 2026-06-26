@echo off
:: Activates NLP\.venv and sets working dir to PYTHON\ (parent),
:: so scripts can be run as:  python NLP\script_name.py
cd /d "%~dp0.."
call "%~dp0.venv\Scripts\activate.bat"
echo [NLP venv active]  Run scripts as:  python NLP\text_summarizer.py
cmd /k
