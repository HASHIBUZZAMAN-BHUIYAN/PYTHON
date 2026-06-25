# PYTHON Learning Curriculum — Setup Guide

## Requirements
- Python 3.10+ (you have 3.11.3)
- Windows 10/11, PowerShell or cmd

## Activating the BEGINNER virtual environment

```powershell
cd C:\Users\zen\Documents\GitHub\PYTHON
.\BEGINNER\.venv\Scripts\Activate.ps1
```

After activation your prompt shows `(.venv)`. To deactivate: `deactivate`

## Activating the ADVANCED virtual environment

```powershell
cd C:\Users\zen\Documents\GitHub\PYTHON
.\ADVANCED\.venv\Scripts\Activate.ps1
```

## Running a lesson

```powershell
# Activate the correct venv first, then:
python BEGINNER\day01\lesson.py
python BEGINNER\day01\exercises.py
python BEGINNER\day01\mini_project.py

python ADVANCED\week1\day01\lesson.py
```

## If Activate.ps1 is blocked by PowerShell execution policy

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Run once, then activate normally.

## Reinstalling dependencies from scratch

```powershell
# BEGINNER
python -m venv BEGINNER\.venv
.\BEGINNER\.venv\Scripts\Activate.ps1
pip install -r requirements\beginner-requirements.txt

# ADVANCED
python -m venv ADVANCED\.venv
.\ADVANCED\.venv\Scripts\Activate.ps1
pip install -r requirements\advanced-requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Hardware note

All scripts are tuned for 8 GB RAM, CPU-only (no CUDA). PyTorch was installed
from the CPU-only wheel index. TensorFlow was installed as `tensorflow-cpu`.
Close browser tabs and heavy apps before running any DL/CNN lessons.
