# Day 13 — Modules, Packages, and pip

# ─── 1. IMPORTING STANDARD LIBRARY MODULES ───────────────────────────────────
import math
import os
import sys
import random
import datetime

print(math.pi)
print(math.sqrt(16))
print(math.floor(3.7), math.ceil(3.2))
print(os.getcwd())
print(sys.version)
print(random.randint(1, 10))
print(datetime.date.today())

# ─── 2. SELECTIVE IMPORT ─────────────────────────────────────────────────────
from math import sqrt, pi
from random import choice, shuffle

print(sqrt(25))
print(choice(["red", "green", "blue"]))

# ─── 3. ALIASING ─────────────────────────────────────────────────────────────
import datetime as dt
import collections as col

today = dt.date.today()
print(today)

counter = col.Counter("aabbccca")
print(counter)

# ─── 4. EXPLORING A MODULE ───────────────────────────────────────────────────
print(dir(math))
print(help(math.gcd))

# ─── 5. CREATING YOUR OWN MODULE ─────────────────────────────────────────────
# See concepts/myutils.py  — a module we import below
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "concepts"))

import myutils    # our custom module

print(myutils.celsius_to_fahrenheit(100))
print(myutils.is_palindrome("racecar"))
print(myutils.greet("Hashib"))

# ─── 6. if __name__ == "__main__" ─────────────────────────────────────────────
# When a file is run directly, __name__ == "__main__"
# When it's imported, __name__ == the module name.
# This lets you have code that only runs when the file is the entry point,
# not when it's imported as a library.
print(__name__)

# ─── 7. PACKAGES ─────────────────────────────────────────────────────────────
# A package is a folder with an __init__.py file.
# Structure:
#   mypackage/
#       __init__.py
#       module_a.py
#       module_b.py
#
# from mypackage import module_a
# from mypackage.module_b import some_function
# Python finds packages on sys.path.

# ─── 8. USEFUL STANDARD LIBRARY MODULES ──────────────────────────────────────
# math          — sqrt, pi, floor, ceil, gcd, log, sin, cos
# os            — file/dir operations, environ, getcwd, path
# sys           — argv, path, version, exit
# random        — randint, choice, shuffle, sample
# datetime      — date, time, datetime, timedelta
# collections   — Counter, defaultdict, namedtuple, deque
# itertools     — combinations, permutations, chain, product
# functools     — reduce, lru_cache, partial
# json          — loads, dumps (read/write JSON strings)
# re            — regular expressions
# time          — sleep, time, perf_counter
# pathlib       — Path — modern OS-independent file paths
# csv           — reader, writer, DictReader
# argparse      — command-line argument parsing
# logging       — logging at different levels (debug/info/warning/error)

# ─── 9. pip QUICKREF ─────────────────────────────────────────────────────────
# pip install package_name
# pip install package==1.2.3      # specific version
# pip uninstall package_name
# pip list                        # show installed packages
# pip freeze > requirements.txt   # save current env
# pip install -r requirements.txt # install from file

# ─── 10. VIRTUAL ENVIRONMENTS ────────────────────────────────────────────────
# Each project gets its own isolated Python environment so packages don't conflict.
# python -m venv .venv
# .venv\Scripts\Activate.ps1      (Windows PowerShell)
# source .venv/bin/activate        (Mac/Linux)
# deactivate                       (exit venv)
