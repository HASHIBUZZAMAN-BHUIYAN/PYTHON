# Day 14 — Capstone Review
# This file is a guided tour of every concept used in mini_project.py.
# Read it alongside the project code.

# ─── CONCEPTS USED IN THE CAPSTONE ───────────────────────────────────────────
# The contact book intentionally uses every topic from the 2-week course.
# Here's a quick reference with the relevant day number.

print("=== Day 14 Concept Map ===\n")

concepts = {
    "Variables & types (Day 1-2)":  "name, phone, email are strings; counts are ints",
    "Conditionals (Day 3)":         "matches() checks multiple conditions with 'or'",
    "Loops (Day 4)":                "for loops to iterate contacts, find matches",
    "Strings (Day 5)":              ".lower(), .strip(), .split(), f-strings, format",
    "Lists & tuples (Day 6)":       "_contacts list, tags list, MENU list of tuples",
    "Dicts (Day 7)":                "to_dict(), from_dict(), DictWriter/Reader",
    "Functions (Day 8)":            "All cmd_* functions, prompt_contact()",
    "File I/O + CSV (Day 9)":       "_load(), _save() use csv.DictReader/DictWriter",
    "Error handling (Day 10)":      "ValidationError, ContactNotFoundError, try/except",
    "OOP I (Day 11)":               "Contact class with __init__, methods, __str__",
    "OOP II (Day 12)":              "@classmethod from_dict, @property if added",
    "Modules (Day 13)":             "import csv, os, datetime — stdlib modules",
}

for concept, usage in concepts.items():
    print(f"  {concept}")
    print(f"    → {usage}\n")

print("Run mini_project.py to use the contact book!")
