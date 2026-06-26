# Day 09 — File Handling
import csv, os

# ─── 1. WRITING A TEXT FILE ──────────────────────────────────────────────────
with open("sample.txt", "w", encoding="utf-8") as f:
    f.write("Line 1\n")
    f.write("Line 2\n")
    f.write("Line 3\n")
print("Written sample.txt")

# ─── 2. READING ENTIRE FILE ──────────────────────────────────────────────────
with open("sample.txt", "r", encoding="utf-8") as f:
    content = f.read()
print(repr(content))

# ─── 3. READ LINE BY LINE ────────────────────────────────────────────────────
with open("sample.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.rstrip())

# ─── 4. READ ALL LINES INTO A LIST ───────────────────────────────────────────
with open("sample.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
print(lines)

clean_lines = [l.rstrip() for l in lines]
print(clean_lines)

# ─── 5. APPENDING TO A FILE ──────────────────────────────────────────────────
with open("sample.txt", "a", encoding="utf-8") as f:
    f.write("Line 4 (appended)\n")

# ─── 6. CHECKING IF FILE EXISTS ──────────────────────────────────────────────
print(os.path.exists("sample.txt"))
print(os.path.exists("nope.txt"))

# ─── 7. WRITING CSV ──────────────────────────────────────────────────────────
students = [
    ["Name", "Math", "Science"],
    ["Alice", 90, 85],
    ["Bob",   78, 92],
    ["Carol", 88, 79],
]

with open("students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(students)
print("Written students.csv")

# ─── 8. READING CSV ──────────────────────────────────────────────────────────
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    print("Header:", header)
    for row in reader:
        print(row)

# ─── 9. DICTREADER / DICTWRITER ──────────────────────────────────────────────
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(dict(row))

# Write with DictWriter
fields = ["Name", "Age", "City"]
people = [{"Name": "Diana", "Age": 22, "City": "Dhaka"},
          {"Name": "Eve",   "Age": 19, "City": "Chittagong"}]

with open("people.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(people)
print("Written people.csv")

# ─── 10. CLEAN UP TEMP FILES ─────────────────────────────────────────────────
for fname in ("sample.txt", "students.csv", "people.csv"):
    if os.path.exists(fname):
        os.remove(fname)
print("Cleaned up.")
