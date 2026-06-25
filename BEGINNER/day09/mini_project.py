# Day 09 Mini-Project — CSV Journal
# A simple daily journal that persists entries to a CSV file.

import csv, os
from datetime import date

JOURNAL_FILE = "journal.csv"
FIELDS = ["date", "mood", "entry"]

def load():
    if not os.path.exists(JOURNAL_FILE):
        return []
    with open(JOURNAL_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save(entries):
    with open(JOURNAL_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(entries)

entries = load()

while True:
    print("\n1.New entry  2.View all  3.Search  4.Quit")
    ch = input("Choice: ").strip()
    if ch == "1":
        mood  = input("Mood (1-5): ")
        entry = input("Today's note: ")
        entries.append({"date": str(date.today()), "mood": mood, "entry": entry})
        save(entries)
        print("Saved.")
    elif ch == "2":
        if not entries:
            print("No entries yet.")
        else:
            for e in entries:
                print(f"[{e['date']}] Mood:{e['mood']} — {e['entry']}")
    elif ch == "3":
        q = input("Search text: ").lower()
        results = [e for e in entries if q in e["entry"].lower()]
        if results:
            for e in results:
                print(f"[{e['date']}] {e['entry']}")
        else:
            print("No matches.")
    elif ch == "4":
        break
