# Day 14 — Solutions (Capstone Extensions)
# These are add-on snippets, not a complete rewrite. They show HOW to extend.

from datetime import date, datetime

# Exercise 1 — Import from CSV
def import_from_csv(book, filepath):
    import csv
    added = 0
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                from day14.mini_project import Contact, DuplicateContactError
                book.add(Contact.from_dict(row))
                added += 1
            except Exception:
                pass
    print(f"Imported {added} contacts.")

# Exercise 3 — Upcoming birthdays snippet
def upcoming_birthdays(contacts, days=30):
    today = date.today()
    result = []
    for c in contacts:
        if not getattr(c, 'birthday', None):
            continue
        bday = datetime.strptime(c.birthday, "%Y-%m-%d").date()
        next_bday = bday.replace(year=today.year)
        if next_bday < today:
            next_bday = next_bday.replace(year=today.year + 1)
        if (next_bday - today).days <= days:
            result.append((c, (next_bday - today).days))
    return sorted(result, key=lambda x: x[1])

# Exercise 4 — Stats snippet
def print_stats(book):
    contacts = list(book.all_sorted())
    n = len(contacts)
    if n == 0:
        print("No contacts.")
        return
    all_tags = [tag for c in contacts for tag in c.tags]
    from collections import Counter
    tag_counts = Counter(all_tags)
    this_month = date.today().strftime("%Y-%m")
    added_this_month = sum(1 for c in contacts if c.created.startswith(this_month))
    print(f"Total contacts   : {n}")
    print(f"Avg tags/contact : {len(all_tags)/n:.1f}")
    print(f"Most common tag  : {tag_counts.most_common(1)[0] if tag_counts else 'none'}")
    print(f"Added this month : {added_this_month}")

print("Day 14 solution snippets loaded. See comments for integration notes.")
