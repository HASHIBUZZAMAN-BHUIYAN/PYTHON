#!/usr/bin/env python3
# Day 14 Capstone — Command-Line Contact Book
# Combines: OOP, file I/O, error handling, functions, all data structures.

import csv, os, sys
from datetime import datetime

# ─── Custom Exceptions ────────────────────────────────────────────────────────
class ContactNotFoundError(Exception): pass
class DuplicateContactError(Exception): pass
class ValidationError(Exception): pass

# ─── Contact Class ────────────────────────────────────────────────────────────
class Contact:
    def __init__(self, name, phone, email="", tags=""):
        self.name   = self._validate_name(name)
        self.phone  = self._validate_phone(phone)
        self.email  = email.strip()
        self.tags   = [t.strip().lower() for t in tags.split(",") if t.strip()] if tags else []
        self.created = datetime.now().strftime("%Y-%m-%d")

    def _validate_name(self, name):
        name = name.strip()
        if len(name) < 2:
            raise ValidationError("Name must be at least 2 characters")
        return name

    def _validate_phone(self, phone):
        phone = "".join(c for c in phone if c.isdigit() or c in "+-() ")
        if len("".join(c for c in phone if c.isdigit())) < 7:
            raise ValidationError("Phone must have at least 7 digits")
        return phone

    def matches(self, query):
        q = query.lower()
        return (q in self.name.lower()
                or q in self.phone
                or q in self.email.lower()
                or any(q in tag for tag in self.tags))

    def to_dict(self):
        return {"name": self.name, "phone": self.phone,
                "email": self.email, "tags": ",".join(self.tags),
                "created": self.created}

    @classmethod
    def from_dict(cls, d):
        c = cls.__new__(cls)
        c.name    = d["name"]
        c.phone   = d["phone"]
        c.email   = d.get("email","")
        c.tags    = [t for t in d.get("tags","").split(",") if t]
        c.created = d.get("created","")
        return c

    def __str__(self):
        tags_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"{self.name:<20} {self.phone:<18} {self.email:<25}{tags_str}"

    def __repr__(self):
        return f"Contact({self.name!r}, {self.phone!r})"


# ─── ContactBook Class ────────────────────────────────────────────────────────
class ContactBook:
    CSV_FIELDS = ["name", "phone", "email", "tags", "created"]

    def __init__(self, filepath="contacts.csv"):
        self.filepath = filepath
        self._contacts = []
        self._load()

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._contacts.append(Contact.from_dict(row))
        except Exception as e:
            print(f"Warning: could not load {self.filepath}: {e}")

    def _save(self):
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            w.writeheader()
            for c in self._contacts:
                w.writerow(c.to_dict())

    def add(self, contact):
        if any(c.name.lower() == contact.name.lower() for c in self._contacts):
            raise DuplicateContactError(f"'{contact.name}' already exists")
        self._contacts.append(contact)
        self._save()

    def delete(self, name):
        for i, c in enumerate(self._contacts):
            if c.name.lower() == name.lower():
                removed = self._contacts.pop(i)
                self._save()
                return removed
        raise ContactNotFoundError(f"'{name}' not found")

    def find(self, query):
        return [c for c in self._contacts if c.matches(query)]

    def get(self, name):
        for c in self._contacts:
            if c.name.lower() == name.lower():
                return c
        raise ContactNotFoundError(f"'{name}' not found")

    def update(self, name, **kwargs):
        c = self.get(name)
        for key, val in kwargs.items():
            if key == "phone": val = c._validate_phone(val)
            if key == "tags":  val = [t.strip().lower() for t in val.split(",") if t.strip()]
            setattr(c, key, val)
        self._save()

    def all_sorted(self):
        return sorted(self._contacts, key=lambda c: c.name.lower())

    def all_tags(self):
        tags = set()
        for c in self._contacts:
            tags.update(c.tags)
        return sorted(tags)

    def __len__(self): return len(self._contacts)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def print_header(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def print_contact(c, idx=None):
    prefix = f"{idx:3}. " if idx is not None else "     "
    print(f"{prefix}{c}")

def prompt_contact():
    name  = input("Name         : ").strip()
    phone = input("Phone        : ").strip()
    email = input("Email (opt)  : ").strip()
    tags  = input("Tags (a,b,c) : ").strip()
    return Contact(name, phone, email, tags)

def cmd_add(book):
    print_header("ADD CONTACT")
    try:
        c = prompt_contact()
        book.add(c)
        print(f"✓ Added: {c.name}")
    except (ValidationError, DuplicateContactError) as e:
        print(f"Error: {e}")

def cmd_list(book):
    print_header(f"ALL CONTACTS ({len(book)})")
    if not book:
        print("  (empty)")
        return
    print(f"  {'NAME':<20} {'PHONE':<18} {'EMAIL':<25} TAGS")
    print("  " + "-"*72)
    for i, c in enumerate(book.all_sorted(), 1):
        print_contact(c, i)

def cmd_search(book):
    q = input("Search: ").strip()
    results = book.find(q)
    print_header(f"RESULTS for '{q}' ({len(results)})")
    for i, c in enumerate(results, 1):
        print_contact(c, i)
    if not results:
        print("  No matches.")

def cmd_delete(book):
    name = input("Delete contact name: ").strip()
    try:
        c = book.delete(name)
        print(f"✓ Deleted: {c.name}")
    except ContactNotFoundError as e:
        print(f"Error: {e}")

def cmd_edit(book):
    name = input("Edit contact name: ").strip()
    try:
        c = book.get(name)
        print(f"Editing {c.name} (press Enter to keep current value)")
        new_phone = input(f"Phone [{c.phone}]: ").strip() or c.phone
        new_email = input(f"Email [{c.email}]: ").strip() or c.email
        new_tags  = input(f"Tags  [{','.join(c.tags)}]: ").strip() or ",".join(c.tags)
        book.update(name, phone=new_phone, email=new_email, tags=new_tags)
        print("✓ Updated.")
    except (ContactNotFoundError, ValidationError) as e:
        print(f"Error: {e}")

def cmd_tags(book):
    tags = book.all_tags()
    if not tags:
        print("No tags yet."); return
    print_header("ALL TAGS")
    for tag in tags:
        contacts = book.find(tag)
        print(f"  {tag:<15}: {', '.join(c.name for c in contacts)}")

MENU = [
    ("1", "List all",   cmd_list),
    ("2", "Add",        cmd_add),
    ("3", "Search",     cmd_search),
    ("4", "Edit",       cmd_edit),
    ("5", "Delete",     cmd_delete),
    ("6", "Browse tags",cmd_tags),
    ("7", "Quit",       None),
]

def main():
    book = ContactBook("contacts.csv")
    print(f"\n=== Contact Book ({len(book)} contacts) ===")
    while True:
        print("\n" + "  ".join(f"{k}.{label}" for k, label, _ in MENU))
        ch = input("→ ").strip()
        matched = next((fn for k, _, fn in MENU if k == ch), None)
        if ch == "7":
            print("Goodbye!")
            break
        elif matched:
            matched(book)
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
