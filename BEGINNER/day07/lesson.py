# Day 07 — Dictionaries and Sets

# ─── 1. DICTIONARIES ─────────────────────────────────────────────────────────
# A dict maps keys → values. Keys must be immutable (str, int, tuple, etc.)

person = {"name": "Alice", "age": 25, "city": "Dhaka"}
print(person["name"])       # Alice
print(person.get("age"))    # 25
print(person.get("email", "N/A"))  # N/A (default if key missing)

# ─── 2. ADD / UPDATE / DELETE ────────────────────────────────────────────────
person["email"] = "alice@example.com"
person["age"] = 26
del person["city"]
print(person)

# pop() removes and returns
val = person.pop("email", None)
print(val, person)

# ─── 3. MEMBERSHIP TEST ──────────────────────────────────────────────────────
print("name" in person)    # True  — checks keys
print("Alice" in person)   # False — values are not checked by 'in'

# ─── 4. ITERATING ────────────────────────────────────────────────────────────
scores = {"Math": 90, "Science": 85, "English": 78}

for key in scores:
    print(key)

for value in scores.values():
    print(value)

for k, v in scores.items():
    print(f"{k}: {v}")

# ─── 5. USEFUL DICT METHODS ──────────────────────────────────────────────────
print(scores.keys())
print(scores.values())
print(len(scores))

# update() merges another dict
scores.update({"Art": 88, "Math": 95})   # adds Art, overwrites Math
print(scores)

# ─── 6. DICT COMPREHENSIONS ──────────────────────────────────────────────────
squares = {x: x**2 for x in range(1, 6)}
print(squares)    # {1:1, 2:4, 3:9, 4:16, 5:25}

# Filter: only passing scores
passing = {k: v for k, v in scores.items() if v >= 85}
print(passing)

# ─── 7. NESTED DICTS ─────────────────────────────────────────────────────────
students = {
    "Alice": {"grade": "A", "score": 95},
    "Bob":   {"grade": "B", "score": 82},
}
print(students["Alice"]["score"])   # 95

# ─── 8. SETS ─────────────────────────────────────────────────────────────────
# Unordered, no duplicates, mutable.
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)   # union         {1,2,3,4,5,6}
print(a & b)   # intersection  {3,4}
print(a - b)   # difference    {1,2}
print(a ^ b)   # symmetric diff {1,2,5,6}

# Adding / removing
a.add(99)
a.discard(2)   # remove if present (no error if missing)
a.remove(3)    # remove — raises KeyError if missing
print(a)

# ─── 9. SET COMPREHENSION ────────────────────────────────────────────────────
words = ["apple", "banana", "apple", "cherry", "banana"]
unique = {w.lower() for w in words}
print(unique)    # {'apple', 'banana', 'cherry'} (order may vary)

# frozenset — immutable set (can be used as dict key)
fs = frozenset({1, 2, 3})
print(fs)
