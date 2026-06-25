# Day 03 — Conditionals

# ─── 1. BASIC if / elif / else ───────────────────────────────────────────────
score = 72

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Score {score} → Grade {grade}")   # C

# ─── 2. INDENTATION MATTERS ──────────────────────────────────────────────────
# Python uses indentation (4 spaces) to define blocks — NOT braces.
x = 5
if x > 0:
    print("positive")    # inside the if
    print("still inside")
print("always runs")     # outside the if

# ─── 3. BOOLEAN LOGIC IN CONDITIONS ─────────────────────────────────────────
age = 20
has_id = True

if age >= 18 and has_id:
    print("Entry allowed")
else:
    print("Entry denied")

# De Morgan: not (A and B)  ==  (not A) or (not B)
print(not (True and False))   # True
print((not True) or (not False))  # True  (same result)

# ─── 4. MEMBERSHIP OPERATOR: in ──────────────────────────────────────────────
color = "red"
if color in ("red", "green", "blue"):
    print(f"{color} is a primary color")

# ─── 5. IDENTITY OPERATOR: is ────────────────────────────────────────────────
value = None
if value is None:
    print("No value provided")

# ─── 6. NESTED CONDITIONALS ──────────────────────────────────────────────────
temp = 28
humidity = 80

if temp > 30:
    if humidity > 70:
        print("Hot and humid")
    else:
        print("Hot but dry")
else:
    print("Comfortable")

# ─── 7. TERNARY (one-line if-else) ───────────────────────────────────────────
n = -4
label = "negative" if n < 0 else "non-negative"
print(label)

# ─── 8. TRUTHINESS ───────────────────────────────────────────────────────────
# Any value can be used as a condition.
# Falsy: 0, 0.0, "", None, [], {}, ()
name = ""
if not name:
    print("Name is empty — please provide one")

items = [1, 2, 3]
if items:
    print(f"List has {len(items)} items")

# ─── 9. CHAINED COMPARISONS ──────────────────────────────────────────────────
val = 15
if 10 <= val <= 20:
    print(f"{val} is between 10 and 20")   # Pythonic — works as expected
