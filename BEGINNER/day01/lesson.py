# Day 01 — Python Basics: Variables, Data Types, Input/Output
# Run this top to bottom to see every concept in action.

# ─── 1. COMMENTS ────────────────────────────────────────────────────────────
# A comment starts with #. Python ignores it.
# Use comments to explain WHY, not what.

# ─── 2. PRINT ────────────────────────────────────────────────────────────────
print("Hello, Python!")
print(42)
print(3.14)
print(True)

# ─── 3. VARIABLES ────────────────────────────────────────────────────────────
name = "Alice"
age = 20
height = 1.68
is_student = True

print(name, age, height, is_student)

# Variable names: lowercase, underscores, no spaces, no leading digit.
first_name = "Bob"
score_1 = 95

# ─── 4. DATA TYPES ───────────────────────────────────────────────────────────
# int   – whole numbers
x = 10
print(type(x))

# float – decimal numbers
pi = 3.14159
print(type(pi))

# str   – text (single or double quotes)
greeting = "Hello"
city = 'Dhaka'
print(type(greeting))

# bool  – True or False (capital T/F)
is_raining = False
print(type(is_raining))

print(type(age))

# ─── 5. F-STRINGS (formatted string literals) ────────────────────────────────
print(f"My name is {name} and I am {age} years old.")
print(f"Pi is approximately {pi:.2f}")  # :.2f = 2 decimal places

# ─── 6. INPUT FROM USER ──────────────────────────────────────────────────────
# input() always returns a STRING, even if the user types a number.
user_name = input("Enter your name: ")
print(f"Welcome, {user_name}!")

user_age = input("Enter your age: ")
# Convert the string to int before doing arithmetic:
user_age = int(user_age)
print(f"Next year you will be {user_age + 1}.")

# ─── 7. MULTIPLE ASSIGNMENT ──────────────────────────────────────────────────
a, b, c = 1, 2, 3
print(a, b, c)

x = y = 0
print(x, y)

# ─── 8. CONSTANTS (convention only) ──────────────────────────────────────────
# Python has no true constants; use ALL_CAPS as a signal "don't change this".
MAX_SCORE = 100
GRAVITY = 9.81

# ─── 9. PUTTING IT TOGETHER ──────────────────────────────────────────────────
print("\n--- Summary ---")
print(f"Name    : {name}")
print(f"Age     : {age}")
print(f"Height  : {height} m")
print(f"Student : {is_student}")
