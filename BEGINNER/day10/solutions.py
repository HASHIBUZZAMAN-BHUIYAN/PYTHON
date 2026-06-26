# Day 10 — Solutions

# Exercise 1
def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a whole number.")

# Exercise 2
def safe_read(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            return [l.rstrip() for l in f]
    except FileNotFoundError:
        return []

print(safe_read("nonexistent.txt"))

# Exercise 3
class ValidationError(Exception): pass

def validate_username(name):
    if len(name) < 3 or len(name) > 20:
        raise ValidationError("Username must be 3-20 characters")
    if " " in name:
        raise ValidationError("Username cannot contain spaces")
    if name[0].isdigit():
        raise ValidationError("Username cannot start with a digit")
    return True

for test in ["ok", "ab", "1user", "bad name", "validUser"]:
    try:
        validate_username(test)
        print(f"{test!r}: valid")
    except ValidationError as e:
        print(f"{test!r}: {e}")

# Exercise 4
def divide_all(numbers, divisor):
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide list elements by zero")
    return [n / divisor for n in numbers]

try:
    print(divide_all([10, 20, 30], 0))
except ZeroDivisionError as e:
    print(f"Caught: {e}")

# Exercise 5
def parse_csv_number(s):
    try:
        return float(s.strip())
    except ValueError as e:
        raise RuntimeError(f"Could not parse {s!r} as a number") from e

try:
    parse_csv_number("abc")
except RuntimeError as e:
    print(e)
    print("Caused by:", e.__cause__)
