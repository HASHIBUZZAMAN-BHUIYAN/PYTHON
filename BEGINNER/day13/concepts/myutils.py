# myutils.py — a simple custom module (imported by day13/lesson.py)

def celsius_to_fahrenheit(c):
    return c * 9/5 + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]

def greet(name):
    return f"Hello, {name}! Welcome to Python modules."

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

if __name__ == "__main__":
    # This only runs when the file is executed directly, not when imported.
    print("myutils self-test:")
    print(celsius_to_fahrenheit(0))
    print(is_palindrome("racecar"))
    print(clamp(15, 0, 10))
