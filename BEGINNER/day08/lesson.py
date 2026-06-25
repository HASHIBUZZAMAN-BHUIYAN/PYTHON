# Day 08 — Functions

# ─── 1. DEFINING AND CALLING ─────────────────────────────────────────────────
def greet(name):
    """Return a greeting string."""
    return f"Hello, {name}!"

print(greet("Alice"))

def add(a, b):
    return a + b

result = add(3, 4)
print(result)   # 7

# ─── 2. DEFAULT PARAMETERS ───────────────────────────────────────────────────
def power(base, exp=2):
    return base ** exp

print(power(3))     # 9  (exp defaults to 2)
print(power(2, 10)) # 1024

# ─── 3. KEYWORD ARGUMENTS ────────────────────────────────────────────────────
def introduce(name, age, city="Unknown"):
    print(f"{name}, {age}, from {city}")

introduce("Bob", 30, city="Dhaka")
introduce(age=25, name="Alice")   # order doesn't matter with keywords

# ─── 4. *args — VARIABLE POSITIONAL ARGUMENTS ────────────────────────────────
def total(*args):
    return sum(args)

print(total(1, 2, 3, 4, 5))   # 15 — args is a tuple

# ─── 5. **kwargs — VARIABLE KEYWORD ARGUMENTS ────────────────────────────────
def show_info(**kwargs):
    for key, val in kwargs.items():
        print(f"  {key}: {val}")

show_info(name="Alice", score=95, passed=True)

# ─── 6. COMBINING ALL PARAMETER TYPES ────────────────────────────────────────
# Order: positional, *args, keyword-only defaults, **kwargs
def mixed(a, b, *args, sep=", ", **kwargs):
    print(f"a={a}, b={b}, extra={args}, sep={sep!r}, extra_kw={kwargs}")

mixed(1, 2, 3, 4, sep=" | ", x=10)

# ─── 7. RETURNING MULTIPLE VALUES ────────────────────────────────────────────
def min_max(lst):
    return min(lst), max(lst)   # returns a tuple

lo, hi = min_max([3, 1, 4, 1, 5, 9])
print(lo, hi)

# ─── 8. SCOPE — LEGB RULE ────────────────────────────────────────────────────
# Local → Enclosing → Global → Built-in
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)    # local
    inner()
    print(x)        # enclosing

outer()
print(x)            # global

# global keyword (rarely needed — prefer passing/returning values)
counter = 0
def increment():
    global counter
    counter += 1

increment()
increment()
print(counter)   # 2

# ─── 9. LAMBDA FUNCTIONS ─────────────────────────────────────────────────────
# One-expression anonymous function
square = lambda x: x ** 2
print(square(5))   # 25

# Useful as a sort key
words = ["banana", "fig", "apple", "cherry"]
words.sort(key=lambda w: len(w))
print(words)    # ['fig', 'fig'..] sorted by length

# ─── 10. RECURSION ───────────────────────────────────────────────────────────
def factorial(n):
    if n <= 1:        # base case
        return 1
    return n * factorial(n - 1)   # recursive case

print(factorial(6))   # 720

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print([fibonacci(i) for i in range(10)])  # [0,1,1,2,3,5,8,13,21,34]
