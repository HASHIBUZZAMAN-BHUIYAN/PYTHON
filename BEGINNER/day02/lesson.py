# Day 02 — Operators, Expressions, Type Casting

# ─── 1. ARITHMETIC OPERATORS ─────────────────────────────────────────────────
a, b = 17, 5
print(a + b)    # 22  addition
print(a - b)    # 12  subtraction
print(a * b)    # 85  multiplication
print(a / b)    # 3.4 true division (always float)
print(a // b)   # 3   floor division (integer quotient)
print(a % b)    # 2   modulus (remainder)
print(a ** b)   # 1419857  exponentiation

# ─── 2. AUGMENTED ASSIGNMENT ─────────────────────────────────────────────────
x = 10
x += 3
x -= 2
x *= 4
x //= 5
x **= 2
print(x)  # 64

# ─── 3. COMPARISON OPERATORS (return bool) ───────────────────────────────────
print(7 == 7)   # True
print(7 != 8)   # True
print(9 > 5)    # True
print(3 < 2)    # False
print(5 >= 5)   # True
print(4 <= 3)   # False

# ─── 4. LOGICAL OPERATORS ────────────────────────────────────────────────────
# and  – both must be True
# or   – at least one must be True
# not  – flips True/False
print(True and False)   # False
print(True or False)    # True
print(not True)         # False

age = 20
has_ticket = True
can_enter = age >= 18 and has_ticket
print(f"Can enter: {can_enter}")

# ─── 5. OPERATOR PRECEDENCE ──────────────────────────────────────────────────
# PEMDAS/BODMAS: ** then */% then +- then comparisons then logical
result = 2 + 3 * 4 ** 2 - 1   # 2 + 3*16 - 1 = 49
print(result)                  # 49
print((2 + 3) * 4)             # parentheses first → 20

# ─── 6. TYPE CASTING ─────────────────────────────────────────────────────────
# int()
print(int(3.9))     # 3  — truncates, does NOT round
print(int("42"))    # 42
print(int(True))    # 1
print(int(False))   # 0

# float()
print(float(7))     # 7.0
print(float("3.14"))# 3.14

# str()
print(str(100))     # "100"
print(str(9.5))     # "9.5"

# bool()
# Falsy values: 0, 0.0, "", None, [], {}, ()
print(bool(0))      # False
print(bool(""))     # False
print(bool("hi"))   # True
print(bool(42))     # True

# ─── 7. COMMON PITFALL ───────────────────────────────────────────────────────
# input() returns a string — always cast before arithmetic
raw = "5"
print(int(raw) + 3)  # 8  ✓

# ─── 8. USEFUL BUILT-INS ─────────────────────────────────────────────────────
print(abs(-7))        # 7   absolute value
print(round(3.567, 2))# 3.57
print(max(4, 9, 2))   # 9
print(min(4, 9, 2))   # 2
print(divmod(17, 5))  # (3, 2)  quotient and remainder at once
