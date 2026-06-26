# Day 10 — Error Handling

# ─── 1. UNHANDLED EXCEPTION ──────────────────────────────────────────────────
# print(1 / 0)   # ZeroDivisionError — crashes if uncommented

# ─── 2. BASIC try / except ───────────────────────────────────────────────────
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Can't divide by zero!")

# ─── 3. CATCH SPECIFIC EXCEPTIONS ───────────────────────────────────────────
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

print(safe_divide(10, 2))
print(safe_divide(10, 0))

# ─── 4. MULTIPLE EXCEPT CLAUSES ──────────────────────────────────────────────
def parse_number(s):
    try:
        return int(s)
    except ValueError:
        print(f"'{s}' is not a valid integer")
        return None
    except TypeError:
        print("Expected a string, got something else")
        return None

print(parse_number("42"))
print(parse_number("abc"))
print(parse_number(None))

# ─── 5. CATCHING THE EXCEPTION OBJECT ────────────────────────────────────────
try:
    x = int("not_a_number")
except ValueError as e:
    print(f"ValueError caught: {e}")

# ─── 6. except Exception — catch-all (avoid broad catches in production) ─────
try:
    data = [1, 2, 3]
    print(data[10])
except Exception as e:
    print(f"Something went wrong: {type(e).__name__}: {e}")

# ─── 7. else CLAUSE — runs if NO exception ───────────────────────────────────
try:
    num = int("123")
except ValueError:
    print("Bad input")
else:
    print(f"Parsed successfully: {num}")

# ─── 8. finally — always runs ────────────────────────────────────────────────
def read_file(path):
    f = None
    try:
        f = open(path, "r")
        return f.read()
    except FileNotFoundError:
        return "File not found"
    finally:
        if f:
            f.close()
        print("finally ran")

print(read_file("nonexistent.txt"))

# ─── 9. RAISING EXCEPTIONS ───────────────────────────────────────────────────
def set_age(age):
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0 or age > 150:
        raise ValueError(f"Age {age} is out of realistic range")
    return age

try:
    set_age(-5)
except ValueError as e:
    print(e)

try:
    set_age("twenty")
except TypeError as e:
    print(e)

# ─── 10. CUSTOM EXCEPTIONS ───────────────────────────────────────────────────
class InsufficientFundsError(Exception):
    def __init__(self, amount, balance):
        self.amount  = amount
        self.balance = balance
        super().__init__(f"Cannot withdraw ${amount:.2f}: balance is ${balance:.2f}")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(amount, self.balance)
        self.balance -= amount
        return amount

acct = BankAccount(100)
try:
    acct.withdraw(150)
except InsufficientFundsError as e:
    print(e)

# ─── 11. COMMON BUILT-IN EXCEPTIONS ─────────────────────────────────────────
# ValueError    — right type, wrong value (int("abc"))
# TypeError     — wrong type for operation
# IndexError    — list[99] when list has fewer items
# KeyError      — dict["missing_key"]
# FileNotFoundError — open("no_such_file.txt")
# AttributeError    — "hello".nonexistent()
# ZeroDivisionError — 1/0
# NameError         — using undefined variable
