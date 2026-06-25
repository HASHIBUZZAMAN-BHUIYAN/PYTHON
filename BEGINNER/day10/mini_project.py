# Day 10 Mini-Project — Robust Calculator with History
# Demonstrates error handling in a real interactive program.

class CalcError(Exception): pass

def calculate(expr):
    """Parse and evaluate a simple 'a op b' expression string."""
    parts = expr.strip().split()
    if len(parts) != 3:
        raise CalcError("Format: <number> <op> <number>  e.g.  3 + 4")
    a_str, op, b_str = parts
    try:
        a = float(a_str)
        b = float(b_str)
    except ValueError:
        raise CalcError(f"'{a_str}' or '{b_str}' is not a valid number")
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/":
        if b == 0: raise ZeroDivisionError("Division by zero")
        return a / b
    if op == "**": return a ** b
    if op == "%":
        if b == 0: raise ZeroDivisionError("Modulo by zero")
        return a % b
    raise CalcError(f"Unknown operator '{op}'. Use +, -, *, /, **, %")

history = []
print("=== Robust Calculator ===  (type 'history' or 'quit')")

while True:
    try:
        expr = input("> ").strip()
        if expr.lower() == "quit":
            break
        if expr.lower() == "history":
            if not history:
                print("No history yet.")
            else:
                for line in history[-10:]:
                    print(" ", line)
            continue
        result = calculate(expr)
        line = f"{expr} = {result}"
        history.append(line)
        print(line)
    except (CalcError, ZeroDivisionError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
