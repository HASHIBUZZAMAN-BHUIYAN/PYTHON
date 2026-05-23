# ─────────────────────────────────────────────
#  SECTION 2: Type Conversion (Casting)
# ─────────────────────────────────────────────

print("\n=== Type Conversion ===")

a = int("42")           # string → int
b = float("3.14")       # string → float
c = str(100)            # int → string
d = bool(0)             # 0 → False
e = bool(99)            # non-zero → True

print(f"int('42')    = {a}  | type: {type(a)}")
print(f"float('3.14')= {b}  | type: {type(b)}")
print(f"str(100)     = {c}  | type: {type(c)}")
print(f"bool(0)      = {d}  | type: {type(d)}")
print(f"bool(99)     = {e}  | type: {type(e)}")
