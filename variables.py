# ─────────────────────────────────────────────
#  SECTION 1: Variables & Data Types
# ─────────────────────────────────────────────

name       = "PyBot"          # str
age        = 1                # int
version    = 3.11             # float
is_ready   = True             # bool
gpu        = None             # NoneType

print("=== Variables ===")
print(type(name))
print(type(age))
print(type(version))
print(type(is_ready))
print(type(gpu))

# f-string formatting (you will use this 1000x in AI code)
print(f"\nHello! I am {name}, version {version}, age {age}.")
print(f"AI Ready: {is_ready} | GPU: {gpu}")