# ─────────────────────────────────────────────
#  SECTION 5: Comparison & Logic
# ─────────────────────────────────────────────

print("\n=== Comparisons ===")

accuracy = 0.94
threshold = 0.90

print(f"accuracy > threshold : {accuracy > threshold}")
print(f"accuracy == 1.0      : {accuracy == 1.0}")
print(f"accuracy >= threshold: {accuracy >= threshold}")

# Logical operators — used in training conditions
is_trained    = True
has_gpu       = False
loss_low      = True

print(f"\nReady to deploy (trained AND loss low): {is_trained and loss_low}")
print(f"Can train fast (trained OR has gpu)   : {is_trained or has_gpu}")
print(f"NOT has_gpu                           : {not has_gpu}")

# Chained comparisons — very Pythonic
val = 0.75
print(f"\n0.5 < {val} < 1.0 : {0.5 < val < 1.0}")