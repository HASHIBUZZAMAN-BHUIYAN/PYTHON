# ─────────────────────────────────────────────
#  SECTION 3: Arithmetic Operators
# ─────────────────────────────────────────────

print("\n=== Arithmetic ===")

x, y = 17, 5

print(f"{x} + {y}  = {x + y}")
print(f"{x} - {y}  = {x - y}")
print(f"{x} * {y}  = {x * y}")
print(f"{x} / {y}  = {x / y}")       # true division → float
print(f"{x} // {y} = {x // y}")      # floor division → int  ← used in ML indexing
print(f"{x} % {y}  = {x % y}")       # modulo/remainder
print(f"{x} ** {y} = {x ** y}")      # power ← used in math formulas everywhere

# Real AI use case: neuron layer size calculation
input_size   = 784       # 28x28 pixel image (MNIST dataset)
hidden_units = 128
output_size  = 10        # 10 digits (0–9)

total_params = (input_size * hidden_units) + hidden_units + \
               (hidden_units * output_size) + output_size

print(f"\nSimple neural net parameter count: {total_params:,}")
