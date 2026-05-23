# ============================================================
#  DAY 1 — Python Basics Practice
#  Your 2-Month AI/Robotics Journey Starts Here
#  Type every line. Run with F5. Break things. Fix them.
# ============================================================


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


# ─────────────────────────────────────────────
#  SECTION 4: String Operations
# ─────────────────────────────────────────────

print("\n=== Strings ===")

sentence = "  Artificial Intelligence is the Future  "

print(sentence.strip())                    # remove whitespace
print(sentence.strip().lower())            # lowercase
print(sentence.strip().upper())            # uppercase
print(sentence.strip().split())            # split into list of words
print(sentence.strip().replace("Future", "Present"))
print(f"Length: {len(sentence.strip())}")
print(f"Starts with 'Art': {sentence.strip().startswith('Art')}")
print(f"Word count: {len(sentence.strip().split())}")

# Slicing — like cutting a tensor in deep learning
s = "DeepLearning"
print(f"\nFull: {s}")
print(f"First 4: {s[:4]}")
print(f"Last 8: {s[4:]}")
print(f"Every 2nd letter: {s[::2]}")
print(f"Reversed: {s[::-1]}")

# Join — used heavily in NLP tokenization
tokens = ["I", "love", "Python", "for", "AI"]
sentence_rebuilt = " ".join(tokens)
print(f"\nJoined: {sentence_rebuilt}")


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


# ─────────────────────────────────────────────
#  SECTION 6: User Input (interactive)
# ─────────────────────────────────────────────

print("\n=== User Input ===")

user_name  = input("Enter your name: ")
birth_year = int(input("Enter your birth year: "))

current_year   = 2026
age_calculated = current_year - birth_year
days_alive     = age_calculated * 365

print(f"\nWelcome, {user_name}!")
print(f"You are approximately {age_calculated} years old.")
print(f"That's roughly {days_alive:,} days of experience — now add Python!")


# ─────────────────────────────────────────────
#  SECTION 7: Mini Projects
# ─────────────────────────────────────────────

print("\n=== Mini Project 1: BMI Calculator ===")

weight_kg = float(input("Your weight in kg: "))
height_m  = float(input("Your height in meters (e.g. 1.75): "))

bmi = weight_kg / (height_m ** 2)

print(f"BMI = {bmi:.2f}")

if bmi < 18.5:
    category = "Underweight"
elif bmi < 25.0:
    category = "Normal weight"
elif bmi < 30.0:
    category = "Overweight"
else:
    category = "Obese"

print(f"Category: {category}")


print("\n=== Mini Project 2: AI Model Accuracy Checker ===")

# Simulating what a real ML evaluation script does
predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]   # model output
ground_truth = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]   # correct labels

correct = 0
for pred, truth in zip(predictions, ground_truth):
    if pred == truth:
        correct += 1

accuracy = correct / len(predictions) * 100
print(f"Predictions : {predictions}")
print(f"Ground Truth: {ground_truth}")
print(f"Correct: {correct}/{len(predictions)}")
print(f"Accuracy: {accuracy:.1f}%")


print("\n=== Mini Project 3: Simple Encoder ===")

# Caesar cipher — foundation of understanding encryption in secure AI systems
message = input("Enter a message to encode: ")
shift   = 3

encoded = ""
for char in message:
    if char.isalpha():
        base  = ord('A') if char.isupper() else ord('a')
        encoded += chr((ord(char) - base + shift) % 26 + base)
    else:
        encoded += char

print(f"Original : {message}")
print(f"Encoded  : {encoded}")


# ─────────────────────────────────────────────
#  CHALLENGES — Do these BEFORE moving to Day 2
# ─────────────────────────────────────────────
#
#  CHALLENGE 1:
#    Write a temperature converter. Ask the user for a temperature
#    and whether it's in Celsius or Fahrenheit, then convert it.
#    Formula C→F: F = C * 9/5 + 32
#    Formula F→C: C = (F - 32) * 5/9
#
#  CHALLENGE 2:
#    Write a "robot power calculator". A robot has:
#      - battery_voltage = 24.0 (volts)
#      - current_draw    = 15.0 (amps)
#    Calculate: power = voltage * current (watts)
#    Calculate: runtime_hours if battery_capacity = 100 (Wh)
#    runtime = battery_capacity / power
#    Print a full status report using f-strings.
#
#  CHALLENGE 3:
#    Take a full name as input (e.g. "Elon Musk").
#    Print: initials (E.M.), reversed name, name in ALL CAPS,
#    number of characters (no spaces), and the name repeated 3x
#    with a dash separator: "Elon Musk-Elon Musk-Elon Musk"
#
#  When done, type: "Day 1 complete, give me Day 2"
# ─────────────────────────────────────────────

print("\n Day 1 loaded. Start from Section 1 and work down.")
print(" Type everything. Run often. Break things on purpose.")
print(" Complete all 3 challenges before Day 2.\n")