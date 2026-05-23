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
