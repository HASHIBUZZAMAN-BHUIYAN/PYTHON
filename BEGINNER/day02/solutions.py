# Day 02 — Solutions

# Exercise 1
n1 = float(input("First number: "))
op = input("Operator (+,-,*,/): ")
n2 = float(input("Second number: "))
if op == '+':   print(n1 + n2)
elif op == '-': print(n1 - n2)
elif op == '*': print(n1 * n2)
elif op == '/':
    if n2 == 0: print("Error: division by zero")
    else:       print(n1 / n2)
else:           print("Unknown operator")

# Exercise 2
n = int(input("Enter an integer: "))
print("Even" if n % 2 == 0 else "Odd")

# Exercise 3 answers
# 10 + 2*3 = 16
# (10+2)*3 = 36
# 2**(3**2) = 2**9 = 512  (right-associative)
# 10%3+1 = 1+1 = 2

# Exercise 4
value = "3.7"
print(int(float(value) * 2))  # 7

# Exercise 5
weight = float(input("Weight (kg): "))
height = float(input("Height (m): "))
bmi = weight / height ** 2
if bmi < 18.5:          cat = "Underweight"
elif bmi < 25:          cat = "Normal"
else:                   cat = "Overweight"
print(f"BMI: {bmi:.1f} — {cat}")
